"""Single-agent LangGraph DAG.

Nodes:

    receive_command  →  query_perception  →  agent_reason  →  dispatch
                                                  ↑              ↓
                                                  └──── feedback ┘

One LLM call per `agent_reason` invocation. Tools are bound at the
LLM layer; `dispatch` executes whichever tool the LLM emitted, captures
the result, and either terminates the turn (report / defer) or feeds
back into `agent_reason` for another turn (capped by max_turns).

The DAG is condition-agnostic; the only thing that varies between
`baseline` and `variance_aware` is the prompt rendering inside
`query_perception` and the system prompt assembled in `agent_reason`.

`build_runtime` accepts the LLM and perceiver as injected dependencies.
For tests, callers pass a MockLLM and a MockPerceiver. For the live
experiment, callers pass a `langchain_ollama.ChatOllama` and a
`MCDropoutYOLO` / `DeterministicYOLO`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Literal, Protocol, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from uagent.agent.prompts import (
    gate_from_variance,
    render_baseline_detection,
    render_variance_detection,
    system_prompt,
)
from uagent.agent.tools import LLM_TOOLS, TOOL_REGISTRY, ToolResult
from uagent.perception.posterior import Detection, Posterior

if TYPE_CHECKING:
    import numpy as np


Condition = Literal["baseline", "variance_aware"]


class _Perceiver(Protocol):
    def predict(self, image: "np.ndarray") -> list:  # noqa: D401 - interface only
        ...


class _LLM(Protocol):
    def bind_tools(self, tools: list) -> "_LLM": ...
    def invoke(self, messages: list) -> AIMessage: ...


class State(TypedDict, total=False):
    """LangGraph state — flat dict so langgraph can checkpoint cleanly.

    `image` is typed `Any` because langgraph evaluates type hints via
    `get_type_hints` at build time and can't resolve forward-refs to
    optionally-imported names (numpy under TYPE_CHECKING).
    """

    image: Any                          # numpy.ndarray BGR frame at runtime
    user_command: str
    condition: Condition
    perception_block: str
    posterior_meta: dict[str, Any]
    messages: list                      # list[BaseMessage]
    last_tool_result: ToolResult | None
    turn: int
    max_turns: int
    done: bool
    final_action: str
    final_args: dict[str, Any]
    reasoning_chain: str


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def build_runtime(
    *,
    condition: Condition,
    perceiver: _Perceiver,
    llm: _LLM,
    variance_thresholds: dict[str, float],
    max_turns: int = 4,
    live_tools: bool = False,
) -> Any:
    """Compile a LangGraph runtime for one condition.

    Args:
        condition: "baseline" or "variance_aware".
        perceiver: object with .predict(image) returning list[Detection]
            (baseline) or list[Posterior] (variance_aware).
        llm: object with .bind_tools(tools) → llm and .invoke(messages)
            → AIMessage (mock or ChatOllama).
        variance_thresholds: {"low_max": float, "high_min": float}.
        max_turns: hard cap on reasoning turns per command.
        live_tools: if True, tool wrappers route through ROS; else mock.
    """

    if condition not in ("baseline", "variance_aware"):
        raise ValueError(f"unknown condition: {condition!r}")

    sys_text = system_prompt(condition)
    low_max = float(variance_thresholds["low_max"])
    high_min = float(variance_thresholds["high_min"])

    # StructuredTool shells with friendly names — what the LLM emits.
    bound_llm = llm.bind_tools(LLM_TOOLS)

    # ---------- nodes ----------

    def receive_command(state: State) -> dict:
        return {
            "turn": 0,
            "done": False,
            "messages": [],
            "last_tool_result": None,
            "max_turns": max_turns,
            "condition": condition,
            "reasoning_chain": "",
        }

    def query_perception(state: State) -> dict:
        image = state["image"]
        results = perceiver.predict(image)

        if condition == "variance_aware":
            top: Posterior | None = max(
                results, key=lambda p: p.mean_confidence, default=None
            ) if results else None
            if top is None:
                return {
                    "perception_block": render_variance_detection(None, None),
                    "posterior_meta": {},
                }
            gate = gate_from_variance(
                top.epistemic_variance, low_max=low_max, high_min=high_min,
                posterior_label=top.label,
            )
            return {
                "perception_block": render_variance_detection(top, gate),
                "posterior_meta": {
                    "label": top.label,
                    "bbox": list(top.bbox),
                    "mean_confidence": top.mean_confidence,
                    "epistemic_variance": top.epistemic_variance,
                    "K": top.K,
                    "gate_level": gate.level,
                },
            }

        # baseline
        top_d: Detection | None = max(
            results, key=lambda d: d.confidence, default=None
        ) if results else None
        if top_d is None:
            return {
                "perception_block": render_baseline_detection(None),
                "posterior_meta": {},
            }
        return {
            "perception_block": render_baseline_detection(top_d),
            "posterior_meta": {
                "label": top_d.label,
                "bbox": list(top_d.bbox),
                "confidence": top_d.confidence,
            },
        }

    def agent_reason(state: State) -> dict:
        feedback_block = ""
        if state.get("last_tool_result") is not None:
            tr: ToolResult = state["last_tool_result"]    # type: ignore[assignment]
            feedback_block = (
                f"\n\nLast tool result: {tr.tool_name} → {tr.detail} "
                f"(ok={tr.ok})"
            )

        user_content = (
            f"{state.get('user_command', '')}\n\n"
            f"{state.get('perception_block', '')}"
            f"{feedback_block}"
        )

        messages = [SystemMessage(sys_text), HumanMessage(user_content)]
        response: AIMessage = bound_llm.invoke(messages)
        chain = state.get("reasoning_chain", "")
        if response.content:
            chain = (chain + "\n" if chain else "") + str(response.content)
        return {
            "messages": state.get("messages", []) + [response],
            "reasoning_chain": chain,
        }

    def dispatch(state: State) -> dict:
        last = state["messages"][-1]
        tool_calls = getattr(last, "tool_calls", None) or []

        if not tool_calls:
            # No tool call → treat as implicit defer with the model's text as reason.
            from uagent.agent.tools import DeferInput, defer as defer_fn
            text = str(last.content) if getattr(last, "content", None) else "no tool call emitted"
            res = defer_fn(DeferInput(reason=text), live=live_tools)
            new_turn = state["turn"] + 1
            return {
                "last_tool_result": res,
                "turn": new_turn,
                "done": True,
                "final_action": "defer",
                "final_args": {"reason": text},
            }

        # Take the first tool call.
        tc = tool_calls[0]
        tool_name = tc["name"]
        tool_args = dict(tc.get("args", {}))

        if tool_name not in TOOL_REGISTRY:
            res = ToolResult(
                tool_name=tool_name, risk_tier="low", ok=False,
                detail=f"unknown tool {tool_name!r}", terminates_turn=False,
            )
            new_turn = state["turn"] + 1
            done = new_turn >= state.get("max_turns", max_turns)
            return {
                "last_tool_result": res,
                "turn": new_turn,
                "done": done,
                **({"final_action": "error", "final_args": {"tool": tool_name}} if done else {}),
            }

        spec = TOOL_REGISTRY[tool_name]
        try:
            validated = spec.input_schema(**tool_args)
        except Exception as e:  # validation error
            res = ToolResult(
                tool_name=tool_name, risk_tier=spec.risk_tier, ok=False,
                detail=f"tool args invalid: {e}", terminates_turn=False,
            )
            new_turn = state["turn"] + 1
            done = new_turn >= state.get("max_turns", max_turns)
            return {
                "last_tool_result": res, "turn": new_turn, "done": done,
                **({"final_action": "error", "final_args": tool_args} if done else {}),
            }

        result: ToolResult = spec.handler(validated, live=live_tools)
        new_turn = state["turn"] + 1
        terminate = result.terminates_turn or new_turn >= state.get("max_turns", max_turns)
        update: dict[str, Any] = {
            "last_tool_result": result,
            "turn": new_turn,
            "done": terminate,
        }
        if terminate:
            update["final_action"] = tool_name
            update["final_args"] = tool_args
        return update

    def should_loop(state: State) -> str:
        return "end" if state.get("done") else "agent_reason"

    # ---------- assemble ----------

    g = StateGraph(State)
    g.add_node("receive_command", receive_command)
    g.add_node("query_perception", query_perception)
    g.add_node("agent_reason", agent_reason)
    g.add_node("dispatch", dispatch)
    g.set_entry_point("receive_command")
    g.add_edge("receive_command", "query_perception")
    g.add_edge("query_perception", "agent_reason")
    g.add_edge("agent_reason", "dispatch")
    g.add_conditional_edges(
        "dispatch", should_loop, {"end": END, "agent_reason": "agent_reason"}
    )
    return g.compile()
