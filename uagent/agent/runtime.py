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
`query_perception` → `agent_reason` boundary.

Implementation lands in step 4.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    import numpy as np

    from uagent.perception.detector import DeterministicYOLO, MCDropoutYOLO

from uagent.agent.tools import ToolResult


Condition = Literal["baseline", "variance_aware"]


@dataclass
class AgentState:
    """LangGraph state — all fields the DAG threads between nodes."""

    image: "np.ndarray | None" = None        # camera frame in BGR
    user_command: str = ""
    condition: Condition = "baseline"
    perception_block: str = ""               # rendered DETECTION text
    posterior_meta: dict[str, Any] = field(default_factory=dict)  # for logging
    messages: list[dict[str, Any]] = field(default_factory=list)
    last_tool_result: ToolResult | None = None
    turn: int = 0
    max_turns: int = 4
    done: bool = False
    final_action: str = ""                   # tool name that ended the turn
    final_args: dict[str, Any] = field(default_factory=dict)
    reasoning_chain: str = ""


def build_runtime(
    *,
    condition: Condition,
    perceiver: "MCDropoutYOLO | DeterministicYOLO",
    llm_model: str,
    llm_host: str,
    variance_thresholds: dict[str, float],
    max_turns: int = 4,
    live_tools: bool = False,
) -> object:
    """Compile the LangGraph runtime for a single condition.

    Returns a `langgraph.graph.CompiledGraph`. Caller invokes it with
    an `AgentState` populated with `image` and `user_command`; the
    graph runs until `done=True` and returns the terminal state.
    """
    raise NotImplementedError("agent runtime is implemented in step 4")
