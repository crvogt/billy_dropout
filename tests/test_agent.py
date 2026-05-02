"""Agent unit tests with a mock LLM and mock perceiver.

Three real tests:

  1. Mock LLM emitting `defer(...)` is routed to terminal state in one turn.
  2. Mock LLM emitting `move_forward(0.3)` produces a non-terminal turn,
     and the next turn's prompt contains the previous tool result feedback.
  3. Variance-aware prompt rendering produces a different DETECTION block
     than baseline rendering on the same posterior input. (LLM-free.)
"""

from __future__ import annotations

import numpy as np
from langchain_core.messages import AIMessage

from uagent.agent.prompts import (
    gate_from_variance,
    render_baseline_detection,
    render_variance_detection,
)
from uagent.agent.runtime import build_runtime
from uagent.perception.posterior import Detection, Posterior


# ---------------------------------------------------------------------------
# Mock LLM and Mock Perceiver
# ---------------------------------------------------------------------------


class MockLLM:
    """Returns a predetermined sequence of tool calls.

    Each entry in `tool_call_sequence` is `(name, args)` or
    `("text", str)` to simulate a no-tool-call turn.
    """

    def __init__(self, tool_call_sequence: list[tuple[str, dict | str]]) -> None:
        self.tool_call_sequence = list(tool_call_sequence)
        self.idx = 0
        self.invocations: list[list] = []

    def bind_tools(self, tools: list) -> "MockLLM":
        return self

    def invoke(self, messages: list) -> AIMessage:
        if self.idx >= len(self.tool_call_sequence):
            raise RuntimeError(f"MockLLM ran out of responses at idx {self.idx}")
        self.invocations.append(messages)
        name, payload = self.tool_call_sequence[self.idx]
        self.idx += 1
        if name == "text":
            return AIMessage(content=str(payload))
        return AIMessage(
            content="",
            tool_calls=[{"name": name, "args": dict(payload), "id": f"mock_{self.idx}"}],
        )


class MockBaselinePerceiver:
    def __init__(self, detections: list[Detection]) -> None:
        self.detections = list(detections)

    def predict(self, image: np.ndarray) -> list[Detection]:
        return self.detections


class MockVariancePerceiver:
    def __init__(self, posteriors: list[Posterior]) -> None:
        self.posteriors = list(posteriors)

    def predict(self, image: np.ndarray) -> list[Posterior]:
        return self.posteriors


THRESHOLDS = {"low_max": 0.05, "high_min": 0.15}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_defer_terminates_turn() -> None:
    """LLM emits a single defer call → graph reaches done in one reasoning turn."""
    llm = MockLLM([("defer", {"reason": "perception too uncertain"})])
    perceiver = MockVariancePerceiver([
        Posterior(label="robot", bbox=(10, 10, 50, 50),
                  mean_confidence=0.4, epistemic_variance=0.2, K=20),
    ])

    runtime = build_runtime(
        condition="variance_aware",
        perceiver=perceiver,
        llm=llm,
        variance_thresholds=THRESHOLDS,
        max_turns=4,
        live_tools=False,
    )
    result = runtime.invoke({
        "image": np.zeros((10, 10, 3), dtype=np.uint8),
        "user_command": "test",
    })

    assert result["done"] is True
    assert result["final_action"] == "defer"
    assert result["final_args"] == {"reason": "perception too uncertain"}
    assert result["turn"] == 1
    assert llm.idx == 1
    # Variance-aware system prompt should mention the high-variance rule.
    sys_msg_text = str(llm.invocations[0][0].content)
    assert "epistemic_variance" in sys_msg_text


def test_move_forward_loops_with_feedback() -> None:
    """Non-terminal tool → second turn sees the tool result in the prompt."""
    llm = MockLLM([
        ("move_forward", {"distance_m": 0.3}),
        ("report", {"message": "arrived"}),
    ])
    perceiver = MockBaselinePerceiver([
        Detection(label="robot", bbox=(100, 100, 200, 200), confidence=0.85),
    ])

    runtime = build_runtime(
        condition="baseline",
        perceiver=perceiver,
        llm=llm,
        variance_thresholds=THRESHOLDS,
        max_turns=4,
        live_tools=False,
    )
    result = runtime.invoke({
        "image": np.zeros((10, 10, 3), dtype=np.uint8),
        "user_command": "approach the robot",
    })

    assert result["done"] is True
    assert result["final_action"] == "report"
    assert result["turn"] == 2
    assert llm.idx == 2

    # Second invocation's user message must contain feedback about the prior tool call.
    second_user_msg = str(llm.invocations[1][1].content)
    assert "Last tool result" in second_user_msg
    assert "move_forward" in second_user_msg
    assert "0.30 m" in second_user_msg


def test_prompt_conditions_differ_on_same_posterior() -> None:
    """Pure prompt-rendering test; no LLM, no graph."""
    det = Detection(label="robot", bbox=(320, 240, 480, 400), confidence=0.78)
    post = Posterior(
        label="robot", bbox=(320, 240, 480, 400),
        mean_confidence=0.78, epistemic_variance=0.18, K=20,
    )
    gate = gate_from_variance(post.epistemic_variance, low_max=0.05, high_min=0.15)

    base = render_baseline_detection(det)
    var = render_variance_detection(post, gate)

    assert "confidence: 0.78" in base
    assert "mean_confidence: 0.78" in var
    assert "epistemic_variance: 0.180" in var
    assert "HIGH variance" in var
    assert base != var
