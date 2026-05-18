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
    SYSTEM_PROMPT_VARIANCE,
    SYSTEM_PROMPT_VARIANCE_FREE,
    gate_from_variance,
    render_baseline_detection,
    render_variance_detection,
    system_prompt,
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


# Arbitrary test thresholds for the runtime-level tests below. These are
# NOT the production thresholds in configs/default.yaml (which are
# 0.003 / 0.010, recalibrated 2026-05-11 against runs/step_b). The unit
# tests do not depend on the production values; they only exercise the
# gate logic on inputs they pick. The byte-identity test
# (test_variance_free_perception_block_matches_directive) sweeps the
# production thresholds explicitly.
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


def test_thought_channel_lands_in_reasoning_chain() -> None:
    """Mock LLM emits content with a thought block + a tool call; the
    runtime should extract the thought into reasoning_chain rather than
    storing the raw content. Closes the coverage gap created when the
    raw-content fallback was removed from agent_reason."""

    class _ThoughtLLM:
        def bind_tools(self, tools: list) -> "_ThoughtLLM":
            return self

        def invoke(self, messages: list) -> AIMessage:
            return AIMessage(
                content="<|channel>thought\nthe model thought this\n<channel|>",
                tool_calls=[{"name": "defer", "args": {"reason": "x"}, "id": "id1"}],
            )

    perceiver = MockBaselinePerceiver([
        Detection(label="robot", bbox=(100, 100, 200, 200), confidence=0.85),
    ])
    runtime = build_runtime(
        condition="baseline",
        perceiver=perceiver,
        llm=_ThoughtLLM(),
        variance_thresholds=THRESHOLDS,
        max_turns=4,
        live_tools=False,
    )
    result = runtime.invoke({
        "image": np.zeros((10, 10, 3), dtype=np.uint8),
        "user_command": "test",
    })
    assert result["reasoning_chain"] == "the model thought this"
    assert result["final_action"] == "defer"


def test_no_tool_call_strips_thought_from_defer_reason() -> None:
    """If the model emits a thought block but no tool call, the dispatcher's
    implicit-defer path must scrub the thought out of final_args.reason —
    otherwise internal reasoning leaks into a field other tooling treats as
    user-facing."""

    class _NoToolThoughtLLM:
        def bind_tools(self, tools: list) -> "_NoToolThoughtLLM":
            return self

        def invoke(self, messages: list) -> AIMessage:
            return AIMessage(
                content=(
                    "<|channel>thought\nsecret internal reasoning\n<channel|>\n"
                    "Visible final answer."
                ),
                tool_calls=[],
            )

    perceiver = MockBaselinePerceiver([
        Detection(label="robot", bbox=(100, 100, 200, 200), confidence=0.85),
    ])
    runtime = build_runtime(
        condition="baseline",
        perceiver=perceiver,
        llm=_NoToolThoughtLLM(),
        variance_thresholds=THRESHOLDS,
        max_turns=4,
        live_tools=False,
    )
    result = runtime.invoke({
        "image": np.zeros((10, 10, 3), dtype=np.uint8),
        "user_command": "test",
    })
    assert result["final_action"] == "defer"
    assert result["final_args"]["reason"] == "Visible final answer."
    # And the thought DID land in reasoning_chain.
    assert "secret internal reasoning" in result["reasoning_chain"]


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


def test_variance_free_perception_block_matches_directive() -> None:
    """The detection block for variance_aware_free MUST be byte-identical
    to variance_aware's at every gate level. The two conditions form a
    controlled comparison on system-prompt content alone; any drift in
    perception rendering — including a per-condition gate-string tail
    introduced by a future refactor — would confound the directive-vs-
    free contrast.
    """
    # Sweep one variance below low_max, one between, one above high_min
    # so all three interpretation labels are exercised. The labels differ
    # in length, so this also catches a renderer that padded one tail and
    # not the other.
    for variance, expected_level in [
        (0.001, "LOW"),
        (0.006, "MEDIUM"),
        (0.020, "HIGH"),
    ]:
        post = Posterior(
            label="door", bbox=(100, 80, 240, 360),
            mean_confidence=0.72, epistemic_variance=variance, K=20,
        )
        gate = gate_from_variance(variance, low_max=0.003, high_min=0.010)
        assert gate.level == expected_level

        directive = render_variance_detection(post, gate, condition="variance_aware")
        free = render_variance_detection(post, gate, condition="variance_aware_free")
        assert directive == free, (
            f"variance_aware and variance_aware_free diverged at gate "
            f"level {expected_level}; only the system prompt may differ"
        )
        # And pin that the interpretation tail actually appears in both.
        assert f"{expected_level} variance" in directive
        assert f"{expected_level} variance" in free

    # No-detection path also identical between the two.
    none_directive = render_variance_detection(None, None, condition="variance_aware")
    none_free = render_variance_detection(None, None, condition="variance_aware_free")
    assert none_directive == none_free


def test_variance_free_gate_none_raises_consistent_with_directive() -> None:
    """Both variance-aware conditions reject `gate=None` with a Posterior.
    If a future refactor special-cases one condition, that drift becomes
    a silent rendering bug — pinned here."""
    import pytest as _pt

    post = Posterior(
        label="door", bbox=(0, 0, 10, 10),
        mean_confidence=0.5, epistemic_variance=0.005, K=20,
    )
    with _pt.raises(ValueError):
        render_variance_detection(post, None, condition="variance_aware")
    with _pt.raises(ValueError):
        render_variance_detection(post, None, condition="variance_aware_free")


def test_system_prompt_returns_free_for_free_condition() -> None:
    """system_prompt('variance_aware_free') returns the free-form prompt,
    not the directive. Guards against an accidental string-equality
    fallthrough that would route both conditions to the same prompt."""
    assert system_prompt("variance_aware_free") == SYSTEM_PROMPT_VARIANCE_FREE
    assert system_prompt("variance_aware") == SYSTEM_PROMPT_VARIANCE
    assert SYSTEM_PROMPT_VARIANCE != SYSTEM_PROMPT_VARIANCE_FREE


def test_variance_free_prompt_excludes_directive_phrases() -> None:
    """The free prompt's entire reason for existing is that it does NOT
    prescribe a mapping. If a future edit copy-pastes directive phrasing
    into the free constant, the controlled comparison silently degrades
    into a near-duplicate of directive — this catches that at the
    constant level, not at runtime.

    The forbidden phrase list is the directive's load-bearing mapping
    vocabulary; broaden if directive ever rephrases."""
    forbidden = [
        "Action mapping by interpretation (mandatory)",
        "LOW              → emit",
        "MEDIUM           → emit",
        "HIGH             → emit",
        "Do not construct your own thresholds",
        "must not be used in",
    ]
    for phrase in forbidden:
        assert phrase not in SYSTEM_PROMPT_VARIANCE_FREE, (
            f"directive-only phrase {phrase!r} leaked into "
            f"SYSTEM_PROMPT_VARIANCE_FREE; the free condition must remain "
            f"discretionary"
        )
    # And the positive marker: the free prompt MUST explicitly disclaim a
    # mapping. If a future edit drops this sentence the condition is no
    # longer "free" in the methodological sense.
    assert "no prescribed mapping" in SYSTEM_PROMPT_VARIANCE_FREE.lower()


def test_runtime_accepts_variance_aware_free() -> None:
    """build_runtime accepts the new condition and routes a Posterior
    through the variance-aware perception branch."""
    llm = MockLLM([("defer", {"reason": "test"})])
    perceiver = MockVariancePerceiver([
        Posterior(label="door", bbox=(10, 10, 50, 50),
                  mean_confidence=0.4, epistemic_variance=0.2, K=20),
    ])
    runtime = build_runtime(
        condition="variance_aware_free",
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
    # Confirm the free-form system prompt (not the directive) is what
    # the LLM actually saw — the controlled-comparison invariant.
    sys_msg_text = str(llm.invocations[0][0].content)
    assert "There is no prescribed mapping" in sys_msg_text
    assert "Action mapping by interpretation (mandatory)" not in sys_msg_text
