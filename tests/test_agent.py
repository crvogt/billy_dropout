"""Agent unit tests with a mock LLM.

  1. Mock LLM emitting `defer(...)` is routed to terminal state in one turn.
  2. Mock LLM emitting `move_forward(0.3)` produces a non-terminal turn,
     and the next turn's prompt contains the previous tool result.
  3. Variance-aware prompt rendering produces a different DETECTION block
     than baseline rendering on the same posterior input.

Implementations land in step 4.
"""

from __future__ import annotations

import pytest


@pytest.mark.skip(reason="implemented in step 4")
def test_defer_terminates_turn() -> None:
    raise NotImplementedError


@pytest.mark.skip(reason="implemented in step 4")
def test_move_forward_loops_with_feedback() -> None:
    raise NotImplementedError


def test_prompt_conditions_differ_on_same_posterior() -> None:
    """Pure prompt-rendering test; needs no LLM. Real now, not stub."""
    from uagent.agent.prompts import (
        gate_from_variance,
        render_baseline_detection,
        render_variance_detection,
    )
    from uagent.perception.posterior import Detection, Posterior

    det = Detection(label="robot", bbox=(320, 240, 480, 400), confidence=0.78)
    post = Posterior(
        label="robot",
        bbox=(320, 240, 480, 400),
        mean_confidence=0.78,
        epistemic_variance=0.18,
        K=20,
    )
    gate = gate_from_variance(post.epistemic_variance, low_max=0.05, high_min=0.15)

    base = render_baseline_detection(det)
    var = render_variance_detection(post, gate)

    assert "confidence: 0.78" in base
    assert "mean_confidence: 0.78" in var
    assert "epistemic_variance: 0.180" in var
    assert "HIGH variance" in var
    assert base != var
