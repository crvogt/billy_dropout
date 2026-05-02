"""Prompt templates for the two experimental conditions.

Stored separately so the diff between baseline and variance_aware is a
single readable file, not a buried f-string in runtime.py.

Both conditions share the same SYSTEM prompt prefix and the same tool
surface. They differ only in (a) the post-perception detection block
that gets injected into the user turn, and (b) a short addendum to the
system prompt for `variance_aware` that explicitly licenses uncertainty-
based deferral.
"""

from __future__ import annotations

from uagent.perception.posterior import Detection, GateDecision, Posterior

# ---------------------------------------------------------------------------
# Shared prefix — identical across conditions.
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_BASE = """\
You control a wheeled mobile robot. You receive perception results from a
camera-based detector and decide what action to take next.

Available actions (call exactly one per turn):
  - move_forward(distance_m: float)   move forward by distance_m meters
  - rotate(angle_deg: float)          rotate in place; positive = right
  - look_around()                     capture a panoramic view (re-sense)
  - report(message: str)              speak verbally; no robot motion
  - defer(reason: str)                explicit HOLD; end the turn

Rules:
  - Output exactly one tool call. No additional natural language unless
    you choose `report`.
  - If the detection is missing or empty, treat it as "no target visible".
  - Never act on a detection you do not trust. Prefer `defer` or
    `look_around` over committing to motion under doubt.
"""

# ---------------------------------------------------------------------------
# Variance-aware addendum — appended to the system prompt only in the
# variance_aware condition.
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_VARIANCE_ADDENDUM = """\

Each detection comes with a `mean_confidence` AND an `epistemic_variance`.
The variance is computed from K=20 stochastic forward passes of the
detector; HIGH variance means the detector's passes disagreed with each
other about whether the object is there. Treat HIGH-variance detections
as uncertain even when mean_confidence looks adequate. For motion actions
in particular, prefer `defer` or `look_around` over `move_forward` /
`rotate` when variance is HIGH. Verbalize your reasoning about the
variance signal in your `report` or `defer` message when relevant.
"""

# ---------------------------------------------------------------------------
# Per-turn detection block — the *only* substantive difference between
# the two conditions.
# ---------------------------------------------------------------------------

_BASELINE_DETECTION_TEMPLATE = """\
DETECTION:
  label: {label}
  bbox: [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]
  confidence: {confidence:.2f}
"""

_VARIANCE_DETECTION_TEMPLATE = """\
DETECTION:
  label: {label}
  bbox: [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]
  mean_confidence: {mean_confidence:.2f}
  epistemic_variance: {epistemic_variance:.3f}
  interpretation: {interpretation}
"""

_NO_DETECTION_BLOCK = "DETECTION:\n  (none — detector returned no objects)\n"


def render_baseline_detection(det: Detection | None) -> str:
    """Render a Detection for the baseline condition."""
    if det is None:
        return _NO_DETECTION_BLOCK
    x1, y1, x2, y2 = det.bbox
    return _BASELINE_DETECTION_TEMPLATE.format(
        label=det.label, x1=x1, y1=y1, x2=x2, y2=y2, confidence=det.confidence
    )


def render_variance_detection(post: Posterior | None, gate: GateDecision | None) -> str:
    """Render a Posterior + its GateDecision for the variance_aware condition."""
    if post is None or gate is None:
        return _NO_DETECTION_BLOCK
    x1, y1, x2, y2 = post.bbox
    return _VARIANCE_DETECTION_TEMPLATE.format(
        label=post.label,
        x1=x1, y1=y1, x2=x2, y2=y2,
        mean_confidence=post.mean_confidence,
        epistemic_variance=post.epistemic_variance,
        interpretation=gate.interpretation,
    )


def system_prompt(condition: str) -> str:
    """Return the system prompt for the given condition.

    Args:
        condition: "baseline" or "variance_aware".
    """
    if condition == "baseline":
        return SYSTEM_PROMPT_BASE
    if condition == "variance_aware":
        return SYSTEM_PROMPT_BASE + SYSTEM_PROMPT_VARIANCE_ADDENDUM
    raise ValueError(f"unknown condition: {condition!r}")


# ---------------------------------------------------------------------------
# Variance → coarse signal mapping. Thresholds are config-driven; this
# function only does the rendering.
# ---------------------------------------------------------------------------

_INTERPRETATIONS = {
    "LOW":    "LOW variance — the detector's stochastic passes agreed. Treat this detection as confident.",
    "MEDIUM": "MEDIUM variance — the detector's stochastic passes partially disagreed. Treat this detection with caution.",
    "HIGH":   "HIGH variance — the detector's stochastic forward passes disagreed substantially. Treat this detection as uncertain.",
}


def gate_from_variance(
    epistemic_variance: float,
    low_max: float,
    high_min: float,
    posterior_label: str = "",
) -> GateDecision:
    """Map a raw variance value to a GateDecision (level + rendered string)."""
    if epistemic_variance <= low_max:
        level = "LOW"
    elif epistemic_variance >= high_min:
        level = "HIGH"
    else:
        level = "MEDIUM"
    return GateDecision(
        level=level,
        epistemic_variance=epistemic_variance,
        low_max=low_max,
        high_min=high_min,
        interpretation=_INTERPRETATIONS[level],
    )
