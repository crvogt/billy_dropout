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

from typing import Literal

from uagent.perception.posterior import Detection, GateDecision, Posterior

# ---------------------------------------------------------------------------
# Shared prefix — identical across conditions.
#
# Two thinking-mode signals are present:
#
#   1. The literal `<|think|>` control token at the very start of the
#      system message. Per the Gemma 4 spec, this enables structured
#      thinking via the `<|channel>thought ... <channel|>` envelope on
#      compliant inference paths. Kept as a forward-compatibility hedge.
#
#   2. An explicit verbal directive to emit a literal `<think>...</think>`
#      block on every turn. The sandbox in `uagent/sandbox/gemma4_thinking/`
#      established that `<|think|>` alone is NOT respected by the e2b/e4b
#      Gemma 4 variants over Ollama's HTTP surface, while the verbal
#      `<think>` instruction is. Both surface forms are recognized by
#      `uagent.agent.parsing.parse_thought_channel` (strict envelope first,
#      then literal `<think>` as a fallback).
#
# The directive must remain in the BASE prompt — both conditions need
# reasoning chains for the experiment's reasoning-coverage success
# criterion. Variance-specific language stays out of the base entirely;
# only the addendum below introduces variance / epistemic / gate
# vocabulary, preserving baseline's variance-blindness.
#
# Refs: https://ollama.com/library/gemma4
#       https://ai.google.dev/gemma/docs/core/prompt-formatting-gemma4
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_BASE = """\
<|think|>
You control a wheeled mobile robot. Your task: navigate toward a
doorway when one is visible. You receive perception results from a
camera-based detector and decide what action to take next.

Reasoning is required. Before any tool call OR final reply, emit a
literal `<think>...</think>` block (these exact characters, with the
opening and closing tags) containing one to three sentences of your
reasoning. The block is mandatory on every turn, including turns
where the decision seems obvious; do not omit it even on a confident
commit. Omitting it produces a malformed event that fails the
experiment's reasoning-coverage check. After the closing </think>
tag, emit exactly one tool call (or, for `report`, the final message).

Available actions (call exactly one per turn):
  - move_forward(distance_m: float)   move forward by distance_m meters
  - rotate(angle_deg: float)          rotate in place; positive = right
  - look_around()                     capture a panoramic view (re-sense)
  - report(message: str)              speak verbally; no robot motion
  - defer(reason: str)                explicit HOLD; end the turn

When to choose each action — these are distinct cases:

  - move_forward / rotate: on detections you have reason to trust —
    adequate confidence. Commit to the motion; repeated re-sensing
    on a clear detection wastes time.

  - look_around: when a different viewing angle would disambiguate
    the detection — partial occlusion, a detection at the edge of
    the frame, or a single mid-confidence fire whose framing suggests
    a wider view could disambiguate. `look_around` is NOT a default
    delay action; only call it when a new viewpoint is the specific
    fix. The `look_around` tool performs a panoramic visual scan from
    the current position. Do not chain `rotate` calls to manually
    search for a different angle when `look_around` is the dedicated
    tool for that purpose.

  - defer: when confidence is very low (below ~0.4) and no
    additional viewing angle would help disambiguate the scene. End
    the turn rather than commit motion on unreliable data. When both
    `look_around` and `defer` could apply, prefer `defer`;
    `look_around` is for mid-confidence ambiguity.

  - report: a verbal observation; no robot motion. Use sparingly.

Output exactly one tool call. Outside the `<think>` block, emit no
additional natural language unless you choose `report`. Decisiveness
on trustworthy detections is part of the job; caution is for
genuinely ambiguous inputs, not the default. If the detection is
missing or empty (no doorway in view), call `look_around` to search.
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
in particular, prefer `defer` (or `look_around` if a new viewpoint would
specifically help) over `move_forward` / `rotate` when variance is HIGH.
Verbalize your reasoning about the variance signal in your `report` or
`defer` message when relevant.

When the gate level is LOW AND `mean_confidence` is adequate, the
detector's confidence is well-calibrated and you should commit to the
appropriate motion action. LOW gate does not override the low-confidence
defer rule. If `mean_confidence` is very low (below ~0.4), defer — even
when variance is LOW. Caution is reserved for MEDIUM and HIGH gate levels.
"""

# ---------------------------------------------------------------------------
# Per-turn detection block — single source of truth.
#
# The *only* substantive difference between the two conditions is the
# condition-specific tail (baseline = confidence-only; variance_aware =
# mean_confidence + variance + interpretation). The header lines
# (DETECTION:, label, bbox) are byte-identical between conditions by
# construction — they come from the same _DETECTION_HEADER template,
# formatted with the same field-format strings.
#
# Any future drift in formatting, whitespace, or field ordering between
# conditions must happen here, in this one file. The two `render_*`
# helpers below are thin wrappers and MUST NOT be changed to bypass
# `build_perception_prompt`.
# ---------------------------------------------------------------------------

_DETECTION_HEADER = """\
DETECTION:
  label: {label}
  bbox: [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]
"""

_BASELINE_TAIL = "  confidence: {confidence:.2f}\n"

_VARIANCE_TAIL = """\
  mean_confidence: {mean_confidence:.2f}
  epistemic_variance: {epistemic_variance:.3f}
  interpretation: {interpretation}
"""

_NO_DETECTION_BLOCK = "DETECTION:\n  (none — detector returned no objects)\n"


def build_perception_prompt(
    detection: Detection | Posterior | None,
    condition: Literal["baseline", "variance_aware"],
    gate: GateDecision | None = None,
) -> str:
    """Render a detection block for the agent's prompt context.

    Single source of truth for the two experimental conditions' perception
    strings. The header (DETECTION/label/bbox) is byte-identical between
    conditions; only the tail differs.

    Args:
        detection: ``Detection`` for ``baseline``, ``Posterior`` for
            ``variance_aware``. ``None`` yields the no-detection block.
        condition: ``"baseline"`` or ``"variance_aware"``.
        gate: required when ``condition == "variance_aware"`` and
            ``detection`` is not None; ignored otherwise.
    """
    if detection is None:
        return _NO_DETECTION_BLOCK

    x1, y1, x2, y2 = detection.bbox
    header = _DETECTION_HEADER.format(
        label=detection.label, x1=x1, y1=y1, x2=x2, y2=y2
    )

    if condition == "baseline":
        if not isinstance(detection, Detection):
            raise TypeError(
                f"baseline condition requires Detection; "
                f"got {type(detection).__name__}"
            )
        return header + _BASELINE_TAIL.format(confidence=detection.confidence)

    if condition == "variance_aware":
        if not isinstance(detection, Posterior):
            raise TypeError(
                f"variance_aware condition requires Posterior; "
                f"got {type(detection).__name__}"
            )
        if gate is None:
            raise ValueError(
                "variance_aware condition requires a GateDecision (gate=None)"
            )
        return header + _VARIANCE_TAIL.format(
            mean_confidence=detection.mean_confidence,
            epistemic_variance=detection.epistemic_variance,
            interpretation=gate.interpretation,
        )

    raise ValueError(f"unknown condition: {condition!r}")


# ---------------------------------------------------------------------------
# Backward-compat thin wrappers. MUST delegate to build_perception_prompt
# so the source-of-truth invariant holds across the codebase.
# ---------------------------------------------------------------------------


def render_baseline_detection(det: Detection | None) -> str:
    """Render a Detection for the baseline condition (delegates to build_perception_prompt)."""
    return build_perception_prompt(det, "baseline")


def render_variance_detection(
    post: Posterior | None, gate: GateDecision | None
) -> str:
    """Render a Posterior + GateDecision for variance_aware (delegates to build_perception_prompt)."""
    if post is None:
        return build_perception_prompt(None, "variance_aware")
    return build_perception_prompt(post, "variance_aware", gate=gate)


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
