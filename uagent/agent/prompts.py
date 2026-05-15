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
# Variance-aware system prompt — STANDALONE, replaces the base entirely.
#
# Architectural shift from the prior addendum-on-base design: an earlier
# 4B run with the addendum produced reasoning chains 100% of the time but
# LOW-gate ID-positive events fired `move_forward` only 48% of the time
# and HIGH-gate OOD events fired `defer` 0% of the time. The chains
# invented parallel thresholds from `mean_confidence` rather than using
# the calibrated interpretation label. Conclusion: small LLMs do not
# reliably reason ABOUT variance even when the variance signal is in
# the perception block; instead, give them an explicit mechanical
# mapping to execute.
#
# This is a deliberate methodological choice for the paper, NOT a hidden
# implementation detail — see docs/paper_notes.md (Section III.D, "Why
# the variance_aware prompt is explicit").
#
# The baseline prompt is unchanged: it remains a narrative prompt with
# no mechanical mapping, no variance vocabulary, and no gate references.
# This preserves the baseline's variance-blindness for the controlled
# comparison.
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_VARIANCE = """\
<|think|>
You are a robot navigation agent. You receive perception output from
a door detector and must select one tool to call.

Reasoning requirement:
Always emit a `<think>...</think>` block before your tool call. State
the interpretation level you observed, the action it maps to, and one
observation from the detection (bbox position, mean_confidence, etc.)
that you observed. One to two sentences. Required for every event.

Detection format:
  label: door
  bbox: [x1, y1, x2, y2]
  mean_confidence: <float 0-1>
  epistemic_variance: <float>
  interpretation: <LOW | MEDIUM | HIGH> variance — <description>

The `interpretation` field renders verbatim as (for example):
  interpretation: LOW variance — the detector's stochastic passes agreed. Treat this detection as confident.
The description text inside the `interpretation` field is
informational. The mapping table below is authoritative.

The interpretation label was calibrated against the detector's
posterior on held-out data. Do not construct your own thresholds
from `mean_confidence`; the interpretation already incorporates that
information.

Action mapping by interpretation (mandatory):

  LOW              → emit `move_forward(distance_m=0.5)`
  MEDIUM           → emit `look_around()`
  HIGH             → emit `defer(reason="HIGH variance — detection unreliable")`
  empty detection  → emit `look_around()`

Why this mapping:
  LOW interpretation means the detector's stochastic forward passes
  agreed. The detection is reliable. Act on it.

  MEDIUM interpretation means the passes showed moderate disagreement.
  Gather more information before committing.

  HIGH interpretation means the passes disagreed substantially. The
  detection is unreliable. Do not act on it.

  Empty detection means there is no signal to act on. `look_around`
  re-senses without committing motion.

The tools `rotate` and `report` are available but must not be used in
this condition.
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
        # Standalone, not BASE + addendum. See the SYSTEM_PROMPT_VARIANCE
        # header comment for the methodological rationale.
        return SYSTEM_PROMPT_VARIANCE
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
