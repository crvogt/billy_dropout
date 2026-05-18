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

# Conditions that share the variance-aware perception block. Centralized
# so the runtime branch in `query_perception`, the dispatch helpers in
# this file, and the perceiver-keying loop in `scripts/run_experiment.py`
# cannot drift on which conditions feed the MC Dropout pipeline.
VARIANCE_CONDITIONS: tuple[str, ...] = ("variance_aware", "variance_aware_free")
ALL_CONDITIONS: tuple[str, ...] = ("baseline",) + VARIANCE_CONDITIONS

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
# Variance-aware FREE prompt — mission-framed, no prescribed mapping.
#
# Companion to SYSTEM_PROMPT_VARIANCE (directive). The pair tests two
# propagation styles for uncertainty into a small LLM agent:
#   - directive: a calibrated mechanical mapping the LLM executes;
#   - free:      the same posterior surfaces, but the LLM exercises
#                discretion under mission framing.
#
# The directive condition was designed in response to a 4B pilot where
# narrative variance prompting failed to produce gate-conditional motion
# (see III.D of paper_notes.md). The free condition exists to defuse the
# methodological objection that the directive reduces the LLM to a
# switch statement. If free produces signal-driven behavior comparable
# to directive, that is the paper's strongest result; if it does not,
# the directive vs free contrast itself is the contribution.
#
# Implementation note: the per-turn DETECTION block for this condition
# is byte-identical to the directive condition's (mean_confidence +
# epistemic_variance + interpretation label). Only this system prompt
# differs. The interpretation label is rendered for the LLM but no
# mapping is supplied; the LLM is told explicitly that no mapping is
# prescribed.
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_VARIANCE_FREE = """\
<|think|>
You are an autonomous ground vehicle. Your mission is to find a
doorway, approach it, and pass through.

For each perception cycle, you receive output from an onboard door
detector and must choose one action.

The detector reports three values about each detection:
- mean_confidence: the detector's average confidence across K=20
  stochastic forward passes
- epistemic_variance: the variance of those confidence estimates
  across passes — a measure of how much the passes disagreed
- interpretation: a coarse summary (LOW, MEDIUM, or HIGH variance)
  computed by thresholds on epistemic_variance

These describe how the detector behaved on this particular image.
How you use these signals to choose an action is your decision.
There is no prescribed mapping from variance level to action.

Available actions:
- move_forward(distance_m): commit to forward motion
- rotate(angle_deg): turn in place
- look_around(): request a panoramic re-perception
- defer(reason): explicit hold; end the cycle without acting
- report(message): verbal observation. Use only when explicitly
  asked to make a verbal observation, not as a stalling action.

Choose the action that best serves the mission. Consider that
committing to motion on a wrong detection is harder to recover
from than re-sensing; that re-sensing or deferring slows mission
progress; and that the detector's reliability varies across images.
The balance is yours to strike.

Always emit a <think>...</think> block before your tool call. State
your reasoning about the perception, your interpretation of the
detector signals, and why this action is the right choice given the
mission. Required for every event.
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
    condition: Literal["baseline", "variance_aware", "variance_aware_free"],
    gate: GateDecision | None = None,
) -> str:
    """Render a detection block for the agent's prompt context.

    Single source of truth for the experimental conditions' perception
    strings. The header (DETECTION/label/bbox) is byte-identical between
    conditions; only the tail differs.

    ``variance_aware`` and ``variance_aware_free`` produce byte-identical
    output — the conditions differ only in the system prompt (directive
    mapping vs free-form discretion under mission framing). Both require
    a ``Posterior`` and a ``GateDecision``.

    Args:
        detection: ``Detection`` for ``baseline``, ``Posterior`` for
            ``variance_aware`` / ``variance_aware_free``. ``None`` yields
            the no-detection block.
        condition: ``"baseline"``, ``"variance_aware"``, or
            ``"variance_aware_free"``.
        gate: required when ``condition`` is a variance-aware variant and
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

    if condition in VARIANCE_CONDITIONS:
        if not isinstance(detection, Posterior):
            raise TypeError(
                f"{condition} condition requires Posterior; "
                f"got {type(detection).__name__}"
            )
        if gate is None:
            raise ValueError(
                f"{condition} condition requires a GateDecision (gate=None)"
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
    post: Posterior | None,
    gate: GateDecision | None,
    condition: Literal["variance_aware", "variance_aware_free"] = "variance_aware",
) -> str:
    """Render a Posterior + GateDecision for a variance-aware condition.

    Both ``variance_aware`` and ``variance_aware_free`` produce identical
    detection-block output (see ``build_perception_prompt``). The
    ``condition`` parameter is accepted for clarity at call sites and
    routed through unchanged.
    """
    if post is None:
        return build_perception_prompt(None, condition)
    return build_perception_prompt(post, condition, gate=gate)


def system_prompt(condition: str) -> str:
    """Return the system prompt for the given condition.

    Args:
        condition: "baseline", "variance_aware", or "variance_aware_free".
    """
    if condition == "baseline":
        return SYSTEM_PROMPT_BASE
    if condition == "variance_aware":
        # Standalone, not BASE + addendum. See the SYSTEM_PROMPT_VARIANCE
        # header comment for the methodological rationale.
        return SYSTEM_PROMPT_VARIANCE
    if condition == "variance_aware_free":
        return SYSTEM_PROMPT_VARIANCE_FREE
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
