"""Data classes for perception output.

`Posterior` is the canonical artifact the agent prompt consumes. Field
shape is fixed by the paper's experimental design — see prompt template
in uagent/agent/prompts.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

BBox = tuple[float, float, float, float]   # (x1, y1, x2, y2) in pixel coords


@dataclass(frozen=True)
class Detection:
    """Single forward-pass detection from a deterministic YOLO call."""

    label: str
    bbox: BBox
    confidence: float                       # post-softmax / post-sigmoid score


@dataclass(frozen=True)
class Posterior:
    """Aggregated MC Dropout posterior over K stochastic forward passes."""

    label: str
    bbox: BBox                              # (x1, y1, x2, y2) — mean of K passes
    mean_confidence: float                  # in [0, 1] — mean of K class scores
    epistemic_variance: float               # variance of K class scores
    K: int                                  # number of passes used


VarianceLevel = Literal["LOW", "MEDIUM", "HIGH"]


@dataclass(frozen=True)
class GateDecision:
    """Interpretation of a Posterior's epistemic_variance into a coarse signal.

    Used only by the variance_aware prompt condition to render the
    `interpretation` line. The agent consumes the human-readable text;
    the level + thresholds let downstream metrics audit the gate.
    """

    level: VarianceLevel
    epistemic_variance: float
    low_max: float                          # threshold from config
    high_min: float                         # threshold from config
    interpretation: str                     # the rendered prompt line
