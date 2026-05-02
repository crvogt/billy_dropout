"""Temperature scaling for the deterministic baseline.

Calibrates the `DeterministicYOLO` point-estimate confidences against a
held-out validation split (robot_dataset/val/), minimizing NLL. The
`baseline` prompt condition reads post-calibration confidence; this is
what makes the baseline a fair comparator (it is not "raw YOLO scores"
but "best-effort calibrated point estimate").

Implementation lands in step 3.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


class TemperatureScaler:
    """Single scalar temperature T; calibrated_p = softmax(logits / T)."""

    def __init__(self, temperature: float = 1.0) -> None:
        raise NotImplementedError("calibration is implemented in step 3")

    @classmethod
    def fit(
        cls,
        logits: "np.ndarray",
        labels: "np.ndarray",
        n_bins: int = 15,
        max_iter: int = 200,
    ) -> "TemperatureScaler":
        """Fit T on a held-out set by minimizing NLL."""
        raise NotImplementedError

    def transform(self, confidence: float) -> float:
        """Apply temperature to a single post-softmax confidence."""
        raise NotImplementedError


def expected_calibration_error(
    confidences: "np.ndarray",
    correct: "np.ndarray",
    n_bins: int = 15,
) -> float:
    """Standard ECE — used by the calibration test and reliability diagram."""
    raise NotImplementedError("calibration is implemented in step 3")
