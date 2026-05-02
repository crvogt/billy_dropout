"""YOLO wrappers — MC Dropout (K stochastic passes) and deterministic.

Both classes load the same checkpoint. `MCDropoutYOLO` forces dropout
modules into train() mode while keeping BatchNorm frozen, runs K passes,
and aggregates per-detection mean and variance over class scores.
`DeterministicYOLO` does a single eval-mode pass and returns a point
estimate; `calibration.TemperatureScaler` is applied downstream.

Implementations are filled in during step 3 of the build (see prompt
order-of-operations). Step 2 ships these as typed stubs.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from uagent.perception.posterior import Detection, Posterior

if TYPE_CHECKING:
    import numpy as np


class MCDropoutYOLO:
    """K-pass stochastic YOLO. Returns Posterior per detection.

    Args:
        weights_path: path to a .pt checkpoint that has nn.Dropout layers in
            its detection head (Branch B retrained weights). If passed a
            checkpoint without dropout, K passes degenerate to a single
            deterministic answer (epistemic_variance == 0); detector_test
            will surface this.
        K: number of stochastic forward passes per image.
        imgsz: square inference resolution.
        conf_threshold: post-aggregation confidence floor.
        iou_threshold: NMS IoU threshold.
        device: "cuda" or "cpu".
    """

    def __init__(
        self,
        weights_path: str | Path,
        K: int = 20,
        imgsz: int = 640,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: str = "cuda",
    ) -> None:
        raise NotImplementedError("perception layer is implemented in step 3")

    def predict(self, image: "np.ndarray") -> list[Posterior]:
        """Run K stochastic passes; return aggregated posteriors."""
        raise NotImplementedError


class DeterministicYOLO:
    """Single-pass YOLO. Returns Detection per result.

    Used as the calibrated point-estimate baseline. Pair with
    `calibration.TemperatureScaler` to produce the `confidence` field
    that the `baseline` prompt condition consumes.
    """

    def __init__(
        self,
        weights_path: str | Path,
        imgsz: int = 640,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: str = "cuda",
    ) -> None:
        raise NotImplementedError("perception layer is implemented in step 3")

    def predict(self, image: "np.ndarray") -> list[Detection]:
        """Run a single deterministic forward pass."""
        raise NotImplementedError
