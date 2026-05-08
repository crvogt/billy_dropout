"""Temperature scaling for the deterministic baseline.

`TemperatureScaler.fit` minimizes NLL on a held-out (logits, labels)
set via L-BFGS over a scalar T. `transform` applies it to a single
post-sigmoid binary confidence by routing through logit-space:

    logit = log(p / (1 - p))
    p_cal = sigmoid(logit / T)

`expected_calibration_error` is standard 15-bin ECE used by both the
calibration test and the reliability diagram in the paper figures.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


class TemperatureScaler:
    def __init__(self, temperature: float = 1.0) -> None:
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        self.temperature = float(temperature)

    @classmethod
    def fit(
        cls,
        logits: "np.ndarray",
        labels: "np.ndarray",
        max_iter: int = 200,
    ) -> "TemperatureScaler":
        """Fit T on a held-out set by minimizing cross-entropy NLL.

        Args:
            logits: shape (n,) for binary or (n, n_classes) for multi-class.
                    Binary inputs are promoted to 2-class by stacking against zero.
            labels: shape (n,) int in [0, n_classes-1].
            max_iter: L-BFGS iteration cap.
        """
        import numpy as np
        import torch

        logits_a = np.asarray(logits, dtype=np.float32)
        labels_a = np.asarray(labels, dtype=np.int64)

        if logits_a.ndim == 1:
            # Binary → 2-class with zero anchor for class 0.
            logits_a = np.stack([np.zeros_like(logits_a), logits_a], axis=1)
        elif logits_a.ndim != 2:
            raise ValueError(f"expected logits of rank 1 or 2, got {logits_a.ndim}")

        logits_t = torch.from_numpy(logits_a)
        labels_t = torch.from_numpy(labels_a)

        # Optimize log_T so T = exp(log_T) is positive by construction.
        # Direct-on-T parameterization let L-BFGS overshoot below 0 on
        # imbalanced detection-calibration sets.
        log_T = torch.nn.Parameter(torch.log(torch.tensor(1.5, dtype=torch.float32)))
        optimizer = torch.optim.LBFGS([log_T], lr=0.1, max_iter=max_iter)
        loss_fn = torch.nn.CrossEntropyLoss()

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            T = torch.exp(log_T)
            loss = loss_fn(logits_t / T, labels_t)
            loss.backward()
            return loss

        optimizer.step(closure)
        return cls(temperature=float(torch.exp(log_T).detach().item()))

    def transform(self, confidence: float) -> float:
        """Apply T to a single binary confidence (sigmoid-space)."""
        eps = 1e-7
        c = max(eps, min(1.0 - eps, float(confidence)))
        logit = math.log(c / (1.0 - c))
        return 1.0 / (1.0 + math.exp(-logit / self.temperature))


def expected_calibration_error(
    confidences: "np.ndarray",
    correct: "np.ndarray",
    n_bins: int = 15,
) -> float:
    """Standard ECE over binned (mean confidence, accuracy) pairs."""
    import numpy as np

    confidences = np.asarray(confidences, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)
    n = confidences.shape[0]
    if n == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == n_bins - 1:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences >= lo) & (confidences < hi)
        m = mask.sum()
        if m == 0:
            continue
        bin_acc = correct[mask].mean()
        bin_conf = confidences[mask].mean()
        ece += (m / n) * abs(bin_acc - bin_conf)
    return float(ece)
