"""Perception unit tests.

Three things to confirm:
  1. K stochastic passes produce non-trivial variance on a fixture image
     (skipped until Branch B dropout-trained weights exist).
  2. DeterministicYOLO is reproducible: same image, same checkpoint,
     same output across two calls.
  3. TemperatureScaler.fit reduces ECE on a synthetic miscalibrated set.

Implementations land in step 3.
"""

from __future__ import annotations

import pytest


@pytest.mark.skip(reason="implemented in step 3")
def test_dropout_passes_produce_variance() -> None:
    raise NotImplementedError


@pytest.mark.skip(reason="implemented in step 3")
def test_deterministic_inference_reproducible() -> None:
    raise NotImplementedError


@pytest.mark.skip(reason="implemented in step 3")
def test_temperature_scaling_reduces_ece() -> None:
    raise NotImplementedError
