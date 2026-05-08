"""Tests for dropout injection into the YOLOv8 detection head.

Three invariants:
  1. Injection adds exactly six Dropout2d modules (3 scales x 2 per cls branch).
  2. Each injected module has the requested probability.
  3. With dropouts forced into train mode and the rest of the model in eval,
     K stochastic forward passes on a fixture produce non-identical outputs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

# Stock COCO YOLOv8n weights — used as the architecture donor for testing.
# The doorway-fine-tuned checkpoint at models/doorway_yolov8n_dropout.pt
# does not exist yet; tests use the stock arch which is identical at the
# Detect head structure level.
STOCK_WEIGHTS = Path(
    "/home/carson/libs/billy_paper/ros-llm-docker/llm-agent/yolov8n.pt"
)


pytestmark = pytest.mark.skipif(
    not STOCK_WEIGHTS.exists(),
    reason=f"stock yolov8n.pt missing at {STOCK_WEIGHTS}",
)


@pytest.fixture(scope="module")
def detection_model():
    from ultralytics import YOLO

    return YOLO(str(STOCK_WEIGHTS)).model


def test_inject_adds_exactly_six_modules(detection_model) -> None:
    from uagent.perception.dropout import (
        count_dropout_modules,
        inject_dropout_into_yolov8_cls_head,
    )

    pre = count_dropout_modules(detection_model)
    n_added = inject_dropout_into_yolov8_cls_head(detection_model, p=0.25)
    post = count_dropout_modules(detection_model)

    assert n_added == 6, f"expected 6 modules added, got {n_added}"
    assert post - pre == 6, (
        f"Dropout2d count mismatch: {pre} -> {post} (delta {post - pre}, expected 6)"
    )


def test_inject_uses_requested_probability(detection_model) -> None:
    """Use a fresh model so this test does not depend on the previous one."""
    import torch.nn as nn
    from ultralytics import YOLO

    from uagent.perception.dropout import inject_dropout_into_yolov8_cls_head

    fresh = YOLO(str(STOCK_WEIGHTS)).model
    inject_dropout_into_yolov8_cls_head(fresh, p=0.25)

    ps = [m.p for m in fresh.modules() if isinstance(m, nn.Dropout2d)]
    assert ps == [0.25] * 6, f"expected six p=0.25 modules, got {ps}"


def _two_forwards(yolo, tensor):
    """Two raw forwards with dropout train mode forced before each."""
    import torch

    from uagent.perception.dropout import enable_dropout_train_mode

    def _flatten(out):
        if isinstance(out, (tuple, list)):
            return torch.cat([t.flatten() for t in out if isinstance(t, torch.Tensor)])
        return out.flatten()

    enable_dropout_train_mode(yolo.model)
    with torch.no_grad():
        a = _flatten(yolo.model(tensor))
    enable_dropout_train_mode(yolo.model)
    with torch.no_grad():
        b = _flatten(yolo.model(tensor))
    return a, b


def _make_input_tensor(seed: int = 0):
    import torch

    rng = np.random.default_rng(seed=seed)
    img_np = rng.integers(0, 255, size=(640, 640, 3), dtype=np.uint8)
    return (
        torch.from_numpy(img_np[..., ::-1].copy())
        .permute(2, 0, 1)
        .unsqueeze(0)
        .float()
        / 255.0
    )


def test_dropout_inference_produces_variance() -> None:
    """Two raw forward passes must differ when dropout is on.

    Calls ``yolo.model(tensor)`` directly rather than ``yolo.predict()``
    because ultralytics' Predictor calls ``model.eval()`` internally
    before forward, which silently resets our dropout modules to eval
    mode and produces deterministic outputs.
    """
    import torch
    from ultralytics import YOLO

    from uagent.perception.dropout import inject_dropout_into_yolov8_cls_head

    yolo = YOLO(str(STOCK_WEIGHTS))
    inject_dropout_into_yolov8_cls_head(yolo.model, p=0.25)

    a, b = _two_forwards(yolo, _make_input_tensor())
    assert not torch.allclose(a, b, atol=1e-6), (
        "raw forward outputs were byte-identical across K=2 passes; "
        "injected dropout is not firing"
    )


def test_no_dropout_means_no_variance() -> None:
    """Negative control: identical forward calls without injection must match.

    Without this, the variance test only proves "something is stochastic" —
    not specifically that the *injected* modules are. Both tests together
    establish causation: stock model => deterministic, injected model =>
    stochastic.
    """
    import torch
    from ultralytics import YOLO

    yolo = YOLO(str(STOCK_WEIGHTS))  # no injection

    a, b = _two_forwards(yolo, _make_input_tensor())
    assert torch.allclose(a, b, atol=1e-6), (
        "stock yolov8n.pt produced different outputs across two raw forwards; "
        "there is a stochastic source other than the injected dropout that "
        "would also pass test_dropout_inference_produces_variance"
    )
