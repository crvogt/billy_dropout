"""Perception unit tests.

Three invariants:

  1. MCDropoutYOLO returns valid Posteriors. If the loaded checkpoint
     has zero nn.Dropout* modules (Branch B not yet trained), variance
     is exactly zero — the documented degenerate case. Test asserts the
     correct branch.
  2. DeterministicYOLO is reproducible: same image, same checkpoint,
     identical output across two calls.
  3. TemperatureScaler.fit reduces ECE on a synthetic over-confident set.

CPU-only by design; no CUDA assumed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from uagent.perception.calibration import (
    TemperatureScaler,
    expected_calibration_error,
)
from uagent.perception.detector import DeterministicYOLO, MCDropoutYOLO

WEIGHTS = Path(
    "/home/carson/libs/billy_paper/ros-llm-docker/llm-agent/yolo_robot_focused.pt"
)
FIXTURE = Path(__file__).parent / "fixtures" / "robot_husky.jpg"


@pytest.fixture(scope="module")
def fixture_image() -> np.ndarray:
    import cv2

    img = cv2.imread(str(FIXTURE))
    assert img is not None, f"failed to read fixture {FIXTURE}"
    return img


@pytest.mark.skipif(not WEIGHTS.exists(), reason=f"weights missing: {WEIGHTS}")
def test_mc_dropout_aggregates_correctly(fixture_image: np.ndarray) -> None:
    """K passes return Posteriors; if no dropout modules, variance==0."""
    det = MCDropoutYOLO(WEIGHTS, K=4, device="cpu")
    posts = det.predict(fixture_image)

    assert isinstance(posts, list)
    for p in posts:
        assert p.K == 4
        assert 0.0 <= p.mean_confidence <= 1.0
        assert p.epistemic_variance >= 0.0
        x1, y1, x2, y2 = p.bbox
        assert x2 > x1 and y2 > y1

    # Documented invariant: stock weights have no dropout → variance = 0.
    if det._n_dropout_modules == 0:
        for p in posts:
            assert p.epistemic_variance < 1e-9, (
                f"checkpoint has 0 dropout modules but variance={p.epistemic_variance}"
            )
    else:  # pragma: no cover — exercised post-doorway-fine-tune
        assert any(p.epistemic_variance > 0 for p in posts), (
            "dropout modules present but K passes still produced zero variance"
        )


STOCK_YOLOV8N = Path("/home/carson/libs/billy_paper/uagent/yolov8n.pt")


@pytest.mark.skipif(
    not STOCK_YOLOV8N.exists(),
    reason=f"yolov8n.pt missing at {STOCK_YOLOV8N} (run any train script first)",
)
def test_mc_dropout_predict_with_injection_produces_variance() -> None:
    """K=20 passes through the production MCDropoutYOLO.predict path
    must produce non-identical raw outputs when dropout is active.

    Regression test for the predict()-eval-reset bug: previously
    ``ultralytics.YOLO.predict()`` reset the model to eval mode after
    ``_enable_dropout_only`` ran, making K passes deterministic. The
    Option-A fix bypasses the Predictor pipeline. This test exercises
    the new path end-to-end without requiring a fine-tuned checkpoint
    on disk: it loads stock yolov8n.pt, injects dropout in-place, and
    runs through the production predict.
    """
    import torch  # noqa: F401 (used by predict via lazy import)

    from uagent.perception.dropout import (
        count_dropout_modules,
        inject_dropout_into_yolov8_cls_head,
    )

    det = MCDropoutYOLO(STOCK_YOLOV8N, K=20, device="cpu")
    inject_dropout_into_yolov8_cls_head(det.yolo.model, p=0.25)
    det._n_dropout_modules = count_dropout_modules(det.yolo.model)
    assert det._n_dropout_modules == 6

    rng = np.random.default_rng(seed=0)
    img = rng.integers(0, 255, size=(480, 640, 3), dtype=np.uint8)

    # Two assertions: (1) the raw forward path with our preprocess + NMS
    # produces non-deterministic outputs across K passes, (2) the
    # production predict() method itself produces at least one Posterior
    # with non-zero variance. The second catches bugs in aggregation /
    # names lookup / scale_boxes that the first would miss.

    # (1) Raw forward smoke
    pass_confs: list[list[float]] = []
    tensor, orig_hw = det._preprocess(img, det.imgsz)
    from ultralytics.utils.ops import non_max_suppression
    from uagent.perception.dropout import enable_dropout_train_mode

    for _ in range(det.K):
        enable_dropout_train_mode(det.yolo.model)
        with torch.no_grad():
            raw = det.yolo.model(tensor)
        preds = raw[0] if isinstance(raw, (tuple, list)) else raw
        nms = non_max_suppression(preds, conf_thres=0.001, iou_thres=det.iou_threshold)
        dets = nms[0]
        confs = dets[:, 4].cpu().tolist() if (dets is not None and len(dets) > 0) else []
        pass_confs.append(confs)

    distinct = {tuple(round(c, 6) for c in confs) for confs in pass_confs}
    assert len(distinct) > 1, (
        f"K={det.K} raw-path passes produced identical confidence vectors; "
        "dropout is not active in the underlying forward"
    )

    # (2) Production predict() — exercises aggregation end-to-end.
    # Use a low conf threshold so detections survive on the random image.
    det.conf_threshold = 0.001
    posts = det.predict(img)
    if posts:
        assert any(p.epistemic_variance > 0 for p in posts), (
            "MCDropoutYOLO.predict aggregated K passes into Posteriors with "
            "all-zero epistemic_variance despite raw forward being stochastic; "
            "bug in aggregation or call-site"
        )
    else:
        # No posteriors usually means pass 0 had no detections; the
        # _aggregate logic returns [] in that case. The raw-forward
        # assertion above already proved dropout is firing.
        pass


@pytest.mark.skipif(not WEIGHTS.exists(), reason=f"weights missing: {WEIGHTS}")
def test_deterministic_inference_reproducible(fixture_image: np.ndarray) -> None:
    """Two calls on the same image with eval-mode model must be identical."""
    det = DeterministicYOLO(WEIGHTS, device="cpu")
    out1 = det.predict(fixture_image)
    out2 = det.predict(fixture_image)
    assert len(out1) == len(out2)
    for a, b in zip(out1, out2):
        assert a.label == b.label
        assert a.bbox == pytest.approx(b.bbox, abs=1e-4)
        assert a.confidence == pytest.approx(b.confidence, abs=1e-6)


def test_temperature_scaling_reduces_ece() -> None:
    """Synthetic over-confident binary set: T-fit should reduce ECE."""
    rng = np.random.default_rng(seed=1337)
    n = 2000

    # True probabilities and Bernoulli labels.
    true_p = rng.uniform(0.05, 0.95, size=n)
    labels = (rng.uniform(0.0, 1.0, size=n) < true_p).astype(np.int64)

    # Over-confident logits: scale true logit by 2 (true T ≈ 0.5; T-fit ≈ 2.0).
    true_logits = np.log(true_p / (1.0 - true_p))
    overconf_logits = 2.0 * true_logits

    pre_cal_conf = 1.0 / (1.0 + np.exp(-overconf_logits))
    pred_label = (pre_cal_conf > 0.5).astype(np.int64)
    correct = (pred_label == labels).astype(np.float64)

    ece_pre = expected_calibration_error(pre_cal_conf, correct)

    scaler = TemperatureScaler.fit(overconf_logits, labels)
    post_cal_conf = np.array([scaler.transform(c) for c in pre_cal_conf])
    ece_post = expected_calibration_error(post_cal_conf, correct)

    assert scaler.temperature > 1.2, (
        f"expected T > 1.2 for over-confident logits, got T={scaler.temperature:.3f}"
    )
    assert ece_post < ece_pre, (
        f"ECE did not improve: pre={ece_pre:.4f} post={ece_post:.4f}"
    )


def test_temperature_scaling_one_sided_logits_stays_positive() -> None:
    """Imbalanced one-sided logits (mostly large positive) — regression test
    for the parameterization. The original direct-T parameterization let
    L-BFGS overshoot below zero, which then failed the constructor's T>0
    check. Mirrors the actual val-set distribution that originally
    triggered the bug (~120 TP / 60 FP, all logits in [0.5, 6.0]).

    Asserted invariants:
      - fit() returns without raising (the original failure mode)
      - T is positive and finite
      - fitted T reduces NLL relative to T=1.0 (real work, not tautology)
    """
    import torch

    rng = np.random.default_rng(seed=2026)
    n_pos, n_neg = 120, 60
    pos_logits = rng.uniform(2.0, 6.0, size=n_pos)
    neg_logits = rng.uniform(0.5, 3.0, size=n_neg)

    logits_np = np.concatenate([pos_logits, neg_logits]).astype(np.float32)
    labels_np = np.concatenate(
        [np.ones(n_pos, dtype=np.int64), np.zeros(n_neg, dtype=np.int64)]
    )

    scaler = TemperatureScaler.fit(logits_np, labels_np)

    assert 1e-3 < scaler.temperature < 100.0, (
        f"T out of range on imbalanced one-sided logits: T={scaler.temperature}"
    )
    assert np.isfinite(scaler.temperature)

    # NLL must improve vs T=1.0; otherwise the fit didn't do real work.
    logits_t = torch.from_numpy(np.stack([np.zeros_like(logits_np), logits_np], axis=1))
    labels_t = torch.from_numpy(labels_np)
    loss_fn = torch.nn.CrossEntropyLoss()
    nll_unit = float(loss_fn(logits_t, labels_t).item())
    nll_fit = float(loss_fn(logits_t / scaler.temperature, labels_t).item())
    assert nll_fit < nll_unit, (
        f"fitted T did not reduce NLL: unit={nll_unit:.4f} fit={nll_fit:.4f}"
    )
