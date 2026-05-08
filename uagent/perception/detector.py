"""YOLO wrappers — MC Dropout (K stochastic passes) and deterministic.

Both classes load the same checkpoint via ultralytics. `MCDropoutYOLO`
forces nn.Dropout* modules into train() mode while keeping BatchNorm
frozen, runs K passes of post-NMS detection, and aggregates per-cluster
mean and variance over class scores. `DeterministicYOLO` does a single
eval-mode pass and returns a point estimate; `calibration.TemperatureScaler`
is applied downstream.

If a checkpoint without dropout layers is passed, K passes degenerate
to identical outputs and `epistemic_variance` is exactly 0. The
detector test asserts this invariant explicitly.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from uagent.perception.posterior import Detection, Posterior

if TYPE_CHECKING:
    import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_device(requested: str) -> str:
    """If 'cuda' is requested but unavailable, fall back to CPU with a warning.

    Avoids a confusing ``RuntimeError`` on the first ``.to(device)`` call
    when running on a dev box without working CUDA. Used by both detector
    constructors.
    """
    import warnings

    import torch

    if requested == "cuda" and not torch.cuda.is_available():
        warnings.warn(
            "device='cuda' requested but torch.cuda.is_available() is False; "
            "falling back to device='cpu'.",
            RuntimeWarning,
            stacklevel=2,
        )
        return "cpu"
    return requested


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    a_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    b_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = a_area + b_area - inter
    return inter / union if union > 0 else 0.0


def _extract_detections(result: Any) -> list[dict]:
    """Pull (label, bbox, confidence) from one ultralytics Results object."""
    out: list[dict] = []
    if result.boxes is None:
        return out
    names = result.names
    for box in result.boxes:
        cls_idx = int(box.cls[0].item())
        out.append({
            "label": names[cls_idx],
            "bbox": tuple(float(v) for v in box.xyxy[0].cpu().numpy().tolist()),
            "confidence": float(box.conf[0].item()),
        })
    return out


# ---------------------------------------------------------------------------
# MC Dropout YOLO
# ---------------------------------------------------------------------------


class MCDropoutYOLO:
    """K-pass stochastic YOLO. Returns Posterior per clustered detection.

    Aggregation: pass 0 defines reference clusters. Each subsequent
    pass's detections are matched into clusters by (label match) AND
    (IoU > iou_match_threshold). Passes that produce no matching
    detection contribute confidence=0 to that cluster — absence is
    information.
    """

    def __init__(
        self,
        weights_path: str | Path,
        K: int = 20,
        imgsz: int = 640,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        iou_match_threshold: float = 0.5,
        device: str = "cuda",
    ) -> None:
        from ultralytics import YOLO

        self.weights_path = Path(weights_path)
        self.K = int(K)
        self.imgsz = int(imgsz)
        self.conf_threshold = float(conf_threshold)
        self.iou_threshold = float(iou_threshold)
        self.iou_match_threshold = float(iou_match_threshold)
        self.device = str(device)

        if self.imgsz % 32 != 0:
            raise ValueError(
                f"imgsz must be a multiple of 32 for YOLOv8 stride alignment; got {self.imgsz}"
            )

        self.device = _resolve_device(self.device)

        self.yolo = YOLO(str(self.weights_path))
        self._n_dropout_modules: int = self._count_dropout_modules()

    def _count_dropout_modules(self) -> int:
        import torch.nn as nn

        n = 0
        for m in self.yolo.model.modules():
            if isinstance(m, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d, nn.AlphaDropout)):
                n += 1
        return n

    def _enable_dropout_only(self) -> None:
        """Eval mode for everything; train mode for Dropout2d modules.

        Delegates to the shared helper in ``uagent.perception.dropout`` so
        the test suite and the production path use one implementation.
        Only Dropout2d is relevant for our YOLOv8 placement.
        """
        from uagent.perception.dropout import enable_dropout_train_mode

        enable_dropout_train_mode(self.yolo.model)

    def predict(self, image: "np.ndarray") -> list[Posterior]:
        """Run K stochastic forward passes; return aggregated posteriors.

        Bypasses ultralytics' ``YOLO.predict()`` because the Predictor
        pipeline calls ``model.eval()`` during setup, which silently
        resets the Dropout2d modules to eval mode and produces
        deterministic outputs. We invoke ``self.yolo.model(tensor)``
        directly and reuse ultralytics' NMS + box-scaling utilities for
        a like-for-like detection result.

        Verified empirically 2026-05-08 that direct invocation preserves
        the train-mode dropout flag across the forward pass.
        """
        import torch
        from ultralytics.utils.ops import non_max_suppression, scale_boxes

        from uagent.perception.dropout import enable_dropout_train_mode

        self.yolo.model.to(self.device)
        tensor, orig_hw = self._preprocess(image, self.imgsz)
        tensor = tensor.to(self.device)

        names = (
            self.yolo.names if hasattr(self.yolo, "names") else self.yolo.model.names
        )

        all_passes: list[list[dict]] = []
        for _ in range(self.K):
            enable_dropout_train_mode(self.yolo.model)
            with torch.no_grad():
                raw = self.yolo.model(tensor)
            # Detect.forward in eval mode returns (predictions, features);
            # raw[0] is the [B, 4+nc, n_anchors] prediction tensor that NMS
            # consumes. enable_dropout_train_mode flips only the Dropout2d
            # submodules — the parent training flag stays False, so
            # the eval-format tuple is what we get.
            preds = raw[0] if isinstance(raw, (tuple, list)) else raw

            nms_out = non_max_suppression(
                preds,
                conf_thres=self.conf_threshold,
                iou_thres=self.iou_threshold,
            )
            dets_t = nms_out[0]  # [n_dets, 6]: xyxy, conf, cls

            if dets_t is None or len(dets_t) == 0:
                all_passes.append([])
                continue

            boxes = scale_boxes(
                (self.imgsz, self.imgsz), dets_t[:, :4], orig_hw
            )

            pass_dets = []
            for i in range(len(dets_t)):
                x1, y1, x2, y2 = boxes[i].cpu().tolist()
                pass_dets.append(
                    {
                        "label": names[int(dets_t[i, 5].item())],
                        "bbox": (x1, y1, x2, y2),
                        "confidence": float(dets_t[i, 4].item()),
                    }
                )
            all_passes.append(pass_dets)

        return self._aggregate(all_passes)

    @staticmethod
    def _preprocess(
        image: "np.ndarray", imgsz: int
    ) -> "tuple[torch.Tensor, tuple[int, int]]":
        """Letterbox to (imgsz, imgsz), pad with 114, BGR→RGB, /255 → tensor.

        Returns (tensor[1,3,imgsz,imgsz], original_hw) so scale_boxes can
        unscale detections back to the source image's coordinate frame.

        Input must be a 3-channel uint8 BGR image (H, W, 3). Grayscale,
        float, or RGBA inputs are rejected — convert upstream.
        """
        import cv2
        import numpy as np
        import torch

        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                f"image must be (H, W, 3) BGR; got shape {image.shape}"
            )
        if image.dtype != np.uint8:
            raise ValueError(
                f"image must be uint8; got {image.dtype}. Convert upstream."
            )

        h0, w0 = image.shape[:2]
        r = min(imgsz / h0, imgsz / w0)
        new_h, new_w = int(round(h0 * r)), int(round(w0 * r))
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        pad_h = imgsz - new_h
        pad_w = imgsz - new_w
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left
        padded = cv2.copyMakeBorder(
            resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )

        rgb = padded[..., ::-1].copy()
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        return tensor, (h0, w0)

    def _aggregate(self, all_passes: list[list[dict]]) -> list[Posterior]:
        if not all_passes or not all_passes[0]:
            return []

        # Reference clusters from pass 0.
        clusters: list[dict] = [
            {"label": d["label"], "bboxes": [d["bbox"]], "confs": [d["confidence"]]}
            for d in all_passes[0]
        ]

        # Match passes 1..K-1 into clusters by label + IoU.
        for pass_dets in all_passes[1:]:
            assigned = [False] * len(clusters)
            for det in pass_dets:
                best_iou = 0.0
                best_idx = -1
                for i, c in enumerate(clusters):
                    if c["label"] != det["label"]:
                        continue
                    if assigned[i]:
                        continue
                    iou = _iou(c["bboxes"][0], det["bbox"])
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = i
                if best_idx >= 0 and best_iou >= self.iou_match_threshold:
                    clusters[best_idx]["bboxes"].append(det["bbox"])
                    clusters[best_idx]["confs"].append(det["confidence"])
                    assigned[best_idx] = True
            # Unassigned clusters in this pass: pad zero (absence is info).
            for i, c in enumerate(clusters):
                if not assigned[i]:
                    c["confs"].append(0.0)

        posteriors: list[Posterior] = []
        for c in clusters:
            confs = c["confs"]
            n = len(confs)
            mean_conf = sum(confs) / n
            var = sum((x - mean_conf) ** 2 for x in confs) / n
            n_bboxes = len(c["bboxes"])
            mean_bbox = tuple(
                sum(b[i] for b in c["bboxes"]) / n_bboxes for i in range(4)
            )
            posteriors.append(
                Posterior(
                    label=c["label"],
                    bbox=mean_bbox,  # type: ignore[arg-type]
                    mean_confidence=float(mean_conf),
                    epistemic_variance=float(var),
                    K=self.K,
                )
            )
        return posteriors


# ---------------------------------------------------------------------------
# Deterministic YOLO
# ---------------------------------------------------------------------------


class DeterministicYOLO:
    """Single-pass YOLO. Calibrated downstream by TemperatureScaler."""

    def __init__(
        self,
        weights_path: str | Path,
        imgsz: int = 640,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: str = "cuda",
    ) -> None:
        from ultralytics import YOLO

        self.weights_path = Path(weights_path)
        self.imgsz = int(imgsz)
        self.conf_threshold = float(conf_threshold)
        self.iou_threshold = float(iou_threshold)
        self.device = _resolve_device(str(device))

        self.yolo = YOLO(str(self.weights_path))

    def predict(self, image: "np.ndarray") -> list[Detection]:
        self.yolo.model.eval()
        results = self.yolo.predict(
            image,
            imgsz=self.imgsz,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
        )
        out: list[Detection] = []
        for det in _extract_detections(results[0]):
            out.append(
                Detection(
                    label=det["label"],
                    bbox=det["bbox"],  # type: ignore[arg-type]
                    confidence=det["confidence"],
                )
            )
        return out
