"""Fit temperature scaling for the deterministic baseline.

Runs DeterministicYOLO over the doorway dataset's calibration split,
builds (confidence, TP/FP) labels by IoU >= 0.5 matching against the
YOLO-format ground truth, fits TemperatureScaler on the corresponding
logits, and writes the scalar back into the config under
`calibration.temperature`. Reports ECE before and after.

Class-agnostic IoU matching: any GT box of any class is a valid target
for any predicted detection. The doorway dataset is filtered to a single
class (door, index 0) by scripts/prepare_doorway_data.py, so this is
effectively single-class matching in practice.

Usage:
    python scripts/calibrate_baseline.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import yaml

from uagent.perception.calibration import (
    TemperatureScaler,
    expected_calibration_error,
)
from uagent.perception.detector import DeterministicYOLO

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
IOU_MATCH_THRESHOLD = 0.5
MIN_DETECTIONS = 30
# Hardcoded as a safeguard, not a knob: this is the minimum minority-class
# count below which the L-BFGS fit is unidentifiable regardless of dataset.
# Promote to config only if there's a per-experiment reason to relax it.
MIN_MINORITY_COUNT = 20
# Confidence values are clipped to this range before logit conversion to
# stop a single saturated detection (conf == 1.0 → logit ~16) from
# dominating the L-BFGS objective and biasing T.
CONF_CLIP = (1e-3, 1.0 - 1e-3)


def _iou_xyxy(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    a_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    b_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = a_area + b_area - inter
    return inter / union if union > 0 else 0.0


def _read_gt_boxes(
    label_path: Path, img_w: int, img_h: int
) -> list[tuple[float, float, float, float]]:
    """YOLO-format labels file -> pixel-xyxy bboxes (class-agnostic)."""
    boxes: list[tuple[float, float, float, float]] = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cx, cy, w, h = (float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))
        x1 = (cx - w / 2.0) * img_w
        y1 = (cy - h / 2.0) * img_h
        x2 = (cx + w / 2.0) * img_w
        y2 = (cy + h / 2.0) * img_h
        boxes.append((x1, y1, x2, y2))
    return boxes


def _confidence_to_logit(p: float) -> float:
    p = max(CONF_CLIP[0], min(CONF_CLIP[1], float(p)))
    return math.log(p / (1.0 - p))


def _write_temperature(yaml_path: Path, T: float) -> None:
    """Replace `calibration.temperature` in-place; preserve comments.

    Locates the `calibration:` top-level block and rewrites the
    `temperature:` line within it. Stops scanning at the next top-level
    key so an `llm.temperature` (or any other section's `temperature:`)
    cannot be matched accidentally.
    """
    lines = yaml_path.read_text(encoding="utf-8").splitlines(keepends=True)
    in_calibration = False
    for i, line in enumerate(lines):
        is_top_level = bool(line) and not line[0].isspace() and line.lstrip() and not line.lstrip().startswith("#")
        if is_top_level:
            if in_calibration:
                break  # left calibration block without finding the key
            in_calibration = line.lstrip().startswith("calibration:")
            continue
        if in_calibration:
            m = re.match(r"^(\s+)temperature:(\s+)(\S+)(.*)$", line)
            if m:
                indent, sp, _old, rest = m.groups()
                lines[i] = f"{indent}temperature:{sp}{T:.4f}{rest}\n"
                _atomic_write_text(yaml_path, "".join(lines))
                return
    raise ValueError(
        f"no `calibration.temperature` key in {yaml_path}; "
        "the canonical config (configs/default.yaml) has it; override files "
        "may not. Calibrate against default.yaml, or add the key to your override."
    )


def _atomic_write_text(target: Path, text: str) -> None:
    """Write to a sibling tempfile and os.replace so partial writes don't corrupt the config."""
    fd, tmp_path = tempfile.mkstemp(
        prefix=target.name + ".",
        suffix=".tmp",
        dir=str(target.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp_path, target)
    except Exception:
        # If the rename succeeded the temp is gone; otherwise clean up.
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, help="path to YAML config")
    args = p.parse_args()

    cfg_path = Path(args.config)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    try:
        weights = Path(cfg["paths"]["yolo_weights"])
        dataset_root = Path(cfg["paths"]["doorway_dataset_root"])
        perc = cfg["perception"]
        n_bins = int(cfg["calibration"]["ece_n_bins"])
        fit_conf_threshold = float(cfg["calibration"]["fit_conf_threshold"])
    except KeyError as e:
        sys.exit(
            f"missing required config key: {e} in {cfg_path}. "
            "Override files (workstation.yaml, agx_dev.yaml) inherit from "
            "configs/default.yaml — calibrate against default.yaml, or copy "
            "the missing block into your override."
        )

    # Calibration is fit on the dedicated calibration split, separate from
    # the train/val splits used for fine-tuning + mAP. Produced by
    # scripts/prepare_doorway_data.py.
    cal_images_dir = dataset_root / "calibration" / "images"
    cal_labels_dir = dataset_root / "calibration" / "labels"
    if not cal_images_dir.is_dir():
        sys.exit(f"calibration images dir not found: {cal_images_dir}")
    if not cal_labels_dir.is_dir():
        sys.exit(f"calibration labels dir not found: {cal_labels_dir}")
    if not weights.exists():
        sys.exit(f"YOLO weights not found: {weights}")

    print(
        f"loading DeterministicYOLO from {weights} "
        f"(device={perc['device']}, fit conf >= {fit_conf_threshold})"
    )
    detector = DeterministicYOLO(
        weights,
        imgsz=int(perc["imgsz"]),
        conf_threshold=fit_conf_threshold,
        iou_threshold=float(perc["iou_threshold"]),
        device=str(perc["device"]),
    )

    image_paths = sorted(
        path for path in cal_images_dir.iterdir() if path.suffix.lower() in IMAGE_EXTS
    )
    print(f"calibrating over {len(image_paths)} images from {cal_images_dir}")
    if len(image_paths) < 100:
        print(
            f"WARNING: only {len(image_paths)} images in calibration split. For "
            "paper-grade calibration we should hold out >= 100 images; this run "
            "is a placeholder until more data is collected."
        )

    confidences: list[float] = []
    logits: list[float] = []
    labels: list[int] = []

    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  skip: failed to read {img_path.name}")
            continue
        h, w = img.shape[:2]
        gt = _read_gt_boxes(cal_labels_dir / (img_path.stem + ".txt"), w, h)
        for det in detector.predict(img):
            best_iou = max((_iou_xyxy(det.bbox, g) for g in gt), default=0.0)
            label = 1 if best_iou >= IOU_MATCH_THRESHOLD else 0
            confidences.append(det.confidence)
            logits.append(_confidence_to_logit(det.confidence))
            labels.append(label)

    n = len(confidences)
    if n < MIN_DETECTIONS:
        sys.exit(
            f"only {n} detections collected; need >= {MIN_DETECTIONS}. "
            "Verify the model produces detections at the configured "
            "conf_threshold and that GT labels are present."
        )

    n_pos = sum(labels)
    n_neg = n - n_pos
    print(f"collected {n} detections: {n_pos} TP / {n_neg} FP")
    minority_count = min(n_pos, n_neg)
    if minority_count < MIN_MINORITY_COUNT:
        sys.exit(
            f"degenerate label split (minority class count = {minority_count}, "
            f"need >= {MIN_MINORITY_COUNT}); T is unidentifiable. Lower "
            "calibration.fit_conf_threshold or collect more val data."
        )

    conf_arr = np.asarray(confidences, dtype=np.float64)
    correct = np.asarray(labels, dtype=np.float64)
    ece_pre = expected_calibration_error(conf_arr, correct, n_bins=n_bins)

    scaler = TemperatureScaler.fit(
        np.asarray(logits, dtype=np.float32),
        np.asarray(labels, dtype=np.int64),
    )
    post = np.array([scaler.transform(c) for c in conf_arr], dtype=np.float64)
    ece_post = expected_calibration_error(post, correct, n_bins=n_bins)

    print(f"T = {scaler.temperature:.4f}")
    if scaler.temperature < 1.0:
        print(
            f"  Note: T<1 sharpens predictions. Likely cause: fit_conf_threshold"
            f"={fit_conf_threshold} is well below runtime conf_threshold="
            f"{perc['conf_threshold']}, so the fit sees many low-confidence FPs"
            " that L-BFGS pushes further down. Calibration is still valid for "
            "the runtime confidence range; raise fit_conf_threshold if you want "
            "a fit closer to the runtime distribution."
        )
    print(f"ECE before: {ece_pre:.4f}")
    print(f"ECE after:  {ece_post:.4f}")
    if ece_post >= ece_pre:
        print(
            "  warning: ECE did not improve. May indicate degenerate labels "
            "(all-TP or all-FP) or that the baseline is already well-calibrated."
        )

    _write_temperature(cfg_path, scaler.temperature)
    print(f"wrote calibration.temperature -> {cfg_path}")


if __name__ == "__main__":
    main()
