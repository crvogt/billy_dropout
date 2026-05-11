"""Fit temperature scaling for the deterministic baseline.

Runs DeterministicYOLO over the doorway dataset's calibration split,
builds (confidence, TP/FP) labels by IoU >= 0.5 matching against the
YOLO-format ground truth, fits TemperatureScaler on the corresponding
logits, and writes the scalar back into the config under
`calibration.temperature`. Reports ECE and NLL before and after, both
aggregated and broken down by source (Giraff-X / Gibson / DoorDetect)
inferred from the image filename prefix.

The per-source breakdown is the signal that informs the
single-temperature-vs-per-source-temperature decision at the user
level: if one source has a substantially worse post-T ECE than the
others, a per-source T may be warranted.

Class-agnostic IoU matching: any GT box of any class is a valid target
for any predicted detection. The doorway dataset is filtered to a single
class (door, index 0) by scripts/prepare_doorway_data.py, so this is
effectively single-class matching in practice.

Usage:
    python scripts/calibrate_baseline.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import tempfile
from collections import defaultdict
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

# Source attribution from output-stem prefix, same convention as
# prepare_doorway_data.py / per_source_val.py / step_b.
SOURCE_PREFIXES = {"gx_": "giraff_x", "gb_": "gibson", "dx_": "doordetect"}


def _source_from_filename(name: str) -> str:
    for prefix, source in SOURCE_PREFIXES.items():
        if name.startswith(prefix):
            return source
    return "unknown"


def _binary_nll(confidences: np.ndarray, labels: np.ndarray) -> float:
    """Negative log-likelihood for the binary TP/FP detection set.

    NLL = -mean(y log p + (1-y) log(1-p)), with p clipped for stability.
    Returns nan on empty input.
    """
    if confidences.size == 0:
        return float("nan")
    eps = 1e-12
    p = np.clip(confidences, eps, 1.0 - eps)
    nll = -(labels * np.log(p) + (1.0 - labels) * np.log(1.0 - p))
    return float(np.mean(nll))


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

    # Pre-flight: bail before we burn detector inference on a malformed split.
    unknown_pre = [p.name for p in image_paths if _source_from_filename(p.name) == "unknown"]
    if unknown_pre:
        sys.exit(
            f"{len(unknown_pre)} image(s) in {cal_images_dir} lack a known "
            f"source prefix (gx_/gb_/dx_); first few: {unknown_pre[:3]}. "
            "scripts/prepare_doorway_data.py enforces the prefixes — this "
            "calibration split is malformed."
        )

    print(f"calibrating over {len(image_paths)} images from {cal_images_dir}")
    if len(image_paths) < 100:
        print(
            f"WARNING: only {len(image_paths)} images in calibration split. For "
            "paper-grade calibration we should hold out >= 100 images; this run "
            "is a placeholder until more data is collected."
        )

    # Per-detection records keyed by source so we can compute aggregate
    # AND per-source ECE / NLL with one pass over the data.
    records_by_source: dict[str, list[dict]] = defaultdict(list)

    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  skip: failed to read {img_path.name}")
            continue
        source = _source_from_filename(img_path.name)
        h, w = img.shape[:2]
        gt = _read_gt_boxes(cal_labels_dir / (img_path.stem + ".txt"), w, h)
        for det in detector.predict(img):
            best_iou = max((_iou_xyxy(det.bbox, g) for g in gt), default=0.0)
            label = 1 if best_iou >= IOU_MATCH_THRESHOLD else 0
            records_by_source[source].append(
                {
                    "confidence": det.confidence,
                    "logit": _confidence_to_logit(det.confidence),
                    "label": label,
                }
            )

    # Flatten in stable insertion order; per-source views derive from
    # records_by_source directly. The aggregated arrays below feed the
    # L-BFGS fit (which is order-invariant) and the aggregate ECE/NLL.
    flat_records = [r for rs in records_by_source.values() for r in rs]
    confidences = [r["confidence"] for r in flat_records]
    logits = [r["logit"] for r in flat_records]
    labels = [r["label"] for r in flat_records]

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
    print(f"  by source: " + ", ".join(
        f"{s}={len(rs)} (TP={sum(r['label'] for r in rs)}, "
        f"FP={len(rs) - sum(r['label'] for r in rs)})"
        for s, rs in sorted(records_by_source.items())
    ))
    minority_count = min(n_pos, n_neg)
    if minority_count < MIN_MINORITY_COUNT:
        sys.exit(
            f"degenerate label split (minority class count = {minority_count}, "
            f"need >= {MIN_MINORITY_COUNT}); T is unidentifiable. Lower "
            "calibration.fit_conf_threshold or collect more val data."
        )

    conf_arr = np.asarray(confidences, dtype=np.float64)
    correct = np.asarray(labels, dtype=np.float64)
    ece_pre_agg = expected_calibration_error(conf_arr, correct, n_bins=n_bins)
    nll_pre_agg = _binary_nll(conf_arr, correct)

    scaler = TemperatureScaler.fit(
        np.asarray(logits, dtype=np.float32),
        np.asarray(labels, dtype=np.int64),
    )
    post = np.array([scaler.transform(c) for c in conf_arr], dtype=np.float64)
    ece_post_agg = expected_calibration_error(post, correct, n_bins=n_bins)
    nll_post_agg = _binary_nll(post, correct)

    # Per-source metrics use the SAME single T that was fit on the aggregated
    # set. This is the "one temperature scalar" baseline the user wants to
    # compare against. If post-T per-source ECEs differ substantially, that's
    # the signal that a per-source T may be warranted.
    per_source_metrics: dict[str, dict] = {}
    for source, recs in sorted(records_by_source.items()):
        s_conf = np.array([r["confidence"] for r in recs], dtype=np.float64)
        s_label = np.array([r["label"] for r in recs], dtype=np.float64)
        s_post = np.array([scaler.transform(c) for c in s_conf], dtype=np.float64)
        per_source_metrics[source] = {
            "n_detections": int(len(recs)),
            "n_tp": int(s_label.sum()),
            "n_fp": int(len(recs) - s_label.sum()),
            "ece_pre": float(expected_calibration_error(s_conf, s_label, n_bins=n_bins)),
            "ece_post": float(expected_calibration_error(s_post, s_label, n_bins=n_bins)),
            "nll_pre": _binary_nll(s_conf, s_label),
            "nll_post": _binary_nll(s_post, s_label),
        }

    print()
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

    print()
    header = f"{'source':<11}{'n':>6}{'ECE pre':>10}{'ECE post':>11}{'NLL pre':>10}{'NLL post':>11}"
    print("=== calibration metrics ===")
    print(header)
    print("-" * len(header))
    print(
        f"{'aggregate':<11}{n:>6}{ece_pre_agg:>10.4f}{ece_post_agg:>11.4f}"
        f"{nll_pre_agg:>10.4f}{nll_post_agg:>11.4f}"
    )
    for source in sorted(per_source_metrics):
        m = per_source_metrics[source]
        print(
            f"{source:<11}{m['n_detections']:>6}"
            f"{m['ece_pre']:>10.4f}{m['ece_post']:>11.4f}"
            f"{m['nll_pre']:>10.4f}{m['nll_post']:>11.4f}"
        )

    if ece_post_agg >= ece_pre_agg:
        print(
            "  warning: aggregate ECE did not improve. May indicate degenerate "
            "labels (all-TP or all-FP) or that the baseline is already well-calibrated."
        )

    results_dict = {
        "temperature": float(scaler.temperature),
        "fit_conf_threshold": fit_conf_threshold,
        "runtime_conf_threshold": float(perc["conf_threshold"]),
        "n_detections": n,
        "n_tp": int(n_pos),
        "n_fp": int(n_neg),
        "aggregate": {
            "ece_pre": float(ece_pre_agg),
            "ece_post": float(ece_post_agg),
            "nll_pre": float(nll_pre_agg),
            "nll_post": float(nll_post_agg),
        },
        "per_source": per_source_metrics,
    }
    # Anchor results path on the script's location, not on cfg_path, so a
    # temp config under /tmp doesn't write results under '/'.
    repo_root = Path(__file__).resolve().parent.parent
    results_path = repo_root / "runs" / "calibration_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(results_dict, indent=2), encoding="utf-8")
    print(f"wrote {results_path}")

    _write_temperature(cfg_path, scaler.temperature)
    print(f"wrote calibration.temperature -> {cfg_path}")


if __name__ == "__main__":
    main()
