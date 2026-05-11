"""Visual spot-check: overlay predicted boxes on ground-truth boxes
for a small sample of val images, with IoU annotated.

Samples N val images deterministically (seeded), stratified across the
three sources (Giraff-X / Gibson / DoorDetect). Runs DeterministicYOLO
on each, draws GT boxes in RED and predicted boxes in GREEN with their
confidence and IoU-to-nearest-GT printed. Writes paired PNGs under
runs/spot_checks/.

The 0.5:0.95 mAP number (0.781 aggregate, 0.306 on DoorDetect) suggests
boxes are mostly tight, but a five-minute visual confirmation removes
the doubt. We're particularly interested in DoorDetect images, where
the per-source mAP is lowest.

Usage:
    python scripts/spot_check_predictions.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

SOURCE_PREFIXES = {
    "giraff_x": "gx_",
    "gibson": "gb_",
    "doordetect": "dx_",
}
SAMPLE_PER_SOURCE = {"giraff_x": 3, "gibson": 3, "doordetect": 2}  # 8 images total
SEED = 1337


def _iou(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
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


def _read_gt_boxes_pixel(
    label_path: Path, img_w: int, img_h: int
) -> list[tuple[float, float, float, float]]:
    """YOLO normalized cxcywh -> pixel xyxy."""
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


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, help="path to YAML config")
    args = p.parse_args()

    cfg_path = Path(args.config).resolve()
    repo_root = cfg_path.parent.parent
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    try:
        weights_path = (repo_root / cfg["paths"]["yolo_weights"]).resolve()
        dataset_root = (repo_root / cfg["paths"]["doorway_dataset_root"]).resolve()
        imgsz = int(cfg["perception"]["imgsz"])
        conf_threshold = float(cfg["perception"]["conf_threshold"])
        iou_threshold = float(cfg["perception"]["iou_threshold"])
        device = str(cfg["perception"]["device"])
    except KeyError as e:
        sys.exit(f"missing required config key: {e}")

    if not weights_path.exists():
        sys.exit(f"trained checkpoint not found: {weights_path}")

    val_images = dataset_root / "val" / "images"
    val_labels = dataset_root / "val" / "labels"

    out_dir = repo_root / "runs" / "spot_checks"
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed=SEED)
    picked: list[tuple[str, Path]] = []
    for source, prefix in SOURCE_PREFIXES.items():
        candidates = sorted(
            p for p in val_images.iterdir() if p.name.startswith(prefix)
        )
        if not candidates:
            print(f"  WARNING: no val images for source {source}")
            continue
        k = min(SAMPLE_PER_SOURCE[source], len(candidates))
        idx = rng.choice(len(candidates), size=k, replace=False)
        for i in sorted(idx):
            picked.append((source, candidates[i]))

    print(f"sampled {len(picked)} val images for spot-check; writing to {out_dir}")

    from uagent.perception.detector import DeterministicYOLO

    detector = DeterministicYOLO(
        weights_path,
        imgsz=imgsz,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        device=device,
    )

    summary_lines: list[str] = []
    for source, img_path in picked:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  skip (cannot read): {img_path.name}")
            continue
        h, w = img.shape[:2]
        gt_boxes = _read_gt_boxes_pixel(val_labels / f"{img_path.stem}.txt", w, h)
        detections = detector.predict(img)

        overlay = img.copy()
        # GT in red
        for gt in gt_boxes:
            x1, y1, x2, y2 = (int(v) for v in gt)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # Predictions in green with IoU + conf annotation
        det_summaries: list[str] = []
        for det in detections:
            x1, y1, x2, y2 = (int(v) for v in det.bbox)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
            best_iou = max(
                (_iou(det.bbox, gt) for gt in gt_boxes), default=0.0
            )
            label_text = f"{det.confidence:.2f} iou={best_iou:.2f}"
            cv2.putText(
                overlay,
                label_text,
                (x1, max(15, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )
            det_summaries.append(
                f"conf={det.confidence:.3f} iou={best_iou:.3f}"
            )

        out_path = out_dir / f"{source}__{img_path.stem}.png"
        cv2.imwrite(str(out_path), overlay)

        line = (
            f"  {source:<11} {img_path.name:<50} "
            f"GT={len(gt_boxes)} pred={len(detections)} | "
            + ("; ".join(det_summaries) if det_summaries else "no detections")
        )
        print(line)
        summary_lines.append(line)

    summary_path = out_dir / "summary.txt"
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print()
    print(f"overlays + summary written to {out_dir}")


if __name__ == "__main__":
    main()
