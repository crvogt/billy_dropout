"""Per-source mAP@0.5 and mAP@0.5:0.95 on the doorway val split.

Detects Gibson same-scene-leak suspicion. If mAP is uniformly high across
all three sources, the detector generalizes evenly. If Gibson is much
higher than Giraff-X or DoorDetect, the headline whole-val number was
buoyed by overfitting to synthetic-render features.

Builds three temp val subsets under --scratch by output-stem prefix
(gx_ = Giraff-X, gb_ = Gibson, dx_ = DoorDetect), each with symlinks
to the parent val/ split. Runs ultralytics' .val() once per source.
Prints a side-by-side table and writes a JSON manifest.

Usage:
    python scripts/per_source_val.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import yaml

SOURCE_PREFIXES = {
    "giraff_x": "gx_",
    "gibson": "gb_",
    "doordetect": "dx_",
}


def _resolve_device(requested: str) -> str:
    import torch

    if requested == "cuda" and not torch.cuda.is_available():
        print("WARNING: cuda requested but unavailable; falling back to cpu")
        return "cpu"
    return requested


def _build_subset(
    val_images: Path, val_labels: Path, prefix: str, dst: Path
) -> int:
    """Symlink per-source val images + labels into dst/{images,labels}/.

    Returns count of images linked.
    """
    img_out = dst / "images"
    lbl_out = dst / "labels"
    img_out.mkdir(parents=True)
    lbl_out.mkdir(parents=True)
    n = 0
    for img in val_images.iterdir():
        if not img.name.startswith(prefix):
            continue
        lbl = val_labels / (img.stem + ".txt")
        if not lbl.exists():
            continue
        os.symlink(img.resolve(), img_out / img.name)
        os.symlink(lbl.resolve(), lbl_out / lbl.name)
        n += 1
    return n


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, help="path to YAML config")
    p.add_argument(
        "--scratch",
        default="/tmp/per_source_val",
        help="temp dir for per-source subsets + val artifacts",
    )
    args = p.parse_args()

    cfg_path = Path(args.config).resolve()
    repo_root = cfg_path.parent.parent
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    try:
        weights_path = (repo_root / cfg["paths"]["yolo_weights"]).resolve()
        dataset_root = (repo_root / cfg["paths"]["doorway_dataset_root"]).resolve()
        imgsz = int(cfg["perception"]["imgsz"])
        device = _resolve_device(str(cfg["perception"]["device"]))
    except KeyError as e:
        sys.exit(f"missing required config key: {e}")

    if not weights_path.exists():
        sys.exit(
            f"trained checkpoint not found: {weights_path}\n"
            "Run training on Thor and rsync the result back first."
        )
    val_images = dataset_root / "val" / "images"
    val_labels = dataset_root / "val" / "labels"
    if not val_images.is_dir() or not val_labels.is_dir():
        sys.exit(f"val dir incomplete at {dataset_root}/val")

    scratch = Path(args.scratch).resolve()
    if scratch.exists():
        shutil.rmtree(scratch)
    scratch.mkdir(parents=True)

    print("building per-source subsets...")
    counts: dict[str, int] = {}
    data_yamls: dict[str, Path] = {}
    for source, prefix in SOURCE_PREFIXES.items():
        src_dir = scratch / source
        n = _build_subset(val_images, val_labels, prefix, src_dir)
        counts[source] = n
        data_yaml = src_dir / "data.yaml"
        # NOTE: train points at val/images deliberately — val() needs the
        # key set, but this data.yaml MUST NOT be reused by .train() (it
        # would train on the val set). Per-source training is not a use
        # case here; this script is val-only.
        data_yaml.write_text(
            yaml.safe_dump(
                {
                    "path": str(src_dir),
                    "train": "images",  # do not reuse for training!
                    "val": "images",
                    "nc": 1,
                    "names": ["door"],
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        data_yamls[source] = data_yaml
        print(f"  {source}: {n} val images")

    print()
    print(f"loading {weights_path} on device={device}")
    from ultralytics import YOLO

    yolo = YOLO(str(weights_path))

    results: dict[str, dict] = {}
    for source, dy in data_yamls.items():
        print()
        print(f"--- validating on {source} (n={counts[source]}) ---")
        metrics = yolo.val(
            data=str(dy),
            imgsz=imgsz,
            device=device,
            verbose=False,
            project=str(scratch),
            name=f"val_{source}",
            exist_ok=True,
        )
        results[source] = {
            "n_images": counts[source],
            "map50": float(metrics.box.map50),
            "map5095": float(metrics.box.map),
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
        }

    print()
    print("=== per-source val metrics ===")
    header = f"{'source':<12}{'n':>6}{'mAP@.5':>10}{'mAP@.5:.95':>14}{'P':>8}{'R':>8}"
    print(header)
    print("-" * len(header))
    for source in SOURCE_PREFIXES:
        r = results[source]
        print(
            f"{source:<12}{r['n_images']:>6}"
            f"{r['map50']:>10.3f}{r['map5095']:>14.3f}"
            f"{r['precision']:>8.3f}{r['recall']:>8.3f}"
        )

    out_json = repo_root / "runs" / "per_source_val_results.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print()
    print(f"saved to {out_json}")
    print(f"ultralytics per-source val artifacts under {scratch}")


if __name__ == "__main__":
    main()
