"""Filter DoorDetect to the "door" class only and split for training.

Reads source images + YOLO-format labels from data/doordetect/, keeps only
images that contain at least one class-0 ("door") annotation, drops the
non-door label lines, and emits a deterministic 70/15/15 train/val/
calibration split under data/doordetect_door/.

The calibration split feeds scripts/calibrate_baseline.py; train+val
feed scripts/train_dropout_yolo.py via the emitted data.yaml.

Usage:
    python scripts/prepare_doorway_data.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import yaml

DOOR_CLASS_INDEX = 0
SPLIT_FRACTIONS = (0.70, 0.15, 0.15)  # train, val, calibration
SPLIT_NAMES = ("train", "val", "calibration")
SEED = 1337
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def _filter_door_lines(label_text: str) -> list[str]:
    """Return only the lines whose class index is DOOR_CLASS_INDEX."""
    out = []
    for line in label_text.splitlines():
        parts = line.strip().split()
        if len(parts) >= 5 and parts[0] == str(DOOR_CLASS_INDEX):
            out.append(line.strip())
    return out


def _find_image_for_label(images_dir: Path, stem: str) -> Path | None:
    for ext in IMAGE_EXTS:
        candidate = images_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def _link_or_copy(src: Path, dst: Path) -> None:
    """Symlink src -> dst when possible (saves disk), copy otherwise.

    Caller is responsible for ensuring dst does not pre-exist; main()
    rmtrees the output dir before calling.
    """
    try:
        os.symlink(src.resolve(), dst)
    except OSError:
        shutil.copy2(src, dst)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, help="path to YAML config")
    p.add_argument(
        "--source",
        default="data/doordetect",
        help="DoorDetect clone root (default: data/doordetect, relative to config dir)",
    )
    p.add_argument(
        "--output",
        default="data/doordetect_door",
        help="output dataset root (default: data/doordetect_door)",
    )
    args = p.parse_args()

    cfg_path = Path(args.config).resolve()
    repo_root = cfg_path.parent.parent  # configs/foo.yaml -> repo root
    source = (repo_root / args.source).resolve()
    output = (repo_root / args.output).resolve()

    src_images = source / "images"
    src_labels = source / "labels"
    if not src_images.is_dir():
        sys.exit(f"source images dir not found: {src_images}")
    if not src_labels.is_dir():
        sys.exit(f"source labels dir not found: {src_labels}")

    label_files = sorted(src_labels.glob("*.txt"))
    print(f"scanning {len(label_files)} label files in {src_labels}")

    keep: list[tuple[Path, Path, list[str]]] = []  # (image_path, label_path, door_lines)
    skipped_no_door = 0
    skipped_missing_image = 0

    for lf in label_files:
        text = lf.read_text(encoding="utf-8")
        door_lines = _filter_door_lines(text)
        if not door_lines:
            skipped_no_door += 1
            continue
        img = _find_image_for_label(src_images, lf.stem)
        if img is None:
            skipped_missing_image += 1
            continue
        keep.append((img, lf, door_lines))

    n_keep = len(keep)
    print(
        f"kept {n_keep} images (skipped {skipped_no_door} without door annotations, "
        f"{skipped_missing_image} with missing image files)"
    )

    if n_keep < 100:
        print(
            f"WARNING: only {n_keep} images contain a 'door' annotation. The full "
            "DoorDetect set has 1,213 images but most are dominated by handle / "
            "cabinet door / refrigerator door labels. Fine-tuning on this small "
            "corpus may underperform; consider supplementing with DoorDetect-Class "
            "(Ramoa et al. 2020) or the Antonazzi et al. 2022 robot-perspective "
            "doors dataset before step B variance validation."
        )

    rng = np.random.default_rng(seed=SEED)
    indices = np.arange(n_keep)
    rng.shuffle(indices)

    n_train = int(round(SPLIT_FRACTIONS[0] * n_keep))
    n_val = int(round(SPLIT_FRACTIONS[1] * n_keep))
    n_cal = n_keep - n_train - n_val
    split_sizes = (n_train, n_val, n_cal)

    cuts = np.cumsum(split_sizes)
    splits = {
        SPLIT_NAMES[0]: indices[: cuts[0]],
        SPLIT_NAMES[1]: indices[cuts[0] : cuts[1]],
        SPLIT_NAMES[2]: indices[cuts[1] :],
    }

    print(
        f"splits @ seed={SEED}: train={n_train} val={n_val} calibration={n_cal}"
    )

    if output.exists():
        print(f"removing existing output dir {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True)

    for split_name, idxs in splits.items():
        img_out = output / split_name / "images"
        lbl_out = output / split_name / "labels"
        img_out.mkdir(parents=True)
        lbl_out.mkdir(parents=True)
        for i in idxs:
            img_src, _, door_lines = keep[i]
            stem = img_src.stem
            _link_or_copy(img_src, img_out / img_src.name)
            (lbl_out / f"{stem}.txt").write_text(
                "\n".join(door_lines) + "\n", encoding="utf-8"
            )

    data_yaml_path = output / "data.yaml"
    data_yaml = {
        "path": str(output),
        "train": "train/images",
        "val": "val/images",
        "nc": 1,
        "names": ["door"],
    }
    data_yaml_path.write_text(yaml.safe_dump(data_yaml, sort_keys=False), encoding="utf-8")
    print(f"wrote {data_yaml_path}")
    print(f"done. dataset rooted at {output}")


if __name__ == "__main__":
    main()
