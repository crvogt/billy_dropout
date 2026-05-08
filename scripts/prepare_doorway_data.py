"""Build the multi-source doorway training corpus.

Combines three sources:

  * DoorDetect (Arduengo et al. 2021) — real, general indoor.
    YOLO-format labels at data/doordetect/. Filtered to class 0 (door).

  * Giraff-X / final_doors_dataset_real (Antonazzi et al. 2024) — real,
    robot-perspective. Per-sample directory format with gzipped numpy
    bbox arrays at data/antonazzi/final_doors_dataset_real/.

  * Gibson / final_doors_dataset (Antonazzi et al. 2024) — photorealistic
    synthetic, robot-perspective. Same custom format at
    data/antonazzi/final_doors_dataset/. Capped at training.gibson_cap to
    keep synthetic data from dominating the real signal (per addendum).

For Antonazzi sources the bbox file is a gzipped .npy (despite the
.tar.gz extension), shape (n_doors, 5), columns [label, x1, y1, w, h]
in pixel coordinates. Two-class label scheme (0=Closed, 1=Open) is
collapsed to single-class door (class 0) per addendum.

Output: data/doorway_combined/ with train/val/calibration splits
(80/10/10), stratified by source so each split has proportional
representation. Plus a YOLO data.yaml and a provenance manifest at
data/manifests/training_image_ids.json for OOD bucket exclusion.

Usage:
    python scripts/prepare_doorway_data.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import os
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import yaml

DOOR_CLASS = 0
SPLIT_NAMES = ("train", "val", "calibration")
SPLIT_FRACTIONS = (0.80, 0.10, 0.10)
DD_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


@dataclass
class TrainingItem:
    """One image in the training corpus, with its labels and provenance."""

    source: str  # "doordetect" | "giraff_x" | "gibson"
    image_path: Path  # absolute path to source image
    label_lines: list[str]  # YOLO-format text lines, class 0 only
    output_stem: str  # unique stem for the combined-output directory
    provenance_id: str  # source-internal identifier for the manifest


# ---------------------------------------------------------------------------
# Source loaders
# ---------------------------------------------------------------------------


def load_doordetect(root: Path) -> list[TrainingItem]:
    """Read DoorDetect's flat YOLO-format directory; keep only door-class lines."""
    items: list[TrainingItem] = []
    src_imgs = root / "images"
    src_lbls = root / "labels"
    if not src_imgs.is_dir() or not src_lbls.is_dir():
        sys.exit(f"DoorDetect not found at {root} (expected images/ and labels/)")

    for lbl in sorted(src_lbls.glob("*.txt")):
        text = lbl.read_text(encoding="utf-8")
        door_lines = [
            line.strip()
            for line in text.splitlines()
            if line.strip() and line.strip().split()[0] == str(DOOR_CLASS)
        ]
        if not door_lines:
            continue
        img_path = None
        for ext in DD_IMAGE_EXTS:
            candidate = src_imgs / f"{lbl.stem}{ext}"
            if candidate.exists():
                img_path = candidate
                break
        if img_path is None:
            continue
        items.append(
            TrainingItem(
                source="doordetect",
                image_path=img_path,
                label_lines=door_lines,
                output_stem=f"dx_{lbl.stem}",
                provenance_id=lbl.stem,  # Open Images V4 ID (16-hex)
            )
        )
    return items


def _convert_pixel_xywh_to_yolo(
    bbox_arr: np.ndarray, img_w: int, img_h: int
) -> list[str]:
    """Convert (label, x1, y1, w, h) pixel rows to YOLO normalized cxcywh.

    Collapses any source label to class 0 (door — addendum: Closed/Open →
    single class). Clips the pixel rect to image bounds *before*
    normalizing, so the resulting YOLO box is guaranteed to be inside
    [0, 1]^2 (clipping the normalized cx/w independently can leave the
    right or bottom edge outside the image).
    """
    lines: list[str] = []
    for row in bbox_arr:
        # Source label discarded — single-class doors per addendum.
        _, x1, y1, bw, bh = (
            int(row[0]),
            int(row[1]),
            int(row[2]),
            int(row[3]),
            int(row[4]),
        )
        x2 = x1 + bw
        y2 = y1 + bh
        x1_c = max(0, min(img_w, x1))
        y1_c = max(0, min(img_h, y1))
        x2_c = max(0, min(img_w, x2))
        y2_c = max(0, min(img_h, y2))
        bw_c = x2_c - x1_c
        bh_c = y2_c - y1_c
        if bw_c <= 0 or bh_c <= 0:
            continue
        cx_n = (x1_c + bw_c / 2.0) / img_w
        cy_n = (y1_c + bh_c / 2.0) / img_h
        w_n = bw_c / img_w
        h_n = bh_c / img_h
        lines.append(f"{DOOR_CLASS} {cx_n:.6f} {cy_n:.6f} {w_n:.6f} {h_n:.6f}")
    return lines


def load_antonazzi(
    root: Path, source_name: str, prefix: str
) -> list[TrainingItem]:
    """Walk a final_doors_dataset[_real]/ root; emit door-bearing samples
    with bboxes converted to YOLO format.

    Layout: <env_or_house>/<bucket>/{bgr_image,bounding_boxes}/. Bbox files
    end in .tar.gz but are actually gzipped .npy. Empty arrays mean no
    doors visible in that frame; those samples are skipped.
    """
    if not root.is_dir():
        sys.exit(f"Antonazzi source not found at {root}")

    items: list[TrainingItem] = []
    for env in sorted(root.iterdir()):
        if not env.is_dir():
            continue
        for sub in sorted(p for p in env.iterdir() if p.is_dir()):
            bbox_dir = sub / "bounding_boxes"
            img_dir = sub / "bgr_image"
            if not bbox_dir.is_dir() or not img_dir.is_dir():
                continue

            for bf in sorted(bbox_dir.glob("*.tar.gz")):
                try:
                    with gzip.open(bf, "rb") as f:
                        arr = np.load(io.BytesIO(f.read()))
                except Exception:
                    continue
                if arr.size == 0:
                    continue

                # Filename pattern: bbox is bounding_boxes_<id>_(<localid>).tar.gz;
                # the matching image is bgr_image_<id>_(<localid>).png.
                stem_suffix = bf.name[len("bounding_boxes_") :].rsplit(".tar.gz", 1)[0]
                img_path = img_dir / f"bgr_image_{stem_suffix}.png"
                if not img_path.exists():
                    continue

                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                h, w = img.shape[:2]
                lines = _convert_pixel_xywh_to_yolo(arr, w, h)
                if not lines:
                    continue

                items.append(
                    TrainingItem(
                        source=source_name,
                        image_path=img_path,
                        label_lines=lines,
                        output_stem=f"{prefix}_{env.name}_{stem_suffix}",
                        provenance_id=f"{env.name}/{stem_suffix}",
                    )
                )
    return items


# ---------------------------------------------------------------------------
# Mixture, split, write
# ---------------------------------------------------------------------------


def _deterministic_subsample(
    items: list[TrainingItem], cap: int, seed: int
) -> list[TrainingItem]:
    """Reproducibly pick `cap` items from the input list."""
    if len(items) <= cap:
        return items
    rng = np.random.default_rng(seed=seed)
    idx = np.arange(len(items))
    rng.shuffle(idx)
    keep = sorted(idx[:cap])
    return [items[i] for i in keep]


def stratified_split(
    items: list[TrainingItem], fractions: tuple[float, float, float], seed: int
) -> dict[str, list[TrainingItem]]:
    """Group by source, deterministically shuffle each group, split each.

    Each split contains the requested proportion of every source. No
    image appears in two splits (shuffle then index-slice on unique ids).
    """
    rng = np.random.default_rng(seed=seed)
    by_source: dict[str, list[TrainingItem]] = defaultdict(list)
    for item in items:
        by_source[item.source].append(item)

    splits: dict[str, list[TrainingItem]] = {name: [] for name in SPLIT_NAMES}
    for source in sorted(by_source):
        group = by_source[source]
        n = len(group)
        idx = np.arange(n)
        rng.shuffle(idx)
        n_train = int(round(fractions[0] * n))
        n_val = int(round(fractions[1] * n))
        n_cal = n - n_train - n_val
        if min(n_train, n_val, n_cal) <= 0:
            raise ValueError(
                f"source '{source}' has too few items (n={n}) for an "
                f"{fractions} split: would yield train={n_train} val={n_val} "
                f"calibration={n_cal}. Increase the source or adjust fractions."
            )
        cuts = (n_train, n_train + n_val)
        for name, ix in zip(
            SPLIT_NAMES,
            (idx[: cuts[0]], idx[cuts[0] : cuts[1]], idx[cuts[1] :]),
        ):
            splits[name].extend(group[i] for i in ix)
    return splits


def _link_or_copy(src: Path, dst: Path) -> None:
    try:
        os.symlink(src.resolve(), dst)
    except OSError:
        shutil.copy2(src, dst)


def write_split(items: list[TrainingItem], out_dir: Path) -> None:
    img_out = out_dir / "images"
    lbl_out = out_dir / "labels"
    img_out.mkdir(parents=True)
    lbl_out.mkdir(parents=True)
    for item in items:
        ext = item.image_path.suffix
        _link_or_copy(item.image_path, img_out / f"{item.output_stem}{ext}")
        (lbl_out / f"{item.output_stem}.txt").write_text(
            "\n".join(item.label_lines) + "\n", encoding="utf-8"
        )


def write_provenance_manifest(
    splits: dict[str, list[TrainingItem]], manifest_path: Path
) -> None:
    """Per-source provenance for OOD bucket exclusion (addendum spec)."""
    by_source: dict[str, list[str]] = defaultdict(list)
    for split_items in splits.values():
        for item in split_items:
            by_source[item.source].append(item.provenance_id)

    manifest = {
        "doordetect": {
            "open_images_ids": sorted(set(by_source.get("doordetect", []))),
            "count": len(by_source.get("doordetect", [])),
            "note": (
                "Filenames are Open Images V4 IDs. The OOD bucket script "
                "(step D) must skip any Open Images V7 image whose ID is "
                "in this list."
            ),
        },
        "giraff_x": {
            "internal_ids": sorted(set(by_source.get("giraff_x", []))),
            "count": len(by_source.get("giraff_x", [])),
            "note": "Robot-internal sample IDs; no Open Images overlap risk.",
        },
        "gibson": {
            "internal_ids": sorted(set(by_source.get("gibson", []))),
            "count": len(by_source.get("gibson", [])),
            "note": "Synthetic-render sample IDs; no Open Images overlap risk.",
        },
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, help="path to YAML config")
    args = p.parse_args()

    cfg_path = Path(args.config).resolve()
    repo_root = cfg_path.parent.parent
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    try:
        seed = int(cfg["dataset"]["split_seed"])
        gibson_cap = int(cfg["dataset"]["gibson_cap"])
    except KeyError as e:
        sys.exit(f"missing required config key: {e}. See configs/default.yaml.")

    doordetect_root = repo_root / "data" / "doordetect"
    giraff_root = repo_root / "data" / "antonazzi" / "final_doors_dataset_real"
    gibson_root = repo_root / "data" / "antonazzi" / "final_doors_dataset"
    output = repo_root / "data" / "doorway_combined"
    manifest_path = repo_root / "data" / "manifests" / "training_image_ids.json"

    print("loading DoorDetect...")
    dd_items = load_doordetect(doordetect_root)
    print(f"  {len(dd_items)} door-bearing images")

    print("loading Giraff-X...")
    gx_items = load_antonazzi(giraff_root, "giraff_x", "gx")
    print(f"  {len(gx_items)} door-bearing samples")

    print("loading Gibson...")
    gb_items_full = load_antonazzi(gibson_root, "gibson", "gb")
    print(f"  {len(gb_items_full)} door-bearing samples (pre-cap)")
    gb_items = _deterministic_subsample(gb_items_full, gibson_cap, seed=seed)
    if len(gb_items) < len(gb_items_full):
        print(f"  capped Gibson to {len(gb_items)} via deterministic subsample")

    all_items = dd_items + gx_items + gb_items
    print(f"combined pool: {len(all_items)} door-bearing images")

    splits = stratified_split(all_items, SPLIT_FRACTIONS, seed=seed)
    for name, lst in splits.items():
        by_src: dict[str, int] = defaultdict(int)
        for it in lst:
            by_src[it.source] += 1
        print(f"  {name}: {len(lst)} ({dict(by_src)})")

    if output.exists():
        print(f"removing existing output dir {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True)

    for name, lst in splits.items():
        write_split(lst, output / name)

    data_yaml = {
        "path": str(output),
        "train": "train/images",
        "val": "val/images",
        "nc": 1,
        "names": ["door"],
    }
    (output / "data.yaml").write_text(
        yaml.safe_dump(data_yaml, sort_keys=False), encoding="utf-8"
    )
    print(f"wrote {output / 'data.yaml'}")

    write_provenance_manifest(splits, manifest_path)
    print(f"wrote provenance manifest -> {manifest_path}")
    print("done.")


if __name__ == "__main__":
    main()
