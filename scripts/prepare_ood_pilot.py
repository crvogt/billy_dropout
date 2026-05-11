"""Build the Step B OOD pilot from Open Images V7 (validation split).

Downloads bbox + image-URL metadata CSVs once, filters by the addendum's
rules into three sub-buckets, downloads N images per bucket, and runs a
pre-check that >= 50% of pilot images produce a detection above the
deterministic detector's runtime conf threshold. If fewer fire, exits
with refinement hints — under-firing imagery wastes the variance signal
test (you want detector confidence to be in the regime where the agent
would normally act).

Sub-buckets (addendum-locked):
  distractors  — Mirror, Picture frame, Cabinetry, Window. Door-shaped
                 objects that are NOT doors. Filter: aspect h/w in
                 [1.0, 3.0], area >= 5% of image, no co-Door annotation.
  partial      — Door class with distributional-tail filter: area < 8%
                 OR width < 60% of height (truncated) OR >=2 Door boxes
                 in image.
  novel        — Door class with environment-shift annotations:
                 concurrent Building / House / Skyscraper / Office
                 building annotation in the image.

Excludes any OIv4 ImageID present in
data/manifests/training_image_ids.json (DoorDetect training set was
sourced from OIv4; defensive even though we pull from validation here).

Usage:
    python scripts/prepare_ood_pilot.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml


# ---------------------------------------------------------------------------
# Constants: class IDs + URL endpoints + filter thresholds.
# ---------------------------------------------------------------------------

OIV4_VALIDATION_BASE = "https://storage.googleapis.com/openimages/2018_04/validation"
OIV7_CLASS_DESCRIPTIONS_URL = (
    "https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions-boxable.csv"
)
BBOX_CSV_URL = f"{OIV4_VALIDATION_BASE}/validation-annotations-bbox.csv"
IMAGES_CSV_URL = f"{OIV4_VALIDATION_BASE}/validation-images-with-rotation.csv"

# Class IDs verified against oidv7-class-descriptions-boxable.csv. The
# addendum's /m/0642b4 was Cupboard; the description "tall closed cabinets"
# clearly intends Cabinetry, so we use the verified Cabinetry ID instead.
# Office building ID was not provided in the addendum text and is verified
# here too.
DOOR_CLASS = "/m/02dgv"          # Door
DISTRACTOR_CLASSES = {
    "/m/054_l":  "Mirror",
    "/m/06z37_": "Picture frame",
    "/m/01s105": "Cabinetry",    # was /m/0642b4 (Cupboard) — addendum errata
    "/m/0d4v4":  "Window",
}
ENV_SHIFT_CLASSES = {
    "/m/0cgh4":  "Building",          # 984 val annotations
    "/m/03jm5":  "House",             # 246
    "/m/079cl":  "Skyscraper",        # 57
    "/m/021sj1": "Office building",   # 45  (verified MID; addendum gave none)
    "/m/01fdzj": "Tower",             # 82  (added: exterior landmark)
    "/m/0d5gx":  "Castle",            # 12
    "/m/04h7h":  "Lighthouse",        # 8
    "/m/0crjs":  "Convenience store", # 65  (storefront → exterior context)
}

# Bucket-assignment precedence when an image qualifies for multiple buckets.
# distractors uses disjoint LabelName classes from partial/novel, so its
# position here is structurally inert. The real claim is novel-before-partial:
# both pull from the Door class, and overlap is possible whenever a Door has
# an env-shift co-annotation AND meets the size/aspect tail constraint —
# higher-priority bucket wins.
BUCKET_PRIORITY = ("distractors", "novel", "partial")

# Filter thresholds.
DISTRACTOR_ASPECT_HW_MIN = 1.0
DISTRACTOR_ASPECT_HW_MAX = 3.0
DISTRACTOR_MIN_AREA = 0.05
DISTRACTOR_MAX_AREA = 0.50       # exclude scene-level whole-frame labels
PARTIAL_MAX_AREA = 0.08
PARTIAL_WIDTH_HEIGHT_RATIO = 0.60
# OIv7 boxable annotations sometimes contain whole-image scene-level labels
# (bbox area ~= 1.0) that aren't object instances. These are noise for our
# purpose. Cap any bucket at this max-area threshold.
SCENE_LEVEL_MAX_AREA = 0.80

N_PER_BUCKET = 10
MIN_FIRE_RATE = 0.50  # >= 50% of pilot images must produce a confident detection


@dataclass
class Candidate:
    image_id: str
    source_class: str
    bbox: tuple[float, float, float, float]  # normalized XMin, YMin, XMax, YMax
    bbox_area: float
    rationale: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _check_curl_present() -> None:
    """The script shells out to curl for HTTP. Fail fast if it's missing."""
    if shutil.which("curl") is None:
        sys.exit(
            "curl not found on PATH. Install curl or modify the script to "
            "use urllib.request / requests for downloads."
        )


def _download(url: str, dst: Path, timeout: int = 60) -> None:
    """One-shot HTTP download; uses curl since requests isn't a hard dep."""
    import subprocess

    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        print(f"  cached: {dst}")
        return
    print(f"  downloading {url} -> {dst}")
    rc = subprocess.run(
        [
            "curl", "-sSL", "--fail",
            "--max-time", str(timeout),
            "-o", str(dst),
            url,
        ],
        check=False,
    ).returncode
    if rc != 0:
        if dst.exists():
            dst.unlink()
        sys.exit(f"download failed (rc={rc}): {url}")


def _resolve_device(requested: str) -> str:
    import torch

    if requested == "cuda" and not torch.cuda.is_available():
        print("WARNING: cuda requested but unavailable; falling back to cpu")
        return "cpu"
    return requested


# ---------------------------------------------------------------------------
# Filter logic
# ---------------------------------------------------------------------------


def _build_distractor_candidates(bbox_df, excluded_ids: set[str]) -> list[Candidate]:
    """Mirror/Picture frame/Cabinetry/Window with door-shape aspect + size."""
    out: list[Candidate] = []
    # Pre-compute set of ImageIDs that have a Door annotation (excluded for distractors).
    door_image_ids = set(bbox_df[bbox_df["LabelName"] == DOOR_CLASS]["ImageID"])
    distractor_df = bbox_df[bbox_df["LabelName"].isin(DISTRACTOR_CLASSES)]
    for _, row in distractor_df.iterrows():
        iid = row["ImageID"]
        if iid in excluded_ids or iid in door_image_ids:
            continue
        x1, x2, y1, y2 = row["XMin"], row["XMax"], row["YMin"], row["YMax"]
        w = max(1e-6, x2 - x1)
        h = max(1e-6, y2 - y1)
        aspect_hw = h / w
        area = w * h
        if not (DISTRACTOR_ASPECT_HW_MIN <= aspect_hw <= DISTRACTOR_ASPECT_HW_MAX):
            continue
        if area < DISTRACTOR_MIN_AREA or area > DISTRACTOR_MAX_AREA:
            continue
        out.append(
            Candidate(
                image_id=iid,
                source_class=DISTRACTOR_CLASSES[row["LabelName"]],
                bbox=(x1, y1, x2, y2),
                bbox_area=area,
                rationale=(
                    f"class={DISTRACTOR_CLASSES[row['LabelName']]} "
                    f"aspect_hw={aspect_hw:.2f} area={area:.3f}"
                ),
            )
        )
    return out


def _build_partial_candidates(bbox_df, excluded_ids: set[str]) -> list[Candidate]:
    """Door class, distributional-tail filter."""
    out: list[Candidate] = []
    door_df = bbox_df[bbox_df["LabelName"] == DOOR_CLASS]
    door_counts_per_image = door_df.groupby("ImageID").size().to_dict()
    for _, row in door_df.iterrows():
        iid = row["ImageID"]
        if iid in excluded_ids:
            continue
        x1, x2, y1, y2 = row["XMin"], row["XMax"], row["YMin"], row["YMax"]
        w = max(1e-6, x2 - x1)
        h = max(1e-6, y2 - y1)
        area = w * h
        if area > SCENE_LEVEL_MAX_AREA:
            continue  # scene-level whole-frame Door label, not an instance
        wh_ratio = w / h
        n_doors = door_counts_per_image.get(iid, 0)
        reasons = []
        if area < PARTIAL_MAX_AREA:
            reasons.append(f"small_area={area:.3f}")
        if wh_ratio < PARTIAL_WIDTH_HEIGHT_RATIO:
            reasons.append(f"truncated_wh={wh_ratio:.2f}")
        if n_doors >= 2:
            reasons.append(f"cluttered_n_doors={n_doors}")
        if not reasons:
            continue
        out.append(
            Candidate(
                image_id=iid,
                source_class="Door",
                bbox=(x1, y1, x2, y2),
                bbox_area=area,
                rationale="; ".join(reasons),
            )
        )
    return out


def _build_novel_candidates(bbox_df, excluded_ids: set[str]) -> list[Candidate]:
    """Door class with environment-shift co-annotations."""
    out: list[Candidate] = []
    env_image_ids = set(bbox_df[bbox_df["LabelName"].isin(ENV_SHIFT_CLASSES)]["ImageID"])
    door_df = bbox_df[bbox_df["LabelName"] == DOOR_CLASS]
    for _, row in door_df.iterrows():
        iid = row["ImageID"]
        if iid in excluded_ids or iid not in env_image_ids:
            continue
        x1, x2, y1, y2 = row["XMin"], row["XMax"], row["YMin"], row["YMax"]
        w = max(1e-6, x2 - x1)
        h = max(1e-6, y2 - y1)
        area = w * h
        if area > SCENE_LEVEL_MAX_AREA:
            continue  # scene-level whole-frame Door label, not an instance
        out.append(
            Candidate(
                image_id=iid,
                source_class="Door+env_shift",
                bbox=(x1, y1, x2, y2),
                bbox_area=area,
                rationale="exterior/building co-annotation",
            )
        )
    return out


def _select_top_n_per_image(
    cands: list[Candidate], n: int
) -> list[Candidate]:
    """Pick N image_ids, prioritizing larger bbox area (more likely to fire).

    Deduplicates by image_id (keeping the largest-area candidate per image)
    so the pilot is N distinct images.
    """
    by_id: dict[str, Candidate] = {}
    for c in cands:
        existing = by_id.get(c.image_id)
        if existing is None or c.bbox_area > existing.bbox_area:
            by_id[c.image_id] = c
    ordered = sorted(by_id.values(), key=lambda c: -c.bbox_area)
    return ordered[:n]


# ---------------------------------------------------------------------------
# Download + precheck
# ---------------------------------------------------------------------------


def _download_pilot_image(image_id: str, url: str, dst: Path, timeout: int = 30) -> bool:
    """Download one image. Returns True on success, False on any failure."""
    import subprocess

    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 0:
        return True
    rc = subprocess.run(
        [
            "curl", "-sSL", "--fail",
            "--max-time", str(timeout),
            "-o", str(dst),
            url,
        ],
        check=False,
    ).returncode
    if rc != 0:
        if dst.exists():
            dst.unlink()
        return False
    if dst.stat().st_size == 0:
        dst.unlink()
        return False
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, help="path to YAML config")
    p.add_argument(
        "--metadata-dir",
        default="data/openimages_metadata",
        help="cache dir for OIv7 CSVs (~32 MB total, downloaded once)",
    )
    p.add_argument(
        "--pilot-dir",
        default="data/ood_pilot",
        help="output dir for pilot images + manifest",
    )
    p.add_argument(
        "--n-per-bucket", type=int, default=N_PER_BUCKET, help="images per sub-bucket"
    )
    p.add_argument(
        "--exclude-manifest",
        default=None,
        help="optional path to an existing manifest.json whose image_ids "
             "should also be excluded (e.g. data/ood_pilot/manifest.json "
             "when building the full set so it doesn't reuse pilot images)",
    )
    args = p.parse_args()

    cfg_path = Path(args.config).resolve()
    repo_root = cfg_path.parent.parent
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    try:
        weights_path = (repo_root / cfg["paths"]["yolo_weights"]).resolve()
        imgsz = int(cfg["perception"]["imgsz"])
        conf_threshold = float(cfg["perception"]["conf_threshold"])
        iou_threshold = float(cfg["perception"]["iou_threshold"])
        device = _resolve_device(str(cfg["perception"]["device"]))
    except KeyError as e:
        sys.exit(f"missing required config key: {e}")

    if not weights_path.exists():
        sys.exit(f"trained checkpoint not found: {weights_path}")

    metadata_dir = repo_root / args.metadata_dir
    pilot_dir = repo_root / args.pilot_dir
    manifest_path = repo_root / "data" / "manifests" / "training_image_ids.json"

    _check_curl_present()

    # 1. Cache OIv7 metadata CSVs.
    print("--- caching OIv7 metadata ---")
    bbox_csv = metadata_dir / "validation-annotations-bbox.csv"
    images_csv = metadata_dir / "validation-images-with-rotation.csv"
    _download(BBOX_CSV_URL, bbox_csv)
    _download(IMAGES_CSV_URL, images_csv)

    # 2. Load exclusion list (OI IDs from training corpus).
    print("--- loading exclusion list ---")
    excluded_ids: set[str] = set()
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        excluded_ids = set(manifest.get("doordetect", {}).get("open_images_ids", []))
    print(f"  {len(excluded_ids)} training image IDs to exclude")
    if args.exclude_manifest:
        extra_path = (repo_root / args.exclude_manifest).resolve()
        if not extra_path.exists():
            sys.exit(f"--exclude-manifest path not found: {extra_path}")
        try:
            extra = json.loads(extra_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            sys.exit(f"--exclude-manifest is not valid JSON: {extra_path} ({e})")
        if not isinstance(extra, list):
            sys.exit(
                f"--exclude-manifest must contain a JSON array of objects with "
                f"an 'image_id' key; got {type(extra).__name__} at {extra_path}"
            )
        extra_ids = {entry["image_id"] for entry in extra if isinstance(entry, dict) and "image_id" in entry}
        excluded_ids |= extra_ids
        print(f"  + {len(extra_ids)} image IDs from {extra_path} ({len(excluded_ids)} total exclusions)")

    # 3. Parse bbox CSV.
    print("--- parsing bbox annotations ---")
    import pandas as pd

    bbox_df = pd.read_csv(bbox_csv, usecols=["ImageID", "LabelName", "XMin", "XMax", "YMin", "YMax"])
    print(f"  {len(bbox_df)} total bboxes; {bbox_df['ImageID'].nunique()} unique images")

    # 4. Build sub-buckets.
    print("--- filtering candidates ---")
    candidates_by_bucket = {
        "distractors": _build_distractor_candidates(bbox_df, excluded_ids),
        "partial":     _build_partial_candidates(bbox_df, excluded_ids),
        "novel":       _build_novel_candidates(bbox_df, excluded_ids),
    }
    for bucket, cands in candidates_by_bucket.items():
        unique_ids = len({c.image_id for c in cands})
        print(f"  {bucket}: {len(cands)} candidate annotations across {unique_ids} unique images")

    # 5. Select top-N per bucket, with cross-bucket dedup by precedence.
    # novel ⊂ partial (both Door class with overlapping conditions). Without
    # this, deterministic largest-area selection picks the same Doors for
    # both, double-counting image_ids in the output.
    selected_by_bucket: dict[str, list[Candidate]] = {}
    claimed_ids: set[str] = set()
    for bucket in BUCKET_PRIORITY:
        cands = [c for c in candidates_by_bucket[bucket] if c.image_id not in claimed_ids]
        chosen = _select_top_n_per_image(cands, args.n_per_bucket)
        selected_by_bucket[bucket] = chosen
        claimed_ids.update(c.image_id for c in chosen)
        if len(chosen) < args.n_per_bucket:
            print(
                f"  WARNING: only {len(chosen)} unique images for bucket '{bucket}' "
                f"(requested {args.n_per_bucket}). Filter may be too strict, "
                "or higher-priority buckets claimed shared candidates."
            )

    # 6. Look up image URLs.
    print("--- resolving image URLs ---")
    images_df = pd.read_csv(images_csv, usecols=["ImageID", "OriginalURL"])
    url_by_id = dict(zip(images_df["ImageID"], images_df["OriginalURL"]))

    # 7. Download images per bucket.
    if pilot_dir.exists():
        print(f"  removing existing {pilot_dir}")
        shutil.rmtree(pilot_dir)
    pilot_dir.mkdir(parents=True)

    manifest_entries: list[dict] = []
    for bucket, chosen in selected_by_bucket.items():
        bucket_dir = pilot_dir / bucket
        print(f"--- downloading {bucket} ({len(chosen)} images) ---")
        downloaded = 0
        for cand in chosen:
            url = url_by_id.get(cand.image_id)
            if not url:
                print(f"  skip {cand.image_id}: no URL in metadata")
                continue
            dst = bucket_dir / f"{cand.image_id}.jpg"
            ok = _download_pilot_image(cand.image_id, url, dst)
            if not ok:
                print(f"  skip {cand.image_id}: download failed")
                continue
            downloaded += 1
            manifest_entries.append({
                "image_id": cand.image_id,
                "bucket": bucket,
                "source_class": cand.source_class,
                "bbox_normalized_xyxy": list(cand.bbox),
                "bbox_area": cand.bbox_area,
                "rationale": cand.rationale,
                "image_path": str(dst.relative_to(repo_root)),
                "original_url": url,
            })
        print(f"  {bucket}: downloaded {downloaded}/{len(chosen)}")

    pilot_manifest_path = pilot_dir / "manifest.json"
    pilot_manifest_path.write_text(
        json.dumps(manifest_entries, indent=2), encoding="utf-8"
    )

    # 8. Fire-rate precheck.
    print()
    print(f"--- precheck: running DeterministicYOLO @ conf>={conf_threshold} ---")
    from uagent.perception.detector import DeterministicYOLO

    detector = DeterministicYOLO(
        weights_path,
        imgsz=imgsz,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        device=device,
    )
    fires_by_bucket: dict[str, int] = defaultdict(int)
    total_by_bucket: dict[str, int] = defaultdict(int)
    for entry in manifest_entries:
        bucket = entry["bucket"]
        img = cv2.imread(str(repo_root / entry["image_path"]))
        if img is None:
            continue
        total_by_bucket[bucket] += 1
        dets = detector.predict(img)
        if dets:
            fires_by_bucket[bucket] += 1
            entry["precheck_fires"] = len(dets)
            entry["precheck_max_conf"] = max(d.confidence for d in dets)
        else:
            entry["precheck_fires"] = 0
            entry["precheck_max_conf"] = 0.0

    # Re-save manifest with precheck columns.
    pilot_manifest_path.write_text(
        json.dumps(manifest_entries, indent=2), encoding="utf-8"
    )

    print()
    print("=== fire rate per bucket ===")
    total_fires = 0
    total_count = 0
    weak_buckets: list[str] = []
    for bucket in selected_by_bucket:
        fires = fires_by_bucket[bucket]
        total = total_by_bucket[bucket]
        total_fires += fires
        total_count += total
        rate = fires / total if total else 0.0
        marker = ""
        if total > 0 and rate < 0.30:
            marker = "  ← weak bucket (<30% fires)"
            weak_buckets.append(bucket)
        print(f"  {bucket:<12} {fires}/{total} ({rate:.0%}){marker}")
    combined_rate = total_fires / total_count if total_count else 0.0
    print(f"  {'combined':<12} {total_fires}/{total_count} ({combined_rate:.0%})")
    if weak_buckets:
        print(
            f"  warning: {len(weak_buckets)} bucket(s) below 30% fire rate "
            f"({', '.join(weak_buckets)}). Combined rate alone is insufficient "
            "to validate signal across all OOD sub-types; consider refining "
            "filters for these buckets specifically."
        )

    print()
    print(f"manifest -> {pilot_manifest_path}")

    if combined_rate < MIN_FIRE_RATE:
        sys.exit(
            f"\nERROR: combined fire rate {combined_rate:.0%} is below the "
            f"{MIN_FIRE_RATE:.0%} threshold. The detector ignores too many "
            "of these images; the variance test would be hollow.\n"
            "Refinement options (edit the script constants and re-run):\n"
            "  - widen DISTRACTOR_ASPECT_HW_MIN/MAX (allow more box shapes)\n"
            "  - lower DISTRACTOR_MIN_AREA (smaller objects)\n"
            "  - relax PARTIAL filters (broader 'tail' definition)\n"
            "Or hand-curate the pilot from these candidates and disable this "
            "check by passing --n-per-bucket 0."
        )

    print()
    print("done.")


if __name__ == "__main__":
    main()
