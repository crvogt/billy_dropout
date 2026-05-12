"""Test-set construction for the agent experiment.

Builds a list of (image_path, bucket) items sampled deterministically
from the prepared datasets. Buckets:

  - in_distribution_pos:  doorway val split (mixed Giraff-X / Gibson /
                          DoorDetect; agent sees a calibrated detection).
  - in_distribution_neg:  not yet collected. Skipped from the test set
                          when no source is available.
  - ood_distractors:      data/ood_full/distractors/
  - ood_partial:          data/ood_full/partial/
  - ood_novel:            data/ood_full/novel/

`build_test_set` is deterministic: same `(seed, n_per_bucket)` always
produces the same image set — required so a resumed run works against
the same test set as its interrupted predecessor.

Determinism caveat: the sample is stable *for a fixed directory state*.
Adding or removing images in val/ or ood_full/* between launch and
resume would shift the candidate ordering and change the sample. Don't
modify the dataset mid-run.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass(frozen=True)
class TestSetItem:
    """One image in the experiment test set."""

    image_path: Path
    bucket: str


def _sample_dir(
    image_dir: Path, n: int, rng: np.random.Generator
) -> list[Path]:
    """Deterministic sample of up to `n` images from a directory.

    Sorted-then-sampled so the result is filesystem-order-independent.
    Returns all images if there are fewer than `n` on disk.
    """
    if not image_dir.is_dir():
        return []
    imgs = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not imgs:
        return []
    take = min(n, len(imgs))
    idx = rng.choice(len(imgs), size=take, replace=False)
    return [imgs[int(i)] for i in sorted(idx)]


def build_test_set(
    *,
    doorway_dataset_root: Path,
    ood_root: Path,
    n_per_bucket: int,
    seed: int,
) -> list[TestSetItem]:
    """Sample the experiment test set.

    Args:
        doorway_dataset_root: data/doorway_combined/ root.
        ood_root: data/ood_full/ root.
        n_per_bucket: target images per bucket. If a bucket has fewer on
            disk, take all available.
        seed: numpy RNG seed.
    """
    rng = np.random.default_rng(seed=seed)
    items: list[TestSetItem] = []

    val_dir = doorway_dataset_root / "val" / "images"
    for p in _sample_dir(val_dir, n_per_bucket, rng):
        items.append(TestSetItem(image_path=p, bucket="in_distribution_pos"))

    for sub in ("distractors", "partial", "novel"):
        sub_dir = ood_root / sub
        for p in _sample_dir(sub_dir, n_per_bucket, rng):
            items.append(TestSetItem(image_path=p, bucket=f"ood_{sub}"))

    # in_distribution_neg has no source on disk yet. When MCIndoor20000
    # or Places365-indoors imagery is collected, add another sample
    # block here.

    return items
