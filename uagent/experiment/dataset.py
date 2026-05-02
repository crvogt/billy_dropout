"""Three-bucket OOD test set loader.

Buckets:
  - in_distribution_pos: held-out images from robot_dataset/test/
  - in_distribution_neg: scenes with no robot, drawn from
      vision_benchmark/test_images/objects/ (per inventory phase)
  - ood:                augmented robot_dataset/test/ images plus, when
      collected, a distractors sub-bucket. Constructed by
      scripts/prepare_dataset.py.

Manifest format: data/<bucket>/manifest.json — list of
  {"image_path": str, "label": "robot" | "none", "source": str, ...}

Implementation lands in step 6.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

Bucket = Literal["in_distribution_pos", "in_distribution_neg", "ood"]


@dataclass(frozen=True)
class TestCase:
    image_path: Path
    label: Literal["robot", "none"]
    bucket: Bucket
    source: str                             # provenance (e.g. "robot_dataset/test")


def load_manifest(data_root: str | Path, bucket: Bucket) -> list[TestCase]:
    """Load all test cases for one bucket from data/<bucket>/manifest.json."""
    raise NotImplementedError("dataset loader is implemented in step 6")
