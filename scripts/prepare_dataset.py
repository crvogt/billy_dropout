"""Build the three-bucket test set into data/.

Buckets:
  - in_distribution_pos: copy-link robot_dataset/test/*.jpg
  - in_distribution_neg: copy-link selected non-robot scenes from
      ros-llm-docker/llm-agent/vision_benchmark/test_images/objects/
  - ood: synthesize from robot_dataset/test/ via reproducible
      augmentations (occlusion patches, gaussian blur, low-light gamma,
      JPEG recompression, viewpoint warp). Optionally include a
      pre-collected distractors sub-bucket if `--distractors-dir` is
      passed.

Each bucket gets data/<bucket>/manifest.json. The script is idempotent
and seed-determined.

Implementation lands in step 6.
"""

from __future__ import annotations

import argparse


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--distractors-dir", default=None,
                   help="optional directory of pre-collected OOD distractor images")
    args = p.parse_args()
    raise NotImplementedError("dataset prep is implemented in step 6")


if __name__ == "__main__":
    main()
