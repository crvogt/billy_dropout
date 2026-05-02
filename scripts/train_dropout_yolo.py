"""Branch B: train YOLOv8n with nn.Dropout layers in the detection head.

Inserts dropout into the head, trains via the ultralytics MAP objective
(cross-entropy + L2 weight decay parameterizing prior precision), saves
weights to configs.paths.yolo_dropout_weights.

This script exists in stub form during step 2; user has deferred the
actual training run. Run when ready:

    python scripts/train_dropout_yolo.py --config configs/default.yaml \\
        --dropout-rate 0.10 --epochs 200 --imgsz 640
"""

from __future__ import annotations

import argparse


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, help="path to YAML config")
    p.add_argument("--dropout-rate", type=float, default=0.10)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--inventory-only", action="store_true",
                   help="just inspect existing checkpoints for nn.Dropout layers")
    args = p.parse_args()
    raise NotImplementedError("training is deferred per user direction; stub only")


if __name__ == "__main__":
    main()
