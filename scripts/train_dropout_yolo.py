"""Fine-tune YOLOv8n with embedded dropout in the cls head, on doorways.

Loads the starting checkpoint (yolov8n.pt by default — auto-downloaded by
ultralytics if missing), injects six nn.Dropout2d modules into the
classification branches of the Detect head per the locked placement
spec (uagent/perception/dropout.py), fine-tunes for the configured
number of epochs on the doorway data.yaml, and copies the best.pt
checkpoint to training.output_checkpoint.

Step A.4-6 of the experiment plan.

Run on the AGX Thor or a workstation GPU; not viable on CPU.

Usage:
    python scripts/train_dropout_yolo.py --config configs/default.yaml

Optional --inventory-only flag inspects the starting checkpoint's
existing dropout layer count without training (sanity check before
booking GPU time).
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import yaml


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, help="path to YAML config")
    p.add_argument(
        "--inventory-only",
        action="store_true",
        help="Load starting weights, count dropout modules, exit. No training.",
    )
    args = p.parse_args()

    cfg_path = Path(args.config).resolve()
    repo_root = cfg_path.parent.parent
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    try:
        tcfg = cfg["training"]
        starting_weights = tcfg["starting_weights"]
        data_yaml = (repo_root / tcfg["data_yaml"]).resolve()
        output_ckpt = (repo_root / tcfg["output_checkpoint"]).resolve()
        epochs = int(tcfg["epochs"])
        imgsz = int(tcfg["imgsz"])
        batch = int(tcfg["batch"])
        patience = int(tcfg["patience"])
        device = str(tcfg["device"])
        dropout_p = float(tcfg["dropout_p"])
        project = (repo_root / tcfg["project"]).resolve()
        run_name = str(tcfg["name"])
    except KeyError as e:
        sys.exit(f"missing required training config key: {e}. See configs/default.yaml.")

    if not args.inventory_only and not data_yaml.exists():
        sys.exit(
            f"data.yaml not found at {data_yaml}. "
            "Run scripts/prepare_doorway_data.py first."
        )

    from ultralytics import YOLO

    from uagent.perception.dropout import (
        count_dropout_modules,
        inject_dropout_into_yolov8_cls_head,
    )

    print(f"loading starting weights: {starting_weights}")
    yolo = YOLO(starting_weights)

    n_pre = count_dropout_modules(yolo.model)
    print(f"pre-injection Dropout2d count: {n_pre}")
    if n_pre != 0:
        print(
            f"  warning: starting checkpoint already has {n_pre} Dropout2d modules. "
            "Injection will add another six on top — verify that is intended."
        )

    n_added = inject_dropout_into_yolov8_cls_head(yolo.model, p=dropout_p)
    n_post = count_dropout_modules(yolo.model)
    print(f"injected {n_added} modules; post-injection Dropout2d count: {n_post}")
    if n_post - n_pre != 6:
        sys.exit(f"injection failed sanity check: delta {n_post - n_pre}, expected 6")

    if args.inventory_only:
        print("inventory-only mode; exiting before training.")
        return

    output_ckpt.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"starting fine-tune: epochs={epochs} imgsz={imgsz} batch={batch} "
        f"device={device} data={data_yaml}"
    )
    yolo.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        patience=patience,
        device=device,
        project=str(project),
        name=run_name,
        exist_ok=True,
        verbose=True,
    )

    best = project / run_name / "weights" / "best.pt"
    if not best.exists():
        last = project / run_name / "weights" / "last.pt"
        if last.exists():
            print(f"best.pt missing; falling back to last.pt at {last}")
            best = last
        else:
            sys.exit(f"no checkpoint at {best} or last.pt; training failed silently?")

    # Round-trip sanity check: reload the saved checkpoint and verify the
    # dropout modules survived ultralytics' serialization. Tight invariant —
    # post-injection count must equal post-reload count, regardless of how
    # many dropouts the starting weights had.
    reloaded = YOLO(str(best))
    n_reload = count_dropout_modules(reloaded.model)
    if n_reload != n_post:
        sys.exit(
            f"saved checkpoint at {best} has {n_reload} Dropout2d modules "
            f"but the trained model had {n_post}; ultralytics did not preserve "
            "the architecture across save/load. Investigate before relying on "
            "this checkpoint."
        )

    shutil.copy2(best, output_ckpt)
    print(f"copied {best} -> {output_ckpt}")
    print(f"reloaded checkpoint Dropout2d count: {n_reload}")
    print("done.")


if __name__ == "__main__":
    main()
