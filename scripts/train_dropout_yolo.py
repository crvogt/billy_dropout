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

    # ---- callback-based dropout re-injection ----
    # YOLO.train() rebuilds self.trainer.model from yolo.model.yaml + the
    # in-memory state_dict. The rebuild discards architectural mods that
    # aren't in the YAML — including our six Dropout2d modules — so the
    # in-memory injection above evaporates by the time training starts.
    #
    # We re-inject inside an `on_pretrain_routine_start` callback which
    # fires AFTER the rebuild but BEFORE ModelEMA construction, so the
    # EMA snapshot and every saved checkpoint thereafter contain the
    # dropout modules. Verified empirically against ultralytics 8.3.20.
    def _reinject_into_trainer(trainer) -> None:
        existing = count_dropout_modules(trainer.model)
        if existing == 0:
            n = inject_dropout_into_yolov8_cls_head(trainer.model, p=dropout_p)
            print(
                f"[callback on_pretrain_routine_start] injected {n} "
                "Dropout2d modules into trainer.model (post-rebuild)"
            )
        else:
            print(
                f"[callback on_pretrain_routine_start] trainer.model already "
                f"has {existing} Dropout2d modules; skipping injection"
            )

    def _verify_dropout_active(trainer) -> None:
        n = count_dropout_modules(trainer.model)
        if n != 6:
            raise RuntimeError(
                f"dropout missing from trainer.model at training start: "
                f"count={n}, expected 6. Callback ordering may have changed "
                "in this ultralytics version — investigate before relying on "
                "the checkpoint."
            )
        # EMA snapshot must also have dropouts. If ModelEMA was constructed
        # before our inject callback fired (e.g. a future ultralytics reorder),
        # save_model will serialize a dropout-less EMA and the round-trip
        # check would only catch it post-training. Pre-flight here.
        ema = getattr(trainer, "ema", None)
        ema_module = getattr(ema, "ema", None) if ema is not None else None
        if ema_module is not None:
            n_ema = count_dropout_modules(ema_module)
            if n_ema != 6:
                raise RuntimeError(
                    f"EMA snapshot is dropout-less: count={n_ema}, expected 6. "
                    "ModelEMA was likely constructed before the inject callback "
                    "fired — ultralytics callback ordering has changed."
                )
            print(
                f"[callback on_train_start] verified Dropout2d count: "
                f"{n} (model), {n_ema} (ema)"
            )
        else:
            # On single-GPU runs (our case on Thor), ultralytics always
            # constructs ModelEMA in _setup_train. If we get here, ultralytics
            # behavior has changed in a way that bypasses EMA — fail loudly
            # rather than ship a checkpoint whose EMA copy is dropout-less.
            raise RuntimeError(
                "trainer.ema is missing or unset at on_train_start. "
                "ultralytics ordinarily constructs ModelEMA before this "
                "callback fires; the absence here indicates a version skew "
                "that needs investigation before relying on the checkpoint."
            )

    yolo.add_callback("on_pretrain_routine_start", _reinject_into_trainer)
    yolo.add_callback("on_train_start", _verify_dropout_active)

    # Capture mtime of any pre-existing best.pt so we can detect a stale
    # checkpoint (this run crashed before producing a new one and we'd
    # otherwise reload the previous run's output silently).
    weights_dir = project / run_name / "weights"
    prior_best = weights_dir / "best.pt"
    prior_last = weights_dir / "last.pt"
    prior_best_mtime = prior_best.stat().st_mtime if prior_best.exists() else 0.0
    prior_last_mtime = prior_last.stat().st_mtime if prior_last.exists() else 0.0

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

    best = weights_dir / "best.pt"
    if not best.exists():
        last = weights_dir / "last.pt"
        if last.exists():
            if last.stat().st_mtime <= prior_last_mtime:
                sys.exit(
                    f"last.pt at {last} was not updated by this run "
                    f"(mtime {last.stat().st_mtime} <= prior {prior_last_mtime}). "
                    "A previous failed run may have left a stale checkpoint. "
                    f"Delete {weights_dir} and re-run."
                )
            print(f"best.pt missing; falling back to last.pt at {last}")
            best = last
        else:
            sys.exit(f"no checkpoint at {best} or last.pt; training failed silently?")
    elif best.stat().st_mtime <= prior_best_mtime:
        sys.exit(
            f"best.pt at {best} was not updated by this run "
            f"(mtime {best.stat().st_mtime} <= prior {prior_best_mtime}). "
            "A previous failed run may have left a stale checkpoint. "
            f"Delete {weights_dir} and re-run."
        )

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
