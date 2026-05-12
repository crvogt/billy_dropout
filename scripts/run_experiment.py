"""Orchestrate the agent experiment over (image × condition × trial × model).

Builds the test set via uagent.experiment.dataset.build_test_set, then
for each tuple invokes uagent.experiment.harness.run_single_trial and
appends one JSONL event to runs/<run-name>/events.jsonl.

Resume support
--------------
The script is crash-safe. On startup it scans the existing events.jsonl
(if any) at the configured runs path and skips any (image_path,
condition, trial_index, model) tuple already recorded. If a previous
run crashed mid-flight, re-running with the same --config and --run-name
picks up exactly where it stopped — no work is duplicated.

To restart from scratch (e.g. after a config change that invalidates
prior events), use a different --run-name or delete the existing
events.jsonl manually.

The test set itself is deterministic for a given (dataset.split_seed,
--n-per-bucket), so a resumed run uses the same image set as the
original.

Smoke mode
----------
--smoke sets n_per_bucket=5, trials_per_image=1, and uses the default
tier's model (configs/default.yaml::llm.default_tier). Use to verify
the harness end-to-end before launching the full sweep.

Model selection
---------------
--models accepts either tier names (resolved against
`llm.models.{small,medium,large}` in the config) or raw Ollama tags.
Tier names are convenient for the sweep:

    --models small
    --models small medium large

Raw tags still work for one-offs:

    --models qwen2.5:3b

Mixing is allowed; raw tags pass through unchanged.

Usage:
    # Smoke on the default tier
    python scripts/run_experiment.py --config configs/default.yaml --smoke

    # Full run on the medium tier
    python scripts/run_experiment.py --config configs/default.yaml \\
        --n-per-bucket 100 --models medium --run-name medium_full

    # Full sweep across all three tiers
    python scripts/run_experiment.py --config configs/default.yaml \\
        --n-per-bucket 100 --models small medium large --run-name sweep

    # Resume an interrupted run (just rerun the same command)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml


def _resolve_device(requested: str) -> str:
    import torch

    if requested == "cuda" and not torch.cuda.is_available():
        print(
            "WARNING: device='cuda' requested but unavailable; "
            "falling back to device='cpu'."
        )
        return "cpu"
    return requested


def _load_completed_keys(jsonl_path: Path) -> set[tuple[str, str, int, str]]:
    """Read existing events.jsonl and return resume keys."""
    keys: set[tuple[str, str, int, str]] = set()
    if not jsonl_path.exists():
        return keys
    with jsonl_path.open(encoding="utf-8") as f:
        for line_no, raw in enumerate(f, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                e = json.loads(raw)
                keys.add((
                    e["image_path"],
                    e["condition"],
                    int(e["trial_index"]),
                    e["model"],
                ))
            except (json.JSONDecodeError, KeyError, ValueError) as err:
                print(f"  WARNING: skipping malformed JSONL line {line_no}: {err}")
    return keys


def _append_event_atomic(jsonl_path: Path, line: str) -> None:
    """Append + fsync so a crash mid-flush still gives us the event."""
    import os

    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("a", encoding="utf-8") as f:
        f.write(line)
        f.flush()
        os.fsync(f.fileno())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, help="path to YAML config")
    p.add_argument(
        "--run-name", default="default",
        help="subdir name under runs_root/; used as the resume key namespace.",
    )
    p.add_argument(
        "--n-per-bucket", type=int, default=None,
        help="override images per bucket (default from config)",
    )
    p.add_argument(
        "--trials-per-image", type=int, default=None,
        help="override trials per image per condition (default from config)",
    )
    p.add_argument(
        "--models", nargs="+", default=None,
        help="LLM models or tier names (small/medium/large) to run. "
             "Tier names resolve via llm.models in the config. "
             "Default: [llm.default_tier] from config.",
    )
    p.add_argument(
        "--smoke", action="store_true",
        help="smoke mode: n_per_bucket=5, trials_per_image=1, single model",
    )
    p.add_argument(
        "--live", action="store_true",
        help="route tool wrappers through ROS instead of mocks (deferred)",
    )
    args = p.parse_args()

    cfg_path = Path(args.config).resolve()
    repo_root = Path(__file__).resolve().parent.parent
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    try:
        weights = (repo_root / cfg["paths"]["yolo_weights"]).resolve()
        dropout_weights = (repo_root / cfg["paths"]["yolo_dropout_weights"]).resolve()
        doorway_root = (repo_root / cfg["paths"]["doorway_dataset_root"]).resolve()
        runs_root = (repo_root / cfg["paths"]["runs_root"]).resolve()
        conditions: list[str] = list(cfg["experiment"]["conditions"])
        trials_cfg = int(cfg["experiment"]["trials_per_image"])
        K = int(cfg["perception"]["K"])
        imgsz = int(cfg["perception"]["imgsz"])
        conf_threshold = float(cfg["perception"]["conf_threshold"])
        iou_threshold = float(cfg["perception"]["iou_threshold"])
        device = _resolve_device(str(cfg["perception"]["device"]))
        var_thresholds = {
            "low_max": float(cfg["variance_thresholds"]["low_max"]),
            "high_min": float(cfg["variance_thresholds"]["high_min"]),
        }
        T = float(cfg["calibration"]["temperature"])
        llm_models_by_tier: dict[str, str] = dict(cfg["llm"]["models"])
        llm_default_tier = str(cfg["llm"]["default_tier"])
        llm_host = str(cfg["llm"]["host"])
        llm_temp = float(cfg["llm"]["temperature"])
        llm_num_ctx = int(cfg["llm"]["num_ctx"])
        llm_seed = int(cfg["llm"]["seed"])
        llm_timeout_s = float(cfg["llm"]["request_timeout_s"])
        max_turns = int(cfg["agent"]["max_turns"])
    except KeyError as e:
        sys.exit(f"missing required config key: {e}")

    if llm_default_tier not in llm_models_by_tier:
        sys.exit(
            f"llm.default_tier={llm_default_tier!r} is not one of "
            f"llm.models keys: {sorted(llm_models_by_tier)}"
        )
    llm_default_tag = llm_models_by_tier[llm_default_tier]

    # CLI args always override config; --smoke just changes the default fill-ins.
    if args.smoke:
        n_per_bucket = args.n_per_bucket if args.n_per_bucket is not None else 5
        trials_per_image = (
            args.trials_per_image if args.trials_per_image is not None else 1
        )
    else:
        n_per_bucket = args.n_per_bucket if args.n_per_bucket is not None else 100
        trials_per_image = (
            args.trials_per_image if args.trials_per_image is not None else trials_cfg
        )

    # Resolve --models against the tier table. Tier name → tag; raw tag passes through.
    requested = args.models or [llm_default_tag]
    models: list[str] = []
    for m in requested:
        if m in llm_models_by_tier:
            models.append(llm_models_by_tier[m])
        else:
            models.append(m)
    # Deduplicate while preserving order (if user passed both a tier and its tag).
    seen: set[str] = set()
    models = [m for m in models if not (m in seen or seen.add(m))]

    out_dir = runs_root / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "events.jsonl"

    # Build test set
    from uagent.experiment.dataset import build_test_set

    test_set = build_test_set(
        doorway_dataset_root=doorway_root,
        ood_root=repo_root / "data" / "ood_full",
        n_per_bucket=n_per_bucket,
        seed=int(cfg["dataset"]["split_seed"]),
    )
    if not test_set:
        sys.exit(
            "test set is empty. Verify data/doorway_combined/val/images "
            "and data/ood_full/{distractors,partial,novel}/ are populated."
        )
    print(f"test set: {len(test_set)} images")
    by_bucket: dict[str, int] = {}
    for it in test_set:
        by_bucket[it.bucket] = by_bucket.get(it.bucket, 0) + 1
    for b in sorted(by_bucket):
        print(f"  {b}: {by_bucket[b]}")

    # Resume check
    completed = _load_completed_keys(jsonl_path)
    if completed:
        print(f"resume: {len(completed)} events already in {jsonl_path}")

    # Build all tuples
    from uagent.experiment.harness import relpath_for_key  # avoid circular at top

    all_tuples: list[tuple] = [
        (it, c, t, m)
        for it in test_set
        for c in conditions
        for t in range(trials_per_image)
        for m in models
    ]
    remaining = []
    for it, c, t, m in all_tuples:
        # Use the SAME canonicalization as the harness's event writer.
        # Centralized in harness.relpath_for_key so the two paths can't drift.
        key = (relpath_for_key(it.image_path, repo_root), c, t, m)
        if key not in completed:
            remaining.append((it, c, t, m))
    total = len(all_tuples)
    todo = len(remaining)
    print(f"plan: {total} total trials; {todo} remaining ({total - todo} already done)")

    if todo == 0:
        print("nothing to do; events.jsonl is up to date.")
        return

    # Lazy imports for the heavy stuff
    from langchain_ollama import ChatOllama

    from uagent.experiment.harness import (
        CalibratedPerceiver,
        git_sha,
        run_single_trial,
    )
    from uagent.perception.calibration import TemperatureScaler
    from uagent.perception.detector import DeterministicYOLO, MCDropoutYOLO

    git_sha_str = git_sha(repo_root)
    scaler = TemperatureScaler(temperature=T)

    # Build perceivers per condition (reused across trials/images)
    print("loading perceivers...")
    perceivers: dict[str, CalibratedPerceiver] = {}
    if "baseline" in conditions:
        det = DeterministicYOLO(
            weights_path=weights,
            imgsz=imgsz,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            device=device,
        )
        perceivers["baseline"] = CalibratedPerceiver(det, scaler)
    if "variance_aware" in conditions:
        mc = MCDropoutYOLO(
            weights_path=dropout_weights,
            K=K,
            imgsz=imgsz,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            device=device,
        )
        if mc._n_dropout_modules == 0:
            sys.exit(
                f"loaded {dropout_weights} has 0 Dropout2d modules — "
                "MC Dropout cannot produce variance. Re-train via "
                "scripts/train_dropout_yolo.py."
            )
        perceivers["variance_aware"] = CalibratedPerceiver(mc, scaler)

    user_command = (
        "Detection from the front camera follows. Decide your next action."
    )

    # Build LLMs per model (reused across all trials with that model)
    print(f"loading LLM(s): {models}")
    llms: dict[str, object] = {}
    for m in models:
        llms[m] = ChatOllama(
            model=m,
            base_url=llm_host,
            temperature=llm_temp,
            num_ctx=llm_num_ctx,
            seed=llm_seed,
            client_kwargs={"timeout": llm_timeout_s},
        )

    # Run
    n_done = 0
    n_errors = 0
    for it, condition, trial_index, model in remaining:
        try:
            event = run_single_trial(
                image_path=it.image_path,
                bucket=it.bucket,
                condition=condition,
                trial_index=trial_index,
                model=model,
                perceiver=perceivers[condition],
                llm=llms[model],
                variance_thresholds=var_thresholds,
                user_command=user_command,
                max_turns=max_turns,
                git_sha_str=git_sha_str,
                repo_root=repo_root,
            )
            _append_event_atomic(jsonl_path, event.to_jsonl())
            n_done += 1
            print(
                f"  [{n_done}/{todo}] {it.bucket:<22} {condition:<16} "
                f"trial={trial_index} model={model} "
                f"action={event.agent_action} latency={event.latency_ms:.0f}ms"
            )
        except Exception as e:
            n_errors += 1
            print(
                f"  [{n_done + n_errors}/{todo}] ERROR on {it.image_path.name} "
                f"({condition}, trial={trial_index}, {model}): {e}"
            )

    print()
    print(f"done: {n_done} events written, {n_errors} errors")
    print(f"events.jsonl: {jsonl_path}")


if __name__ == "__main__":
    main()
