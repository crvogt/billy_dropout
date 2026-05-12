"""Pre-flight viability check for a candidate LLM before the full experiment.

Runs three checks on a single Ollama model tag (or tier name resolved
against llm.models in the config):

  Test 1 — Tool-call validity on 10 events (5 in_distribution_pos +
           5 ood_distractors, all baseline condition). Pass criterion:
           tool-call validity ≥ 90% AND zero hard crashes.

  Test 2 — Variance-aware comprehension on 5 OOD events under the
           variance_aware condition. Dumps each event's reasoning chain
           to stdout for visual review. Pass/fail here is qualitative
           and reported, not gated.

  Test 3 — Aggregate latency from tests 1+2 (mean and p95). Reported,
           not gated.

Exit code 0 iff Test 1 passes its gate AND no event raised.

--quick skips Test 2 (validity + latency only). Use for iteration on
a flaky model where you haven't yet decided to invest in reasoning
chain review.

Writes one JSONL event per trial to
runs/verify_<sanitized_tag>_<ts>/events.jsonl (same schema as the full
experiment) so downstream tooling works unchanged.

Usage:
    python scripts/verify_model.py --config configs/default.yaml \\
        --model gemma4:e4b
    python scripts/verify_model.py --config configs/default.yaml \\
        --model small               # tier name resolves via config
    python scripts/verify_model.py --config configs/default.yaml \\
        --model gemma4:31b --quick  # skip reasoning chain dump
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
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


def _sanitize(tag: str) -> str:
    """Filesystem-safe filename component."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", tag)


def _append_atomic(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(line)
        f.flush()
        os.fsync(f.fileno())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, help="path to YAML config")
    p.add_argument(
        "--model",
        required=True,
        help="Ollama model tag (e.g. gemma4:e4b) or tier name "
             "(small/medium/large) — resolved against llm.models.",
    )
    p.add_argument(
        "--quick", action="store_true",
        help="Skip Test 2 (reasoning chain dump); run validity + latency only.",
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
        models_by_tier: dict[str, str] = dict(cfg["llm"]["models"])
        llm_host = str(cfg["llm"]["host"])
        llm_temp = float(cfg["llm"]["temperature"])
        llm_num_ctx = int(cfg["llm"]["num_ctx"])
        llm_seed = int(cfg["llm"]["seed"])
        llm_timeout_s = float(cfg["llm"]["request_timeout_s"])
        split_seed = int(cfg["dataset"]["split_seed"])
        max_turns = int(cfg["agent"]["max_turns"])
    except KeyError as e:
        sys.exit(f"missing required config key: {e}")

    # Resolve tier name → tag
    model_tag = models_by_tier.get(args.model, args.model)
    print(f"verifying model: {model_tag}")
    if not weights.exists():
        sys.exit(f"yolo weights missing: {weights}")
    if not dropout_weights.exists():
        sys.exit(f"yolo_dropout_weights missing: {dropout_weights}")

    ts = time.strftime("%Y%m%dT%H%M%S")
    out_dir = runs_root / f"verify_{_sanitize(model_tag)}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "events.jsonl"

    # Lazy imports for the heavy stuff
    from langchain_ollama import ChatOllama

    from uagent.experiment.dataset import build_test_set
    from uagent.experiment.harness import (
        CalibratedPerceiver,
        git_sha,
        run_single_trial,
    )
    from uagent.experiment.metrics import (
        iter_events,
        latency_summary,
        tool_call_validity_rate,
    )
    from uagent.perception.calibration import TemperatureScaler
    from uagent.perception.detector import DeterministicYOLO, MCDropoutYOLO

    # Build sources
    print("loading detector + LLM...")
    scaler = TemperatureScaler(temperature=T)
    det = DeterministicYOLO(
        weights_path=weights, imgsz=imgsz, conf_threshold=conf_threshold,
        iou_threshold=iou_threshold, device=device,
    )
    perceiver_baseline = CalibratedPerceiver(det, scaler)

    mc = MCDropoutYOLO(
        weights_path=dropout_weights, K=K, imgsz=imgsz,
        conf_threshold=conf_threshold, iou_threshold=iou_threshold, device=device,
    )
    if mc._n_dropout_modules == 0:
        sys.exit(
            "MCDropoutYOLO has 0 Dropout2d modules — checkpoint is missing "
            "the embedded dropout. Re-train via scripts/train_dropout_yolo.py."
        )
    perceiver_variance = CalibratedPerceiver(mc, scaler)

    llm = ChatOllama(
        model=model_tag, base_url=llm_host, temperature=llm_temp,
        num_ctx=llm_num_ctx, seed=llm_seed,
        client_kwargs={"timeout": llm_timeout_s},
    )

    git = git_sha(repo_root)
    user_command = (
        "Detection from the front camera follows. Decide your next action."
    )

    # Build the test set; pick 5 ID-pos + 5 OOD-distractors for Test 1,
    # 5 OOD images (mix of sub-buckets) for Test 2.
    test_set = build_test_set(
        doorway_dataset_root=doorway_root,
        ood_root=repo_root / "data" / "ood_full",
        n_per_bucket=5,
        seed=split_seed,
    )
    test1_items = [
        it for it in test_set
        if it.bucket in ("in_distribution_pos", "ood_distractors")
    ]
    # Round-robin across the three OOD sub-buckets so the reasoning-chain
    # dump covers distractors / partial / novel rather than concentrating
    # in whichever bucket appears first in test_set order. For N_OOD=5
    # across 3 buckets the natural mix is 2/2/1 (distractors/partial/novel).
    _ood_buckets = ("ood_distractors", "ood_partial", "ood_novel")
    _ood_by_bucket: dict[str, list] = {b: [] for b in _ood_buckets}
    for it in test_set:
        if it.bucket in _ood_by_bucket:
            _ood_by_bucket[it.bucket].append(it)
    test2_items = []
    _idx = 0
    while len(test2_items) < 5 and _idx < 50:
        bucket = _ood_buckets[_idx % len(_ood_buckets)]
        pos = _idx // len(_ood_buckets)
        if pos < len(_ood_by_bucket[bucket]):
            test2_items.append(_ood_by_bucket[bucket][pos])
        _idx += 1

    n_test1 = len(test1_items)
    n_test2 = 0 if args.quick else len(test2_items)
    print(f"plan: Test 1 = {n_test1} events (baseline), "
          f"Test 2 = {n_test2} events (variance_aware)")

    n_crashes = 0

    # --- Test 1 ---
    print()
    print("=== Test 1: tool-call validity (baseline) ===")
    for i, it in enumerate(test1_items, 1):
        try:
            ev = run_single_trial(
                image_path=it.image_path, bucket=it.bucket,
                condition="baseline", trial_index=0, model=model_tag,
                perceiver=perceiver_baseline, llm=llm,
                variance_thresholds=var_thresholds,
                user_command=user_command, max_turns=max_turns,
                git_sha_str=git, repo_root=repo_root,
            )
            _append_atomic(jsonl_path, ev.to_jsonl())
            print(
                f"  [{i}/{n_test1}] {it.bucket:<22} action={ev.agent_action:<14} "
                f"latency={ev.latency_ms:.0f}ms"
            )
        except Exception as e:
            n_crashes += 1
            print(f"  [{i}/{n_test1}] CRASH {it.image_path.name}: {e}")

    # --- Test 2 (optional) ---
    if not args.quick:
        print()
        print("=== Test 2: variance-aware comprehension (reasoning chain dump) ===")
        for i, it in enumerate(test2_items, 1):
            try:
                ev = run_single_trial(
                    image_path=it.image_path, bucket=it.bucket,
                    condition="variance_aware", trial_index=0, model=model_tag,
                    perceiver=perceiver_variance, llm=llm,
                    variance_thresholds=var_thresholds,
                    user_command=user_command, max_turns=max_turns,
                    git_sha_str=git, repo_root=repo_root,
                )
                _append_atomic(jsonl_path, ev.to_jsonl())
                post = ev.posterior
                gate = post.get("gate_level", "—")
                var = post.get("epistemic_variance", float("nan"))
                conf = post.get("mean_confidence", float("nan"))
                print(
                    f"  [{i}/{len(test2_items)}] {it.bucket:<14} "
                    f"gate={gate:<7} var={var:.4f} mean_conf={conf:.2f}  "
                    f"action={ev.agent_action}"
                )
                chain = ev.agent_reasoning_chain.strip()
                if chain:
                    print("    reasoning:")
                    for line in chain.splitlines():
                        print(f"      {line}")
                else:
                    print("    reasoning: (none — empty thought channel or tool-call-only response)")
            except Exception as e:
                n_crashes += 1
                print(f"  [{i}/{len(test2_items)}] CRASH {it.image_path.name}: {e}")

    # --- Compute Test 1 validity + latency ---
    events = list(iter_events(jsonl_path))
    baseline_events = [e for e in events if e["condition"] == "baseline"]
    variance_events = [e for e in events if e["condition"] == "variance_aware"]
    val_rates = tool_call_validity_rate(events)
    lat = latency_summary(events)

    print()
    print("=== summary ===")
    print(f"  events:        {len(events)} written, {n_crashes} crashes")
    print(
        f"  validity:      "
        + ", ".join(f"{c}={r:.0%}" for c, r in sorted(val_rates.items()))
    )
    print(
        f"  latency_ms:    "
        + ", ".join(
            f"{c}: mean={s['mean']:.0f} p95={s['p95']:.0f}"
            for c, s in sorted(lat.items())
        )
    )

    # Per-condition validity counts (and hallucinated tool names). Source
    # the tool list from the metrics module so verify and the metrics
    # report stay in lockstep if the tool surface ever changes.
    from uagent.experiment.metrics import TOOL_NAMES as _TOOL_NAMES
    TOOL_NAMES = set(_TOOL_NAMES)
    hallucinations: dict[str, int] = {}
    for e in events:
        a = e.get("agent_action", "")
        if a not in TOOL_NAMES:
            hallucinations[a] = hallucinations.get(a, 0) + 1
    if hallucinations:
        print(f"  hallucinated/invalid tool names: {hallucinations}")
    else:
        print(f"  hallucinated/invalid tool names: none")

    # Gate
    baseline_validity = val_rates.get("baseline", 0.0)
    pass_gate = baseline_validity >= 0.90 and n_crashes == 0
    print()
    print(f"verdict: {'PASS' if pass_gate else 'FAIL'} "
          f"(baseline validity {baseline_validity:.0%} ≥ 90%? "
          f"{'yes' if baseline_validity >= 0.90 else 'no'}; "
          f"crashes {n_crashes})")
    print(f"artifacts: {out_dir}")
    sys.exit(0 if pass_gate else 1)


if __name__ == "__main__":
    main()
