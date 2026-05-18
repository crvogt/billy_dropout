"""Pre-flight verify for condition 3 (variance_aware_free).

Locked criteria (from the 2026-05-17 design note):

    1. Reasoning chains non-empty and substantive — they must state
       something about the variance signal (not just "I will move").
    2. LOW-gate ID positives produce SOME `move_forward` (>0% but not
       necessarily 100%; agent discretion is expected).
    3. HIGH-gate OOD events produce mostly `look_around` or `defer`
       (>=60% of HIGH-gate OOD events).
    4. At least some events reference the MISSION (doorway / approach /
       pass through / goal), not just the gate label.

This script samples 5 `in_distribution_pos` images and 5 `ood_distractors`
images from the existing test set construction (same `dataset.split_seed`
as the full run), invokes them under the `variance_aware_free` condition
on the configured 4B model (`gemma4:e4b`), and reports pass/fail per
criterion plus the full reasoning chains so a human can read them.

Output:
  runs/verify_condition3/events.jsonl  (full event log)
  stdout                               (pass/fail report)

Usage:
  python scripts/verify_condition3.py --config configs/default.yaml

Exits 0 if all four criteria pass; 1 otherwise. The exit code is for
scripting — a human MUST eyeball the reasoning chains before launching
the full run.
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


# Mission-aware-reasoning vocabulary. The agent's chain must reference the
# objective directly, not just the gate label. Conservatively broad — any
# of these tokens (case-insensitive) counts. Extend if the prompt evolves.
MISSION_VOCAB = [
    "doorway", "door", "approach", "pass through", "navigat",
    "mission", "goal", "objective", "target", "find", "reach",
]

# Variance-signal vocabulary. A reasoning chain that mentions the gate
# label or any of these scores as "substantive". Catches both the gate
# label form ("LOW variance") and freer paraphrase ("the passes agreed",
# "uncertain detection", "epistemic_variance is small").
VARIANCE_VOCAB = [
    "variance", "uncertain", "uncertainty", "confidence", "confident",
    "agreed", "disagree", "stochastic", "passes", "epistemic",
    "low", "medium", "high",
]


def _resolve_device(requested: str) -> str:
    import torch

    if requested == "cuda" and not torch.cuda.is_available():
        print(
            "WARNING: device='cuda' requested but unavailable; "
            "falling back to device='cpu'. MC Dropout will be slow."
        )
        return "cpu"
    return requested


def _select_verify_subset(
    test_set: list,
    n_per_bucket: int = 5,
    target_buckets: tuple[str, ...] = ("in_distribution_pos", "ood_distractors"),
) -> list:
    """Slice the deterministic test set to exactly the verify subset.

    Reuses `build_test_set` so the sample is from the same deterministic
    pool as the production run — same seed, same images, no fresh RNG."""
    by_bucket: dict[str, list] = {}
    for it in test_set:
        by_bucket.setdefault(it.bucket, []).append(it)
    out: list = []
    for b in target_buckets:
        items = by_bucket.get(b, [])
        if len(items) < n_per_bucket:
            print(
                f"  WARNING: bucket {b} has only {len(items)} items; "
                f"verify wanted {n_per_bucket}"
            )
        out.extend(items[:n_per_bucket])
    return out


def _has_vocab(text: str, vocab: list[str]) -> bool:
    if not text:
        return False
    t = text.lower()
    return any(v in t for v in vocab)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, help="path to YAML config")
    p.add_argument(
        "--model", default=None,
        help="Ollama tag. Defaults to llm.models.small from the config.",
    )
    p.add_argument(
        "--run-name", default="verify_condition3",
        help="subdir name under runs_root/ for the events.jsonl",
    )
    p.add_argument(
        "--n-per-bucket", type=int, default=5,
        help="images per bucket (default 5; spec is 5 ID pos + 5 OOD distractors)",
    )
    args = p.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

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
        llm_host = str(cfg["llm"]["host"])
        llm_temp = float(cfg["llm"]["temperature"])
        llm_num_ctx = int(cfg["llm"]["num_ctx"])
        llm_seed = int(cfg["llm"]["seed"])
        llm_timeout_s = float(cfg["llm"]["request_timeout_s"])
        max_turns = int(cfg["agent"]["max_turns"])
        model_tag = args.model or str(cfg["llm"]["models"]["small"])
        split_seed = int(cfg["dataset"]["split_seed"])
    except KeyError as e:
        sys.exit(f"missing required config key: {e}")

    # ---- Build test set + pick the verify subset ----
    from uagent.experiment.dataset import build_test_set

    full_test_set = build_test_set(
        doorway_dataset_root=doorway_root,
        ood_root=repo_root / "data" / "ood_full",
        n_per_bucket=max(args.n_per_bucket, 5),
        seed=split_seed,
    )
    subset = _select_verify_subset(
        full_test_set,
        n_per_bucket=args.n_per_bucket,
        target_buckets=("in_distribution_pos", "ood_distractors"),
    )
    if not subset:
        sys.exit("verify subset is empty. Check data/ood_full/distractors/ and val/.")
    print(f"verify subset: {len(subset)} images")
    by_bucket: dict[str, int] = {}
    for it in subset:
        by_bucket[it.bucket] = by_bucket.get(it.bucket, 0) + 1
    for b in sorted(by_bucket):
        print(f"  {b}: {by_bucket[b]}")

    # ---- Build perceivers + LLM ----
    from langchain_ollama import ChatOllama

    from uagent.experiment.harness import (
        CalibratedPerceiver,
        git_sha,
        run_single_trial,
    )
    from uagent.perception.calibration import TemperatureScaler
    from uagent.perception.detector import MCDropoutYOLO

    git_sha_str = git_sha(repo_root)
    scaler = TemperatureScaler(temperature=T)

    print("loading MC Dropout perceiver...")
    mc = MCDropoutYOLO(
        weights_path=dropout_weights,
        K=K, imgsz=imgsz,
        conf_threshold=conf_threshold, iou_threshold=iou_threshold,
        device=device,
    )
    if mc._n_dropout_modules == 0:
        sys.exit(
            f"loaded {dropout_weights} has 0 Dropout2d modules. "
            "Re-train via scripts/train_dropout_yolo.py."
        )
    perceiver = CalibratedPerceiver(mc, scaler)

    print(f"loading LLM {model_tag}...")
    llm = ChatOllama(
        model=model_tag, base_url=llm_host,
        temperature=llm_temp, num_ctx=llm_num_ctx, seed=llm_seed,
        client_kwargs={"timeout": llm_timeout_s},
    )

    user_command = (
        "Detection from the front camera follows. Decide your next action."
    )

    # ---- Run trials ----
    out_dir = runs_root / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "events.jsonl"
    # Fresh file — verify is not resumed.
    if jsonl_path.exists():
        jsonl_path.unlink()

    print()
    print(f"=== running {len(subset)} events under variance_aware_free ===")
    events: list[dict] = []
    t0 = time.monotonic()
    for i, it in enumerate(subset, 1):
        try:
            ev = run_single_trial(
                image_path=it.image_path, bucket=it.bucket,
                condition="variance_aware_free", trial_index=0,
                model=model_tag, perceiver=perceiver, llm=llm,
                variance_thresholds=var_thresholds,
                user_command=user_command, max_turns=max_turns,
                git_sha_str=git_sha_str, repo_root=repo_root,
            )
            line = ev.to_jsonl()
            with jsonl_path.open("a", encoding="utf-8") as f:
                f.write(line)
                f.flush()
                os.fsync(f.fileno())
            events.append(json.loads(line))
            gate = ev.posterior.get("gate_level", "—") if ev.posterior else "—"
            print(
                f"  [{i}/{len(subset)}] {it.bucket:<22} "
                f"gate={gate:<7} action={ev.agent_action:<14} "
                f"latency={ev.latency_ms:.0f}ms"
            )
        except Exception as e:
            print(f"  [{i}/{len(subset)}] CRASH {it.image_path.name}: {e}")

    dt = time.monotonic() - t0
    print()
    print(f"=== done in {dt:.1f}s ({len(events)} events) ===")
    print(f"events.jsonl: {jsonl_path}")

    # ---- Evaluate the four criteria ----
    print()
    print("=== reasoning chains (verbatim — human eyeball this) ===")
    for i, e in enumerate(events, 1):
        gate = e.get("posterior", {}).get("gate_level", "—")
        chain = (e.get("agent_reasoning_chain") or "").strip()
        print(
            f"\n[{i}] {e['bucket']:<22} gate={gate:<7} action={e['agent_action']}"
        )
        if chain:
            for ln in chain.splitlines():
                print(f"    {ln}")
        else:
            print("    (empty)")

    print()
    print("=== pass/fail report ===")

    # Criterion 1: reasoning chain non-empty and substantive.
    c1_non_empty = [bool((e.get("agent_reasoning_chain") or "").strip()) for e in events]
    c1_substantive = [
        _has_vocab(e.get("agent_reasoning_chain") or "", VARIANCE_VOCAB)
        for e in events
    ]
    c1_pass = (
        sum(c1_non_empty) >= max(1, int(0.8 * len(events)))
        and sum(c1_substantive) >= max(1, int(0.6 * len(events)))
    )
    print(
        f"  1. reasoning coverage: "
        f"{sum(c1_non_empty)}/{len(events)} non-empty, "
        f"{sum(c1_substantive)}/{len(events)} reference variance — "
        f"{'PASS' if c1_pass else 'FAIL'}"
    )

    # Criterion 2: LOW-gate ID positives → some move_forward.
    id_low = [
        e for e in events
        if e["bucket"] == "in_distribution_pos"
        and e.get("posterior", {}).get("gate_level") == "LOW"
    ]
    id_low_motion = [e for e in id_low if e.get("agent_action") == "move_forward"]
    c2_pass = bool(id_low) and len(id_low_motion) >= 1
    print(
        f"  2. LOW-gate ID positives → move_forward at least once: "
        f"{len(id_low_motion)}/{len(id_low)} — "
        f"{'PASS' if c2_pass else ('N/A — no LOW-gate ID events sampled' if not id_low else 'FAIL')}"
    )

    # Criterion 3: HIGH-gate OOD events → mostly look_around / defer.
    ood_high = [
        e for e in events
        if e["bucket"].startswith("ood_")
        and e.get("posterior", {}).get("gate_level") == "HIGH"
    ]
    ood_high_abstain = [
        e for e in ood_high
        if e.get("agent_action") in ("look_around", "defer")
    ]
    if not ood_high:
        c3_pass = False  # cannot validate criterion without a HIGH-gate OOD event
        c3_status = "N/A — no HIGH-gate OOD events sampled (consider rerunning with more)"
    else:
        frac = len(ood_high_abstain) / len(ood_high)
        c3_pass = frac >= 0.6
        c3_status = f"{len(ood_high_abstain)}/{len(ood_high)} = {frac:.0%}"
    print(
        f"  3. HIGH-gate OOD → look_around/defer (>=60%): "
        f"{c3_status} — {'PASS' if c3_pass else 'FAIL'}"
    )

    # Criterion 4: mission-aware reasoning.
    mission_aware = [
        e for e in events
        if _has_vocab(e.get("agent_reasoning_chain") or "", MISSION_VOCAB)
    ]
    c4_pass = len(mission_aware) >= max(1, int(0.3 * len(events)))
    print(
        f"  4. mission-aware reasoning (≥30%): "
        f"{len(mission_aware)}/{len(events)} — "
        f"{'PASS' if c4_pass else 'FAIL'}"
    )

    all_pass = c1_pass and c2_pass and c3_pass and c4_pass
    print()
    print(f"OVERALL: {'PASS' if all_pass else 'FAIL'}")
    print(
        "  NOTE: pass/fail is automated heuristic. A human MUST read the "
        "reasoning chains above to confirm the agent is actually engaging "
        "with the mission framing before approving the full run."
    )
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
