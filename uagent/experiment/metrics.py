"""Read-only analytics over runs/<timestamp>/events.jsonl.

Minimal functional set for the smoke test report. Each function takes
the path or the loaded list-of-event-dicts. Full per-bucket × per-
condition × per-model break-downs and Spearman correlations live here.

Usage in code:
    from uagent.experiment.metrics import iter_events, summarize_smoke
    events = list(iter_events("runs/exp_smoke/events.jsonl"))
    print(summarize_smoke(events))
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

TOOL_NAMES = ("move_forward", "rotate", "look_around", "report", "defer")


@dataclass(frozen=True)
class ActionDistribution:
    bucket: str
    condition: str
    model: str
    counts: dict[str, int]
    total: int


def iter_events(jsonl_path: str | Path) -> Iterator[dict]:
    """Yield one event dict per line. Skips malformed lines with a print warning."""
    path = Path(jsonl_path)
    if not path.exists():
        return
    with path.open(encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                yield json.loads(raw)
            except json.JSONDecodeError as e:
                print(f"  WARNING: skipping malformed JSONL line {line_no}: {e}")


def action_distribution(events: list[dict]) -> list[ActionDistribution]:
    """Per-(bucket, condition, model) action distribution."""
    groups: dict[tuple[str, str, str], dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    for e in events:
        key = (e.get("bucket", ""), e.get("condition", ""), e.get("model", ""))
        action = e.get("agent_action", "(none)")
        groups[key][action] += 1

    out: list[ActionDistribution] = []
    for (bucket, condition, model), counts in sorted(groups.items()):
        total = sum(counts.values())
        out.append(
            ActionDistribution(
                bucket=bucket,
                condition=condition,
                model=model,
                counts=dict(counts),
                total=total,
            )
        )
    return out


def tool_call_validity_rate(events: list[dict]) -> dict[str, float]:
    """Fraction of events whose final action is in TOOL_NAMES, per condition.

    Catches cases where the model emitted no tool call or an unknown
    one (dispatcher falls back to 'defer' or 'error').
    """
    per_condition: dict[str, list[bool]] = defaultdict(list)
    for e in events:
        cond = e.get("condition", "")
        action = e.get("agent_action", "")
        per_condition[cond].append(action in TOOL_NAMES)
    return {
        cond: sum(flags) / len(flags) if flags else 0.0
        for cond, flags in per_condition.items()
    }


def latency_summary(events: list[dict]) -> dict[str, dict[str, float]]:
    """Per-condition latency stats (mean, median, p95) in ms."""
    per_condition: dict[str, list[float]] = defaultdict(list)
    for e in events:
        cond = e.get("condition", "")
        per_condition[cond].append(float(e.get("latency_ms", 0.0)))

    out: dict[str, dict[str, float]] = {}
    for cond, vs in per_condition.items():
        if not vs:
            continue
        vs_sorted = sorted(vs)
        n = len(vs_sorted)
        if n % 2 == 1:
            median = vs_sorted[n // 2]
        else:
            median = (vs_sorted[n // 2 - 1] + vs_sorted[n // 2]) / 2.0
        out[cond] = {
            "n": n,
            "mean": sum(vs_sorted) / n,
            "median": median,
            "p95": vs_sorted[max(0, int(n * 0.95) - 1)],
            "max": vs_sorted[-1],
        }
    return out


def differential_abstention(
    events: list[dict],
) -> dict[str, dict[str, float]]:
    """Per-bucket P(defer|condition) deltas relative to baseline.

    Returns ``{bucket -> {condition -> P(defer|condition) - P(defer|baseline)}}``
    aggregated over models. One entry per non-baseline condition observed
    anywhere in ``events`` — the inner-dict key set is the global non-
    baseline condition set, NOT only the conditions seen in that bucket.
    This keeps the report shape stable across buckets so downstream tables
    have consistent columns.

    NaN rules:
      - Buckets with no baseline events emit NaN for every non-baseline
        condition.
      - Conditions absent from a given bucket (but present in another)
        emit NaN for that bucket.

    With three conditions this exposes both ``variance_aware − baseline``
    and ``variance_aware_free − baseline`` so neither is silently dropped
    from downstream reports.
    """
    by_bucket_cond: dict[tuple[str, str], list[bool]] = defaultdict(list)
    for e in events:
        b = e.get("bucket", "")
        c = e.get("condition", "")
        by_bucket_cond[(b, c)].append(e.get("agent_action") == "defer")

    buckets = sorted({b for (b, _) in by_bucket_cond})
    conditions = {c for (_, c) in by_bucket_cond} - {"baseline"}
    out: dict[str, dict[str, float]] = {}
    for b in buckets:
        b_runs = by_bucket_cond.get((b, "baseline"), [])
        p_b = sum(b_runs) / len(b_runs) if b_runs else float("nan")
        deltas: dict[str, float] = {}
        for c in sorted(conditions):
            c_runs = by_bucket_cond.get((b, c), [])
            if not c_runs:
                deltas[c] = float("nan")
                continue
            p_c = sum(c_runs) / len(c_runs)
            deltas[c] = p_c - p_b
        out[b] = deltas
    return out


def summarize_smoke(events: list[dict]) -> str:
    """Human-readable smoke report. Returns a multi-line string."""
    lines: list[str] = []
    lines.append(f"=== smoke summary ({len(events)} events) ===")

    tcv = tool_call_validity_rate(events)
    lines.append("tool-call validity (action ∈ {move_forward, rotate, look_around, report, defer}):")
    for cond, rate in sorted(tcv.items()):
        lines.append(f"  {cond:<16} {rate:.0%}")

    lines.append("latency_ms:")
    for cond, stats in sorted(latency_summary(events).items()):
        lines.append(
            f"  {cond:<16} n={int(stats['n'])} mean={stats['mean']:.0f} "
            f"median={stats['median']:.0f} p95={stats['p95']:.0f} max={stats['max']:.0f}"
        )

    lines.append("action distribution by (bucket, condition):")
    for ad in action_distribution(events):
        kv = ", ".join(f"{k}={v}" for k, v in sorted(ad.counts.items()))
        lines.append(f"  {ad.bucket:<22} {ad.condition:<16} n={ad.total:3d}  {kv}")

    lines.append("differential abstention (condition − baseline):")
    for bucket, deltas in sorted(differential_abstention(events).items()):
        if not deltas:
            lines.append(f"  {bucket:<22} (no non-baseline conditions)")
            continue
        parts = []
        for cond, delta in sorted(deltas.items()):
            marker = "n/a" if math.isnan(delta) else f"{delta:+.2f}"
            parts.append(f"{cond}={marker}")
        lines.append(f"  {bucket:<22} {', '.join(parts)}")

    return "\n".join(lines)
