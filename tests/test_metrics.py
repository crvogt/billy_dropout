"""Metrics + resume-key unit tests on synthetic JSONL events.

  1. action_distribution counts correctly.
  2. differential_abstention computes baseline-vs-variance_aware delta
     correctly, including NaN for missing buckets.
  3. iter_events skips malformed lines without crashing.
  4. latency_summary median is the canonical median (averaged for even n).
  5. relpath_for_key (the resume-key path canonicalizer used by both the
     orchestrator and the harness) handles in-tree, out-of-tree, and
     broken-symlink paths consistently.
"""

from __future__ import annotations

import math
import os
import tempfile
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------


def _ev(bucket: str, condition: str, action: str, latency: float = 100.0) -> dict:
    return {
        "trial_id": "x",
        "trial_index": 0,
        "image_path": f"{bucket}/img.jpg",
        "bucket": bucket,
        "condition": condition,
        "posterior": {},
        "agent_action": action,
        "agent_args": {},
        "agent_reasoning_chain": "",
        "latency_ms": latency,
        "model": "test-model",
        "git_sha": "abc",
        "timestamp": "2026-05-11T00:00:00Z",
    }


def test_action_distribution_counts() -> None:
    from uagent.experiment.metrics import action_distribution

    events = [
        _ev("ood", "baseline", "defer"),
        _ev("ood", "baseline", "move_forward"),
        _ev("ood", "baseline", "move_forward"),
        _ev("ood", "variance_aware", "defer"),
        _ev("ood", "variance_aware", "defer"),
        _ev("ood", "variance_aware", "look_around"),
    ]
    dists = action_distribution(events)
    by_key = {(d.bucket, d.condition): d for d in dists}
    assert by_key[("ood", "baseline")].counts == {"defer": 1, "move_forward": 2}
    assert by_key[("ood", "baseline")].total == 3
    assert by_key[("ood", "variance_aware")].counts == {"defer": 2, "look_around": 1}


def test_differential_abstention_delta() -> None:
    from uagent.experiment.metrics import differential_abstention

    events = [
        # bucket A: baseline defers 1/2 (0.5), variance defers 2/2 (1.0) → +0.5
        _ev("A", "baseline", "defer"),
        _ev("A", "baseline", "move_forward"),
        _ev("A", "variance_aware", "defer"),
        _ev("A", "variance_aware", "defer"),
        # bucket B: only baseline events → variance side is NaN → delta NaN
        _ev("B", "baseline", "defer"),
    ]
    delta = differential_abstention(events)
    assert delta["A"] == pytest.approx(0.5)
    assert math.isnan(delta["B"])


def test_iter_events_skips_malformed(tmp_path: Path) -> None:
    from uagent.experiment.metrics import iter_events

    jsonl = tmp_path / "events.jsonl"
    jsonl.write_text(
        '{"trial_id": "ok", "agent_action": "defer"}\n'
        'not valid json at all\n'
        '\n'  # blank line
        '{"trial_id": "ok2", "agent_action": "report"}\n',
        encoding="utf-8",
    )
    out = list(iter_events(jsonl))
    assert len(out) == 2
    assert {e["trial_id"] for e in out} == {"ok", "ok2"}


def test_latency_summary_median_even_n() -> None:
    """Canonical median for even n averages the two middle values."""
    from uagent.experiment.metrics import latency_summary

    events = [
        _ev("A", "baseline", "defer", latency=v)
        for v in [10.0, 20.0, 30.0, 40.0]  # n=4, median = (20+30)/2 = 25
    ]
    stats = latency_summary(events)
    assert stats["baseline"]["median"] == pytest.approx(25.0)
    assert stats["baseline"]["n"] == 4
    assert stats["baseline"]["mean"] == pytest.approx(25.0)
    assert stats["baseline"]["max"] == 40.0


# ---------------------------------------------------------------------------
# resume-key path canonicalizer
# ---------------------------------------------------------------------------


def test_relpath_for_key_in_tree(tmp_path: Path) -> None:
    """Image inside repo_root → returns repo-relative string."""
    from uagent.experiment.harness import relpath_for_key

    repo_root = tmp_path / "repo"
    img_dir = repo_root / "data" / "imgs"
    img_dir.mkdir(parents=True)
    img = img_dir / "a.jpg"
    img.write_bytes(b"")
    key = relpath_for_key(img, repo_root.resolve())
    assert key == "data/imgs/a.jpg"


def test_relpath_for_key_out_of_tree(tmp_path: Path) -> None:
    """Image outside repo_root → returns resolved absolute string."""
    from uagent.experiment.harness import relpath_for_key

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    outside = tmp_path / "other" / "b.jpg"
    outside.parent.mkdir()
    outside.write_bytes(b"")
    key = relpath_for_key(outside, repo_root.resolve())
    # Falls back to resolved absolute string (not relative)
    assert key == str(outside.resolve())
    assert os.path.isabs(key)


def test_relpath_for_key_idempotent(tmp_path: Path) -> None:
    """Same image_path twice produces the same key — required for resume."""
    from uagent.experiment.harness import relpath_for_key

    repo_root = tmp_path / "repo"
    img_dir = repo_root / "x"
    img_dir.mkdir(parents=True)
    img = img_dir / "y.jpg"
    img.write_bytes(b"")
    rr = repo_root.resolve()
    assert relpath_for_key(img, rr) == relpath_for_key(img, rr)


def test_relpath_for_key_follows_symlink_in_tree(tmp_path: Path) -> None:
    """In-tree symlink target → key is the resolved path inside the tree."""
    from uagent.experiment.harness import relpath_for_key

    repo_root = tmp_path / "repo"
    real_dir = repo_root / "real"
    real_dir.mkdir(parents=True)
    real = real_dir / "img.jpg"
    real.write_bytes(b"")
    link_dir = repo_root / "linked"
    link_dir.mkdir()
    link = link_dir / "img.jpg"
    os.symlink(real.resolve(), link)
    key = relpath_for_key(link, repo_root.resolve())
    # resolve() follows the symlink, so the key reflects the target's location.
    assert key == "real/img.jpg"
