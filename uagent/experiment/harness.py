"""Single-trial experimental runner: (image, condition, trial, model) → JSONL event.

The harness owns ONE event at a time. Orchestration over many events
(buckets × conditions × trials × models) lives in
scripts/run_experiment.py.

JSONL event schema (fixed):
    {
      "trial_id":              str (uuid hex),
      "trial_index":           int,        # for resume key
      "image_path":            str,        # repo-relative
      "bucket":                str,        # in_distribution_pos | ood_*
      "condition":             str,        # baseline | variance_aware | variance_aware_free
      "posterior":             dict,       # condition-specific keys
      "agent_action":          str,        # final tool name
      "agent_args":            dict,
      "agent_reasoning_chain": str,
      "latency_ms":            float,
      "model":                 str,
      "git_sha":               str,
      "timestamp":             str (ISO-8601 UTC),
    }

The resume key is (image_path, condition, trial_index, model). Trial
indices start at 0 within each (image, condition, model) cell.
"""

from __future__ import annotations

import math
import subprocess
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2

from uagent.agent.parsing import parse_thought_channel as parse_thought_channel
from uagent.agent.runtime import build_runtime
from uagent.perception.calibration import TemperatureScaler
from uagent.perception.posterior import Detection, Posterior

# `parse_thought_channel` is re-exported above so callers can continue
# importing it from `uagent.experiment.harness` per the Step F directive,
# while the implementation lives in `uagent.agent.parsing` to avoid a
# circular import with `uagent.agent.runtime`.


# ---------------------------------------------------------------------------
# Calibrated-perceiver wrapper
# ---------------------------------------------------------------------------


class CalibratedPerceiver:
    """Applies temperature scaling to per-detection confidence.

    Wraps either ``DeterministicYOLO`` (returns Detection) or
    ``MCDropoutYOLO`` (returns Posterior). Scales the confidence /
    mean_confidence field through a fitted ``TemperatureScaler``.
    Epistemic variance is left untouched — the variance threshold
    (configs/default.yaml::variance_thresholds) was set on the raw
    variance scale from Step B and is not affected by T-scaling.
    """

    def __init__(self, inner: Any, scaler: TemperatureScaler) -> None:
        self.inner = inner
        self.scaler = scaler

    def predict(self, image: Any) -> list[Any]:
        results = self.inner.predict(image)
        out: list[Any] = []
        for r in results:
            if isinstance(r, Detection):
                out.append(
                    Detection(
                        label=r.label,
                        bbox=r.bbox,
                        confidence=self.scaler.transform(r.confidence),
                    )
                )
            elif isinstance(r, Posterior):
                out.append(
                    Posterior(
                        label=r.label,
                        bbox=r.bbox,
                        mean_confidence=self.scaler.transform(r.mean_confidence),
                        epistemic_variance=r.epistemic_variance,
                        K=r.K,
                    )
                )
            else:
                raise TypeError(
                    f"CalibratedPerceiver: unsupported inner return type "
                    f"{type(r).__name__}"
                )
        return out


# ---------------------------------------------------------------------------
# Event dataclass + helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrialEvent:
    trial_id: str
    trial_index: int
    image_path: str
    bucket: str
    condition: str
    posterior: dict[str, Any] = field(default_factory=dict)
    agent_action: str = "(none)"
    agent_args: dict[str, Any] = field(default_factory=dict)
    agent_reasoning_chain: str = ""
    latency_ms: float = 0.0
    model: str = ""
    git_sha: str = ""
    timestamp: str = ""

    def to_jsonl(self) -> str:
        import json

        return json.dumps(asdict(self), default=_json_default) + "\n"


def _json_default(o: Any) -> Any:
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, float) and math.isnan(o):
        return None
    raise TypeError(f"not JSON-serializable: {type(o).__name__}")


def relpath_for_key(image_path: Path, repo_root: Path) -> str:
    """Resume-key canonical path string.

    Used by BOTH the orchestrator's resume check AND the harness's event
    writer. Centralized here so the two paths cannot drift.

    Strategy:
      1. Resolve the image_path (follows symlinks, makes absolute).
      2. Try to make it relative to repo_root. This is the common case
         (image lives inside the repo tree) and gives short stable keys.
      3. If relative_to raises (image outside the repo tree, different
         mount/bind on a remote box), fall back to the resolved absolute
         string. Stable within one machine; not portable across boxes.
      4. If resolve() itself raises (broken symlink), fall back to the
         raw str(image_path). Last-resort fallback so the key is at least
         deterministic per (input) call site.
    """
    try:
        resolved = image_path.resolve()
    except OSError:
        return str(image_path)
    try:
        return str(resolved.relative_to(repo_root))
    except ValueError:
        return str(resolved)


def git_sha(repo_root: Path) -> str:
    """Best-effort short git rev for the audit trail."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(repo_root),
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return out.strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Per-trial runner
# ---------------------------------------------------------------------------


def run_single_trial(
    *,
    image_path: Path,
    bucket: str,
    condition: str,
    trial_index: int,
    model: str,
    perceiver: Any,
    llm: Any,
    variance_thresholds: dict[str, float],
    user_command: str,
    max_turns: int,
    git_sha_str: str,
    repo_root: Path,
) -> TrialEvent:
    """Run one (image, condition, trial) end-to-end. Returns the event record.

    Builds the LangGraph runtime per call. The build is cheap relative to
    the LLM invocation (~ms vs seconds), and per-call construction
    isolates state across trials. The perceiver/llm pair is reused across
    trials by the caller.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"cannot read image: {image_path}")

    runtime = build_runtime(
        condition=condition,
        perceiver=perceiver,
        llm=llm,
        variance_thresholds=variance_thresholds,
        max_turns=max_turns,
        live_tools=False,
    )

    start = time.monotonic()
    final_state = runtime.invoke(
        {"image": img, "user_command": user_command}
    )
    latency_ms = (time.monotonic() - start) * 1000.0

    rel_path = relpath_for_key(image_path, repo_root)

    return TrialEvent(
        trial_id=uuid.uuid4().hex,
        trial_index=trial_index,
        image_path=rel_path,
        bucket=bucket,
        condition=condition,
        posterior=dict(final_state.get("posterior_meta", {})),
        agent_action=str(final_state.get("final_action", "(none)")),
        agent_args=dict(final_state.get("final_args", {})),
        agent_reasoning_chain=str(final_state.get("reasoning_chain", "")),
        latency_ms=latency_ms,
        model=model,
        git_sha=git_sha_str,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
