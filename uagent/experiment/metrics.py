"""Metrics computed from runs/<timestamp>/events.jsonl.

Five computations, all read-only over the JSONL:

    1. Per-bucket action distribution per condition.
    2. Differential abstention: P(defer | variance_aware) − P(defer | baseline),
       per bucket. The headline number for the paper.
    3. Gate-variance correlation: within OOD, is P(defer) monotone in
       epistemic_variance?
    4. Action correctness vs ground-truth labels:
         act-on-positive, refuse-on-negative, defer-on-OOD.
    5. Reasoning-chain analysis: keyword-match for variance / uncertainty /
       disagreement terms in agent's verbalization.

Implementation lands in step 7.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass(frozen=True)
class ActionDistribution:
    bucket: str
    condition: str
    counts: dict[str, int]                  # tool_name -> count
    total: int


def iter_events(jsonl_path: str | Path) -> Iterator[dict]:
    """Yield one event dict per line. Skips malformed lines with a warning."""
    raise NotImplementedError("metrics are implemented in step 7")


def action_distribution(events: list[dict]) -> list[ActionDistribution]:
    raise NotImplementedError


def differential_abstention(events: list[dict]) -> dict[str, float]:
    """Returns {bucket -> P(defer|variance_aware) - P(defer|baseline)}."""
    raise NotImplementedError


def gate_variance_correlation(events: list[dict]) -> float:
    """Spearman ρ between epistemic_variance and P(defer) on OOD only."""
    raise NotImplementedError


def action_correctness(events: list[dict]) -> dict[str, dict[str, float]]:
    """Returns {bucket -> {condition -> correctness_rate}}."""
    raise NotImplementedError


def reasoning_chain_keyword_rate(
    events: list[dict],
    keywords: tuple[str, ...] = ("variance", "uncertain", "disagree", "confidence", "noisy"),
) -> dict[str, float]:
    """Returns {condition -> fraction of chains mentioning any keyword}."""
    raise NotImplementedError
