"""Metrics unit tests on synthetic JSONL events.

  1. action_distribution counts correctly on a hand-built event list.
  2. differential_abstention computes baseline-vs-variance_aware delta
     correctly on a synthetic bucket.
  3. iter_events skips a malformed line without crashing.

Implementations land in step 7.
"""

from __future__ import annotations

import pytest


@pytest.mark.skip(reason="implemented in step 7")
def test_action_distribution_counts() -> None:
    raise NotImplementedError


@pytest.mark.skip(reason="implemented in step 7")
def test_differential_abstention_delta() -> None:
    raise NotImplementedError


@pytest.mark.skip(reason="implemented in step 7")
def test_iter_events_skips_malformed() -> None:
    raise NotImplementedError
