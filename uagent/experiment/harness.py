"""End-to-end experimental runner.

For each (image, condition, trial) triple:

    1. Load image from manifest.
    2. Run perceiver (MCDropoutYOLO for variance_aware,
       DeterministicYOLO + TemperatureScaler for baseline).
    3. Render the appropriate detection block via prompts.py.
    4. Invoke the compiled LangGraph runtime; capture terminal action.
    5. Append one JSONL event to runs/<timestamp>/events.jsonl atomically.

JSONL schema is fixed; see README and schema documented at top of
`run_one_trial`. The file is append-only and crash-safe: each event is
written + fsync'd before the next trial starts.

Implementation lands in step 7.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class HarnessConfig:
    config_path: Path
    runs_root: Path
    seed: int


def run_experiment(cfg: HarnessConfig) -> Path:
    """Run the full experiment end-to-end. Returns the output JSONL path."""
    raise NotImplementedError("harness is implemented in step 7")
