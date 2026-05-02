"""Main experimental runner.

Wraps uagent.experiment.harness.run_experiment with CLI argument
parsing and config loading.

    python scripts/run_experiment.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--live", action="store_true",
                   help="route tools through ROS instead of mocks")
    p.add_argument("--smoke", action="store_true",
                   help="N=1, single image per bucket (sanity)")
    args = p.parse_args()
    raise NotImplementedError("harness is implemented in step 7")


if __name__ == "__main__":
    main()
