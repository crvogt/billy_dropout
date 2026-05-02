"""Fit temperature scaling for the deterministic baseline.

Runs DeterministicYOLO over robot_dataset/val/, collects (logit, label)
pairs, fits T to minimize NLL, writes the scalar back into the config
under `calibration.temperature`. Reports ECE before and after.

Implementation lands in step 3 (perception layer).
"""

from __future__ import annotations

import argparse


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    args = p.parse_args()
    raise NotImplementedError("calibration is implemented in step 3")


if __name__ == "__main__":
    main()
