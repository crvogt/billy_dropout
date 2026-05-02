"""Run the agent live on the robot for one recorded qualitative example.

Connects to rosbridge, captures one camera frame, runs both conditions
back-to-back on the same frame, prints the agent's tool selection and
reasoning chain side-by-side. Used to record the qualitative figure for
the paper.

Not part of the main harness; off the experimental critical path.
Implementation lands after step 7.
"""

from __future__ import annotations

import argparse


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    args = p.parse_args()
    raise NotImplementedError("live demo lands after step 7")


if __name__ == "__main__":
    main()
