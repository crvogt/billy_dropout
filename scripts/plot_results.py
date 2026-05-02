"""Generate the four paper figures from runs/<timestamp>/events.jsonl.

Figures (saved as PDFs in the same runs directory):

    fig1_reliability.pdf      detector reliability diagram (calibration)
    fig2_diff_abstention.pdf  differential abstention bar chart per bucket
    fig3_gate_variance.pdf    gate-variance scatter (OOD only)
    fig4_qualitative.pdf      one paired-trace qualitative example

    python scripts/plot_results.py runs/20260501T161234/

Implementation lands in step 9.
"""

from __future__ import annotations

import argparse


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir", help="path to runs/<timestamp>/")
    args = p.parse_args()
    raise NotImplementedError("plotting is implemented in step 9")


if __name__ == "__main__":
    main()
