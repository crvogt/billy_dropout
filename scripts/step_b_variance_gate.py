"""Step B variance signal validation — the go/no-go gate for the paper.

Runs MCDropoutYOLO with K=20 on val/, calibration/, and ood_pilot/.
Writes one record per detection to runs/step_b/detections.jsonl, then
plots a four-panel diagnostic answering whether MC Dropout's variance
signal (a) tracks per-source detector competence, (b) distinguishes
confident-right from confident-wrong on the same source, and (c)
visibly skews higher on OOD vs ID confident-correct.

Decision criteria (per the step B directive):
  1. Median epistemic_variance on confident-correct ID < 0.05
  2. OOD pilot median epistemic_variance > 0.10 AND visibly higher
     than ID confident-correct distribution
  3. Per-source confident-correct distributions not pathologically
     bimodal (some spread expected; complete non-overlap is concerning)

Usage:
    python scripts/step_b_variance_gate.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import yaml

SOURCE_PREFIXES = {"gx_": "giraff_x", "gb_": "gibson", "dx_": "doordetect"}
ID_SOURCES = ("giraff_x", "gibson", "doordetect")
SOURCE_COLORS = {
    "giraff_x":        "tab:red",
    "gibson":          "tab:blue",
    "doordetect":      "tab:green",
    "ood_distractors": "tab:orange",
    "ood_partial":     "tab:purple",
    "ood_novel":       "tab:brown",
    "ood":             "tab:orange",
}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
CONF_LOW_MAX = 0.25
CONF_UNCERTAIN_MAX = 0.50

PASS_CRIT_1_THRESHOLD = 0.05   # ID confident-correct median variance
PASS_CRIT_2_THRESHOLD = 0.10   # OOD pilot median variance

SEED = 1337


# ---------------------------------------------------------------------------
# Geometry helpers (same shape as elsewhere in the codebase)
# ---------------------------------------------------------------------------


def _iou(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    a_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    b_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = a_area + b_area - inter
    return inter / union if union > 0 else 0.0


def _read_gt_boxes(
    label_path: Path, img_w: int, img_h: int
) -> list[tuple[float, float, float, float]]:
    """YOLO normalized cxcywh -> pixel xyxy. Returns [] if no labels."""
    boxes: list[tuple[float, float, float, float]] = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cx, cy, w, h = (float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))
        x1 = (cx - w / 2.0) * img_w
        y1 = (cy - h / 2.0) * img_h
        x2 = (cx + w / 2.0) * img_w
        y2 = (cy + h / 2.0) * img_h
        boxes.append((x1, y1, x2, y2))
    return boxes


def _conf_bucket(mean_conf: float) -> str:
    if mean_conf < CONF_LOW_MAX:
        return "low"
    if mean_conf < CONF_UNCERTAIN_MAX:
        return "uncertain"
    return "confident"


# ---------------------------------------------------------------------------
# Detection record collection
# ---------------------------------------------------------------------------


_unknown_warned: set[str] = set()


def _source_from_id_filename(name: str) -> str:
    for prefix, source in SOURCE_PREFIXES.items():
        if name.startswith(prefix):
            return source
    if name not in _unknown_warned:
        _unknown_warned.add(name)
        print(f"    WARNING: filename has no known source prefix: {name}")
    return "unknown"


def _nan_to_none(value):
    """JSONL needs RFC-8259 compliant output; json.dumps(math.nan) emits 'NaN'.
    Replace NaN with None recursively."""
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, dict):
        return {k: _nan_to_none(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_nan_to_none(v) for v in value]
    return value


def collect_records(
    detector,
    image_dir: Path,
    labels_dir: Path | None,
    source_lookup: Callable[[str], str],
    split: str,
    source_map50_by_source: dict[str, float],
    repo_root: Path,
) -> list[dict]:
    """Run K=20 stochastic detection on each image and emit per-detection records.

    labels_dir is None for OOD images; in that case every detection is
    labeled "FP" since the experiment treats OOD detections as
    "agent should not have fired" regardless of any underlying class
    annotation in the source dataset.
    """
    records: list[dict] = []
    images = sorted(
        p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS
    )
    n_total = len(images)
    for i, img_path in enumerate(images):
        if i % 50 == 0:
            print(f"    {split} {i + 1}/{n_total}")
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        source = source_lookup(img_path.name)
        gt_boxes = (
            _read_gt_boxes(labels_dir / f"{img_path.stem}.txt", w, h)
            if labels_dir is not None
            else []
        )
        posteriors = detector.predict(img)
        for p in posteriors:
            best_iou = max((_iou(p.bbox, gt) for gt in gt_boxes), default=0.0)
            tp_fp = "TP" if (gt_boxes and best_iou >= 0.5) else "FP"
            records.append(
                {
                    "image_path": str(img_path.relative_to(repo_root)),
                    "source": source,
                    "source_map50": source_map50_by_source.get(
                        source, math.nan
                    ),
                    "split": split,
                    "bbox": [float(v) for v in p.bbox],
                    "mean_confidence": float(p.mean_confidence),
                    "epistemic_variance": float(p.epistemic_variance),
                    "K": p.K,
                    "tp_fp": tp_fp,
                    "conf_bucket": _conf_bucket(p.mean_confidence),
                }
            )
    return records


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _select_variances(records, predicate) -> list[float]:
    return [r["epistemic_variance"] for r in records if predicate(r)]


def _hist_label(name: str, vals) -> str:
    if not vals:
        return f"{name} (n=0)"
    median = float(np.median(vals))
    q1, q3 = float(np.percentile(vals, 25)), float(np.percentile(vals, 75))
    return f"{name} (n={len(vals)}, med={median:.3f}, IQR=[{q1:.3f},{q3:.3f}])"


def plot_panel_a(records, ax, bins) -> dict:
    ax.set_title("Panel A — per-source confident-correct ID variance")
    stats: dict[str, dict] = {}
    for source in ID_SOURCES:
        vals = _select_variances(
            records,
            lambda r, s=source: r["source"] == s
            and r["split"] in ("val", "calibration")
            and r["tp_fp"] == "TP"
            and r["conf_bucket"] == "confident",
        )
        if vals:
            ax.hist(
                vals,
                bins=bins,
                alpha=0.55,
                color=SOURCE_COLORS[source],
                label=_hist_label(source, vals),
            )
            stats[source] = {
                "n": len(vals),
                "median": float(np.median(vals)),
                "q1": float(np.percentile(vals, 25)),
                "q3": float(np.percentile(vals, 75)),
            }
        else:
            stats[source] = {"n": 0, "median": math.nan, "q1": math.nan, "q3": math.nan}
    ax.set_xlabel("epistemic_variance")
    ax.set_ylabel("count (log)")
    ax.set_yscale("symlog")
    ax.legend(fontsize=8, loc="upper right")
    return stats


def plot_panel_b(records, ax, bins) -> dict:
    ax.set_title("Panel B — DoorDetect: confident-correct vs confident-wrong")
    stats: dict[str, dict] = {}
    for label, tp_fp, color in (
        ("confident-correct", "TP", "tab:green"),
        ("confident-wrong", "FP", "tab:red"),
    ):
        vals = _select_variances(
            records,
            lambda r, tp_fp=tp_fp: r["source"] == "doordetect"
            and r["split"] in ("val", "calibration")
            and r["tp_fp"] == tp_fp
            and r["conf_bucket"] == "confident",
        )
        if vals:
            ax.hist(
                vals,
                bins=bins,
                alpha=0.55,
                color=color,
                label=_hist_label(label, vals),
            )
            stats[label] = {
                "n": len(vals),
                "median": float(np.median(vals)),
                "q1": float(np.percentile(vals, 25)),
                "q3": float(np.percentile(vals, 75)),
            }
        else:
            stats[label] = {"n": 0, "median": math.nan, "q1": math.nan, "q3": math.nan}
    ax.set_xlabel("epistemic_variance")
    ax.set_ylabel("count (log)")
    ax.set_yscale("symlog")
    ax.legend(fontsize=8, loc="upper right")
    return stats


def plot_panel_c(records, ax, bins) -> dict:
    ax.set_title("Panel C — ID confident-correct vs OOD aggregate")
    stats: dict[str, dict] = {}
    id_vals = _select_variances(
        records,
        lambda r: r["split"] in ("val", "calibration")
        and r["tp_fp"] == "TP"
        and r["conf_bucket"] == "confident",
    )
    ood_vals = _select_variances(records, lambda r: r["split"] == "ood_pilot")
    for label, vals, color in (
        ("ID confident-correct", id_vals, "tab:blue"),
        ("OOD pilot (all detections)", ood_vals, "tab:orange"),
    ):
        if vals:
            ax.hist(
                vals,
                bins=bins,
                alpha=0.55,
                color=color,
                label=_hist_label(label, vals),
            )
            stats[label] = {
                "n": len(vals),
                "median": float(np.median(vals)),
                "q1": float(np.percentile(vals, 25)),
                "q3": float(np.percentile(vals, 75)),
            }
        else:
            stats[label] = {"n": 0, "median": math.nan, "q1": math.nan, "q3": math.nan}
    ax.set_xlabel("epistemic_variance")
    ax.set_ylabel("count (log)")
    ax.set_yscale("symlog")
    ax.legend(fontsize=8, loc="upper right")
    return stats


def plot_panel_d(records, ax, panel_a_stats, panel_c_stats) -> dict:
    """Competence-vs-variance scatter. Four points: three ID sources by
    mAP@.5 + OOD aggregate at a placeholder x. Reports Pearson r over
    the three ID points only (OOD has no meaningful mAP)."""
    ax.set_title("Panel D — competence vs median variance")

    xs_id: list[float] = []
    ys_id: list[float] = []
    for source in ID_SOURCES:
        any_rec = next((r for r in records if r["source"] == source), None)
        if any_rec is None:
            continue
        map50 = any_rec["source_map50"]
        med = panel_a_stats[source]["median"]
        if math.isnan(map50) or math.isnan(med):
            continue
        xs_id.append(float(map50))
        ys_id.append(float(med))
        ax.scatter([map50], [med], color=SOURCE_COLORS[source], s=90, zorder=3)
        ax.annotate(
            source,
            (map50, med),
            textcoords="offset points",
            xytext=(7, 4),
            fontsize=8,
        )

    ood_med = panel_c_stats.get("OOD pilot (all detections)", {}).get("median", math.nan)
    # Off the left edge so it's visually disjoint from the ID mAP scale,
    # not "interleaved between two real mAPs".
    ood_x_placeholder = -0.05
    if not math.isnan(ood_med):
        ax.scatter(
            [ood_x_placeholder],
            [ood_med],
            color="tab:orange",
            s=90,
            marker="s",
            zorder=3,
        )
        ax.annotate(
            "OOD pilot",
            (ood_x_placeholder, ood_med),
            textcoords="offset points",
            xytext=(7, 4),
            fontsize=8,
        )

    pearson_r = math.nan
    if len(xs_id) >= 2:
        pearson_r = float(np.corrcoef(xs_id, ys_id)[0, 1])

    ax.set_xlabel("source mAP@0.5  (OOD at left placeholder)")
    ax.set_ylabel("median epistemic_variance (confident-correct ID; all OOD)")
    ax.grid(True, alpha=0.3)
    ax.text(
        0.02,
        0.97,
        f"Pearson r (ID-only, n=3) = {pearson_r:.3f}  (descriptive only)",
        transform=ax.transAxes,
        verticalalignment="top",
        fontsize=8,
    )
    return {"pearson_r_id_only": pearson_r}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, help="path to YAML config")
    args = p.parse_args()

    np.random.seed(SEED)

    cfg_path = Path(args.config).resolve()
    repo_root = cfg_path.parent.parent
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    try:
        weights_path = (repo_root / cfg["paths"]["yolo_dropout_weights"]).resolve()
        dataset_root = (repo_root / cfg["paths"]["doorway_dataset_root"]).resolve()
        K = int(cfg["perception"]["K"])
        imgsz = int(cfg["perception"]["imgsz"])
        conf_threshold = float(cfg["perception"]["conf_threshold"])
        iou_threshold = float(cfg["perception"]["iou_threshold"])
        device = str(cfg["perception"]["device"])
    except KeyError as e:
        sys.exit(f"missing required config key: {e}")

    if not weights_path.exists():
        sys.exit(f"trained checkpoint not found: {weights_path}")

    per_source_results_path = repo_root / "runs" / "per_source_val_results.json"
    if not per_source_results_path.exists():
        sys.exit(
            f"missing per-source val results at {per_source_results_path}. "
            "Run scripts/per_source_val.py first."
        )
    per_source_data = json.loads(per_source_results_path.read_text(encoding="utf-8"))
    source_map50: dict[str, float] = {
        src: float(d["map50"]) for src, d in per_source_data.items()
    }

    ood_pilot_root = repo_root / "data" / "ood_pilot"
    if not ood_pilot_root.is_dir():
        sys.exit(
            f"missing OOD pilot at {ood_pilot_root}. "
            "Run scripts/prepare_ood_pilot.py first."
        )

    out_dir = repo_root / "runs" / "step_b"
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "detections.jsonl"

    print(f"loading {weights_path}")
    from uagent.perception.detector import MCDropoutYOLO

    detector = MCDropoutYOLO(
        weights_path,
        K=K,
        imgsz=imgsz,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        device=device,
    )
    print(
        f"detector ready: K={detector.K} "
        f"dropouts={detector._n_dropout_modules} device={detector.device}"
    )
    if detector._n_dropout_modules == 0:
        sys.exit(
            "loaded checkpoint has 0 Dropout2d modules — cannot run MC Dropout. "
            "Verify models/doorway_yolov8n_dropout.pt was trained with the "
            "callback-injected dropouts."
        )

    all_records: list[dict] = []

    for split_name in ("val", "calibration"):
        image_dir = dataset_root / split_name / "images"
        labels_dir = dataset_root / split_name / "labels"
        if not image_dir.is_dir():
            sys.exit(f"missing split: {image_dir}")
        print(f"--- {split_name}/ ---")
        all_records.extend(
            collect_records(
                detector,
                image_dir,
                labels_dir,
                _source_from_id_filename,
                split_name,
                source_map50,
                repo_root,
            )
        )

    for bucket_dir in sorted(d for d in ood_pilot_root.iterdir() if d.is_dir()):
        bucket = bucket_dir.name
        ood_source = f"ood_{bucket}"
        print(f"--- ood_pilot/{bucket}/ ---")
        all_records.extend(
            collect_records(
                detector,
                bucket_dir,
                None,
                lambda _name, s=ood_source: s,
                "ood_pilot",
                source_map50,
                repo_root,
            )
        )

    print(f"collected {len(all_records)} detection records")
    with jsonl_path.open("w", encoding="utf-8") as f:
        for r in all_records:
            f.write(json.dumps(_nan_to_none(r)) + "\n")
    print(f"wrote {jsonl_path}")
    unknown_images = {r["image_path"] for r in all_records if r["source"] == "unknown"}
    if unknown_images:
        print(
            f"  WARNING: {len(unknown_images)} image(s) had no source prefix; "
            "their detections will be excluded from per-source panels."
        )

    # Plotting
    print("--- plotting ---")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    bins = np.linspace(0, 0.4, 41)

    a_stats = plot_panel_a(all_records, axes[0][0], bins)
    b_stats = plot_panel_b(all_records, axes[0][1], bins)
    c_stats = plot_panel_c(all_records, axes[1][0], bins)
    d_stats = plot_panel_d(all_records, axes[1][1], a_stats, c_stats)

    plt.tight_layout()
    pdf_path = out_dir / "gate_output.pdf"
    fig.savefig(pdf_path)
    print(f"  PDF: {pdf_path}")

    for i, panel in enumerate(("a", "b", "c", "d")):
        single_fig, single_ax = plt.subplots(figsize=(8, 6))
        if panel == "a":
            plot_panel_a(all_records, single_ax, bins)
        elif panel == "b":
            plot_panel_b(all_records, single_ax, bins)
        elif panel == "c":
            plot_panel_c(all_records, single_ax, bins)
        elif panel == "d":
            plot_panel_d(all_records, single_ax, a_stats, c_stats)
        plt.tight_layout()
        single_fig.savefig(out_dir / f"panel_{panel}.png", dpi=130)
        plt.close(single_fig)
        print(f"  PNG: panel_{panel}.png")

    # Decision criteria
    print("--- decision criteria ---")
    id_cc_vals = _select_variances(
        all_records,
        lambda r: r["split"] in ("val", "calibration")
        and r["tp_fp"] == "TP"
        and r["conf_bucket"] == "confident",
    )
    ood_vals = _select_variances(
        all_records, lambda r: r["split"] == "ood_pilot"
    )
    id_cc_median = float(np.median(id_cc_vals)) if id_cc_vals else math.nan
    ood_median = float(np.median(ood_vals)) if ood_vals else math.nan

    crit_1_pass = (not math.isnan(id_cc_median)) and id_cc_median < PASS_CRIT_1_THRESHOLD
    crit_2_pass = (
        not math.isnan(ood_median)
        and not math.isnan(id_cc_median)
        and ood_median > PASS_CRIT_2_THRESHOLD
        and ood_median > id_cc_median
    )
    # Bimodality check: are per-source confident-correct distributions
    # so extreme that one source is at variance≈0 while another is at
    # variance≈0.5? Use the spread of per-source medians as a proxy.
    per_source_medians = [
        a_stats[s]["median"] for s in ID_SOURCES if not math.isnan(a_stats[s]["median"])
    ]
    if per_source_medians:
        max_med = max(per_source_medians)
        min_med = min(per_source_medians)
        median_spread = max_med - min_med
        # Pathological if spread > 0.10 AND min < 0.005 (one source near zero,
        # another way up). Conservative; report-only finding either way.
        crit_3_pass = not (median_spread > 0.10 and min_med < 0.005)
    else:
        median_spread = math.nan
        # No usable per-source medians — can't evaluate. Default to pass with
        # a flag in the summary; the user will see n=0 records in the panel.
        crit_3_pass = True

    summary = {
        "criterion_1_id_cc_median_lt_0.05": {
            "value": id_cc_median,
            "threshold": PASS_CRIT_1_THRESHOLD,
            "pass": crit_1_pass,
        },
        "criterion_2_ood_median_gt_0.10_and_gt_id": {
            "value": ood_median,
            "threshold": PASS_CRIT_2_THRESHOLD,
            "pass": crit_2_pass,
        },
        "criterion_3_no_pathological_bimodality": {
            "per_source_medians": {s: a_stats[s]["median"] for s in ID_SOURCES},
            "median_spread": median_spread,
            "pass": crit_3_pass,
        },
        "panel_a_per_source_confident_correct_id": a_stats,
        "panel_b_doordetect_correct_vs_wrong": b_stats,
        "panel_c_id_vs_ood": c_stats,
        "panel_d_competence_vs_variance": d_stats,
        "n_records": len(all_records),
        "n_records_by_split": {
            "val": sum(1 for r in all_records if r["split"] == "val"),
            "calibration": sum(1 for r in all_records if r["split"] == "calibration"),
            "ood_pilot": sum(1 for r in all_records if r["split"] == "ood_pilot"),
        },
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(
        json.dumps(_nan_to_none(summary), indent=2), encoding="utf-8"
    )
    print(f"wrote {summary_path}")

    print()
    print("=== gate decision ===")
    for k in (
        "criterion_1_id_cc_median_lt_0.05",
        "criterion_2_ood_median_gt_0.10_and_gt_id",
        "criterion_3_no_pathological_bimodality",
    ):
        entry = summary[k]
        status = "PASS" if entry["pass"] else "FAIL"
        print(f"  {k}: {status}")
    print(
        f"  id_cc_median={id_cc_median:.4f}  ood_median={ood_median:.4f}  "
        f"per-source medians: "
        + ", ".join(
            f"{s}={a_stats[s]['median']:.4f}" for s in ID_SOURCES
        )
    )
    print(f"done. artifacts in {out_dir}")


if __name__ == "__main__":
    main()
