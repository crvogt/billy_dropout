# Paper notes — placement, errata, and findings to retrieve at Step G

Created during step C (2026-05-11). The points below specify where in the
paper draft (per the standard IEEE MILCOM structure: I Introduction,
II Related Work, III Method, IV Results, V Discussion, VI Conclusion)
specific experimental artifacts and findings belong. Retrieve at Step G
when drafting the manuscript.

## Section III — Method

**Detector-side calibration verification belongs here.** Panel D from
Step B (`runs/step_b/panel_d.png`) — the competence-vs-median-variance
scatter showing the near-monotone relationship across Giraff-X / Gibson
/ DoorDetect plus the OOD pilot point — is verification that the MC
Dropout posterior is doing useful work, not a Results finding. Suggested
phrasing:

> The MC Dropout posterior's variance scales monotonically with the
> detector's per-source generalization performance (Pearson r = −0.98
> over the three in-distribution sources; n=3, descriptive only),
> confirming that the propagated signal is a usable measure of detector
> competence rather than a binary ID/OOD flag. The OOD pilot point
> extends this trend along the same monotone curve.

**Locked numerical details for the methods description:**
- K = 20 stochastic forward passes
- Six `nn.Dropout2d(p=0.25)` modules inserted in the classification
  branches (cv3) of YOLOv8's decoupled detection head at all three
  scales — two per branch (before the second-to-last and final 1×1
  conv). Regression branches remain deterministic. Placement follows
  the Stochastic-YOLO recipe (Azevedo et al. 2020) translated to
  YOLOv8's decoupled head.
- Temperature scaling: single T fit by L-BFGS via `exp(log_T)`
  reparameterization (`uagent/perception/calibration.py`). The
  reparameterization is necessary; direct-T parameterization can let
  L-BFGS overshoot below zero on imbalanced TP/FP distributions and
  fail the constructor invariant. Document this if Stochastic-YOLO
  / temperature-scaling implementations are described in any detail.
- Variance thresholds for the variance_aware prompt's interpretation
  label (LOW / MEDIUM / HIGH) recalibrated empirically:
  `low_max = 0.003`, `high_min = 0.010`, set from observed Step B
  distributions (median ID-confident-correct = 0.0026, OOD pilot
  Q3 = 0.018). The originally-spec'd thresholds (0.05 / 0.15) were
  ~10× too high under the realised variance scale; under those, the
  HIGH band would have been unreachable.

**OOD construction details — verify before publication:**
- Open Images V7 class IDs verified against
  `oidv7-class-descriptions-boxable.csv`:
  - Mirror `/m/054_l`, Picture frame `/m/06z37_`,
    **Cabinetry `/m/01s105`** (the training-data addendum specified
    `/m/0642b4` which is actually *Cupboard* in OIv7 — addendum errata
    flagged at step B; verified MID used in the scripts),
    Window `/m/0d4v4`, Door `/m/02dgv`.
  - Env-shift co-annotations: Building `/m/0cgh4`, House `/m/03jm5`,
    Skyscraper `/m/079cl`, **Office building `/m/021sj1`** (the
    addendum did not specify a MID for Office building; verified).
- Scene-level annotation exclusion: OIv7 boxable annotations sometimes
  contain whole-image scene-level labels (bbox area ≈ 1.0). These are
  NOT object instances and are NOT door-shaped. The OOD downloader
  applies `SCENE_LEVEL_MAX_AREA = 0.80` to all sub-buckets and an
  additional `DISTRACTOR_MAX_AREA = 0.50` for distractors. Mention this
  filter in the OOD construction subsection — necessary for
  reproducibility.

## Section IV — Results

**Panel C from Step B (`runs/step_b/panel_c.png`) is the gate-check
figure** for the perception-signal claim: ID-confident-correct (median
0.0026, IQR [0.0014, 0.0049]) vs OOD pilot aggregate (median 0.0085,
IQR [0.0035, 0.0181]). The OOD distribution skews higher (3.3×
median ratio) with the upper IQR extending into the 0.018 range that
ID-confident-correct rarely reaches.

Headline Results in Section IV are still about *agent behavior* (the
core experimental contribution, generated in Steps E and F):
differential abstention rates between baseline and variance_aware
conditions, gate-variance correlation within OOD, action correctness
against expected actions per bucket. Panels A–D from Step B are
supporting evidence for the perception-side prerequisites, not the
headline finding.

## Section V — Discussion

**Panel B from Step B (`runs/step_b/panel_b.png`)** — DoorDetect's
confident-wrong vs confident-correct distributions — gets an honest
mention as motivation for why the agent reasons over the *full
posterior* (mean confidence + variance + categorical interpretation
label) rather than gating purely on variance. The single-source
within-source variance signal is modest:

- DoorDetect confident-correct median: 0.0049 (n=57)
- DoorDetect confident-wrong median:   0.0060 (n=19)
- Ratio: 1.22× (correct vs wrong)

The variance signal exists but does not, on its own, reliably
discriminate within-source confident-right from confident-wrong. The
multi-signal composite (raw posterior + interpretation label + the
agent's reasoning chain) is what the experiment evaluates, not the
variance scalar alone.

**Per-source temperature-scaling result is a Discussion-worthy
finding.** Step C calibration (`runs/calibration_results.json`):

| Source | n | ECE pre | ECE post | NLL pre | NLL post |
|---|---|---|---|---|---|
| aggregate | 1845 | 0.0615 | **0.0510** | 0.3103 | 0.3068 |
| Giraff-X | 1105 | 0.0779 | **0.0648** | 0.2791 | 0.2698 |
| Gibson | 552 | 0.0671 | **0.0580** | 0.3554 | 0.3591 |
| DoorDetect | 188 | 0.0690 | **0.0731** | 0.3615 | 0.3706 |

The single temperature scalar (T = 0.859) improves aggregate ECE and
helps both Antonazzi sources, but *actively worsens* calibration on
DoorDetect (188 detections, 38 TP / 150 FP — predominantly FP). The
T<1 sharpening pushes already-confident-but-wrong DoorDetect detections
further toward 1.0, increasing the gap between predicted probability
and actual TP rate. Per-source T would partially fix this — but at
inference time the agent has no signal indicating source membership for
OOD inputs (that's the whole point of the experiment). Single T is
operationally correct; the DoorDetect ECE-worsening is exactly the
miscalibration regime variance signaling is supposed to address.

**Connecting threads: deterministic calibration cannot fix OOD
miscalibration; the variance signal does.** The agent experiment in
Section IV demonstrates whether the LLM can use the variance signal
productively. This Discussion paragraph links the calibration-side
empirical observation (T scaling fails on DoorDetect) to the
agent-side motivation (need a signal robust to source membership).

## Open question (do not resolve in advance)

The Step B variance scale (median 0.0026 for ID confident-correct,
0.0085 for OOD) is small in absolute terms. Two hypotheses for how
the LLM will handle this:

- **H1:** The LLM attends to the categorical interpretation label
  (LOW / MEDIUM / HIGH) and the raw variance number is decoration.
  The recalibrated thresholds make this work.
- **H2:** The LLM treats small raw variance values as "essentially
  zero" regardless of the categorical label, and the experiment
  produces Pattern A (no differential behavior).

Step F results will reveal which world we are in. If H2, a targeted
follow-up with aggressive dropout (p=0.5, possibly wider placement)
is in scope. Do not preempt.
