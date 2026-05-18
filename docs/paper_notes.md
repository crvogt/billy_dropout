# Paper notes — placement, errata, and findings to retrieve at Step G

Created during step C (2026-05-11). The points below specify where in the
paper draft (per the standard IEEE MILCOM structure: I Introduction,
II Related Work, III Method, IV Results, V Discussion, VI Conclusion)
specific experimental artifacts and findings belong. Retrieve at Step G
when drafting the manuscript.

## Section III — Method

### III.A — Pilot/full OOD separation

Variance-interpretation thresholds were calibrated on the 26-image
OOD pilot (Step B2, `data/ood_pilot/manifest.json`). Agent behavior
is evaluated on a disjoint 100-image OOD test set
(`data/ood_full/manifest.json`) drawn from the same source (Open
Images V7 validation split) with pilot image IDs explicitly excluded
via the downloader's `--exclude-manifest` flag. This pilot/full
separation prevents calibration→evaluation leakage on the variance
thresholds.

Full-set bucket counts (post-attrition + post-dedup):

| bucket | downloaded | precheck fires |
|---|---|---|
| distractors | 45 | 27 (60%) |
| novel | 18 | 12 (66%) |
| partial | 37 | 27 (73%) |
| **combined** | **100** | **66 (66%)** |

Cross-bucket image-ID deduplication is enforced via a
`BUCKET_PRIORITY` ordering (novel-before-partial; both pull from
the Door class).

### Detector-side calibration verification

Panel D from
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

### III.B — LLM model selection

We use the Gemma 4 family (Google DeepMind, Apache 2.0, released
May 2026 on Ollama) across three size tiers: `gemma4:e4b` (small,
~4B effective; edge variant), `gemma4:26b` (medium; mixture-of-experts
with ~4B active params per token at 26B total), and `gemma4:31b`
(large, dense). Selection criteria:

  - **Open weights, native function-calling support** — required for
    the LangGraph tool-bound agent path.
  - **Multimodal capability** — kept available for future agent
    variants that consume the image directly, though the present
    experiment feeds only the detector's text-rendered posterior.
  - **Western-origin** — motivated by venue requirements; one of the
    paper's claims is a deployable system, and provenance matters
    for that claim.
  - **Coherent size spectrum within a single family** — critical for
    the SWaP-scaling claim. Cross-family comparisons (e.g., Qwen vs
    Llama vs Gemma) would confound architecture and training data with
    model size; staying inside one family lets the size dimension be
    the only varying factor in the sweep.

Inference parameters held constant across tiers: `temperature=0.1`,
`num_ctx=4096`, `seed=42`. Reproducibility under Ollama is approximate
even at fixed seed (less consistent than vLLM); the three trials per
event partially compensate. Pre-flight verification per tier via
`scripts/verify_model.py` gates the full run on tool-call validity ≥
90% on a 10-event baseline check.

### III.C — Locked numerical details for the methods description

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

### III.D — Variance-aware conditions: directive vs free

The experiment uses TWO variance-aware conditions, named here as in
code: `variance_aware` (paper-facing name: **directive**) and
`variance_aware_free` (**free**). Both receive the identical perception
block — same `mean_confidence`, `epistemic_variance`, and calibrated
LOW/MEDIUM/HIGH interpretation label — and differ only in the system
prompt. The byte-identity of the perception block between the two is
a controlled-comparison invariant pinned by
`tests/test_agent.py::test_variance_free_perception_block_matches_directive`.

The two conditions test different *propagation styles* for uncertainty
into a small LLM agent: directive supplies a calibrated mechanical
mapping the LLM executes; free supplies the same posterior and asks
the LLM to exercise mission-framed discretion.

#### III.D.1 — Why the directive prompt is explicit

The variance_aware_directive condition uses an explicit mechanical
mapping from the calibrated interpretation label to a tool call (LOW →
`move_forward(distance_m=0.5)`; MEDIUM → `look_around()`; HIGH →
`defer(reason=...)`; empty detection → `look_around()`). The
prompt instructs the LLM to follow the mapping deterministically and
not to construct parallel thresholds from `mean_confidence`. The
baseline condition has no such mapping; it gets a narrative system
prompt with only the confidence scalar and chooses actions
unconditionally.

The explicit mapping is a deliberate design choice driven by a single
pilot prompt run on the 4B tier (final-prompt-v1, narrative variance
addendum with `<think>` reasoning required):

  - Reasoning chains non-empty on 100% of events.
  - LOW-gate ID-positive events emitted `move_forward` on only 48%
    of trials (target: >90%).
  - HIGH-gate OOD events emitted `defer` on 0% of trials.
  - Reasoning chains showed the model inventing its own thresholds
    from `mean_confidence` (e.g., "0.42 is below my threshold of
    0.5 so I will defer") rather than using the calibrated
    interpretation label.

Run path: `runs/medium_4b_final_<timestamp>/events.jsonl` for the
final-prompt-v1 pilot. **TODO before Step G:** resolve `<timestamp>`
to the actual run directory and re-verify the four numbers above
directly from the events file.

The interpretation: small LLMs (4B-class) do not reliably reason
about variance unprompted even when the variance signal is rendered
in the perception block; the categorical LOW/MEDIUM/HIGH label was
designed to make variance legible but the LLM disregarded it and
reverted to confidence-thresholding. The redesign replaces "reason
about variance" with "execute a calibrated mapping."

Counter-position. A plausible reading is that the variance_aware
condition has been reduced to a Python `if/elif` and there is no
LLM role left. The LLM's actual role under the v2 prompt is to parse
the heterogeneous perception block (which may or may not contain a
detection, with varying bbox shapes and confidence scalars), branch
on the categorical label, and emit a schema-valid tool call with
auditable per-event provenance — tasks an `if/elif` cannot perform
on free-form prompt input. The contribution is operational, not
cognitive: an LLM with a calibrated mapping and a variance signal
yields the behavioral suppression the paper's thesis requires, even
at small scales where unconstrained variance reasoning fails.

The methodological contrast:

  - Baseline evaluates the LLM's action selection under a
    confidence-only perception block, with the LLM exercising full
    discretion.
  - Variance-aware evaluates whether a constrained LLM — instructed
    to defer to a calibrated mapping rather than reason from scalars
    — produces behaviorally appropriate motion suppression under OOD.

The reasoning chain in the v2 prompt is a compliance trace rather
than a reasoning artifact. It provides interpretable per-event
provenance (which interpretation level the model observed, which
action it selected) and a compliance signal: chains that reference
the interpretation label and the mapped action confirm the mapping
is being followed; chains that invent parallel thresholds signal
mapping deviation.

The decision tree (LOW/MEDIUM/HIGH → action) is hand-calibrated by
the Step B variance-distribution analysis (`runs/step_b/summary.json`)
using the variance thresholds locked in III.C above; it is NOT a
learned policy over the LLM's outputs.

#### III.D.2 — Why the free prompt exists alongside directive

The directive condition by construction reduces the LLM's role on a
detection-present turn to executing a three-branch dispatch. A
plausible methodological objection is that the comparison "directive
vs baseline" therefore measures the calibrated mapping, not the LLM's
ability to reason over uncertainty: a Python `if/elif` substituted
for the LLM would yield similar abstention rates, and the LLM is
present only to parse heterogeneous perception strings and emit
schema-valid tool calls. (We address this directly in III.D.1's
counter-position: the LLM's role under directive *is* operational
parsing + tool-call emission, and that is a real contribution at
small scales. But the objection persists.)

The free condition exists to test the alternative propagation style.
The agent receives the same MC Dropout posterior — `mean_confidence`,
`epistemic_variance`, and the LOW/MEDIUM/HIGH interpretation label —
under mission framing ("find a doorway, approach it, and pass
through") and is told explicitly:

> How you use these signals to choose an action is your decision.
> There is no prescribed mapping from variance level to action.

If the free condition produces signal-driven motion suppression on
OOD comparable to the directive condition, the paper's headline
contribution is strengthened: small LLMs *can* reason over a surfaced
posterior given mission framing. If it does not, the directive-vs-
free contrast itself is the contribution — it quantifies how much of
the directive condition's behavior is attributable to the calibrated
mapping rather than to the variance signal itself, and frames an
explicit design recommendation for deployers ("propagate variance as
a directive, not as information, when working at the 4B scale").

The free vs directive comparison is the **Section V Discussion**
methodological contribution (see V.A below). The headline Section IV
result remains the differential abstention between variance-aware
(either variant) and baseline.

**Tool-surface asymmetry between directive and free (methodological
caveat).** The directive prompt instructs the agent that `rotate` and
`report` are available but must not be used in this condition,
effectively restricting the action set to `{move_forward, look_around,
defer}`. The free prompt allows all five tools. This asymmetry exists
because the directive prompt's mapping is exhaustive over the three
gate levels + empty-detection case and any additional tools would be
ambiguous; the free prompt cannot ban tools it isn't prescribing the
use of. The asymmetry is not a confound on the headline P(defer | OOD)
metric — `rotate` and `report` are not abstain actions and are
relatively rare in the free condition — but it does mean that any
"directive matches free on full action distribution" claim must be
reported on the move-vs-non-commit subset, treating
`{rotate, report, look_around, defer}` as a single non-commit bucket.
Methods text and Results plots that compare action distributions
across conditions should adopt the move-vs-non-commit collapse and
state it explicitly.

#### III.D.3 — Replicability of the perception block across the two variance conditions

Because the MC Dropout posterior is stochastic (K=20 forward passes with
training-mode dropouts), the perception block is identical *given the
same Posterior input* but the Posterior itself depends on torch RNG
state, which advances across calls. The two variance conditions on the
same `(image_path, trial_index)` therefore see slightly different
posteriors unless the run pipeline either (a) snapshots and replays the
directive run's posteriors for the free run, or (b) seeds the perceiver
deterministically per event. The current implementation does neither;
the methodological note in this subsection must be filled in once the
project lead decides which approach to use, or the per-event posterior
disagreement is reported as a methodological limitation in V.B.

## Section IV — Results

**Panel C from Step B (`runs/step_b/panel_c.png`) is the gate-check
figure** for the perception-signal claim: ID-confident-correct (median
0.0026, IQR [0.0014, 0.0049]) vs OOD pilot aggregate (median 0.0085,
IQR [0.0035, 0.0181]). The OOD distribution skews higher (3.3×
median ratio) with the upper IQR extending into the 0.018 range that
ID-confident-correct rarely reaches.

Headline Results in Section IV are still about *agent behavior* (the
core experimental contribution, generated in Steps E and F):
differential abstention rates between baseline and the variance-aware
conditions (directive and free; see III.D), gate-variance correlation
within OOD, action correctness against expected actions per bucket.
Panels A–D from Step B are supporting evidence for the perception-side
prerequisites, not the headline finding.

With three conditions in flight, IV tables must report the
condition-specific deltas explicitly:
`P(defer | OOD, variance_aware_directive) − P(defer | OOD, baseline)`
and `P(defer | OOD, variance_aware_free) − P(defer | OOD, baseline)`.
The directive–free delta itself is the V.A contribution; in IV it is
reported as a derived row rather than the headline.

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

**Single-temperature calibration on a heterogeneous distribution —
known failure mode, and the strongest motivation for the variance
approach.** Step C calibration (`runs/calibration_results.json`):

| Source | n | share | ECE pre | ECE post | NLL pre | NLL post |
|---|---|---|---|---|---|---|
| aggregate | 1845 | — | 0.0615 | **0.0510** | 0.3103 | 0.3068 |
| Giraff-X | 1105 | 60% | 0.0779 | **0.0648** | 0.2791 | 0.2698 |
| Gibson | 552 | 30% | 0.0671 | **0.0580** | 0.3554 | 0.3591 |
| DoorDetect | 188 | **10%** | 0.0690 | **0.0731** | 0.3615 | 0.3706 |

The single temperature scalar (T = 0.859) improves aggregate ECE and
helps both Antonazzi sources, but *actively worsens* calibration on
DoorDetect — the minority sub-distribution at 10% of the calibration
mix. This is a known failure mode of single-temperature calibration on
heterogeneous data: temperature scaling optimizes against the
calibration set's distribution mix, so sub-distributions that are
minority in that mix can be made worse, not better, by the fit. The
finding generalizes beyond our specific dataset: any deployment with
calibration data that doesn't proportionally match the runtime
distribution faces the same risk.

Per-source T would partially repair the DoorDetect-specific gap, but
that solution presumes the deployment system can identify source
membership at inference time. For genuinely OOD inputs (the case the
paper is built around), source identity is precisely the unknown.

### V.A — Directive vs free propagation of uncertainty (added 2026-05-17)

The two variance-aware conditions (directive and free; see III.D)
together address a methodological question the paper would otherwise
leave open: **when propagating a calibrated uncertainty signal into a
small LLM agent, is the right interface a hard rule the model executes,
or information the model reasons over discretionarily?**

The directive condition operationalizes uncertainty as a calibrated
mapping the LLM is instructed to follow. The free condition propagates
the same posterior but withholds the mapping, framing the choice as a
mission-aware judgment. The headline differential against baseline
(P(defer | OOD)) is reported for both conditions; the directive–free
delta is what speaks to the propagation-style question.

Three possible outcomes and their interpretations:

1. **Free ≈ directive on the headline metrics.** Small LLMs can reason
   over a surfaced posterior given mission framing — the directive's
   mapping is a sufficiency demonstration but not a necessity. The
   paper's deployment recommendation is "surface the posterior; mapping
   is optional, not load-bearing."

2. **Free < directive on the headline metrics.** Mission framing alone
   is insufficient at this scale; the calibrated mapping is doing the
   work. The paper recommends propagating uncertainty as a directive,
   not as information, when working at the 4B–26B scale. This is the
   most likely outcome consistent with the 2026-05-13 pilot
   (Section III.D.1).

3. **Free > directive on the headline metrics** (motion-rate retained
   on ID while OOD abstention is matched). Less likely; if observed,
   warrants follow-up because it suggests the directive's hard mapping
   is over-suppressing ID motion in ways the free condition's
   discretion avoids.

The interpretation reported in V.A must include the **reasoning chain
analysis under free**: chains that reference the calibrated label
("LOW variance ... I will move") versus chains that reason over the
mission ("doorway visible, approach"). The chain composition tells a
deployer whether free is succeeding because the LLM is doing what
directive prescribes anyway, or because it is genuinely composing the
posterior with the mission state.

**Confound to acknowledge:** the free condition's tool surface is a
superset of the directive's (free allows all five tools; directive
restricts to three — see III.D.2). The directive-vs-free action-
distribution comparison must collapse non-commit actions
(`rotate`, `report`, `look_around`, `defer`) into a single bucket to
remain interpretable. The headline P(defer | OOD) is unaffected, but
secondary distribution tables must adopt the collapse and state it.

**This motivates the variance signal as an alternative path to the
same problem.** Rather than try to predict the right temperature for
an unknown deployment distribution — a problem temperature scaling
fundamentally cannot solve when the distribution is open-set — the
MC Dropout posterior exposes the detector's *intrinsic* uncertainty
regardless of how the softmax was calibrated. The variance signal
does not depend on knowing which calibration regime the input fell
into; it derives from the model's own response to that input. Step B's
−0.98 competence-variance correlation (Section III) is empirical
evidence that this intrinsic signal tracks the same generalization
gradient that defeats deterministic calibration here. The agent
experiment in Section IV evaluates whether the LLM can act on this
signal productively when categorical calibration breaks down.

### V.B — Future work

- **Cross-family generalization of the propagation mechanism.** The
  experiment uses three tiers within the Gemma 4 family. Whether the
  same variance-aware prompt structure produces the same behavioral
  shift on a different open-weights family (e.g., Llama 3.2 Vision,
  IBM Granite Vision, or successor Gemma releases) is the subject of
  follow-on work. The mechanism — text-rendering of an MC Dropout
  posterior into the prompt — does not depend on Gemma-specific
  internals; the open question is whether comparably-sized models from
  other families read the variance abstraction equally well.

## Pre-specified success criteria — final prompt overnight runs (recorded 2026-05-13)

Recorded **before** the runs complete to lock the criteria against
post-hoc rationalization. The final-prompt overnight runs cover two
tiers (`gemma4:e4b` and `gemma4:26b`); 31B is deferred and out of
scope for this gate.

The runs are evaluated against four pre-specified criteria. All four
must hold on **both** tiers for the experiment to count as a positive
result. If any criterion fails on either tier, the failure is reported
and analyzed before any iteration; we do not iterate criteria after
seeing the data.

  1. **Reasoning chain coverage.** `agent_reasoning_chain` non-empty on
     ≥ 80% of events, both tiers, both conditions. This is the prompt-
     compliance gate — the prior 26B run scored 0% on this axis and
     drove the prompt redesign. Below 80% means the model is still
     ignoring the reasoning directive and the audit trail the paper
     relies on is incomplete.

  2. **ID-positives motion rate under variance_aware ≥ 85%, both
     tiers.** "Motion" = `move_forward` ∪ `rotate`. The
     `in_distribution_pos` bucket should produce confident commit
     behavior — if the variance_aware addendum over-suppresses motion
     on the easy cases, the agent isn't usable. This is the
     decisive-action-license check.

  3. **OOD-novel motion rate under variance_aware ≤ 15%, both tiers.**
     The `ood_novel` bucket should produce suppression — defer / look_
     around dominate, motion is rare. This is the variance-aware
     condition's main contribution: the variance signal moving the
     agent off of motion on inputs the detector cannot reliably
     classify.

  4. **Within OOD variance_aware: gate-conditional motion split.**
     LOW gate events: motion rate > 70%. HIGH gate events: motion
     rate < 30%. This is the within-OOD differential — the
     experimental claim is not just that OOD motion is suppressed in
     aggregate, but that the suppression is *signal-driven*. LOW gate
     events inside OOD should still commit; HIGH gate events should
     not. A flat distribution here means the LLM is reading the
     bucket, not the gate.

Failure modes and their interpretations:

  - Criterion 1 fails → prompt-compliance problem; the rest of the
    metrics are not interpretable until reasoning is being emitted.
    Halt; do not analyze 2-4.
  - Criterion 2 fails (motion too LOW on ID positives) → the LOW-gate
    decisive-action license isn't reaching the model, OR the
    look_around / defer disambiguation is over-rotating the model
    toward caution. Diff against the prior 26B run.
  - Criterion 3 fails (motion too HIGH on OOD novel) → the variance
    addendum isn't moving behavior; either the gate signal isn't being
    read or the addendum's caution language is too weak.
  - Criterion 4 fails (no gate-conditional split inside OOD) → H2 from
    the open-question section below: the LLM treats small raw variance
    values as "essentially zero" regardless of categorical label.
    Aggressive-dropout follow-up becomes in-scope.

The 31B tier is **not** launched on the basis of overnight results
alone, regardless of how the four criteria score. 31B requires
explicit user approval after analysis. Wall-clock budget for 31B
(~4-5× the 26B latency from prior viability) is not justified without
the user explicitly authorizing the spend.

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
