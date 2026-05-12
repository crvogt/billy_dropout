# Thor execution checklist — Gemma 4 sweep

Runnable top-to-bottom. Executes on the AGX Thor box. The repo + data
are assumed already sync'd from the dev box (see `docs/step_a_thor_run.md`
for the rsync invocation). This doc covers the Gemma-4-era experiment.

## 0. Prerequisites

```bash
# Ollama version. Gemma 4 requires Ollama 0.20.0 or newer (per
# `ollama show`'s "requires" field on gemma4:e4b).
ollama --version
# Expect: 0.20.0 or higher. If lower, upgrade per
#   https://ollama.com/download/linux

# Ensure the doorway-experiment branch is current.
cd ~/uagent
git fetch origin
git checkout doorway-experiment
git pull --ff-only

# Venv exists and deps are installed (one-time):
source .venv/bin/activate
pip install -e . --quiet

# Sanity-check the harness still imports.
python -c "from uagent.experiment.harness import run_single_trial; print('ok')"
```

## 1. Pull Gemma 4 models (one-time, ~tens of minutes total)

```bash
# Tags locked in configs/default.yaml::llm.models
ollama pull gemma4:e4b      # small, ~3 GB
ollama pull gemma4:26b      # medium, ~16 GB MoE
ollama pull gemma4:31b      # large, ~20 GB dense

ollama list | grep gemma4   # expect three rows
```

## 2. Verify each tier before launching the full experiment

This is the viability check we couldn't run locally. Each invocation
runs ~15 events total (10 Test-1 baseline + 5 Test-2 variance_aware),
writes JSONL events to `runs/verify_<tag>_<ts>/`, prints reasoning
chains to stdout, and exits non-zero on failure.

```bash
# Small (gemma4:e4b). Quick to run; sets the floor for what we expect.
python scripts/verify_model.py --config configs/default.yaml --model small
echo "small exit=$?"

# Medium (gemma4:26b). The headline tier; this is the one that has to pass.
python scripts/verify_model.py --config configs/default.yaml --model medium
echo "medium exit=$?"

# Large (gemma4:31b). Slower; the --quick flag skips Test 2 if you want
# to iterate before the full reasoning-chain dump.
python scripts/verify_model.py --config configs/default.yaml --model large
echo "large exit=$?"
```

**Halt if any tier exits non-zero.** See "Decision A" below.

## 3. Full experiment per tier

Order: medium first (the headline), then small + large.

### 3a. Medium tier (the headline run)

```bash
# 100 images × 2 conditions × 3 trials × 1 model = 600 events
python scripts/run_experiment.py \
    --config configs/default.yaml \
    --models medium \
    --run-name medium_gemma4_26b
echo "medium exit=$?"
```

Wall-clock estimate: **~50 minutes upper bound**. This is a rough
scaling from the dev-box `qwen2.5:3b` CPU latency (~20 s/event);
Thor's GPU should be substantially faster — possibly under 20 minutes.
Treat the printed estimates as a budget, not a prediction.

After this finishes, **stop and inspect** before launching small + large.
See "Decision C" below.

### 3b. Small tier (after user approval)

```bash
python scripts/run_experiment.py \
    --config configs/default.yaml \
    --models small \
    --run-name small_gemma4_e4b
```

Wall-clock estimate: **~30 minutes upper bound** (scaled from dev-box
CPU; GPU likely faster).

### 3c. Large tier (after user approval)

```bash
python scripts/run_experiment.py \
    --config configs/default.yaml \
    --models large \
    --run-name large_gemma4_31b
```

Wall-clock estimate: **~90 minutes upper bound** (31B dense is the
slowest tier; even on GPU expect this to take meaningfully longer
than the 26B MoE).

### Resuming an interrupted run

The orchestrator skips events already in `events.jsonl`. To resume after
a crash, **re-run the same command** — the resume key is
`(image_path, condition, trial_index, model)` and the appended-and-fsynced
events.jsonl is the source of truth. The test set is deterministic
(seeded), so the resumed run picks up the same images.

```bash
# Identical to the original launch — resume is implicit.
python scripts/run_experiment.py --config configs/default.yaml \
    --models medium --run-name medium_gemma4_26b
```

## 4. Sync artifacts back to the dev box

After each tier completes, pull the run dir + metrics back. Run from
the **dev box**, not Thor:

```bash
rsync -avz thor:~/uagent/runs/medium_gemma4_26b/ \
    ~/libs/billy_paper/uagent/runs/medium_gemma4_26b/

# (Skip the verify_<tag>_<ts>/ directories if you don't need them
# locally — they're useful only as the audit trail of the viability
# check.)
```

## Decision A — verify_model.py fails on a tier

`verify_model.py` exits non-zero if baseline tool-call validity is
below 90% OR any event raised. Inspect:

```bash
# What's the actual validity rate?
ls runs/verify_*_*/events.jsonl | xargs -I {} python -c "
import json
events = [json.loads(l) for l in open('{}')]
n = len(events)
tools = {'move_forward','rotate','look_around','report','defer'}
ok = sum(1 for e in events if e['agent_action'] in tools)
print('{}: {}/{} = {:.0%} valid'.format('{}', ok, n, ok/n if n else 0))
"
```

**Failure-mode reading:**

- **Hallucinated tool names** (e.g., `agent_action == "navigate"` or
  some other not-in-surface name) → the model didn't internalize the
  tool schema. Likely a prompt-format mismatch. Halt and pull
  `verify_<tag>_<ts>/events.jsonl` back to the dev box for review.

- **Empty `agent_action`** or `agent_action == "(none)"` (the
  dispatcher's null fallback) → the model emitted no tool call. Could
  be: too small for tool use; needs different prompting; or hit a
  refusal pattern.

- **Crashes (`n_crashes > 0`)** → Ollama errors (OOM, timeout, refused
  load). Check Thor memory + `ollama ps`.

**Do not** prompt-engineer around model quirks on the fly. Halt and
report — the user decides whether to pivot models or invest in
adaptation.

## Decision B — variance-aware reasoning is incoherent

Test 2 dumps reasoning chains to stdout. Read them. The signal we're
looking for is the model engaging with the variance abstraction:

- **Coherent engagement** (good): reasoning chain mentions
  `variance`, `epistemic_variance`, the gate level (LOW/MEDIUM/HIGH),
  or paraphrases the `interpretation` string from the prompt. Action
  selection broadly tracks the gate level (HIGH → defer/look_around;
  LOW → move_forward).

- **Surface-level pattern matching** (acceptable): chain doesn't
  mention variance specifically but the action distribution still
  differs between LOW and HIGH gates in a way that matches the
  intended behavior. The model may be reading the categorical
  interpretation label rather than the raw number.

- **Incoherent / no engagement** (halt): chain is empty across all 5
  events, OR action is uniform regardless of gate level, OR chain
  reads as nonsense / refusal / instruction-following failure. This
  is a model-capability finding, not a config bug. **Halt and
  report; do not retune prompts.**

## Decision C — medium-tier full run results

After step 3a (`medium_gemma4_26b`) finishes, summarize:

```bash
python -c "
from uagent.experiment.metrics import iter_events, summarize_smoke
events = list(iter_events('runs/medium_gemma4_26b/events.jsonl'))
print(summarize_smoke(events))
"
```

**Surface to the user:**

1. **Per-bucket abstention (defer rate)** per condition. The
   headline paper metric.
2. **Per-bucket commit rate** (`move_forward`+`rotate` share) per
   condition. The complementary metric, important because the qwen2.5:3b
   smoke surfaced that some models channel uncertainty into
   `look_around` rather than `defer`.
3. **ID-vs-OOD differential** for each metric. The experimental
   contrast.
4. **Reasoning-chain qualitative review**: pick 3 events where
   variance_aware diverged from baseline and quote the reasoning chain.

The user reads this report and decides whether to launch small + large.

## Artifacts checklist

After all three tiers finish, the dev box should have:

```
runs/
  verify_gemma4_e4b_<ts>/events.jsonl   # viability audit trail for small
  verify_gemma4_26b_<ts>/events.jsonl   # medium
  verify_gemma4_31b_<ts>/events.jsonl   # large
  small_gemma4_e4b/events.jsonl         # 600 events, small tier
  medium_gemma4_26b/events.jsonl        # 600 events, medium tier
  large_gemma4_31b/events.jsonl         # 600 events, large tier
```

Plus the existing `runs/per_source_val_results.json`, `runs/calibration_results.json`,
`runs/step_b/` (gate output PDF + summary), and `runs/spot_checks/`.
