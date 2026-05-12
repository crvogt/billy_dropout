# Thor execution checklist — Gemma 4 sweep (two-tier)

Runnable top-to-bottom on the AGX Thor box. Repo + data are assumed
already sync'd from the dev box (see `docs/step_a_thor_run.md` for the
initial rsync invocation). This doc covers the Gemma 4 thinking-mode
experiment, scaled down to **two tiers**: `gemma4:e4b` (small) and
`gemma4:26b` (medium).

**Why two, not three:** the 31B tier was viability-checked successfully
but its p95 latency (~284 s/event) makes the wall time prohibitive for
the full matrix. 31B is retained for future cross-family or supplementary
work; not in the headline experiment.

## What changed since the Step E checkpoint

- Thinking mode enabled: the literal token `<|think|>` is now the first
  content of every system prompt (both `baseline` and `variance_aware`).
- The harness parses the Gemma 4 thought channel
  (`<|channel>thought\n...\n<channel|>`) out of every model response and
  stores it as `agent_reasoning_chain`. The post-`<channel|>` text feeds
  LangChain's existing tool-call path unchanged.
- The implicit-defer fallback (no tool call emitted) now strips the
  thought channel out of the defer reason — so `final_args.reason` no
  longer leaks internal reasoning.

These changes are gated by passing tests (46/0) on `doorway-experiment`.

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

# Sanity-check the harness still imports + tests pass.
source .venv/bin/activate
python -m pytest tests/ -q
# Expect: 46 passed.

python -c "from uagent.experiment.harness import run_single_trial, parse_thought_channel; print('ok')"
```

**Do not run `pip install -e .` on Thor.** It will replace the
JetPack-built torch wheel with the PyPI ARM CPU wheel and break CUDA.
If a new dependency truly needs installing, use `pip install <pkg>
--no-deps` for that single package so torch is never re-resolved.

## 1. Pull Gemma 4 models (one-time)

We pull only the two tiers we'll run end-to-end. 31B can be re-pulled
later if needed.

```bash
# Tags locked in configs/default.yaml::llm.models
ollama pull gemma4:e4b      # small, ~3 GB
ollama pull gemma4:26b      # medium, ~16 GB MoE

ollama list | grep gemma4   # expect two rows
```

## 2. Verify e4b after the thinking-mode change

The harness is materially different from the Step E smoke run: the
system prompt now starts with `<|think|>`, and the response parser is
new. Before launching the full ~5-hour e4b run, confirm e4b's tool-call
validity AND reasoning-chain coherence on the new prompt + parser.

```bash
python scripts/verify_model.py --config configs/default.yaml --model small
echo "small verify exit=$?"
```

This writes `runs/verify_gemma4_e4b_<ts>/events.jsonl` with 15 events
(10 baseline + 5 variance_aware) and prints reasoning chains to stdout.

### STOP AND VERIFY — gate 1 (before full e4b)

Do not launch the full e4b experiment until **all four** of these hold:

1. `verify_model.py` exited **0** (baseline tool-call validity ≥ 90%,
   zero crashes).
2. Variance-aware reasoning chains in stdout are **non-empty** for at
   least 3 of 5 events. An empty chain across the board means the model
   ignored `<|think|>` and is not producing the audit trail the paper
   relies on. See Decision A below.
3. Reasoning chains read as **coherent English** about the detection /
   variance signal — not a refusal, not a garbled token soup, not the
   model's pre-training default rambling about an unrelated topic.
4. Action selection isn't pathologically uniform — at least two distinct
   actions appear across the 15 events.

If any of (1)–(4) fail, **halt and surface to the user**; do not
prompt-engineer around it on the fly.

## 3. e4b full experiment

```bash
# 100 images × 2 conditions × 3 trials × 1 model = 600 events
python scripts/run_experiment.py \
    --config configs/default.yaml \
    --models small \
    --run-name small_gemma4_e4b
echo "small full exit=$?"
```

Wall-clock estimate: **~5 hours**. This scales the qwen2.5:3b dev-box
latency by the e4b/3b parameter ratio and adds overhead for thinking
mode (the model emits a thought block per event, increasing tokens
generated per call). Treat as a budget, not a prediction.

### Resuming an interrupted run

The orchestrator skips events already in `events.jsonl` using the
resume key `(image_path, condition, trial_index, model)`. To resume,
**re-run the same command** — the events file is appended-and-fsynced,
so a partial run is always the source of truth. The test set is
deterministic (seeded), so the resumed run picks up the same images in
the same order.

```bash
# Identical to the original launch — resume is implicit.
python scripts/run_experiment.py --config configs/default.yaml \
    --models small --run-name small_gemma4_e4b
```

### Sync e4b results back to the dev box

Run from the **dev box**, not Thor:

```bash
rsync -avz thor:~/uagent/runs/small_gemma4_e4b/ \
    ~/libs/billy_paper/uagent/runs/small_gemma4_e4b/
```

### STOP AND VERIFY — gate 2 (before 26b)

Once e4b results are back on the dev box, summarize:

```bash
python -c "
from uagent.experiment.metrics import iter_events, summarize_smoke
events = list(iter_events('runs/small_gemma4_e4b/events.jsonl'))
print(summarize_smoke(events))
"
```

Surface to the user:

1. **Per-bucket abstention (defer rate)** per condition. Paper headline.
2. **Per-bucket commit rate** (`move_forward`+`rotate` share) per
   condition. The complementary metric — qwen2.5:3b funneled into
   `look_around` rather than `defer`, so we watch both edges.
3. **ID-vs-OOD differential** for each metric. The experimental contrast.
4. **Empty-reasoning-chain rate** per condition. New: with the
   no-fallback semantic, this exposes "thinking enabled but no thought
   block emitted" anomalies. Should be near 0 for a well-behaved tier.
5. **Reasoning-chain qualitative review**: pick 3 events where
   `variance_aware` diverged from `baseline` and quote the chain
   verbatim. Especially: chains that mention `variance`, `epistemic`,
   or the gate level (LOW/MEDIUM/HIGH).

The user reads this report and decides whether to launch the 26b tier.
This is the gate where the experiment can pivot if the small tier's
reasoning is too thin to bother spending another ~7 hours on 26b.

## 4. 26b full experiment (after user approval)

```bash
python scripts/run_experiment.py \
    --config configs/default.yaml \
    --models medium \
    --run-name medium_gemma4_26b
echo "medium full exit=$?"
```

Wall-clock estimate: **~7 hours**. 26B MoE is compute-light vs. 31B
dense but the per-call latency on Thor is meaningfully higher than e4b.
Resume semantics are identical to e4b (`run_experiment.py` keys on
`(image_path, condition, trial_index, model)`).

### Sync 26b results back

```bash
rsync -avz thor:~/uagent/runs/medium_gemma4_26b/ \
    ~/libs/billy_paper/uagent/runs/medium_gemma4_26b/
```

## Decision A — verify_model.py fails or chains are incoherent

If gate 1 fails:

- **Hallucinated tool names** (`agent_action` ∉ {move_forward, rotate,
  look_around, report, defer}) → the model didn't internalize the tool
  schema. Likely a prompt-format mismatch with thinking mode active.
  Pull `verify_gemma4_e4b_<ts>/events.jsonl` back to the dev box for
  review.

- **Empty `agent_action`** / `"(none)"` → the model emitted no tool
  call. With the new dispatcher path, this becomes an implicit defer;
  the `reason` field will have the model's post-thought text (stripped
  of the thought channel itself). Read it — is it model refusal? Loss
  of instruction-following under thinking mode? Both indicate the
  thinking-mode + tool-call combination isn't behaving.

- **Empty reasoning chains across the board** → the model ignored
  `<|think|>`. Check via:
  ```bash
  python -c "
  from uagent.experiment.metrics import iter_events
  events = list(iter_events('runs/<verify_dir>/events.jsonl'))
  empty = sum(1 for e in events if not e['agent_reasoning_chain'])
  print(f'empty chains: {empty}/{len(events)}')
  "
  ```
  If > 50% empty, the thinking-mode change is not landing. Possible
  causes: Ollama version mismatch (re-check `ollama --version`),
  template-handling difference, or the model emitting the thought block
  in a format other than the documented `<|channel>thought ...
  <channel|>`. Surface to user; do not retune.

- **Crashes (`n_crashes > 0`)** → check Thor memory + `ollama ps`. The
  26B MoE in particular is memory-sensitive.

## Decision B — gate 2 results are ambiguous

If e4b's variance-aware reasoning is uniformly weak (chains don't
mention variance, action distributions are identical between
conditions), there are two reads:

- **Capability ceiling**: e4b is too small to use the variance
  abstraction. This is a finding, not a bug — surface to user; the
  paper can report this as a scaling result if 26b shows a contrast.
- **Prompt design**: the model can engage with variance but our prompt
  isn't surfacing it well. Higher-risk fix; user decides.

Do not retune the prompt on the fly. 7 hours of 26B is not the right
forcing function for a prompt iteration — get the user's call first.

## Artifacts checklist

After both tiers finish and are sync'd back, the dev box should have:

```
runs/
  verify_gemma4_e4b_<ts>/events.jsonl   # viability audit trail (small)
  small_gemma4_e4b/events.jsonl         # 600 events, small tier
  medium_gemma4_26b/events.jsonl        # 600 events, medium tier
```

Plus the existing `runs/per_source_val_results.json`,
`runs/calibration_results.json`, `runs/step_b/`, and
`runs/spot_checks/`.

That's the input set for Step G (plotting and paper figures).
