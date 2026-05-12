# uagent — uncertainty-gated agent (paper experiment)

A clean codebase for one experiment, distilled from the KIBOT thesis
(`~/libs/billy_paper/ros-llm-docker/`):

> When an agentic LLM runs on SWaP-constrained hardware, does propagating
> a calibrated perception posterior — mean confidence plus epistemic
> variance from MC Dropout — into the agent's prompt context measurably
> change its tool-selection behavior on out-of-distribution inputs,
> compared to a point-estimate baseline?

This repo is **not** a refactor of the thesis code. It is a fresh build
that harvests four reusable assets — YOLO weights, the robot dataset, the
rosbridge driving primitives, and the panorama capture — and writes
everything else from scratch.

## Architecture

A **single** LangGraph DAG with four nodes:

```
receive_command  →  query_perception  →  agent_reason  →  dispatch
                                              ↑              ↓
                                              └──── feedback ┘
```

One LLM call per reasoning turn, with the tool surface bound. No
Manager / Vision / Task split. A turn ends when the LLM emits a terminal
action (act-and-stop) or a `defer` HOLD decision (optionally with a
re-sense side effect).

## Two prompt conditions

`baseline` and `variance_aware` differ **only** in how a perception
result enters the prompt. See [uagent/agent/prompts.py](uagent/agent/prompts.py)
for the full templates side-by-side.

- `baseline`: `label / bbox / confidence`
- `variance_aware`: `label / bbox / mean_confidence / epistemic_variance / interpretation`

## Tool surface (5 tools, scope-limited for the paper)

| Tool | Risk tier | Wraps |
|---|---|---|
| `move_forward(distance_m)` | low | `driving_helpers.drive_robot_forward_raw` |
| `rotate(angle_deg)` | low | `driving_helpers.rotate_robot_raw` |
| `look_around()` | low | `vision_helpers.capture_panoramic_frames` (panorama) |
| `report(message)` | low | pure verbal, no robot motion |
| `defer(reason)` | low | explicit HOLD, ends turn |

## How to reproduce a run

```bash
# one-time setup
python3.10 -m venv .venv && source .venv/bin/activate
pip install -e .

# train dropout-YOLO (Branch B — see "Status" below)
python scripts/train_dropout_yolo.py --config configs/default.yaml

# calibrate the deterministic baseline
python scripts/calibrate_baseline.py --config configs/default.yaml

# build the three OOD buckets from robot_dataset/ + augmentations
python scripts/prepare_dataset.py --config configs/default.yaml

# run the experiment
python scripts/run_experiment.py --config configs/default.yaml
# → writes runs/<timestamp>/events.jsonl

# generate the four paper figures
python scripts/plot_results.py runs/<timestamp>/
```

## Configs

| File | Purpose |
|---|---|
| `configs/default.yaml` | canonical config (paper results, all defaults) |

Every path, model name, hyperparameter, and threshold lives in YAML. No
hardcoding in code. Box-specific overrides should be expressed as CLI
flags (`--models`, etc.) rather than partial YAML files — the script
loaders do not deep-merge configs.

## Status

- ✅ Inventory: existing checkpoints have no `nn.Dropout` layers.
  Branch B (retrain YOLOv8n with dropout in head) is required.
- ⏸ Retraining run is **deferred** by user direction.
- ⏸ OOD bucket source (synthesize vs collect) is open.
- 🚧 Skeleton + perception + agent runtime under construction.

## Tests

```bash
pytest tests/
```

Targets <30 s. The detector unit test uses a checked-in fixture image.
Agent unit tests use a mock LLM.
