# Step A — Fine-tuning on AGX Thor

The fine-tune is the only piece of step A that requires GPU. Everything
else (data prep, dropout injection, smoke tests) runs anywhere.

## What you'll do

Run `scripts/train_dropout_yolo.py` on the Thor against the doorway dataset
prepared by `scripts/prepare_doorway_data.py`. The script:

1. Loads `yolov8n.pt` (auto-downloaded by ultralytics on first run)
2. Injects six `nn.Dropout2d(p=0.25)` modules into the cls branches of
   the Detect head (`uagent/perception/dropout.py`)
3. Calls `ultralytics.YOLO.train(...)` on `data/doordetect_door/data.yaml`
4. Verifies dropout modules survived save/load via a reload check
5. Copies the best checkpoint to `models/doorway_yolov8n_dropout.pt`

## One-time setup on Thor

```bash
# Clone the repo and check out the doorway-experiment branch
git clone <uagent-repo-url> ~/uagent
cd ~/uagent
git checkout doorway-experiment

# Create venv with the pinned deps
python3.10 -m venv .venv
source .venv/bin/activate
pip install -e .

# Clone DoorDetect (~700 MB, mostly images-in-git)
git clone --depth 1 https://github.com/MiguelARD/DoorDetect-Dataset.git data/doordetect

# Verify CUDA is visible to torch
python -c "import torch; print('cuda:', torch.cuda.is_available(), 'device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
```

## Per-run procedure

```bash
cd ~/uagent
source .venv/bin/activate

# 1. Prepare data: filter to door class, split 70/15/15 (deterministic, seed=1337)
python scripts/prepare_doorway_data.py --config configs/default.yaml
# Expect: ~251 train / ~54 val / ~54 calibration

# 2. Sanity check: load weights, inject, count modules, exit before training
python scripts/train_dropout_yolo.py --config configs/default.yaml --inventory-only
# Expect: "post-injection Dropout2d count: 6"

# 3. Fine-tune. Default config: 2 epochs, batch=16, imgsz=640, device=cuda.
python scripts/train_dropout_yolo.py --config configs/default.yaml
# Expect: ultralytics' standard training output, then
#   "reloaded checkpoint Dropout2d count: 6"
#   "copied .../weights/best.pt -> models/doorway_yolov8n_dropout.pt"
```

## Validating the trained model

After training, run pytest to catch any regressions in injection /
serialization:

```bash
pytest tests/test_dropout_injection.py -v
```

Inspect ultralytics' val mAP from the training log (it prints
`mAP@0.5` per epoch). Spec target: deterministic mAP > 0.4. If well below,
**do not proceed to step B** — flag for discussion.

## Tuning the run

`configs/default.yaml :: training` controls hyperparameters. The shipped
`epochs: 2` is a smoke value. For a real fine-tune, raise to 50-100 and
re-run; 251 training images is small and 2 epochs almost certainly
underfits. Iterate on epochs / batch / imgsz based on val mAP.

If mAP stalls below 0.4, the spec's optional supplementary datasets are:
- DoorDetect-Class (Ramôa et al. 2020)
- Antonazzi et al. 2022 robot-perspective doors

Add via a second `prepare_*` script or by extending
`scripts/prepare_doorway_data.py`. **Do not augment.**

## Syncing the result back

The checkpoint at `models/doorway_yolov8n_dropout.pt` is what the rest of
the pipeline consumes. To run step B / C / etc. off the Thor:

```bash
# From the dev box, pull the trained checkpoint:
rsync -avz thor:~/uagent/models/doorway_yolov8n_dropout.pt ./models/
```

Or run all subsequent steps on the Thor — the AGX class machine is fine
for inference too.

## Known issue surfaced for step B

`uagent/perception/detector.py::MCDropoutYOLO.predict` currently calls
`ultralytics.YOLO.predict`, which resets the model to eval mode internally
*after* `_enable_dropout_only` runs. Against the doorway-dropout
checkpoint produced by this step, K stochastic passes will be byte-
identical and `epistemic_variance` will be 0. **First task of step B is
to fix this** — see the inline KNOWN BUG block at `detector.py:115`.
The two fix paths are documented there.
