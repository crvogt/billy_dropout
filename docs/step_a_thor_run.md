# Step A — Fine-tuning on AGX Thor

The fine-tune is the only piece of step A that requires GPU. Everything
else (data prep, dropout injection, smoke tests) runs anywhere.

## What you'll do

Run `scripts/train_dropout_yolo.py` on the Thor against the multi-source
doorway corpus prepared by `scripts/prepare_doorway_data.py`. The script:

1. Loads `yolov8n.pt` (auto-downloaded by ultralytics on first run)
2. Injects six `nn.Dropout2d(p=0.25)` modules into the cls branches of
   the Detect head (`uagent/perception/dropout.py`)
3. Calls `ultralytics.YOLO.train(...)` on the combined data.yaml
4. Verifies dropout modules survived save/load via a reload check
5. Copies the best checkpoint to `models/doorway_yolov8n_dropout.pt`

## Data sources

The corpus combines three sources (addendum-locked):

| Source | Counts (door-bearing) | Format | Provenance |
|---|---|---|---|
| **Giraff-X** (real, robot-perspective) | 3,657 | gzipped npy bboxes; 240×320 PNGs | Antonazzi et al. 2024 |
| **Gibson** (synthetic, photorealistic) | capped at 1,500 of 5,457 | same Antonazzi format; 256×256 PNGs | Antonazzi et al. 2024 |
| **DoorDetect** (real, general indoor) | 359 of 386 (rest have missing image files) | YOLO-format directly | Arduengo et al. 2021 |
| **Combined pool** | 5,516 | unified YOLO format under `data/doorway_combined/` | — |

Stratified 80/10/10 split → 4,413 train / 552 val / 551 calibration.

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

# Verify CUDA is visible to torch
python -c "import torch; print('cuda:', torch.cuda.is_available(), 'device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
```

### Datasets on Thor

The simplest path is to rsync the dev-box copies (already extracted
locally at `~/libs/billy_paper/uagent/data/`) rather than re-download:

```bash
# From the dev box:
rsync -avz --progress \
    ~/libs/billy_paper/uagent/data/doordetect/ \
    ~/libs/billy_paper/uagent/data/antonazzi/ \
    thor:~/uagent/data/

# Or, on the Thor, fresh from sources (slower; Antonazzi requires
# manual browser download from the OneDrive links in the AISLab README
# at https://github.com/aislabunimi/robotic-vision-door-detection):
git clone --depth 1 https://github.com/MiguelARD/DoorDetect-Dataset.git ~/uagent/data/doordetect
# Then place giraff_x.zip and gibson.zip under ~/uagent/data/antonazzi/
# and unzip them.
```

## Per-run procedure

```bash
cd ~/uagent
source .venv/bin/activate

# 1. Prepare data: filter to door class across three sources,
#    cap Gibson, stratified 80/10/10 split, write provenance manifest.
#    Deterministic (dataset.split_seed=1337 in default.yaml).
python scripts/prepare_doorway_data.py --config configs/default.yaml
# Expect: combined pool: 5516 door-bearing images
#         train: 4413, val: 552, calibration: 551
#         Provenance manifest at data/manifests/training_image_ids.json

# 2. Sanity check: load weights, inject dropout, count modules, exit before training
python scripts/train_dropout_yolo.py --config configs/default.yaml --inventory-only
# Expect: "post-injection Dropout2d count: 6"

# 3. Fine-tune. Default config: epochs=75, patience=10, batch=16, imgsz=640, device=cuda.
python scripts/train_dropout_yolo.py --config configs/default.yaml
# Expect: ultralytics' standard training output, then
#   "reloaded checkpoint Dropout2d count: 6"
#   "copied .../weights/best.pt -> models/doorway_yolov8n_dropout.pt"
```

Estimated wall time on AGX Thor: ~1–2 hours for the full 75 epochs at
batch=16; early stopping (patience=10) usually fires before then.

## Validating the trained model

After training, run pytest to catch any regressions in injection /
serialization:

```bash
pytest tests/test_dropout_injection.py tests/test_detector.py -v
```

Inspect ultralytics' val mAP from the training log (printed per epoch).
Spec target: deterministic mAP@0.5 > 0.4. If well below, **do not
proceed to step B** — flag for discussion.

## Tuning the run

`configs/default.yaml :: training` controls hyperparameters. Defaults
(epochs=75, patience=10) are sized for the ~5,500-image corpus with
early stopping. If you change the corpus size dramatically (e.g.
re-cap Gibson, add DeepDoors2 or iGibson supplementary sources), revisit:

- **Smaller corpus**: bump epochs higher and/or batch lower.
- **Larger corpus**: epochs and batch may need raising; patience can stay.

If mAP stalls below 0.4, the addendum's optional supplementary sources are:
- DeepDoors2 (Antonazzi et al. 2024 release; earlier real dataset)
- iGibson (Antonazzi et al. 2024 release; lower-fidelity synthetic)

Both are linked in the AISLab README. Extend `prepare_doorway_data.py`
to load them as additional sources. **Do not augment images.**

## Syncing the result back

The checkpoint at `models/doorway_yolov8n_dropout.pt` is what the rest of
the pipeline consumes. To run step B / C / etc. off the Thor:

```bash
# From the dev box, pull the trained checkpoint:
rsync -avz thor:~/uagent/models/doorway_yolov8n_dropout.pt ./models/
```

Or run all subsequent steps on the Thor — the AGX class machine is fine
for inference too.

## Step B preparation — already in place

The `MCDropoutYOLO.predict` eval-mode-reset bug surfaced previously is
**now fixed** (commit on `doorway-experiment` after Option A bypass).
The production K-pass path now uses `model(tensor)` directly with
ultralytics' NMS utilities. The new test
`tests/test_detector.py::test_mc_dropout_predict_with_injection_produces_variance`
covers the regression. Step B can move directly to validating variance
on the doorway-dropout checkpoint.
