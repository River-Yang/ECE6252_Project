# Deepfake Detector Robustness Under Shift

This repo contains the code and saved result artifacts for our final project on
deepfake detection under distribution shift. We train a ResNet50 binary
classifier on FF++ face crops, evaluate it on clean FF++ and DFDC, test several
post-processing shifts, and run a small provenance / watermark policy
simulation on top of detector scores.

The detector uses `0 = real` and `1 = fake`. Inference is run on face frames,
then video-level scores are computed by averaging frame scores from the same
video.

**Main files**

```text
configs/default.yaml            Main configuration file
src/datasets/                   Dataset download, split, frame, and face-crop scripts
src/models/resnet50_binary.py   ResNet50 binary classifier
src/training/train.py           FF++ baseline training
src/training/inference.py       Shared inference and artifact writing
src/evaluation/                 Clean, shift, cross-dataset, and report table scripts
src/policies/                   Provenance / watermark policy simulation
process/                        Optional utilities for materialized shifted copies
data/*/splits/                  Split CSVs used by the experiments
results/tables/                 Result tables used in the report
results/figures/                Figures used in the report
```

There are no notebooks in this version. All experiments are run through Python
scripts.

**Environment**

Recommended Python version: `3.9+`.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If using the official Kaggle DFDC data instead of the Hugging Face mirror,
install and configure the Kaggle CLI as well.

**Before running on a new machine**

Edit `configs/default.yaml` first. The current default data paths point to the
scratch directory used for the original runs:

```text
/home/hice1/yyang3119/scratch/deepfake_data/...
```

The entries most likely to need changes are:

```text
ffpp_real_dir
ffpp_fake_dir
dfdc_real_dir
dfdc_fake_dir
ffpp_frame_manifest
dfdc_frame_manifest
ffpp_face_manifest
dfdc_face_manifest
ffpp_shift_root
hf_cache_dir
```

The output paths can usually remain relative:

```yaml
checkpoint_dir: results/checkpoints
prediction_dir: results/predictions
figure_dir: results/figures
table_dir: results/tables
log_dir: results/logs
```

Another option is to keep `configs/default.yaml` unchanged and use a local copy:

```bash
cp configs/default.yaml configs/local.yaml
vim configs/local.yaml
python -m src.training.train --config configs/local.yaml
```

Do not commit a local config if it contains private paths or credentials.

**Datasets**

FF++ face images:

- Hugging Face: `RohanRamesh/ff-images-dataset`
- Link: https://huggingface.co/datasets/RohanRamesh/ff-images-dataset

DFDC:

- Hugging Face mirror: `191fa07121/deepfake-detection-challenge`
- Link: https://huggingface.co/datasets/191fa07121/deepfake-detection-challenge
- Official Kaggle challenge data: https://www.kaggle.com/competitions/deepfake-detection-challenge

**Data preparation**

Prepare FF++:

```bash
python -m src.datasets.prepare_ffpp_hf --config configs/default.yaml
```

This writes `data/ffpp/splits/ffpp_splits.csv` and the FF++ face manifest at
`paths.ffpp_face_manifest`.

Prepare DFDC from the Hugging Face mirror:

```bash
python -m src.datasets.prepare_dfdc_hf --config configs/default.yaml
python -m src.datasets.build_dfdc_dataset --config configs/default.yaml
python -m src.datasets.extract_frames \
  --config configs/default.yaml \
  --split-csv data/dfdc/splits/dfdc_test.csv \
  --output-root <dfdc_frame_image_dir> \
  --manifest-out <dfdc_frame_manifest>
python -m src.datasets.face_crop \
  --config configs/default.yaml \
  --frame-manifest <dfdc_frame_manifest> \
  --output-root <dfdc_face_image_dir> \
  --manifest-out <dfdc_face_manifest>
```

Use the same DFDC frame and face paths that are set in `configs/default.yaml`.
For Kaggle DFDC, replace the first DFDC command with:

```bash
python -m src.datasets.prepare_dfdc_kaggle --config configs/default.yaml
```

**Reproducing results**

Run these commands in order.

Train the FF++ baseline:

```bash
python -m src.training.train --config configs/default.yaml
```

Expected outputs:

- `results/checkpoints/resnet50_best_auc.pt`
- `results/logs/train_history.csv`

The checkpoint is not committed because it is several hundred MB. To reproduce
the evaluation results from scratch, run training first or place a compatible
checkpoint at `results/checkpoints/resnet50_best_auc.pt`.

Pretrained checkpoint used for the report:
[Google Drive](https://drive.google.com/file/d/1hhRfOEpgjR2nckCx6ElvxkDg2PFbn3nO/view?usp=sharing)

After downloading it, place it at:

```text
results/checkpoints/resnet50_best_auc.pt
```

Run clean FF++ and DFDC evaluation:

```bash
python -m src.evaluation.cross_dataset_eval --config configs/default.yaml
```

Expected outputs:

- `results/tables/ffpp_test_metrics.json`
- `results/tables/dfdc_test_clean_metrics.json`
- `results/tables/cross_dataset_drop.json`

Run FF++ shift evaluation:

```bash
python -m src.evaluation.shift_eval --config configs/default.yaml --datasets ffpp
```

Reported FF++ conditions:

```text
clean
jpeg_q95
jpeg_q75
jpeg_q50
jpeg_q30
blur_s1
blur_s2
gaussian_noise_s8
speckle_s005
pixelate_b8
brightness_07
brightness_13
contrast_07
crop_90
crop_80
resize_reencode
combo_comp_blur_noise
```

Expected outputs:

- `results/tables/shift_metrics_summary.csv`
- `results/figures/shift_auc_bar_ffpp.png`
- `results/figures/shift_fnr_line_ffpp.png`

`combo_comp_blur_noise` is a pre-generated mixed-shift FF++ test set. Its CSV is
configured by `paths.ffpp_external_shift_manifest`. If that file is missing, the
single-shift conditions can still be reproduced.

Run the provenance / watermark policy simulation:

```bash
python -m src.policies.authenticity_policy_eval --config configs/default.yaml
```

Expected outputs:

- `results/tables/authenticity_policy_summary.csv`
- `results/predictions/authenticity_signals_all_conditions.csv`
- `results/predictions/authenticity_policy_decisions.csv`

Build the report tables and figures:

```bash
python -m src.evaluation.final_report_assets --config configs/default.yaml
```

Expected outputs:

- `results/tables/report_core_metrics.csv`
- `results/tables/report_shift_focus.csv`
- `results/tables/report_cross_dataset_drop.csv`
- `results/tables/report_policy_focus.csv`
- `results/figures/report_core_metrics.png`
- `results/figures/report_shift_overview.png`
- `results/figures/report_shift_failure_modes.png`
- `results/figures/report_policy_ffpp.png`
- `results/figures/report_policy_dfdc.png`

**Optional shifted-copy workflow**

Most shift conditions are applied during evaluation. The `process/` folder is
only needed when creating a separate shifted copy of FF++ images on disk.

```bash
python process/apply_shifts.py \
  --input-dir /path/to/ffpp/faces/images/test_shift_pixelate \
  --type pixelate \
  --assignment fixed \
  --seed 42
```

Then evaluate that shifted copy:

```bash
python -m src.evaluation.shift_infra \
  --config configs/default.yaml \
  --checkpoint results/checkpoints/resnet50_best_auc.pt \
  --prefix ffpp_pixelate \
  --shift-name pixelate \
  --shift-process-manifest /path/to/test_shift_pixelate/test_shift_pixelate_manifest.csv
```

More details are in `process/README.md`.

**Notes**

- The default classification threshold is `0.5`.
- The model checkpoint is ignored by Git because of file size.
- Frame-level prediction CSVs are also ignored by Git; video-level predictions,
  tables, and figures are kept for the report.
- The policy experiment is a simulation over detector scores and synthetic
  authenticity signals, not a separately trained classifier.
