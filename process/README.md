# FFPP Shift Processing

Utilities in this folder create FF++ test-set copies with synthetic image shifts
already written to disk. The main evaluation code can apply most shifts at
inference time, so this workflow is only needed for materialized shifted folders.

Basic workflow:

1. Copy the clean FF++ test image folder.
2. Run `apply_shifts.py` on the copy.
3. Run `build_eval_manifests.py` to create eval CSVs that point test rows to the
   shifted images.

The image filenames and relative paths are kept unchanged. The scripts modify
pixel values in place and write a manifest recording the applied shifts.

## Outputs

`apply_shifts.py` writes:

- `test_shift_*/test_shift_*_manifest.csv`
- `test_shift_*/SHIFT_INFO.md`

The manifest is used by later evaluation scripts. `SHIFT_INFO.md` is a short
record of the run settings.

## Supported Shifts

- `dct`: block-based low-pass filtering
- `cosine4`: smoother global low-pass filtering
- `compression`: JPEG recompression
- `noise`: additive Gaussian noise
- `speckle`: multiplicative noise
- `blur`: Gaussian blur
- `resize`: downsample then upsample
- `pixelate`: blocky low-resolution look
- `lighter`: increase brightness
- `dimmer`: decrease brightness
- `contrast_down`: reduce contrast

## Examples

Create a shifted copy:

```bash
cp -r ffpp/faces/images/test ffpp/faces/images/test_shift
python process/apply_shifts.py \
  --input-dir ffpp/faces/images/test_shift \
  --assignment combo \
  --max-shifts-per-image 2 \
  --seed 42
```

Apply one fixed shift to every image:

```bash
python process/apply_shifts.py \
  --input-dir ffpp/faces/images/test_shift \
  --type pixelate \
  --assignment fixed
```

Limit the allowed shifts:

```bash
python process/apply_shifts.py \
  --input-dir ffpp/faces/images/test_shift \
  --type compression noise blur
```

Assignment modes:

- `single`: one sampled shift per image
- `combo`: between 1 and `--max-shifts-per-image` shifts per image
- `fixed`: the same selected shifts for every image

Useful optional controls:

- `--ratio`: modify only part of the folder
- `--mode media`: use the media-style shift preset
- `--prob`: set per-shift probabilities
- `--jpeg-quality`: fix JPEG quality for `compression`

Example with probabilities:

```bash
python process/apply_shifts.py \
  --input-dir ffpp/faces/images/test_shift \
  --type compression blur noise \
  --prob compression=0.95 blur=0.4 noise=0.2
```

Example with fixed JPEG quality:

```bash
python process/apply_shifts.py \
  --input-dir ffpp/faces/images/test_shift \
  --type compression \
  --assignment fixed \
  --jpeg-quality 50
```

## Building Eval Manifests

After shifted folders exist, run:

```bash
python process/build_eval_manifests.py
```

Defaults:

- base manifest: `ffpp/faces/ffpp_faces.csv`
- image root: `ffpp/faces/images`
- shifted folders: `test_shift_*`

Generated eval CSVs keep train and validation rows unchanged and replace only
test `face_path` values with shifted-image paths.

Custom paths:

```bash
python process/build_eval_manifests.py \
  --base-manifest ffpp/faces/ffpp_faces.csv \
  --images-root ffpp/faces/images \
  --output-dir ffpp/faces/images
```

`apply_shifts.py` modifies the folder passed with `--input-dir`. Run it on a
copy, not on the clean source folder.
