# Colab Pro execution guide

Run each cell in order. Choose **Runtime → Change runtime type → GPU** first.
The large run root is stored in Google Drive, so downloads, masks, features,
logs and checkpoints survive a Colab disconnect. No final-test images are read.

## 1. Confirm a GPU before installing anything

```bash
!nvidia-smi
```

Stop if `nvidia-smi` fails. Pro improves availability but does not guarantee a
particular GPU model.

## 2. Clone the frozen branch

```bash
%cd /content
!git clone --branch 'merge-A&B' --single-branch \
  https://github.com/xychen5165/ECE539-Grp5-FoodNutritionEst.git food-nutrition
%cd /content/food-nutrition
!git rev-parse HEAD
!git diff --exit-code 112c26e98f99ee16984532b7798ba42c6bac6911 -- \
  A_food_segmentation/sam_runtime.py \
  B_nutrition_estimation/prepare_input_comparison.py \
  B_nutrition_estimation/train_input_comparison.py \
  B_nutrition_estimation/compare_input_conditions.py \
  tests/test_b_input_comparison.py
```

The final command must print no diff. Record-only commits after `112c26e` are
allowed; implementation changes are not.

## 3. Install the experiment dependencies

Use Colab's CUDA-enabled torch/torchvision instead of replacing them.

```bash
%pip install -q \
  'numpy>=1.26,<3' 'Pillow>=10,<13' \
  open_clip_torch==2.24.0 opencv-python-headless==4.11.0.86 \
  'segment-anything @ git+https://github.com/facebookresearch/segment-anything.git@dca509fe793f601edb92606367a655c15ac00fdf'
```

If Colab asks for a runtime restart, restart, then repeat cell 1 and `%cd` back
to `/content/food-nutrition`. Do not rerun `git clone` over an existing folder.

## 4. Mount Drive and define the persistent run root

```python
from google.colab import drive
drive.mount('/content/drive')
```

```python
from pathlib import Path
import os
RUN_ROOT = Path('/content/drive/MyDrive/ECE539-FoodNutritionEst/005-b-input-comparison')
RUN_ROOT.mkdir(parents=True, exist_ok=True)
os.environ['RUN_ROOT'] = str(RUN_ROOT)
print(RUN_ROOT)
```

## 5. Capture the exact runtime before model/data execution

```bash
!python B_nutrition_estimation/capture_colab_environment.py \
  --output "$RUN_ROOT/environment.json" \
  --run-root "$RUN_ROOT"
!cat "$RUN_ROOT/environment.json"
```

The script fails if CUDA is unavailable, hashes differ, the implementation has
changed since the frozen code commit, or Train/Dev split files differ.

## 6. Download only canonical Train and Dev images

```bash
!python B_nutrition_estimation/prepare_input_comparison.py download \
  --root "$RUN_ROOT" --workers 12
```

This stage is resume-safe. Existing valid RGB files are verified and skipped.

## 7. Generate Part A masks on GPU

```bash
!mkdir -p "$RUN_ROOT/checkpoints"
![ -f "$RUN_ROOT/checkpoints/sam_vit_b_01ec64.pth" ] || \
  wget -q --show-progress \
  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth \
  -O "$RUN_ROOT/checkpoints/sam_vit_b_01ec64.pth"
!sha256sum "$RUN_ROOT/checkpoints/sam_vit_b_01ec64.pth"
```

Expected SAM hash:
`ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912`.

```bash
!python B_nutrition_estimation/prepare_input_comparison.py segment \
  --root "$RUN_ROOT" \
  --sam-checkpoint "$RUN_ROOT/checkpoints/sam_vit_b_01ec64.pth" \
  --device cuda
```

Segmentation is resume-safe per image. It validates nondegenerate masks,
unchanged foreground pixels and RGB-128 background before recording each item.

## 8. Extract both frozen CLIP feature sets

```bash
!python B_nutrition_estimation/prepare_input_comparison.py features \
  --root "$RUN_ROOT" --clip-cache "$RUN_ROOT/checkpoints" \
  --batch-size 64 --device cuda
```

If GPU memory is insufficient, rerun this cell with `--batch-size 32` or 16 and
record that unavoidable preprocessing-budget difference. This stage writes the
final arrays atomically but restarts feature extraction if interrupted.

## 9. Train B0, B1, B2 and B3

Run sequentially so all four share the same runtime and do not compete for GPU.

```bash
!for CONDITION in B0 B1 B2 B3; do \
  python B_nutrition_estimation/train_input_comparison.py "$CONDITION" \
    --input-root "$RUN_ROOT" --output-root "$RUN_ROOT/results" || exit 1; \
done
```

Each condition retains seeds 0/1/2, logs, ID-keyed predictions and checkpoints.
Existing output directories are rejected instead of overwritten.

## 10. Run paired comparisons and package small evidence

```bash
!python B_nutrition_estimation/compare_input_conditions.py \
  --results "$RUN_ROOT/results" --output "$RUN_ROOT/comparison.json"
!python B_nutrition_estimation/package_colab_results.py \
  --run-root "$RUN_ROOT" \
  --output "$RUN_ROOT/005-small-results.zip"
```

Download `005-small-results.zip` from Drive and attach it to the experiment task.
It excludes raw RGBs, masks, feature arrays and checkpoints; those remain in
Drive. The archive contains environment, hashes, logs, metrics, predictions and
the paired comparison needed to complete the English run records.

## Restart and failure policy

- Reconnect, mount Drive, clone/pull the same branch and continue at the failed
  stage. Download and segmentation reuse verified files.
- Never delete or overwrite a failed `results/B*` directory. Rename it with an
  attempt suffix and record the failure before rerunning.
- Do not open, download or evaluate canonical Test images during experiment 005.
- Do not tune settings after seeing B0–B3 results; the protocol is already frozen.

