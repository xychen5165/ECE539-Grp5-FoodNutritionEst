# Colab execution guide for experiment 007/B5

Use a GPU runtime and run in order. Experiment 007 reuses the completed 005/006
Drive root. It does not rerun SAM or crop feature extraction.

```python
%pip install -q \
  'numpy>=1.26,<3' 'Pillow>=10,<13' \
  open_clip_torch==2.24.0 opencv-python-headless==4.11.0.86 \
  'segment-anything @ git+https://github.com/facebookresearch/segment-anything.git@dca509fe793f601edb92606367a655c15ac00fdf'
```

```python
%cd /content
![ -d /content/food-nutrition/.git ] || git clone \
  --branch 'merge-A&B' --single-branch \
  https://github.com/xychen5165/ECE539-Grp5-FoodNutritionEst.git \
  /content/food-nutrition
!git -C /content/food-nutrition pull --ff-only origin 'merge-A&B'
```

Mount Drive and define the retained run root:

```python
from google.colab import drive
drive.mount('/content/drive')
from pathlib import Path
import os
RUN_ROOT = Path('/content/drive/MyDrive/ECE539-FoodNutritionEst/005-b-input-comparison')
os.environ['RUN_ROOT'] = str(RUN_ROOT)
for required in ['clip_original.npz', 'clip_crop.npz', 'results/B4/metrics.json']:
    assert (RUN_ROOT / required).is_file(), required
```

Run the 007 environment check after pulling the frozen implementation:

```bash
!python /content/food-nutrition/B_nutrition_estimation/capture_finetune_environment.py \
  --output "$RUN_ROOT/environment-007.json" --run-root "$RUN_ROOT"
```

Stage image files on the Colab VM to avoid repeated small-file Drive reads:

```bash
!mkdir -p /content/b5-data/images /content/b5-data/crops
!rsync -a "$RUN_ROOT/images/" /content/b5-data/images/
!rsync -a "$RUN_ROOT/crops/" /content/b5-data/crops/
```

Train B5. Do not run another GPU job concurrently. If CUDA reports insufficient
memory before training begins, stop and report it rather than changing settings.

```bash
!python /content/food-nutrition/B_nutrition_estimation/train_clip_finetune.py \
  --image-root /content/b5-data \
  --artifact-root "$RUN_ROOT" \
  --output-root "$RUN_ROOT/results" \
  --clip-cache "$RUN_ROOT/checkpoints"
```

Compare and package small evidence:

```bash
!python /content/food-nutrition/B_nutrition_estimation/compare_finetune_condition.py \
  --results "$RUN_ROOT/results" --output "$RUN_ROOT/comparison-007.json"
!python /content/food-nutrition/B_nutrition_estimation/package_finetune_results.py \
  --run-root "$RUN_ROOT" --output "$RUN_ROOT/007-small-results.zip"
!ls -lh "$RUN_ROOT/007-small-results.zip"
```

Download `007-small-results.zip`. Do not inspect final-test data or tune B5 after
viewing development results.
