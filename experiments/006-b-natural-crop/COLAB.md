# Colab execution guide for experiment 006

Use a GPU runtime. This experiment reuses the completed 005 Drive root and does
not rerun downloads, SAM segmentation, original features, B0 or B2.

```bash
%cd /content/food-nutrition
!git pull origin 'merge-A&B'
!git status --short
!git rev-parse HEAD
```

If this is a fresh runtime, install the same dependencies as experiment 005 and
mount Google Drive. Then define the existing root:

```python
from pathlib import Path
import os
RUN_ROOT = Path('/content/drive/MyDrive/ECE539-FoodNutritionEst/005-b-input-comparison')
os.environ['RUN_ROOT'] = str(RUN_ROOT)
assert (RUN_ROOT / 'clip_original.npz').is_file()
assert (RUN_ROOT / 'results/B0/metrics.json').is_file()
assert (RUN_ROOT / 'results/B2/metrics.json').is_file()
print(RUN_ROOT)
```

Generate deterministic crops locally in the Drive run root:

```bash
!python B_nutrition_estimation/prepare_crop_comparison.py crop \
  --root "$RUN_ROOT" --margin-fraction 0.10
```

Extract frozen CLIP features on the GPU:

```bash
!python B_nutrition_estimation/prepare_crop_comparison.py features \
  --root "$RUN_ROOT" --clip-cache "$RUN_ROOT/checkpoints" \
  --batch-size 64 --device cuda
```

Train B4. An existing `results/B4` is rejected; preserve failed attempts under a
renamed directory before retrying.

```bash
!python B_nutrition_estimation/train_crop_comparison.py \
  --input-root "$RUN_ROOT" --output-root "$RUN_ROOT/results"
```

Compare with the retained B0 and B2 predictions, then package small evidence:

```bash
!python B_nutrition_estimation/compare_crop_condition.py \
  --results "$RUN_ROOT/results" --output "$RUN_ROOT/comparison-006.json"
!python B_nutrition_estimation/package_crop_results.py \
  --run-root "$RUN_ROOT" --output "$RUN_ROOT/006-small-results.zip"
!ls -lh "$RUN_ROOT/006-small-results.zip"
```

Download `006-small-results.zip`. Do not access canonical final-test images or
labels, change the crop margin, or tune settings after viewing B4 results.
