# A - Food segmentation

Component workspace for public-data preparation and frozen SAM integration. The
metadata audit and reproducible subset-manifest step are implemented; SAM
integration remains planned.

## Current runnable step

`audit_nutrition5k.py` parses the two official dish metadata CSV files, checks
the official RGB train/test lists for overlap and missing labels, matches each
dish to its expected overhead `rgb.png` path, and writes a deterministic small
manifest plus an audit summary. It uses only the Python standard library.

The full Nutrition5k archive is about 181 GB. Do not download it for this first
check. Download only `metadata/`, `dish_ids/splits/`, and a few overhead RGB
images from the public bucket, preserving this layout under the ignored
`data/raw/nutrition5k/` directory:

```text
data/raw/nutrition5k/
  metadata/dish_metadata_cafe1.csv
  metadata/dish_metadata_cafe2.csv
  dish_ids/splits/rgb_train_ids.txt
  dish_ids/splits/rgb_test_ids.txt
  imagery/realsense_overhead/<dish_id>/rgb.png  # optional for local match checks
```

Create a label-only planning subset even if images have not been downloaded:

```bash
python3 A_food_segmentation/audit_nutrition5k.py \
  --dataset-root data/raw/nutrition5k \
  --output data/splits/nutrition5k_debug_seed539.csv \
  --train-limit 20 \
  --test-limit 5 \
  --seed 539
```

Add `--require-images` when the manifest must contain only verified local
image-label pairs. The summary JSON records counts and SHA-256 hashes of the
four metadata/split inputs. Raw files and downloaded images remain ignored by
Git.

Run the checks with:

```bash
python3 -m unittest discover -s tests -v
```

This debug manifest is not yet the final train/development/evaluation split.
Before model comparison, the team must agree on a plate-group mapping so related
incremental scans cannot cross the internal train/development boundary.

## Deliverables
- Match Nutrition5k RGB images to nutrient labels and audit related plate groups.
- Freeze shared train/development/final-evaluation manifests before model comparisons.
- Generate candidate masks with frozen SAM ViT-B using an internal point grid; select the food region using development-set area, location, and overlap rules.
- Document mask selection, segmentation quality checks, and the portion-confirmation workflow.

## Interface
Input: RGB meal image.
Output: selected food-region mask and RGB image with a neutral background, preserving image scale, for component B.

Keep component experiments and notes here; place reusable pipeline code in ../src/ and shared data instructions/manifests in ../data/. Follow ../CONTRIBUTING.md. Do not commit raw datasets, checkpoints or private identity mappings.
