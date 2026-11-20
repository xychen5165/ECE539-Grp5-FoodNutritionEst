# Part A debug validation

This folder contains the small feasibility check performed before the team's
canonical Nutrition5k split was available. It is **debug evidence only** and
must not be used as an alternative train/development/test split.

## What this validation checks

1. Official Nutrition5k dish metadata and RGB split lists can be parsed.
2. A few overhead RGB images can be matched to dish-level nutrition labels.
3. Frozen SAM ViT-B can generate candidate masks on those images.
4. A deterministic first-pass rule can select food candidates while recording
   failures such as selecting the plate or paper background.

The three local examples are:

| Dish ID | Metadata description | Official split |
|---|---|---|
| `dish_1556572657` | 36 g olives | train |
| `dish_1556573514` | 23 g mixed greens | train |
| `dish_1556575014` | 62 g olives | train |

The images are official Nutrition5k overhead RGB files. They remain in
`local_data/`, which is ignored by Git as raw data. Generated masks and overlays
remain in the ignored `outputs/` directory. The manifests record dish IDs,
labels, paths, and input hashes without publishing raw images.

## Folder layout

```text
debug_validation/
  audit_nutrition5k.py       exploratory metadata and path audit
  run_sam_debug.py           frozen SAM feasibility runner
  manifests/                 small exploratory manifests and audit summaries
  tests/                     audit-script unit tests
  local_data/                ignored raw metadata and three RGB images
  outputs/                   ignored SAM masks, masked RGB, overlays, metrics
```

## Run SAM on the three local examples

From the repository root:

```bash
.venv/bin/python A_food_segmentation/debug_validation/run_sam_debug.py \
  --checkpoint checkpoints/sam_vit_b_01ec64.pth \
  --output-dir A_food_segmentation/debug_validation/outputs/sam_debug \
  --device cpu \
  --points-per-side 16 \
  A_food_segmentation/debug_validation/local_data/nutrition5k/imagery/realsense_overhead/dish_1556572657/rgb.png \
  A_food_segmentation/debug_validation/local_data/nutrition5k/imagery/realsense_overhead/dish_1556573514/rgb.png \
  A_food_segmentation/debug_validation/local_data/nutrition5k/imagery/realsense_overhead/dish_1556575014/rgb.png
```

The November 19 run generated 29, 25, and 36 candidates and took about 14
seconds per 640 x 480 image on CPU. Manual review found that the final masks
captured the main olives in two images and the main salad cluster in the third.
Thin onion strands in the salad were partly missed.

## Fixed-rule development check

After the three-image feasibility run, the selector was kept fixed and tested
on nine additional images sampled from the canonical development split: three
single-ingredient dishes, three dishes with two to four ingredients, and three
dishes with at least five ingredients. See `manifests/dev9_selection.csv`.

Program execution completed on six images and recorded failures on three rather
than stopping the batch. Manual review of the saved overlays found:

| Rating | Images | Meaning |
|---|---:|---|
| Pass | 0 | Main food region selected without a major non-food region |
| Partial | 2 | Some useful food coverage, but substantial misses or extra regions |
| Fail | 7 | No mask, mostly non-food, or only a small fraction of the meal |

The detailed review is in `manifests/dev9_qualitative_review.csv`. This check
shows that the first selector worked on the simple feasibility examples but
does not generalize to diverse development images. In particular, SAM candidate
confidence alone does not distinguish food from plate surface, fixtures, or a
single small item in a multi-food meal.

The next Part A experiment should keep this nine-image set fixed and compare a
second selection method against the recorded first-pass results. Do not tune a
different rule for each image.

## Canonical data source

Do not use `manifests/nutrition5k_planning_seed539.csv` for model training or
evaluation. It is retained only to document the initial feasibility attempt.
Formal experiments must use the team's reviewed Cafe 1 train/development/test
files produced by the canonical split pipeline.
