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

## Selector comparison

The first method is retained as the `candidate_union` baseline. A second
`plate_complement` selector fills the outer plate contour, subtracts the SAM
plate-surface mask, erodes the rim, and removes components that are too small or
too large. Both selectors use the same SAM ViT-B checkpoint, 16 x 16 point grid,
input images, and automatic-mask thresholds.

| Selector | Program completed | Program failed | Manual pass | Manual partial | Manual fail |
|---|---:|---:|---:|---:|---:|
| `candidate_union` | 6 | 3 | 0 | 2 | 7 |
| `plate_complement` | 5 | 4 | 0 | 2 | 7 |

The second method produced more coherent central masks on two medium-complexity
dishes, but it did not improve the overall manual ratings and failed to produce
a mask on one additional image. On other complex meals it still selected plate
surface or only one small item. Therefore `plate_complement` is recorded as a
negative result, not as a replacement for the baseline.

Detailed ratings are stored in `manifests/dev9_qualitative_review.csv` and
`manifests/dev9_plate_complement_review.csv`; the compact comparison is in
`manifests/dev9_selector_comparison.csv`.

Run either method with `--selector candidate_union` or
`--selector plate_complement`, using separate output directories so the
baseline evidence is not overwritten.

## Third selector using plate geometry and appearance

`appearance_selector.py` implements `plate_appearance`. It scores convex hulls
of SAM candidates using centered position, aspect ratio, compactness, neutral
surface color, and contact with image edges. A seven-pixel inward margin
reduces plate-rim contamination. It estimates the plate's Lab color from
neutral pixels and retains SAM regions with sufficient color/darkness contrast
or support from holes in the plate surface. It unions multiple accepted regions
and removes connected components smaller than 100 pixels. All parameters apply
equally to all nine images; dish IDs and nutrition targets are not selector inputs.

The candidate-union and plate-complement functions are unchanged from commit
`fcd8937`. Their original output directories are retained. The third method's
outputs are in `outputs/sam_dev9_plate_appearance_v1/`.

| Selector | Completed | Program failed | Manual pass | Manual partial | Manual fail |
|---|---:|---:|---:|---:|---:|
| Candidate union baseline | 6 | 3 | 0 | 2 | 7 |
| Plate complement | 5 | 4 | 0 | 2 | 7 |
| Plate appearance v1 | 7 | 2 | 4 | 3 | 2 |

Ratings use the same informal visual criteria as the earlier runs. Pass means
the main visible food is covered without a major non-food region; partial
means useful coverage with visible omissions or extra regions; fail includes
missing output or predominantly incorrect selection. These are one-reviewer
qualitative judgments, not IoU measurements. Images with a runtime failure count
as fail rather than being excluded from the denominator.

This method captures pale food, grapes, sliced food, and the combined
salad/fruit example better. Two complex meals have no plausible plate candidate;
another meal is clipped by the estimated plate hull. The method assumes a
light plate and can still confuse shadows, reflection, and pale food. These
nine images have informed method development, so this is a development result,
not an independent estimate of generalization. Freeze the parameters and use a
new group-disjoint development sample before claiming broader improvement.

Reproduce the fixed subset from the repository root:

```bash
.venv/bin/python A_food_segmentation/debug_validation/run_dev9.py \
  --selector plate_appearance \
  --output-dir A_food_segmentation/debug_validation/outputs/sam_dev9_plate_appearance_rerun
```

The same wrapper accepts `candidate_union` and `plate_complement`. Use a fresh
output directory for each run. All runs use frozen SAM ViT-B, CPU, a 16 x 16
grid, predicted-IoU threshold 0.86, stability threshold 0.92, no crop layers,
and minimum mask region area 100. The JSON summaries and per-image selections
are local evidence; `manifests/dev9_plate_appearance_review.csv` records the
third method's manual judgments. Run tests with:

```bash
.venv/bin/python -m unittest discover -s A_food_segmentation/debug_validation/tests -v
```

## Canonical data source

Do not use `manifests/nutrition5k_planning_seed539.csv` for model training or
evaluation. It is retained only to document the initial feasibility attempt.
Formal experiments must use the team's reviewed Cafe 1 train/development/test
files produced by the canonical split pipeline.
