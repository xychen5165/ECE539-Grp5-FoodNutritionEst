# 001 - Median baseline

Status: Completed

**Result:** normalized nutrient error **0.737** on the development split (95% interval 0.672–0.812). A photo model is useful only if it scores clearly below this.

## Purpose

The baseline ignores the photo and predicts the same "typical" value for every dish. It sets the score that any trained model must beat: a model scoring about the same has not learned anything useful from the images. The plan specifies it in [docs/architecture.md](../../docs/architecture.md) ("Photo baseline: each target's training-set median").

## Procedure

Run with `python src/baseline_median.py`. Scoring code is in [src/evaluate.py](../../src/evaluate.py) and is shared by all models.

### Step 1: Compute the guess from the training split

For each target separately, sort the values of the 2,135 training dishes and take the middle one (position 1,068). This median is the guess for every development dish.

| Target | Training median (guess) |
|---|---|
| Mass | 184.0 g |
| Energy | 213.9 kcal |
| Protein | 12.5 g |
| Carbohydrate | 15.4 g |
| Fat | 9.1 g |

The median is used because it is the constant guess with the lowest average absolute error. Each median comes from its own column, so the five values together do not describe one real dish.

### Step 2: Measure the average error on the development split

For each of the 538 development dishes, take the absolute gap between the true value and the guess. Average the gaps.

Energy example (guess 213.9 kcal):

| Dish | True | Guess | Gap |
|---|---|---|---|
| dish_1557853229 | 195.9 | 213.9 | 17.9 |
| dish_1557861697 | 13.1 | 213.9 | 200.7 |
| dish_1557863012 | 3.1 | 213.9 | 210.8 |
| dish_1558028690 | 23.8 | 213.9 | 190.1 |
| … 534 more dishes | | | |

Sum of all 538 gaps = 92,583.9 kcal, so the average error = 92,583.9 ÷ 538 = **172.1 kcal**.

The same calculation for every target:

| Target | Average error |
|---|---|
| Mass | 120.3 g |
| Energy | 172.1 kcal |
| Protein | 15.1 g |
| Carbohydrate | 12.0 g |
| Fat | 10.6 g |

### Step 3: Combine the four nutrients into one score

**Training average:** the average true value of a nutrient over the 2,135 training dishes (for energy, 255.9 kcal).

Divide each nutrient's average error by its training average, then average the four results. Mass is reported separately because daily tracking and meal planning use only energy and macronutrients. The training averages are fixed, so every model is scaled the same way.

| Nutrient | Average error | Training average | Error ÷ training average |
|---|---|---|---|
| Energy | 172.1 kcal | 255.9 kcal | 0.673 |
| Protein | 15.1 g | 18.4 g | 0.823 |
| Carbohydrate | 12.0 g | 19.2 g | 0.623 |
| Fat | 10.6 g | 12.7 g | 0.830 |

Normalized nutrient error = (0.673 + 0.823 + 0.623 + 0.830) ÷ 4 = **0.737**.

### Step 4: Estimate the uncertainty

Draw 182 development plates at random with replacement, recompute the normalized nutrient error from the saved predictions, and repeat 1,000 times (seed 0). The middle 95% of these scores gives the interval **0.672–0.812**. Whole plates are resampled, not single photos, because photos of the same plate are not independent.

## Setup

- Contributor: xychen5165
- Date: 2026-11-20 (America/Chicago), rerun on the combined-grouping splits
- Code: `src/evaluate.py` and `src/baseline_median.py` (commit `dcde4c8`), splits from `src/build_splits.py` with combined grouping
- Data: [data/splits/train.csv](../../data/splits/train.csv) (hash `b4713f8b5125b1f7`) and [data/splits/dev.csv](../../data/splits/dev.csv) (hash `e3025e1a9f47e963`). The test split is not used.

## Evidence

- [metrics.json](metrics.json): all numbers above, plus the split file hashes. It also lists each error as a percentage of the development average (`mae_pct_of_mean`), for comparison with the Nutrition5k paper.
- [dev_predictions.csv](dev_predictions.csv): true and predicted values for each development dish.

## Notes

- Earlier run on the first splits (ingredient rule only, 2,189 training / 550 development dishes): 0.727 (0.652–0.808). The new splits remove more plate leakage, so the scores are not directly comparable.
- If the split files change, rerun the script; it takes seconds.
- Next: extract CLIP features from the overhead photos and train the regression head and a linear model, scored with the same code.
