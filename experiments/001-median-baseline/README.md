# 001 - Median baseline

Status: Completed

**Result:** normalized nutrient error **0.727** on the development split (95% interval 0.652–0.808). A photo model is useful only if it scores clearly below this.

## Purpose

The baseline ignores the photo and predicts the same "typical" value for every dish. It sets the score that any trained model must beat: a model scoring about the same has not learned anything useful from the images. The plan specifies it in [docs/architecture.md](../../docs/architecture.md) ("Photo baseline: each target's training-set median").

## Procedure

Run with `python src/baseline_median.py`. Scoring code is in [src/evaluate.py](../../src/evaluate.py) and is shared by all models.

### Step 1: Compute the guess from the training split

For each target separately, sort the values of the 2,189 training dishes and take the middle one (position 1,095). This median is the guess for every development dish.

| Target | Training median (guess) |
|---|---|
| Mass | 177.0 g |
| Energy | 204.4 kcal |
| Protein | 11.2 g |
| Carbohydrate | 15.4 g |
| Fat | 8.5 g |

The median is used because it is the constant guess with the lowest average absolute error. Each median comes from its own column, so the five values together do not describe one real dish.

### Step 2: Measure the average error on the development split

For each of the 550 development dishes, take the absolute gap between the true value and the guess. Average the gaps.

Energy example (guess 204.4 kcal):

| Dish | True | Guess | Gap |
|---|---|---|---|
| dish_1557853229 | 195.9 | 204.4 | 8.4 |
| dish_1557853314 | 115.4 | 204.4 | 89.0 |
| dish_1557853429 | 330.2 | 204.4 | 125.8 |
| dish_1557862829 | 5.5 | 204.4 | 198.8 |
| … 546 more dishes | | | |

Sum of all 550 gaps = 89,049.4 kcal, so the average error = 89,049.4 ÷ 550 = **161.9 kcal**.

The same calculation for every target:

| Target | Average error |
|---|---|
| Mass | 118.8 g |
| Energy | 161.9 kcal |
| Protein | 15.5 g |
| Carbohydrate | 11.2 g |
| Fat | 10.0 g |

### Step 3: Combine the four nutrients into one score

**Training average:** the average true value of a nutrient over the 2,189 training dishes (for energy, 251.1 kcal).

Divide each nutrient's average error by its training average, then average the four results. Mass is reported separately because daily tracking and meal planning use only energy and macronutrients. The training averages are fixed, so every model is scaled the same way.

| Nutrient | Average error | Training average | Error ÷ training average |
|---|---|---|---|
| Energy | 161.9 kcal | 251.1 kcal | 0.645 |
| Protein | 15.5 g | 17.7 g | 0.878 |
| Carbohydrate | 11.2 g | 19.2 g | 0.587 |
| Fat | 10.0 g | 12.5 g | 0.797 |

Normalized nutrient error = (0.645 + 0.878 + 0.587 + 0.797) ÷ 4 = **0.727**.

### Step 4: Estimate the uncertainty

Draw 219 development plates at random with replacement, recompute the normalized nutrient error from the saved predictions, and repeat 1,000 times (seed 0). The middle 95% of these scores gives the interval **0.652–0.808**. Whole plates are resampled, not single photos, because photos of the same plate are not independent.

## Setup

- Contributor: xychen5165
- Date: 2026-11-19 (America/Chicago)
- Code: commit `37b8307` plus `src/evaluate.py` and `src/baseline_median.py`
- Data: [data/splits/train.csv](../../data/splits/train.csv) (hash `53487925d4e7f975`) and [data/splits/dev.csv](../../data/splits/dev.csv) (hash `a29c1a96a22eaeaf`). The test split is not used.

## Evidence

- [metrics.json](metrics.json): all numbers above, plus the split file hashes. It also lists each error as a percentage of the development average (`mae_pct_of_mean`), for comparison with the Nutrition5k paper.
- [dev_predictions.csv](dev_predictions.csv): true and predicted values for each development dish.

## Notes

- If the split files change (the plate-grouping time cutoff is still under review), rerun the script; it takes seconds.
- Next: extract CLIP features from the overhead photos and train the regression head and a linear model, scored with the same code.
