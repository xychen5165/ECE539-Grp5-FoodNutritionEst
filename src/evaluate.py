"""Shared evaluation for nutrient predictions. Every model is scored with these functions.

Metrics (see docs/architecture.md, "Training and evaluation"):
    mae                        mean absolute error per target, in the target's units
    mae_pct_of_mean            mean absolute error / mean true value of the evaluated set, x 100
    normalized_nutrient_mae    average over energy, protein, carbohydrate and fat of
                               (mean absolute error / training-set mean); mass is reported separately
    plate_bootstrap_ci         95% interval for normalized_nutrient_mae, resampling whole plate groups
"""

import csv
from pathlib import Path

import numpy as np

TARGETS = ["mass_g", "energy_kcal", "protein_g", "carbohydrate_g", "fat_g"]
NUTRIENTS = ["energy_kcal", "protein_g", "carbohydrate_g", "fat_g"]  # used for tracking and planning


def load_split(path):
    """Return dish IDs, plate groups and an (n, 5) array of targets in TARGETS order."""
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    ids = [r["dish_id"] for r in rows]
    groups = [r["plate_group"] for r in rows]
    y = np.array([[float(r[t]) for t in TARGETS] for r in rows])
    return ids, groups, y


def normalized_nutrient_mae(y_true, y_pred, train_mean):
    cols = [TARGETS.index(t) for t in NUTRIENTS]
    mae = np.abs(y_pred - y_true).mean(axis=0)
    return float(np.mean(mae[cols] / train_mean[cols]))


def plate_bootstrap_ci(y_true, y_pred, groups, train_mean, n_boot=1000, seed=0):
    """95% interval for normalized_nutrient_mae, resampling whole plate groups with replacement."""
    rng = np.random.default_rng(seed)
    group_names = sorted(set(groups))
    rows_of = {g: [] for g in group_names}
    for i, g in enumerate(groups):
        rows_of[g].append(i)
    scores = []
    for _ in range(n_boot):
        drawn = rng.choice(len(group_names), size=len(group_names), replace=True)
        idx = np.concatenate([rows_of[group_names[k]] for k in drawn])
        scores.append(normalized_nutrient_mae(y_true[idx], y_pred[idx], train_mean))
    low, high = np.percentile(scores, [2.5, 97.5])
    return float(low), float(high)


def evaluate(y_true, y_pred, groups, train_mean):
    mae = np.abs(y_pred - y_true).mean(axis=0)
    true_mean = y_true.mean(axis=0)
    return {
        "n_dishes": int(len(y_true)),
        "n_plates": len(set(groups)),
        "mae": {t: round(float(m), 3) for t, m in zip(TARGETS, mae)},
        "mae_pct_of_mean": {t: round(float(100 * m / mu), 1) for t, m, mu in zip(TARGETS, mae, true_mean)},
        "normalized_nutrient_mae": round(normalized_nutrient_mae(y_true, y_pred, train_mean), 4),
        "normalized_nutrient_mae_95ci": [round(v, 4) for v in plate_bootstrap_ci(y_true, y_pred, groups, train_mean)],
    }


def write_predictions(path, ids, groups, y_true, y_pred):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["dish_id", "plate_group"] + [f"true_{t}" for t in TARGETS] + [f"pred_{t}" for t in TARGETS])
        for i, d in enumerate(ids):
            w.writerow([d, groups[i]] + [round(v, 3) for v in y_true[i]] + [round(v, 3) for v in y_pred[i]])
