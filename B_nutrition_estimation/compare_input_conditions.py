"""Create paired B0-B3 development comparisons from retained predictions."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evaluate import NUTRIENTS, TARGETS, load_split, normalized_nutrient_mae


def predictions(path, expected_ids):
    with path.open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if [row["dish_id"] for row in rows] != expected_ids:
        raise ValueError(f"Prediction order mismatch: {path}")
    return np.array([[float(row[f"pred_{target}"]) for target in TARGETS]
                     for row in rows])


def paired_interval(y_true, baseline, candidate, groups, train_mean, seed=0, n_boot=1000):
    group_names = sorted(set(groups))
    rows_of = {group: [] for group in group_names}
    for index, group in enumerate(groups):
        rows_of[group].append(index)
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(n_boot):
        drawn = rng.choice(len(group_names), size=len(group_names), replace=True)
        indices = np.concatenate([rows_of[group_names[index]] for index in drawn])
        deltas.append(normalized_nutrient_mae(y_true[indices], candidate[indices], train_mean)
                      - normalized_nutrient_mae(y_true[indices], baseline[indices], train_mean))
    return [float(value) for value in np.percentile(deltas, [2.5, 97.5])]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path,
                        default=ROOT / "experiments/005-b-input-comparison/results")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "experiments/005-b-input-comparison/comparison.json")
    args = parser.parse_args()
    train_ids, _, y_train = load_split(ROOT / "data/splits/train.csv")
    dev_ids, dev_groups, y_dev = load_split(ROOT / "data/splits/dev.csv")
    train_mean = y_train.mean(axis=0)
    conditions = ["B0", "B1", "B2", "B3"]
    stored = {condition: {} for condition in conditions}
    for condition in conditions:
        for seed in (0, 1, 2):
            stored[condition][seed] = predictions(
                args.results / condition / f"dev_predictions_seed{seed}.csv", dev_ids)
    comparisons = {}
    for baseline, candidate in (("B0", "B1"), ("B0", "B2"), ("B1", "B2"),
                                ("B2", "B3"), ("B0", "B3")):
        key = f"{candidate}_minus_{baseline}"
        comparisons[key] = {}
        for seed in (0, 1, 2):
            base, cand = stored[baseline][seed], stored[candidate][seed]
            nutrient_delta = (np.abs(cand - y_dev).mean(axis=0)
                              - np.abs(base - y_dev).mean(axis=0))
            comparisons[key][str(seed)] = {
                "mae_delta": {target: float(value)
                              for target, value in zip(TARGETS, nutrient_delta)},
                "normalized_nutrient_mae_delta": (
                    normalized_nutrient_mae(y_dev, cand, train_mean)
                    - normalized_nutrient_mae(y_dev, base, train_mean)),
                "paired_plate_bootstrap_95ci": paired_interval(
                    y_dev, base, cand, dev_groups, train_mean),
            }
    output = {"interpretation": "negative deltas favor the candidate",
              "n_dev": len(dev_ids), "n_dev_groups": len(set(dev_groups)),
              "bootstrap_draws": 1000, "bootstrap_seed": 0,
              "comparisons": comparisons}
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
