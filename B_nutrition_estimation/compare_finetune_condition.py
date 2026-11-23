"""Compare partially fine-tuned B5 against frozen-feature B4."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from B_nutrition_estimation.compare_input_conditions import paired_interval, predictions
from src.evaluate import TARGETS, load_split, normalized_nutrient_mae


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _, _, y_train = load_split(ROOT / "data/splits/train.csv")
    dev_ids, groups, y_dev = load_split(ROOT / "data/splits/dev.csv")
    train_mean = y_train.mean(axis=0)
    rows = {}
    for seed in (0, 1, 2):
        b4 = predictions(args.results / "B4" / f"dev_predictions_seed{seed}.csv", dev_ids)
        b5 = predictions(args.results / "B5" / f"dev_predictions_seed{seed}.csv", dev_ids)
        delta = np.abs(b5 - y_dev).mean(0) - np.abs(b4 - y_dev).mean(0)
        rows[str(seed)] = {
            "mae_delta": {name: float(value) for name, value in zip(TARGETS, delta)},
            "normalized_nutrient_mae_delta": normalized_nutrient_mae(y_dev, b5, train_mean) - normalized_nutrient_mae(y_dev, b4, train_mean),
            "paired_plate_bootstrap_95ci": paired_interval(y_dev, b4, b5, groups, train_mean),
        }
    output = {"interpretation": "negative deltas favor B5", "n_dev": len(dev_ids),
              "n_dev_groups": len(set(groups)), "bootstrap_draws": 1000,
              "comparison": {"B5_minus_B4": rows}}
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
