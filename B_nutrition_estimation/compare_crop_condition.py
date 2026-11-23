"""Compare B4 natural crop against retained B0 and B2 predictions."""
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
    output = {"interpretation": "negative deltas favor B4", "n_dev": len(dev_ids),
              "n_dev_groups": len(set(groups)), "comparisons": {}}
    for baseline in ("B0", "B2"):
        rows = {}
        for seed in (0, 1, 2):
            base = predictions(args.results / baseline / f"dev_predictions_seed{seed}.csv", dev_ids)
            crop = predictions(args.results / "B4" / f"dev_predictions_seed{seed}.csv", dev_ids)
            delta = np.abs(crop - y_dev).mean(axis=0) - np.abs(base - y_dev).mean(axis=0)
            rows[str(seed)] = {
                "mae_delta": {name: float(value) for name, value in zip(TARGETS, delta)},
                "normalized_nutrient_mae_delta": normalized_nutrient_mae(y_dev, crop, train_mean) - normalized_nutrient_mae(y_dev, base, train_mean),
                "paired_plate_bootstrap_95ci": paired_interval(y_dev, base, crop, groups, train_mean),
            }
        output["comparisons"][f"B4_minus_{baseline}"] = rows
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
