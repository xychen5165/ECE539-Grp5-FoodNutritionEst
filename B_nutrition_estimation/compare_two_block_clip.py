"""Compare B6 with completed, ID-aligned B5 and retained B4 predictions."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from B_nutrition_estimation.compare_input_conditions import paired_interval, predictions
from src.evaluate import TARGETS, load_split, normalized_nutrient_mae


def checked_predictions(folder, seed, ids, labels):
    path = folder / f"dev_predictions_seed{seed}.csv"
    values = predictions(path, ids)
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    recorded = np.array([[float(row['true_' + key]) for key in TARGETS] for row in rows])
    np.testing.assert_allclose(recorded, labels, rtol=0, atol=0.00051)
    if values.shape != (538, 5) or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('Invalid development predictions: ' + str(path))
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--b6-results', type=Path, required=True)
    parser.add_argument('--b5-results', type=Path)
    parser.add_argument('--b4-results', type=Path,
                        default=ROOT / 'experiments/006-b-natural-crop/results/B4')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    _, _, train = load_split(ROOT / 'data/splits/train.csv')
    ids, groups, labels = load_split(ROOT / 'data/splits/dev.csv')
    scale = train.mean(0)
    comparisons = {}
    baselines = {'B4': args.b4_results}
    if args.b5_results is not None:
        baselines['B5'] = args.b5_results
    for name, folder in baselines.items():
        records = {}
        for seed in (0, 1, 2):
            candidate = checked_predictions(args.b6_results, seed, ids, labels)
            baseline = checked_predictions(folder, seed, ids, labels)
            delta = np.abs(candidate-labels).mean(0) - np.abs(baseline-labels).mean(0)
            records[str(seed)] = {
                'mae_delta': dict(zip(TARGETS, map(float, delta))),
                'normalized_nutrient_mae_delta': normalized_nutrient_mae(labels, candidate, scale)
                    - normalized_nutrient_mae(labels, baseline, scale),
                'paired_plate_bootstrap_95ci': paired_interval(labels, baseline, candidate, groups, scale),
            }
        comparisons['B6_minus_' + name] = records
    output = {'interpretation': 'Negative candidate-minus-baseline deltas favor B6',
              'n_dev': len(ids), 'n_dev_groups': len(set(groups)),
              'bootstrap_draws': 1000, 'bootstrap_seed': 0,
              'primary_comparison_status': 'completed' if args.b5_results else 'pending_missing_completed_B5',
              'comparisons': comparisons, 'final_test_accessed': False}
    args.output.write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
