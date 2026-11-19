"""Median baseline: predict each target's training-set median for every dish.

Usage:
    python src/baseline_median.py

Fits on data/splits/train.csv and scores on data/splits/dev.csv (the test split is not used).
Writes metrics.json and dev_predictions.csv to experiments/001-median-baseline/.
The median is the best constant prediction for mean absolute error, so any useful model must beat it.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from evaluate import TARGETS, evaluate, load_split, write_predictions

REPO_ROOT = Path(__file__).resolve().parents[1]
SPLITS = REPO_ROOT / "data" / "splits"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=str(REPO_ROOT / "experiments" / "001-median-baseline"))
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    _, _, y_train = load_split(SPLITS / "train.csv")
    dev_ids, dev_groups, y_dev = load_split(SPLITS / "dev.csv")

    train_median = np.median(y_train, axis=0)
    train_mean = y_train.mean(axis=0)
    y_pred = np.tile(train_median, (len(y_dev), 1))

    metrics = {
        "model": "training-set median per target",
        "fit_on": "train.csv",
        "evaluated_on": "dev.csv",
        "split_files_sha256_first16": {
            f: hashlib.sha256((SPLITS / f).read_bytes()).hexdigest()[:16] for f in ("train.csv", "dev.csv")
        },
        "train_median": {t: round(float(v), 3) for t, v in zip(TARGETS, train_median)},
        "train_mean": {t: round(float(v), 3) for t, v in zip(TARGETS, train_mean)},
        "dev": evaluate(y_dev, y_pred, dev_groups, train_mean),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    write_predictions(out / "dev_predictions.csv", dev_ids, dev_groups, y_dev, y_pred)
    print(json.dumps(metrics["dev"], indent=2))


if __name__ == "__main__":
    main()
