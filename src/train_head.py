"""Train a regression head on frozen CLIP features to predict the five nutrition targets.

Usage:
    python src/train_head.py --model mlp --loss l1                 # planned head, absolute loss
    python src/train_head.py --model mlp --loss mse                # same, squared loss
    python src/train_head.py --model linear --loss l1              # linear model
    python src/train_head.py --features <npz> --name masked-mlp-l1 # e.g. food-only photo features

Data: features from src/extract_clip_features.py, matched to data/splits/train.csv and dev.csv by
dish ID. The test split is never read.

Method (plan in docs/architecture.md; d2l.ai sections in brackets):
    - Inputs: CLIP features standardized with the training mean and standard deviation [5.7].
    - Targets: mass, energy, protein, carbohydrate, fat, each divided by its training mean so all
      five count equally in the loss.
    - Models: `linear` = 512 -> 5 [3.1]; `mlp` = 512 -> 128 (ReLU) -> 5 (Softplus) [5.1, 5.2],
      optional dropout [5.6]. Softplus keeps predictions positive; linear predictions are clipped at 0.
    - Loss: `l1` (absolute error, matches the reported metric) or `mse` (squared error) [3.1].
    - Optimizer: AdamW, learning rate 0.001, weight decay 0.0001 [3.7, 12.10], batch size 32,
      up to 30 epochs.
    - Model selection: after every epoch, score the development split; keep the epoch with the lowest
      normalized nutrient error [3.6].
    - Seeds 0, 1, 2 by default; results are reported per seed and as mean and standard deviation.

Outputs in experiments/<name>/:
    metrics.json                  settings, best epoch and development metrics per seed, summary
    training_log.csv              training loss and development score for every epoch and seed
    dev_predictions_seed<N>.csv   true and predicted values for each development dish
    checkpoints/seed<N>.pt        best weights per seed (ignored by Git)
"""

import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from evaluate import TARGETS, evaluate, load_split, normalized_nutrient_mae, write_predictions

REPO_ROOT = Path(__file__).resolve().parents[1]
SPLITS = REPO_ROOT / "data" / "splits"


def build_model(kind, n_in, hidden, dropout):
    if kind == "linear":
        return nn.Linear(n_in, len(TARGETS))
    return nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(), nn.Dropout(dropout),
                         nn.Linear(hidden, len(TARGETS)), nn.Softplus())


def predict(model, x, target_scale):
    """Predictions in original units (g, kcal), never negative."""
    model.eval()
    with torch.no_grad():
        return (model(x).clamp(min=0) * target_scale).numpy()


def train_one_seed(seed, args, x_train, y_train_scaled, x_dev, y_dev, dev_groups, target_scale, train_mean):
    torch.manual_seed(seed)
    model = build_model(args.model, x_train.shape[1], args.hidden, args.dropout)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.L1Loss() if args.loss == "l1" else nn.MSELoss()
    generator = torch.Generator().manual_seed(seed)

    best = {"score": float("inf")}
    log = []
    n = len(x_train)
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for idx in torch.randperm(n, generator=generator).split(args.batch_size):
            optimizer.zero_grad()
            loss = loss_fn(model(x_train[idx]), y_train_scaled[idx])
            loss.backward()
            optimizer.step()
            total += loss.item() * len(idx)
        dev_pred = predict(model, x_dev, target_scale)
        score = normalized_nutrient_mae(y_dev, dev_pred, train_mean)
        log.append({"seed": seed, "epoch": epoch, "train_loss": round(total / n, 5), "dev_score": round(score, 5)})
        if score < best["score"]:
            best = {"score": score, "epoch": epoch, "state": copy.deepcopy(model.state_dict()), "pred": dev_pred}

    model.load_state_dict(best["state"])
    metrics = evaluate(y_dev, best["pred"], dev_groups, train_mean)
    return model, best, metrics, log


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--features", default=str(REPO_ROOT.parent / "data" / "nutrition5k" / "features"
                                             / "clip_vitb32_original.npz"))
    p.add_argument("--model", choices=["linear", "mlp"], default="mlp")
    p.add_argument("--loss", choices=["l1", "mse"], default="l1")
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--exclude-flagged-train", action="store_true",
                   help="drop training dishes with energy_mismatch_flag or deprecated_ingredient_flag "
                        "(development is always scored on all dishes)")
    p.add_argument("--name", default=None, help="experiment folder name (default: <features>-<model>-<loss>)")
    args = p.parse_args()

    feat_path = Path(args.features)
    data = np.load(feat_path)
    row_of = {d: i for i, d in enumerate(data["dish_ids"])}
    features = data["features"].astype(np.float32)

    train_ids, _, y_train = load_split(SPLITS / "train.csv")
    dev_ids, dev_groups, y_dev = load_split(SPLITS / "dev.csv")
    if args.exclude_flagged_train:
        with open(SPLITS / "train.csv", encoding="utf-8") as f:
            flagged = {r["dish_id"] for r in csv.DictReader(f)
                       if r["energy_mismatch_flag"] == "1" or r["deprecated_ingredient_flag"] == "1"}
        keep = [i for i, d in enumerate(train_ids) if d not in flagged]
        train_ids, y_train = [train_ids[i] for i in keep], y_train[keep]
    missing = [d for d in train_ids + dev_ids if d not in row_of]
    if missing:
        raise SystemExit(f"{len(missing)} dishes have no features, e.g. {missing[:3]}")

    x_train = features[[row_of[d] for d in train_ids]]
    x_dev = features[[row_of[d] for d in dev_ids]]

    # Standardize inputs and scale targets using training data only.
    mu, sigma = x_train.mean(axis=0), x_train.std(axis=0) + 1e-6
    train_mean = y_train.mean(axis=0)
    x_train_t = torch.tensor((x_train - mu) / sigma)
    x_dev_t = torch.tensor((x_dev - mu) / sigma)
    y_train_scaled = torch.tensor(y_train / train_mean, dtype=torch.float32)
    target_scale = torch.tensor(train_mean, dtype=torch.float32)

    name = args.name or f"{feat_path.stem.replace('clip_vitb32_', '')}-{args.model}-{args.loss}"
    out = REPO_ROOT / "experiments" / name
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)

    per_seed, logs = {}, []
    for seed in args.seeds:
        model, best, metrics, log = train_one_seed(seed, args, x_train_t, y_train_scaled, x_dev_t, y_dev,
                                                    dev_groups, target_scale, train_mean)
        logs += log
        per_seed[seed] = {"best_epoch": best["epoch"], "dev": metrics}
        torch.save({"state_dict": model.state_dict(), "input_mean": mu, "input_std": sigma,
                    "target_scale": train_mean, "targets": TARGETS, "args": vars(args)},
                   out / "checkpoints" / f"seed{seed}.pt")
        write_predictions(out / f"dev_predictions_seed{seed}.csv", dev_ids, dev_groups, y_dev, best["pred"])
        print(f"seed {seed}: best epoch {best['epoch']}, normalized nutrient error "
              f"{metrics['normalized_nutrient_mae']:.4f}", flush=True)

    scores = np.array([s["dev"]["normalized_nutrient_mae"] for s in per_seed.values()])
    energy = np.array([s["dev"]["mae"]["energy_kcal"] for s in per_seed.values()])
    summary = {
        "normalized_nutrient_mae_mean": round(float(scores.mean()), 4),
        "normalized_nutrient_mae_std": round(float(scores.std(ddof=1)), 4) if len(scores) > 1 else None,
        "energy_mae_kcal_mean": round(float(energy.mean()), 1),
    }
    metrics = {
        "settings": vars(args),
        "features_sha256_first16": hashlib.sha256(feat_path.read_bytes()).hexdigest()[:16],
        "split_files_sha256_first16": {f: hashlib.sha256((SPLITS / f).read_bytes()).hexdigest()[:16]
                                       for f in ("train.csv", "dev.csv")},
        "n_train": len(train_ids), "n_dev": len(dev_ids),
        "torch": torch.__version__,
        "per_seed": per_seed,
        "summary": summary,
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    with open(out / "training_log.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["seed", "epoch", "train_loss", "dev_score"])
        w.writeheader()
        w.writerows(logs)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
