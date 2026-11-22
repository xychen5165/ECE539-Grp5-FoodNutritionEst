"""Train one frozen-protocol B0-B3 nutrition regression condition."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evaluate import TARGETS, evaluate, load_split, normalized_nutrient_mae, write_predictions


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class SingleView(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(512, 128), nn.ReLU(), nn.Dropout(0.5),
                                     nn.Linear(128, 5), nn.Softplus())

    def forward(self, original, masked, geometry):
        return self.network(original if masked is None else masked)


class DualView(nn.Module):
    def __init__(self, with_geometry=False):
        super().__init__()
        self.original = nn.Sequential(nn.Linear(512, 64), nn.ReLU())
        self.masked = nn.Sequential(nn.Linear(512, 64), nn.ReLU())
        self.dropout = nn.Dropout(0.5)
        self.output = nn.Sequential(nn.Linear(136 if with_geometry else 128, 5), nn.Softplus())
        self.with_geometry = with_geometry

    def forward(self, original, masked, geometry):
        fused = torch.cat([self.original(original), self.masked(masked)], dim=1)
        if self.with_geometry:
            fused = torch.cat([fused, geometry], dim=1)
        return self.output(self.dropout(fused))


def load_feature(path, expected_ids, width):
    data = np.load(path)
    ids = [str(value) for value in data["dish_ids"]]
    if ids != expected_ids:
        raise ValueError(f"Dish order mismatch in {path}")
    values = data["features"].astype(np.float32)
    if values.shape != (len(ids), width) or not np.isfinite(values).all():
        raise ValueError(f"Invalid features in {path}: {values.shape}")
    return values


def standardize(train, dev):
    mean = train.mean(axis=0)
    std = train.std(axis=0) + 1e-6
    return ((train - mean) / std, (dev - mean) / std, mean, std)


def predict(model, tensors, scale):
    model.eval()
    with torch.inference_mode():
        return (model(*tensors).clamp(min=0) * scale).cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("condition", choices=["B0", "B1", "B2", "B3"])
    parser.add_argument("--input-root", type=Path,
                        default=ROOT / "artifacts/b-input-comparison")
    parser.add_argument("--output-root", type=Path,
                        default=ROOT / "experiments/005-b-input-comparison/results")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.001)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--limit-per-split", type=int,
                        help="Validation only; use the first N train and dev rows")
    args = parser.parse_args()

    train_ids, _, y_train = load_split(ROOT / "data/splits/train.csv")
    dev_ids, dev_groups, y_dev = load_split(ROOT / "data/splits/dev.csv")
    if args.limit_per_split:
        train_ids, y_train = train_ids[:args.limit_per_split], y_train[:args.limit_per_split]
        dev_ids = dev_ids[:args.limit_per_split]
        dev_groups, y_dev = dev_groups[:args.limit_per_split], y_dev[:args.limit_per_split]
    ids = train_ids + dev_ids
    n_train = len(train_ids)
    original = load_feature(args.input_root / "clip_original.npz", ids, 512)
    masked = load_feature(args.input_root / "clip_masked.npz", ids, 512)
    geometry = load_feature(args.input_root / "geometry.npz", ids, 8)
    original_train, original_dev, original_mean, original_std = standardize(
        original[:n_train], original[n_train:])
    masked_train, masked_dev, masked_mean, masked_std = standardize(
        masked[:n_train], masked[n_train:])
    geometry_train, geometry_dev, geometry_mean, geometry_std = standardize(
        geometry[:n_train], geometry[n_train:])
    arrays = {
        "original": (original_train, original_dev),
        "masked": (masked_train, masked_dev),
        "geometry": (geometry_train, geometry_dev),
    }
    tensors = {name: (torch.tensor(pair[0]), torch.tensor(pair[1]))
               for name, pair in arrays.items()}
    target_scale_np = y_train.mean(axis=0).astype(np.float32)
    target_scale = torch.tensor(target_scale_np)
    y_train_scaled = torch.tensor(y_train / target_scale_np, dtype=torch.float32)
    out = args.output_root / args.condition
    out.mkdir(parents=True, exist_ok=False)
    (out / "checkpoints").mkdir()
    all_logs, per_seed = [], {}
    started_all = time.perf_counter()

    for seed in args.seeds:
        torch.manual_seed(seed)
        if args.condition in ("B0", "B1"):
            model = SingleView()
        else:
            model = DualView(with_geometry=args.condition == "B3")
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                      weight_decay=args.weight_decay)
        generator = torch.Generator().manual_seed(seed)
        best = {"score": float("inf")}
        started = time.perf_counter()
        for epoch in range(1, args.epochs + 1):
            model.train()
            loss_sum = 0.0
            for index in torch.randperm(n_train, generator=generator).split(args.batch_size):
                optimizer.zero_grad()
                if args.condition == "B0":
                    batch = (tensors["original"][0][index], None, None)
                elif args.condition == "B1":
                    batch = (tensors["original"][0][index], tensors["masked"][0][index], None)
                else:
                    batch = (tensors["original"][0][index], tensors["masked"][0][index],
                             tensors["geometry"][0][index] if args.condition == "B3" else None)
                loss = nn.functional.l1_loss(model(*batch), y_train_scaled[index])
                loss.backward()
                optimizer.step()
                loss_sum += loss.item() * len(index)
            if args.condition == "B0":
                dev_tensors = (tensors["original"][1], None, None)
            elif args.condition == "B1":
                dev_tensors = (tensors["original"][1], tensors["masked"][1], None)
            else:
                dev_tensors = (tensors["original"][1], tensors["masked"][1],
                               tensors["geometry"][1] if args.condition == "B3" else None)
            dev_prediction = predict(model, dev_tensors, target_scale)
            score = normalized_nutrient_mae(y_dev, dev_prediction, target_scale_np)
            row = {"seed": seed, "epoch": epoch,
                   "train_loss": round(loss_sum / n_train, 6),
                   "dev_normalized_nutrient_mae": round(score, 6)}
            all_logs.append(row)
            if score < best["score"]:
                best = {"score": score, "epoch": epoch,
                        "state": copy.deepcopy(model.state_dict()),
                        "prediction": dev_prediction.copy()}
        model.load_state_dict(best["state"])
        serializable_settings = {key: str(value) if isinstance(value, Path) else value
                                 for key, value in vars(args).items()}
        checkpoint = {"condition": args.condition, "state_dict": model.state_dict(),
                      "target_scale": target_scale_np, "targets": TARGETS,
                      "normalization": {"original_mean": original_mean,
                                        "original_std": original_std,
                                        "masked_mean": masked_mean, "masked_std": masked_std,
                                        "geometry_mean": geometry_mean, "geometry_std": geometry_std},
                      "settings": serializable_settings, "seed": seed,
                      "best_epoch": best["epoch"]}
        torch.save(checkpoint, out / "checkpoints" / f"seed{seed}.pt")
        write_predictions(out / f"dev_predictions_seed{seed}.csv", dev_ids, dev_groups,
                          y_dev, best["prediction"])
        per_seed[str(seed)] = {"best_epoch": best["epoch"],
                               "seconds": time.perf_counter() - started,
                               "dev": evaluate(y_dev, best["prediction"], dev_groups,
                                               target_scale_np)}
        print(f"{args.condition} seed {seed}: {best['score']:.4f}", flush=True)

    with (out / "training_log.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_logs[0]))
        writer.writeheader()
        writer.writerows(all_logs)
    scores = np.array([entry["dev"]["normalized_nutrient_mae"]
                       for entry in per_seed.values()])
    metrics = {
        "status": "completed", "condition": args.condition,
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                cwd=ROOT, text=True).strip(),
        "settings": serializable_settings,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "n_train": n_train, "n_dev": len(dev_ids), "target_order": TARGETS,
        "inputs_sha256": {name: sha256(args.input_root / filename) for name, filename in {
            "original": "clip_original.npz", "masked": "clip_masked.npz",
            "geometry": "geometry.npz"}.items()},
        "split_sha256": {name: sha256(ROOT / f"data/splits/{name}.csv")
                         for name in ("train", "dev")},
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "numpy": np.__version__, "platform": platform.platform()},
        "per_seed": per_seed,
        "summary": {"normalized_nutrient_mae_mean": float(scores.mean()),
                    "normalized_nutrient_mae_std": (float(scores.std(ddof=1))
                                                     if len(scores) > 1 else None),
                    "seconds": time.perf_counter() - started_all},
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")


if __name__ == "__main__":
    main()
