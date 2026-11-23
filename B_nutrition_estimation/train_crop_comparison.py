"""Train B4: parameter-matched original plus natural-crop dual-view fusion."""
from __future__ import annotations

import argparse
import copy
import csv
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
from B_nutrition_estimation.train_input_comparison import (DualView, load_feature,
                                                            predict, sha256, standardize)
from src.evaluate import TARGETS, evaluate, load_split, normalized_nutrient_mae, write_predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.001)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args()

    train_ids, _, y_train = load_split(ROOT / "data/splits/train.csv")
    dev_ids, dev_groups, y_dev = load_split(ROOT / "data/splits/dev.csv")
    ids, n_train = train_ids + dev_ids, len(train_ids)
    original = load_feature(args.input_root / "clip_original.npz", ids, 512)
    crop = load_feature(args.input_root / "clip_crop.npz", ids, 512)
    original_train, original_dev, original_mean, original_std = standardize(original[:n_train], original[n_train:])
    crop_train, crop_dev, crop_mean, crop_std = standardize(crop[:n_train], crop[n_train:])
    tensors = {
        "original": (torch.tensor(original_train), torch.tensor(original_dev)),
        "crop": (torch.tensor(crop_train), torch.tensor(crop_dev)),
    }
    target_scale_np = y_train.mean(axis=0).astype(np.float32)
    target_scale = torch.tensor(target_scale_np)
    y_train_scaled = torch.tensor(y_train / target_scale_np, dtype=torch.float32)
    out = args.output_root / "B4"
    out.mkdir(parents=True, exist_ok=False)
    (out / "checkpoints").mkdir()
    all_logs, per_seed = [], {}
    started_all = time.perf_counter()

    for seed in args.seeds:
        torch.manual_seed(seed)
        model = DualView(False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        generator = torch.Generator().manual_seed(seed)
        best = {"score": float("inf")}
        started = time.perf_counter()
        for epoch in range(1, args.epochs + 1):
            model.train()
            loss_sum = 0.0
            for index in torch.randperm(n_train, generator=generator).split(args.batch_size):
                optimizer.zero_grad()
                output = model(tensors["original"][0][index], tensors["crop"][0][index], None)
                loss = nn.functional.l1_loss(output, y_train_scaled[index])
                loss.backward()
                optimizer.step()
                loss_sum += loss.item() * len(index)
            dev_prediction = predict(model, (tensors["original"][1], tensors["crop"][1], None), target_scale)
            score = normalized_nutrient_mae(y_dev, dev_prediction, target_scale_np)
            all_logs.append({"seed": seed, "epoch": epoch,
                             "train_loss": round(loss_sum / n_train, 6),
                             "dev_normalized_nutrient_mae": round(score, 6)})
            if score < best["score"]:
                best = {"score": score, "epoch": epoch,
                        "state": copy.deepcopy(model.state_dict()),
                        "prediction": dev_prediction.copy()}
        torch.save({"condition": "B4", "state_dict": best["state"],
                    "target_scale": target_scale, "targets": TARGETS,
                    "normalization": {"original_mean": torch.tensor(original_mean),
                                      "original_std": torch.tensor(original_std),
                                      "crop_mean": torch.tensor(crop_mean),
                                      "crop_std": torch.tensor(crop_std)},
                    "settings": vars(args), "seed": seed, "best_epoch": best["epoch"]},
                   out / "checkpoints" / f"seed{seed}.pt")
        write_predictions(out / f"dev_predictions_seed{seed}.csv", dev_ids, dev_groups,
                          y_dev, best["prediction"])
        per_seed[str(seed)] = {"best_epoch": best["epoch"],
                               "seconds": time.perf_counter() - started,
                               "dev": evaluate(y_dev, best["prediction"], dev_groups, target_scale_np)}
        print(f"B4 seed {seed}: {best['score']:.4f}", flush=True)

    with (out / "training_log.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_logs[0]))
        writer.writeheader(); writer.writerows(all_logs)
    scores = np.array([entry["dev"]["normalized_nutrient_mae"] for entry in per_seed.values()])
    serializable = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    metrics = {"status": "completed", "condition": "B4",
               "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
               "settings": serializable, "parameter_count": sum(p.numel() for p in model.parameters()),
               "n_train": n_train, "n_dev": len(dev_ids), "target_order": TARGETS,
               "inputs_sha256": {"original": sha256(args.input_root / "clip_original.npz"),
                                 "crop": sha256(args.input_root / "clip_crop.npz")},
               "environment": {"python": platform.python_version(), "torch": torch.__version__,
                               "numpy": np.__version__, "platform": platform.platform()},
               "per_seed": per_seed,
               "summary": {"normalized_nutrient_mae_mean": float(scores.mean()),
                           "normalized_nutrient_mae_std": (float(scores.std(ddof=1))
                                                             if len(scores) > 1 else None),
                           "seconds": time.perf_counter() - started_all}}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str) + "\n")


if __name__ == "__main__":
    main()
