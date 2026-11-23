"""Train B5 with the last shared CLIP visual block partially fine-tuned."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import platform
import random
import subprocess
import sys
import time

import numpy as np
import open_clip
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from B_nutrition_estimation.extract_clip_features import pad_to_square
from B_nutrition_estimation.train_input_comparison import load_feature
from src.evaluate import TARGETS, evaluate, load_split, normalized_nutrient_mae, write_predictions


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selected_visual_parameters(visual):
    """Return exactly block 11, post-LN and projection parameters."""
    selected = []
    for name, parameter in visual.named_parameters():
        if (name.startswith("transformer.resblocks.11.")
                or name.startswith("ln_post.") or name == "proj"):
            selected.append((name, parameter))
    if not selected or not any(name.startswith("transformer.resblocks.11.")
                               for name, _ in selected):
        raise ValueError("Expected OpenCLIP ViT-B/32 visual block 11")
    return selected


class PairedImages(Dataset):
    def __init__(self, ids, labels, image_root, preprocess):
        self.ids = ids
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.image_root = Path(image_root)
        self.preprocess = preprocess

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        dish_id = self.ids[index]
        original_path = self.image_root / "images" / dish_id / "rgb.png"
        crop_path = self.image_root / "crops" / dish_id / "crop_rgb.png"
        with Image.open(original_path) as image:
            original = self.preprocess(pad_to_square(image.convert("RGB")))
        with Image.open(crop_path) as image:
            crop = self.preprocess(pad_to_square(image.convert("RGB")))
        return original, crop, self.labels[index], index


class B5Model(nn.Module):
    def __init__(self, visual, original_mean, original_std, crop_mean, crop_std):
        super().__init__()
        self.visual = visual
        self.original_branch = nn.Sequential(nn.Linear(512, 64), nn.ReLU())
        self.crop_branch = nn.Sequential(nn.Linear(512, 64), nn.ReLU())
        self.dropout = nn.Dropout(0.5)
        self.output = nn.Sequential(nn.Linear(128, 5), nn.Softplus())
        self.register_buffer("original_mean", torch.tensor(original_mean))
        self.register_buffer("original_std", torch.tensor(original_std))
        self.register_buffer("crop_mean", torch.tensor(crop_mean))
        self.register_buffer("crop_std", torch.tensor(crop_std))
        for parameter in self.visual.parameters():
            parameter.requires_grad = False

    def head_parameters(self):
        for module in (self.original_branch, self.crop_branch, self.output):
            yield from module.parameters()

    def set_clip_trainable(self, trainable):
        for _, parameter in selected_visual_parameters(self.visual):
            parameter.requires_grad = trainable

    def forward(self, original, crop, tune_clip):
        if tune_clip:
            original_feature = self.visual(original)
            crop_feature = self.visual(crop)
        else:
            with torch.no_grad():
                original_feature = self.visual(original)
                crop_feature = self.visual(crop)
        original_z = (original_feature.float() - self.original_mean) / self.original_std
        crop_z = (crop_feature.float() - self.crop_mean) / self.crop_std
        fused = torch.cat([self.original_branch(original_z), self.crop_branch(crop_z)], dim=1)
        return self.output(self.dropout(fused))


def predict(model, loader, device, target_scale, tune_clip):
    model.eval()
    predictions = np.empty((len(loader.dataset), 5), dtype=np.float32)
    with torch.inference_mode():
        for original, crop, _, indices in loader:
            original, crop = original.to(device), crop.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                output = model(original, crop, tune_clip)
            predictions[indices.numpy()] = (output.float().clamp(min=0)
                                             * target_scale).cpu().numpy()
    return predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--clip-cache", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--accumulation-steps", type=int, default=4)
    parser.add_argument("--head-lr", type=float, default=0.001)
    parser.add_argument("--clip-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.001)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("B5 requires a CUDA GPU")
    if args.warmup_epochs < 0 or args.warmup_epochs >= args.epochs:
        parser.error("warmup epochs must be >=0 and less than total epochs")

    train_ids, _, y_train = load_split(ROOT / "data/splits/train.csv")
    dev_ids, dev_groups, y_dev = load_split(ROOT / "data/splits/dev.csv")
    ids, n_train = train_ids + dev_ids, len(train_ids)
    original = load_feature(args.artifact_root / "clip_original.npz", ids, 512)
    crop = load_feature(args.artifact_root / "clip_crop.npz", ids, 512)
    original_mean, original_std = original[:n_train].mean(0), original[:n_train].std(0) + 1e-6
    crop_mean, crop_std = crop[:n_train].mean(0), crop[:n_train].std(0) + 1e-6
    target_scale_np = y_train.mean(axis=0).astype(np.float32)
    target_scale = torch.tensor(target_scale_np, device="cuda")
    y_train_scaled = y_train / target_scale_np
    settings = {key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()}
    out = args.output_root / "B5"
    out.mkdir(parents=True, exist_ok=False)
    (out / "checkpoints").mkdir()
    all_logs, per_seed = [], {}
    started_all = time.perf_counter()

    for seed in args.seeds:
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        clip_model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai", cache_dir=str(args.clip_cache))
        model = B5Model(clip_model.visual, original_mean, original_std,
                        crop_mean, crop_std).to("cuda")
        selected = selected_visual_parameters(model.visual)
        selected_names = [name for name, _ in selected]
        selected_count = sum(parameter.numel() for _, parameter in selected)
        optimizer = torch.optim.AdamW([
            {"params": list(model.head_parameters()), "lr": args.head_lr},
            {"params": [parameter for _, parameter in selected], "lr": args.clip_lr},
        ], weight_decay=args.weight_decay)
        scaler = torch.amp.GradScaler("cuda")
        train_dataset = PairedImages(train_ids, y_train_scaled, args.image_root, preprocess)
        dev_dataset = PairedImages(dev_ids, y_dev / target_scale_np, args.image_root, preprocess)
        generator = torch.Generator().manual_seed(seed)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                  generator=generator, num_workers=args.workers,
                                  pin_memory=True, persistent_workers=args.workers > 0)
        dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size * 2,
                                num_workers=args.workers, pin_memory=True,
                                persistent_workers=args.workers > 0)
        best = {"score": float("inf")}
        started = time.perf_counter()
        for epoch in range(1, args.epochs + 1):
            tune_clip = epoch > args.warmup_epochs
            model.set_clip_trainable(tune_clip)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss_sum = 0.0
            for step, (original_batch, crop_batch, labels, _) in enumerate(train_loader, 1):
                original_batch = original_batch.to("cuda", non_blocking=True)
                crop_batch = crop_batch.to("cuda", non_blocking=True)
                labels = labels.to("cuda", non_blocking=True)
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    output = model(original_batch, crop_batch, tune_clip)
                    loss = nn.functional.l1_loss(output, labels)
                    scaled_loss = loss / args.accumulation_steps
                scaler.scale(scaled_loss).backward()
                if step % args.accumulation_steps == 0 or step == len(train_loader):
                    scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
                loss_sum += loss.item() * len(labels)
            dev_prediction = predict(model, dev_loader, "cuda", target_scale, tune_clip)
            score = normalized_nutrient_mae(y_dev, dev_prediction, target_scale_np)
            row = {"seed": seed, "epoch": epoch,
                   "phase": "partial_clip" if tune_clip else "head_warmup",
                   "train_loss": round(loss_sum / n_train, 6),
                   "dev_normalized_nutrient_mae": round(score, 6)}
            all_logs.append(row)
            print(f"B5 seed {seed} epoch {epoch}/{args.epochs}: {score:.4f}", flush=True)
            if score < best["score"]:
                best = {"score": score, "epoch": epoch, "prediction": dev_prediction.copy(),
                        "head_state": {name: value.detach().cpu().clone()
                                       for name, value in model.state_dict().items()
                                       if not name.startswith("visual.")},
                        "visual_state": {name: parameter.detach().cpu().clone()
                                         for name, parameter in selected}}
        torch.save({"condition": "B5", "selected_visual_state": best["visual_state"],
                    "non_visual_state": best["head_state"], "target_scale": target_scale.cpu(),
                    "targets": TARGETS, "settings": settings, "seed": seed,
                    "best_epoch": best["epoch"]}, out / "checkpoints" / f"seed{seed}.pt")
        write_predictions(out / f"dev_predictions_seed{seed}.csv", dev_ids, dev_groups,
                          y_dev, best["prediction"])
        per_seed[str(seed)] = {"best_epoch": best["epoch"],
                               "seconds": time.perf_counter() - started,
                               "dev": evaluate(y_dev, best["prediction"], dev_groups, target_scale_np)}
        del model, clip_model, optimizer, scaler, train_loader, dev_loader, selected
        torch.cuda.empty_cache()

    with (out / "training_log.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_logs[0]))
        writer.writeheader(); writer.writerows(all_logs)
    scores = np.array([entry["dev"]["normalized_nutrient_mae"] for entry in per_seed.values()])
    metrics = {"status": "completed", "condition": "B5",
               "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
               "settings": settings, "n_train": n_train, "n_dev": len(dev_ids),
               "target_order": TARGETS,
               "parameter_counts": {"head": 66309,
                                    "selected_visual": selected_count},
               "selected_visual_names": selected_names,
               "inputs_sha256": {"original": sha256(args.artifact_root / "clip_original.npz"),
                                 "crop": sha256(args.artifact_root / "clip_crop.npz")},
               "environment": {"python": platform.python_version(), "torch": torch.__version__,
                               "numpy": np.__version__, "platform": platform.platform()},
               "per_seed": per_seed,
               "summary": {"normalized_nutrient_mae_mean": float(scores.mean()),
                           "normalized_nutrient_mae_std": float(scores.std(ddof=1)),
                           "seconds": time.perf_counter() - started_all}}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")


if __name__ == "__main__":
    main()
