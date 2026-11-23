"""Prepare matched original/masked CLIP features for B0-B3.

Only canonical train and development manifests are read. The final-test manifest
is never opened. Large outputs belong under ignored ``artifacts/``.
"""
from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import io
import json
from pathlib import Path
import sys
import time
import urllib.request

import cv2
import numpy as np
import open_clip
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "A_food_segmentation"))
from sam_runtime import (file_sha256, infer_rgb_with_model, load_decoder,
                         load_sam, resolve_device)
from B_nutrition_estimation.extract_clip_features import pad_to_square

PHOTO_URL = ("https://storage.googleapis.com/nutrition5k_dataset/"
             "nutrition5k_dataset/imagery/realsense_overhead/{}/rgb.png")
GEOMETRY_NAMES = ["foreground_fraction", "bbox_width_fraction",
                  "bbox_height_fraction", "bbox_area_fraction",
                  "bbox_fill_fraction", "centroid_x_fraction",
                  "centroid_y_fraction", "log1p_component_count"]


def load_rows(limit=None):
    rows = []
    for split in ("train", "dev"):
        with (ROOT / "data/splits" / f"{split}.csv").open(encoding="utf-8") as stream:
            selected = list(csv.DictReader(stream))
        if limit:
            selected = selected[:limit]
        rows.extend(selected)
    ids = [row["dish_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate dish IDs in canonical train/dev manifests")
    return rows


def valid_rgb(path):
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except Exception:
        return False


def download_one(row, image_root):
    target = image_root / row["dish_id"] / "rgb.png"
    if target.exists() and valid_rgb(target):
        return {"dish_id": row["dish_id"], "status": "cached",
                "sha256": file_sha256(target), "bytes": target.stat().st_size}
    request = urllib.request.Request(PHOTO_URL.format(row["dish_id"]),
                                     headers={"User-Agent": "ECE539-B-input-comparison"})
    error = None
    for attempt in range(3):
        try:
            data = urllib.request.urlopen(request, timeout=45).read()
            with Image.open(io.BytesIO(data)) as image:
                image.verify()
            target.parent.mkdir(parents=True, exist_ok=True)
            partial = target.with_suffix(".part")
            partial.write_bytes(data)
            partial.replace(target)
            return {"dish_id": row["dish_id"], "status": "downloaded",
                    "sha256": file_sha256(target), "bytes": len(data)}
        except Exception as exc:
            error = str(exc)
            time.sleep(attempt + 1)
    return {"dish_id": row["dish_id"], "status": "failed", "error": error}


def download(rows, root, workers):
    image_root = root / "images"
    records = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(download_one, row, image_root) for row in rows]
        for index, future in enumerate(as_completed(futures), 1):
            records.append(future.result())
            if index % 100 == 0 or index == len(futures):
                failures = sum(record["status"] == "failed" for record in records)
                print(f"download {index}/{len(futures)} failures={failures}", flush=True)
    records.sort(key=lambda record: record["dish_id"])
    (root / "downloads.json").write_text(json.dumps(records, indent=2) + "\n")
    failures = [record for record in records if record["status"] == "failed"]
    if failures:
        raise RuntimeError(f"{len(failures)} image downloads failed")


def geometry(mask):
    height, width = mask.shape
    ys, xs = np.nonzero(mask)
    if not len(xs) or len(xs) == mask.size:
        raise ValueError("Degenerate empty/full mask")
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    box_width, box_height = x1 - x0 + 1, y1 - y0 + 1
    box_area = box_width * box_height
    components = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)[0] - 1
    return np.array([
        mask.mean(), box_width / width, box_height / height,
        box_area / mask.size, mask.sum() / box_area,
        xs.mean() / width, ys.mean() / height, np.log1p(components),
    ], dtype=np.float32)


def segment(rows, root, sam_checkpoint, decoder_checkpoint, requested_device):
    device = resolve_device(requested_device)
    model = load_sam(sam_checkpoint, device)
    selected_epoch = load_decoder(model, decoder_checkpoint)
    records, geometries = [], []
    for index, row in enumerate(rows, 1):
        dish_id = row["dish_id"]
        image_path = root / "images" / dish_id / "rgb.png"
        output = root / "masked" / dish_id
        mask_path, masked_path = output / "mask.png", output / "masked_rgb.png"
        if mask_path.exists() and masked_path.exists():
            mask = np.asarray(Image.open(mask_path)) > 0
            rgb = np.asarray(Image.open(image_path).convert("RGB"))
            masked = np.asarray(Image.open(masked_path).convert("RGB"))
            status = "cached"
            seconds = None
        else:
            rgb = np.asarray(Image.open(image_path).convert("RGB"))
            started = time.perf_counter()
            mask, metadata = infer_rgb_with_model(rgb, model, device,
                                                  selected_epoch=selected_epoch)
            seconds = time.perf_counter() - started
            output.mkdir(parents=True, exist_ok=False)
            Image.fromarray(mask.astype(np.uint8) * 255).save(mask_path)
            masked = np.full_like(rgb, 128)
            masked[mask] = rgb[mask]
            Image.fromarray(masked).save(masked_path)
            status = "generated"
        if mask.shape != rgb.shape[:2] or masked.shape != rgb.shape:
            raise ValueError(f"Shape mismatch for {dish_id}")
        if not np.array_equal(masked[mask], rgb[mask]) or not np.all(masked[~mask] == 128):
            raise ValueError(f"Pixel contract failed for {dish_id}")
        vector = geometry(mask)
        geometries.append(vector)
        records.append({"dish_id": dish_id, "split": row["split"], "status": status,
                        "seconds": seconds, "foreground_fraction": float(vector[0]),
                        "image_sha256": file_sha256(image_path),
                        "mask_sha256": file_sha256(mask_path),
                        "masked_sha256": file_sha256(masked_path)})
        if index % 25 == 0 or index == len(rows):
            print(f"segment {index}/{len(rows)}", flush=True)
    (root / "segmentation.json").write_text(json.dumps(records, indent=2) + "\n")
    np.savez(root / "geometry.npz", dish_ids=np.array([r["dish_id"] for r in rows]),
             features=np.stack(geometries), names=np.array(GEOMETRY_NAMES))


def extract_features(rows, root, condition, batch_size, clip_cache, requested_device):
    torch.manual_seed(0)
    device = resolve_device(requested_device)
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai", cache_dir=str(clip_cache))
    model.to(device).eval()
    all_features, started = [], time.perf_counter()
    with torch.inference_mode():
        for offset in range(0, len(rows), batch_size):
            batch = rows[offset:offset + batch_size]
            images = []
            for row in batch:
                if condition == "original":
                    path = root / "images" / row["dish_id"] / "rgb.png"
                else:
                    path = root / "masked" / row["dish_id"] / "masked_rgb.png"
                with Image.open(path) as image:
                    images.append(preprocess(pad_to_square(image.convert("RGB"))))
            batch_tensor = torch.stack(images).to(device)
            all_features.append(model.encode_image(batch_tensor).float().cpu().numpy())
            done = min(offset + batch_size, len(rows))
            if done % (batch_size * 5) == 0 or done == len(rows):
                print(f"{condition} features {done}/{len(rows)}", flush=True)
    features = np.concatenate(all_features)
    if features.shape != (len(rows), 512) or not np.isfinite(features).all():
        raise ValueError(f"Invalid {condition} feature matrix {features.shape}")
    settings = {"condition": condition, "model": "ViT-B-32", "weights": "openai",
                "device": str(device),
                "n_images": len(rows), "seconds": time.perf_counter() - started,
                "preprocessing": "pad square with CLIP mean, resize 224, normalize"}
    np.savez(root / f"clip_{condition}.npz",
             dish_ids=np.array([row["dish_id"] for row in rows]),
             features=features, settings=json.dumps(settings))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["download", "segment", "features", "all"])
    parser.add_argument("--root", type=Path, default=ROOT / "artifacts/b-input-comparison")
    parser.add_argument("--sam-checkpoint", type=Path)
    parser.add_argument("--decoder-checkpoint", type=Path,
                        default=ROOT / "A_food_segmentation/model_weights/food_decoder_no_prompt.pt")
    parser.add_argument("--clip-cache", type=Path,
                        default=ROOT / "artifacts/ab-integration/checkpoints")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit-per-split", type=int)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.limit_per_split)
    (args.root / "selection.json").write_text(json.dumps(rows, indent=2) + "\n")
    if args.action in ("download", "all"):
        download(rows, args.root, args.workers)
    if args.action in ("segment", "all"):
        if not args.sam_checkpoint or not args.sam_checkpoint.is_file():
            parser.error("--sam-checkpoint is required for segmentation")
        segment(rows, args.root, args.sam_checkpoint, args.decoder_checkpoint, args.device)
    if args.action in ("features", "all"):
        extract_features(rows, args.root, "original", args.batch_size, args.clip_cache, args.device)
        extract_features(rows, args.root, "masked", args.batch_size, args.clip_cache, args.device)


if __name__ == "__main__":
    main()
