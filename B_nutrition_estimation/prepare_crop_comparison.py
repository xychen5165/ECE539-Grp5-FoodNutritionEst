"""Prepare natural tight-crop CLIP features for experiment 006/B4."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import open_clip
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from A_food_segmentation.sam_runtime import file_sha256, resolve_device
from B_nutrition_estimation.extract_clip_features import pad_to_square
from B_nutrition_estimation.prepare_input_comparison import load_rows


def crop_from_mask(rgb, mask, margin_fraction=0.10):
    """Return a clamped food bounding-box crop with proportional context."""
    if rgb.shape[:2] != mask.shape:
        raise ValueError("RGB/mask shape mismatch")
    ys, xs = np.nonzero(mask)
    if not len(xs) or len(xs) == mask.size:
        raise ValueError("Degenerate empty/full mask")
    height, width = mask.shape
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    margin_x = max(1, int(np.ceil((x1 - x0) * margin_fraction)))
    margin_y = max(1, int(np.ceil((y1 - y0) * margin_fraction)))
    box = [max(0, x0 - margin_x), max(0, y0 - margin_y),
           min(width, x1 + margin_x), min(height, y1 + margin_y)]
    return rgb[box[1]:box[3], box[0]:box[2]], box


def generate_crops(rows, root, margin_fraction):
    records = []
    for index, row in enumerate(rows, 1):
        dish_id = row["dish_id"]
        image_path = root / "images" / dish_id / "rgb.png"
        mask_path = root / "masked" / dish_id / "mask.png"
        output = root / "crops" / dish_id / "crop_rgb.png"
        if not image_path.is_file() or not mask_path.is_file():
            raise FileNotFoundError(f"Missing 005 image/mask for {dish_id}")
        rgb = np.asarray(Image.open(image_path).convert("RGB"))
        mask = np.asarray(Image.open(mask_path)) > 0
        crop, box = crop_from_mask(rgb, mask, margin_fraction)
        if output.exists():
            saved = np.asarray(Image.open(output).convert("RGB"))
            if not np.array_equal(saved, crop):
                raise ValueError(f"Existing crop differs for {dish_id}")
            status = "cached"
        else:
            output.parent.mkdir(parents=True, exist_ok=False)
            Image.fromarray(crop).save(output)
            status = "generated"
        records.append({"dish_id": dish_id, "split": row["split"], "status": status,
                        "box_xyxy": [int(v) for v in box],
                        "crop_width": int(crop.shape[1]), "crop_height": int(crop.shape[0]),
                        "image_sha256": file_sha256(image_path),
                        "mask_sha256": file_sha256(mask_path),
                        "crop_sha256": file_sha256(output)})
        if index % 100 == 0 or index == len(rows):
            print(f"crop {index}/{len(rows)}", flush=True)
    (root / "crop_metadata.json").write_text(json.dumps({
        "margin_fraction": margin_fraction,
        "definition": "predicted-mask bbox plus 10% bbox-width/height context, clamped",
        "records": records,
    }, indent=2) + "\n")


def extract_features(rows, root, batch_size, clip_cache, requested_device):
    torch.manual_seed(0)
    device = resolve_device(requested_device)
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai", cache_dir=str(clip_cache))
    model.to(device).eval()
    features, started = [], time.perf_counter()
    with torch.inference_mode():
        for offset in range(0, len(rows), batch_size):
            batch = rows[offset:offset + batch_size]
            images = []
            for row in batch:
                path = root / "crops" / row["dish_id"] / "crop_rgb.png"
                with Image.open(path) as image:
                    images.append(preprocess(pad_to_square(image.convert("RGB"))))
            features.append(model.encode_image(torch.stack(images).to(device)).float().cpu().numpy())
            done = min(offset + batch_size, len(rows))
            if done % (batch_size * 5) == 0 or done == len(rows):
                print(f"crop features {done}/{len(rows)}", flush=True)
    values = np.concatenate(features)
    if values.shape != (len(rows), 512) or not np.isfinite(values).all():
        raise ValueError(f"Invalid crop feature matrix {values.shape}")
    settings = {"condition": "crop", "model": "ViT-B-32", "weights": "openai",
                "device": str(device), "n_images": len(rows),
                "seconds": time.perf_counter() - started,
                "preprocessing": "predicted-mask bbox + 10% context; pad square; CLIP"}
    np.savez(root / "clip_crop.npz", dish_ids=np.array([r["dish_id"] for r in rows]),
             features=values, settings=json.dumps(settings))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["crop", "features", "all"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--margin-fraction", type=float, default=0.10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--clip-cache", type=Path)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    args = parser.parse_args()
    if args.margin_fraction < 0:
        parser.error("--margin-fraction must be nonnegative")
    rows = load_rows()
    if args.action in ("crop", "all"):
        generate_crops(rows, args.root, args.margin_fraction)
    if args.action in ("features", "all"):
        extract_features(rows, args.root, args.batch_size,
                         args.clip_cache or args.root / "checkpoints", args.device)


if __name__ == "__main__":
    main()
