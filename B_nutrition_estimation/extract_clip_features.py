"""Extract frozen CLIP ViT-B/32 image features (512 numbers per photo).

Usage:
    python B_nutrition_estimation/extract_clip_features.py                       # original overhead photos
    python B_nutrition_estimation/extract_clip_features.py --image-dir <folder> --name masked   # e.g. food-only photos

Reads <image-dir>/<dish_id>/rgb.png for every Cafe 1 dish with an overhead photo and writes
<data-root>/features/clip_vitb32_<name>.npz with:
    dish_ids   (n,)      dish IDs in the same order as the rows of `features`
    features   (n, 512)  CLIP image embeddings, float32, not normalized
    settings             JSON string with the model, weights and preprocessing

Preprocessing: pad each 640 x 480 photo to a square with CLIP's mean colour (nothing is cropped,
so portion size and scale are kept), then CLIP's own resize to 224 x 224 and normalization.
CLIP is frozen: no weights change (see d2l.ai 14.2 on using pretrained models).
"""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import open_clip
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL, PRETRAINED = "ViT-B-32", "openai"
CLIP_MEAN_RGB = (122, 116, 104)  # CLIP's normalization mean (0.481, 0.458, 0.408) x 255


def pad_to_square(image):
    w, h = image.size
    side = max(w, h)
    square = Image.new("RGB", (side, side), CLIP_MEAN_RGB)
    square.paste(image, ((side - w) // 2, (side - h) // 2))
    return square


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", default=str(REPO_ROOT.parent / "data" / "nutrition5k"))
    p.add_argument("--image-dir", default=None, help="default: <data-root>/imagery/realsense_overhead")
    p.add_argument("--name", default="original", help="label used in the output file name")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--limit", type=int, default=None, help="only the first N dishes (for testing)")
    args = p.parse_args()

    root = Path(args.data_root)
    image_dir = Path(args.image_dir) if args.image_dir else root / "imagery" / "realsense_overhead"
    with open(root / "metadata" / "dish_metadata_cafe1.csv", encoding="utf-8") as f:
        cafe1 = {row[0] for row in csv.reader(f)}
    with open(root / "overhead_dirs.txt", encoding="utf-8") as f:
        overhead = {line.strip() for line in f if line.strip()}
    dish_ids = sorted(cafe1 & overhead)[:args.limit]

    torch.manual_seed(0)
    model, _, preprocess = open_clip.create_model_and_transforms(MODEL, pretrained=PRETRAINED)
    model.eval()

    feats, start = [], time.time()
    with torch.no_grad():
        for i in range(0, len(dish_ids), args.batch_size):
            batch = dish_ids[i:i + args.batch_size]
            images = []
            for d in batch:
                with Image.open(image_dir / d / "rgb.png") as im:
                    images.append(preprocess(pad_to_square(im.convert("RGB"))))
            feats.append(model.encode_image(torch.stack(images)).float().numpy())
            done = min(i + args.batch_size, len(dish_ids))
            if done % (args.batch_size * 10) == 0 or done == len(dish_ids):
                print(f"{done}/{len(dish_ids)} photos ({time.time() - start:.0f} s)", flush=True)
    features = np.concatenate(feats)

    settings = {"model": MODEL, "pretrained": PRETRAINED, "open_clip": open_clip.__version__,
                "torch": torch.__version__, "image_dir": str(image_dir),
                "preprocessing": "pad to square with CLIP mean colour, then CLIP resize 224 + normalize",
                "n_photos": len(dish_ids), "seconds": round(time.time() - start, 1)}
    out = root / "features" / f"clip_vitb32_{args.name}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, dish_ids=np.array(dish_ids), features=features, settings=json.dumps(settings))
    print(f"Saved {features.shape} to {out}")


if __name__ == "__main__":
    main()
