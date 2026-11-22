"""Download Nutrition5k overhead colour photos (rgb.png) for Cafe 1 dishes only.

Usage:
    python src/download_photos.py                 # all Cafe 1 dishes with an overhead photo (3,262)
    python src/download_photos.py --limit 20      # quick test

Dishes come from dish_metadata_cafe1.csv and must appear in overhead_dirs.txt.
Photos are saved to <data-root>/imagery/realsense_overhead/<dish_id>/rgb.png, the dataset's own
layout. Existing valid photos are skipped, so the script can be stopped and rerun. Each photo is
checked to open as an image; failures are listed in <data-root>/imagery/download_failures.txt.
Depth images and videos are not downloaded.
"""

import argparse
import csv
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
PHOTO_URL = ("https://storage.googleapis.com/nutrition5k_dataset/nutrition5k_dataset/"
             "imagery/realsense_overhead/{}/rgb.png")


def is_valid_image(path):
    try:
        with Image.open(path) as im:
            im.verify()
        return True
    except Exception:
        return False


def download(dish_id, out_dir, retries=3):
    out = out_dir / dish_id / "rgb.png"
    if out.exists() and is_valid_image(out):
        return dish_id, "skipped"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".part")
    for attempt in range(1, retries + 1):
        try:
            urllib.request.urlretrieve(PHOTO_URL.format(dish_id), tmp)
            if is_valid_image(tmp):
                tmp.replace(out)
                return dish_id, "downloaded"
            error = "not a valid image"
        except Exception as e:  # network error or missing file
            error = str(e)
        time.sleep(2 * attempt)
    tmp.unlink(missing_ok=True)
    return dish_id, f"failed: {error}"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", default=str(REPO_ROOT.parent / "data" / "nutrition5k"))
    p.add_argument("--limit", type=int, default=None, help="download only the first N dishes (for testing)")
    p.add_argument("--workers", type=int, default=8, help="parallel downloads")
    args = p.parse_args()
    root = Path(args.data_root)

    with open(root / "metadata" / "dish_metadata_cafe1.csv", encoding="utf-8") as f:
        cafe1 = {row[0] for row in csv.reader(f)}
    with open(root / "overhead_dirs.txt", encoding="utf-8") as f:
        overhead = {line.strip() for line in f if line.strip()}
    dishes = sorted(cafe1 & overhead)
    if args.limit:
        dishes = dishes[:args.limit]

    out_dir = root / "imagery" / "realsense_overhead"
    counts, failures = {"downloaded": 0, "skipped": 0, "failed": 0}, []
    start = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(download, d, out_dir) for d in dishes]
        for i, fut in enumerate(as_completed(futures), 1):
            dish_id, status = fut.result()
            if status.startswith("failed"):
                counts["failed"] += 1
                failures.append(f"{dish_id}\t{status}")
            else:
                counts[status] += 1
            if i % 200 == 0 or i == len(dishes):
                print(f"{i}/{len(dishes)} done ({time.time() - start:.0f} s): {counts}", flush=True)

    fail_file = root / "imagery" / "download_failures.txt"
    fail_file.write_text("\n".join(failures) + ("\n" if failures else ""), encoding="utf-8")
    print(f"Finished: {counts}. Failures listed in {fail_file}")


if __name__ == "__main__":
    main()
