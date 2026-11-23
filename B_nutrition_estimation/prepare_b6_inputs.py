"""Restore exact handoff train/dev RGB/crops using public RGBs and acquired metadata.

No Google Drive access, credentials, SAM inference, human masks or test inputs
are used. Every restored file must match the handoff SHA-256 byte for byte.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys
import time
import urllib.request
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evaluate import load_split
from B_nutrition_estimation.verify_b6_inputs import EXPECTED_SPLITS, sha

METADATA_SHA = {
    'crop_metadata.json': '460f2960dbfc248790860098c29977bde7bcf1d64e9cc486e4d962844bb0aab1',
    'downloads.json': 'be582d554de639e493c29ab0faff5b3a7dcdee1ce6b49ead1c2f57857fd13787'}
PUBLIC_RGB = ('https://storage.googleapis.com/nutrition5k_dataset/'
              'nutrition5k_dataset/imagery/realsense_overhead/{}/rgb.png')


def restore_one(dish, row, root, cache):
    target = root / 'images' / dish / 'rgb.png'
    cropped = root / 'crops' / dish / 'crop_rgb.png'
    expected = row['image_sha256']
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if sha(target) != expected:
            raise ValueError('Existing RGB differs from handoff: ' + dish)
        origin = 'already_verified'
    else:
        cached = cache / dish / 'rgb.png' if cache else None
        if cached is not None and cached.is_file() and sha(cached) == expected:
            shutil.copyfile(cached, target)
            origin = 'copied_existing_public_rgb'
        else:
            raw = None
            for attempt in range(3):
                try:
                    request = urllib.request.Request(PUBLIC_RGB.format(dish), headers={'User-Agent': 'ECE539-B6'})
                    with urllib.request.urlopen(request, timeout=45) as response:
                        raw = response.read()
                    if hashlib.sha256(raw).hexdigest() != expected:
                        raise ValueError('Public RGB differs from handoff: ' + dish)
                    break
                except Exception:
                    if attempt == 2:
                        raise
                    time.sleep(attempt + 1)
            target.write_bytes(raw)
            origin = 'downloaded_official_public_rgb'
    cropped.parent.mkdir(parents=True, exist_ok=True)
    if not cropped.exists():
        with Image.open(target) as image:
            crop = image.convert('RGB').crop(tuple(row['box_xyxy']))
            encoded = io.BytesIO()
            crop.save(encoded, format='PNG')
        raw_crop = encoded.getvalue()
        if hashlib.sha256(raw_crop).hexdigest() != row['crop_sha256']:
            raise ValueError('Restored PNG encoding differs from handoff; use original crop file: ' + dish)
        cropped.write_bytes(raw_crop)
    if sha(cropped) != row['crop_sha256']:
        raise ValueError('Existing crop differs from handoff: ' + dish)
    return {'dish_id': dish, 'status': 'verified', 'rgb_source': origin}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root', type=Path, required=True)
    parser.add_argument('--existing-rgb-cache', type=Path)
    parser.add_argument('--workers', type=int, default=16)
    args = parser.parse_args()
    root = args.input_root
    for name, digest in METADATA_SHA.items():
        if sha(root / name) != digest:
            raise ValueError('Acquired handoff metadata changed: ' + name)
    ids = []
    for split, digest in EXPECTED_SPLITS.items():
        path = ROOT / 'data/splits' / (split + '.csv')
        if sha(path) != digest:
            raise ValueError('Canonical split changed: ' + split)
        ids.extend(load_split(path)[0])
    rows = json.loads((root / 'crop_metadata.json').read_text())['records']
    by_id = {row['dish_id']: row for row in rows}
    if len(ids) != 2673 or len(rows) != 2673 or set(ids) != set(by_id):
        raise ValueError('Invalid train/dev handoff membership')
    records = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = {pool.submit(restore_one, dish, by_id[dish], root, args.existing_rgb_cache): dish for dish in ids}
        for completed, future in enumerate(as_completed(jobs), 1):
            dish = jobs[future]
            try:
                records.append(future.result())
            except Exception as error:
                records.append({'dish_id': dish, 'status': 'failed', 'error': str(error)})
            if completed % 100 == 0 or completed == len(ids):
                print(f'{completed}/{len(ids)} pairs processed; failures={sum(r["status"]=="failed" for r in records)}', flush=True)
                (root / 'restoration_progress.json').write_text(json.dumps(records, indent=2) + '\n')
    failed = [row for row in records if row['status'] == 'failed']
    if failed:
        raise SystemExit(f'{len(failed)} input pairs failed verification; measured training is blocked')


if __name__ == '__main__':
    main()
