"""Verify every train/dev input against the supplied handoff before B6 training."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evaluate import load_split

EXPECTED_SPLITS = {
    'train': '0fe38b0f6ff50e7031dea091da19675a76adfba533e44e2bff3276cd9e5b9fbd',
    'dev': 'c6462b3444676502915de5606891b4fbe162e248e186c3df7d7bd4d7ee179a31'}
EXPECTED_FEATURES = {
    'clip_original.npz': '307f2e7cdad43bb635d88e1e0e49b237631e76d0ff32d16fdf8a1098e2aabaa2',
    'clip_crop.npz': '3e7d4a418445c3914f29fd2eb4b13115fc124b94f8c6ebebb26460e748ea55fd'}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    ids = []
    for name, expected in EXPECTED_SPLITS.items():
        path = ROOT / 'data/splits' / (name + '.csv')
        if sha(path) != expected:
            raise ValueError('Canonical split changed: ' + name)
        ids.extend(load_split(path)[0])
    if len(ids) != 2673 or len(set(ids)) != 2673:
        raise ValueError('Unexpected train/dev membership')
    root = args.input_root
    crop_info = json.loads((root / 'crop_metadata.json').read_text())
    downloads = json.loads((root / 'downloads.json').read_text())
    crop_rows = crop_info['records']
    by_id = {r['dish_id']: r for r in crop_rows}
    by_download = {r['dish_id']: r for r in downloads}
    if crop_info['margin_fraction'] != 0.10:
        raise ValueError('Natural-crop margin changed')
    if len(crop_rows) != 2673 or len(downloads) != 2673:
        raise ValueError('Incorrect handoff manifest count')
    if set(by_id) != set(ids) or set(by_download) != set(ids):
        raise ValueError('Handoff membership differs from canonical train/dev')
    for name, expected in EXPECTED_FEATURES.items():
        if sha(root / name) != expected:
            raise ValueError('Handoff feature fingerprint changed: ' + name)
        with np.load(root / name, allow_pickle=False) as feature:
            if feature['dish_ids'].astype(str).tolist() != ids:
                raise ValueError('Feature ordering differs from manifest')
            if feature['features'].shape != (2673, 512) or not np.isfinite(feature['features']).all():
                raise ValueError('Invalid feature matrix')
    for index, dish in enumerate(ids, 1):
        row = by_id[dish]
        original = root / 'images' / dish / 'rgb.png'
        cropped = root / 'crops' / dish / 'crop_rgb.png'
        if sha(original) != row['image_sha256'] or sha(original) != by_download[dish]['sha256']:
            raise ValueError('Original checksum mismatch: ' + dish)
        if sha(cropped) != row['crop_sha256']:
            raise ValueError('Crop checksum mismatch: ' + dish)
        rgb = np.asarray(Image.open(original).convert('RGB'))
        crop = np.asarray(Image.open(cropped).convert('RGB'))
        x0, y0, x1, y1 = row['box_xyxy']
        if not np.array_equal(crop, rgb[y0:y1, x0:x1]):
            raise ValueError('Crop pixels differ from original: ' + dish)
        if index % 500 == 0:
            print(f'Verified {index}/{len(ids)} pairs', flush=True)
    result = {'status': 'passed', 'verified_original_images': 2673,
              'verified_natural_crops': 2673, 'all_image_hashes_match_handoff': True,
              'all_crop_pixels_match_source_rectangle': True,
              'feature_hashes': EXPECTED_FEATURES, 'canonical_split_hashes': EXPECTED_SPLITS,
              'final_test_images_or_labels_accessed': False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
