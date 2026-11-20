"""Select new development groups before viewing any images; optionally download RGB."""
import argparse
import csv
import hashlib
import io
import json
import random
import subprocess
import urllib.request
from pathlib import Path

BASE = Path(__file__).resolve().parent
REPO = BASE.parents[1]
SOURCE_COMMIT = 'dcde4c8330b5ad663845c25f611f096b08a462b2'
SEED = 53920261119
FROZEN_IMPLEMENTATION = 'aa44984'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download', action='store_true')
    args = parser.parse_args()
    for filename in ['appearance_selector.py', 'run_sam_debug.py']:
        frozen = subprocess.check_output(
            ['git', 'show', FROZEN_IMPLEMENTATION + ':A_food_segmentation/debug_validation/' + filename],
            cwd=REPO)
        if sha(frozen) != sha((BASE / filename).read_bytes()):
            raise ValueError('Implementation differs from frozen commit: ' + filename)
    source = subprocess.check_output(
        ['git', 'show', SOURCE_COMMIT + ':data/splits/dev.csv'], cwd=REPO)
    rows = list(csv.DictReader(io.StringIO(source.decode())))
    with (BASE / 'manifests/dev9_selection.csv').open(newline='') as handle:
        old_ids = {row['dish_id'] for row in csv.DictReader(handle)}
    old_ids.update(['dish_1556572657', 'dish_1556573514', 'dish_1556575014'])
    all_split_rows = []
    for split in ['train', 'dev', 'test']:
        data = subprocess.check_output(
            ['git', 'show', SOURCE_COMMIT + ':data/splits/' + split + '.csv'], cwd=REPO)
        all_split_rows.extend(csv.DictReader(io.StringIO(data.decode())))
    excluded_groups = {row['plate_group'] for row in all_split_rows if row['dish_id'] in old_ids}
    groups = {}
    for row in rows:
        if row['dish_id'] not in old_ids and row['plate_group'] not in excluded_groups:
            groups.setdefault(row['plate_group'], []).append(row)
    rng = random.Random(SEED)
    # Choose one representative without viewing any RGB image, then stratify
    # by the representative's ingredient count. Every group occurs once.
    representatives = [rng.choice(sorted(groups[group], key=lambda row: row['dish_id']))
                       for group in sorted(groups)]
    strata = {'single': [], 'medium': [], 'complex': []}
    for row in representatives:
        count = int(row['n_ingredients'])
        group = 'single' if count == 1 else 'medium' if count <= 4 else 'complex'
        strata[group].append(row)
    selected = []
    for group, pool in strata.items():
        for row in rng.sample(pool, 4):
            selected.append({**row, 'complexity_group': group})
    assert len({row['plate_group'] for row in selected}) == 12
    assert not {row['plate_group'] for row in selected} & excluded_groups
    path = BASE / 'manifests/fresh_dev12_selection.csv'
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selected[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(selected)
    provenance = {
        'implementation_commit': FROZEN_IMPLEMENTATION,
        'source_commit': SOURCE_COMMIT, 'source_path': 'data/splits/dev.csv',
        'source_sha256': sha(source), 'seed': SEED,
        'excluded_prior_dish_ids': sorted(old_ids),
        'excluded_plate_groups': sorted(excluded_groups),
        'selected_count': len(selected), 'unique_selected_groups': 12,
        'strata': {'single': '1 ingredient', 'medium': '2-4 ingredients', 'complex': '5+ ingredients'},
        'samples_per_stratum': 4,
        'selector_sha256': sha((BASE / 'appearance_selector.py').read_bytes()),
        'runner_sha256': sha((BASE / 'run_sam_debug.py').read_bytes()),
        'selection_sha256': sha(path.read_bytes()),
        'note': 'Groups are inferred by the pinned source pipeline; group disjointness is not proof of physical identity.',
    }
    (BASE / 'manifests/fresh_dev12_provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(json.dumps(provenance, indent=2), flush=True)
    if args.download:
        from PIL import Image
        downloads = []
        for row in selected:
            url = 'https://storage.googleapis.com/nutrition5k_dataset/nutrition5k_dataset/' + row['image_path']
            target = BASE / 'local_data/nutrition5k' / row['image_path']
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                with urllib.request.urlopen(url, timeout=60) as response:
                    data = response.read()
                with Image.open(io.BytesIO(data)) as image:
                    image.verify()
                target.write_bytes(data)
            data = target.read_bytes()
            with Image.open(target) as image:
                size = list(image.size)
            downloads.append({'dish_id': row['dish_id'], 'url': url, 'bytes': len(data),
                              'sha256': sha(data), 'image_size': size})
            print(row['dish_id'], len(data), 'bytes', flush=True)
        (BASE / 'manifests/fresh_dev12_downloads.json').write_text(json.dumps(downloads, indent=2) + '\n')


if __name__ == '__main__':
    main()
