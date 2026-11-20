"""Verify the frozen run and combine machine outcomes with explicit visual review."""
import csv
import hashlib
import importlib.metadata
import json
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parent


def main():
    manifests = BASE / 'manifests'
    provenance = json.loads((manifests / 'fresh_dev12_provenance.json').read_text())
    for filename, key in [('appearance_selector.py', 'selector_sha256'),
                          ('run_sam_debug.py', 'runner_sha256')]:
        digest = hashlib.sha256((BASE / filename).read_bytes()).hexdigest()
        if digest != provenance[key]:
            raise ValueError('Frozen implementation changed: ' + filename)
    run = json.loads((BASE / 'outputs/fresh_dev12_plate_appearance_v1/summary.json').read_text())
    with (manifests / 'fresh_dev12_selection.csv').open(newline='') as handle:
        selection = {r['dish_id']: r for r in csv.DictReader(handle)}
    with (manifests / 'fresh_dev12_review.csv').open(newline='') as handle:
        review = {r['dish_id']: r for r in csv.DictReader(handle)}
    outcomes = {r['dish_id']: r for r in run['images']}
    if not (len(selection) == len(review) == len(outcomes) == 12
            and selection.keys() == review.keys() == outcomes.keys()):
        raise ValueError('Selection, execution and review must cover the same 12 images')
    for dish_id, row in review.items():
        if row['program_status'] != outcomes[dish_id]['status']:
            raise ValueError('Program status mismatch: ' + dish_id)
        if row['manual_rating'] not in ['pass', 'partial', 'fail']:
            raise ValueError('Invalid review rating: ' + dish_id)
        if row['program_status'] == 'failed' and row['manual_rating'] != 'fail':
            raise ValueError('Failed execution must count as failed review')
    report = {
        'implementation_commit': 'aa44984', 'source_commit': provenance['source_commit'],
        'selector_sha256': provenance['selector_sha256'],
        'runner_sha256': provenance['runner_sha256'],
        'selection_sha256': provenance['selection_sha256'],
        'sample_count': 12, 'unique_plate_groups': len({r['plate_group'] for r in selection.values()}),
        'sam_settings': {'device': run['device'], 'points_per_side': run['points_per_side'],
                         'pred_iou_thresh': .86, 'stability_score_thresh': .92,
                         'crop_n_layers': 0, 'min_mask_region_area': 100},
        'program_counts': dict(Counter(r['status'] for r in run['images'])),
        'manual_counts': dict(Counter(r['manual_rating'] for r in review.values())),
        'by_complexity': {
            group: dict(Counter(review[d]['manual_rating'] for d, r in selection.items()
                                if r['complexity_group'] == group))
            for group in ['single', 'medium', 'complex']},
        'environment': {name: importlib.metadata.version(name)
                        for name in ['torch', 'torchvision', 'numpy', 'Pillow', 'opencv-python-headless']},
        'review_method': 'Assistant visual inspection of original and overlay; teammate confirmation pending; no pixel ground truth or IoU.',
        'images': [{'dish_id': d, 'plate_group': selection[d]['plate_group'],
                    'complexity_group': selection[d]['complexity_group'],
                    **review[d], **({'candidate_count': outcomes[d]['candidate_count'],
                                   'food_area_ratio': outcomes[d]['selected']['food_area_ratio']}
                                  if outcomes[d]['status'] == 'completed'
                                  else {'error': outcomes[d]['error']})}
                   for d in selection],
    }
    target = manifests / 'fresh_dev12_results.json'
    target.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: report[key] for key in ['program_counts', 'manual_counts', 'by_complexity']}, indent=2))


if __name__ == '__main__':
    main()
