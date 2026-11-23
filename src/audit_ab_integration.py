"""Dependency-free A/B integration preflight; does not run neural inference.

Run: python3 src/audit_ab_integration.py --output experiments/004-ab-integration/preflight.json
"""
import argparse
import ast
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
TARGETS = ('mass_g', 'energy_kcal', 'protein_g', 'carbohydrate_g', 'fat_g')


def rows(path):
    with path.open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    checks = []

    def check(name, condition):
        checks.append({'check': name, 'passed': bool(condition)})

    splits = {name: rows(ROOT / 'data/splits' / f'{name}.csv')
              for name in ('train', 'dev', 'test')}
    ids = {name: {r['dish_id'] for r in data} for name, data in splits.items()}
    groups = {name: {r['plate_group'] for r in data} for name, data in splits.items()}
    for name in splits:
        check(f'{name}: unique dish IDs', len(ids[name]) == len(splits[name]))
    for left, right in [('train', 'dev'), ('train', 'test'), ('dev', 'test')]:
        check(f'{left}/{right}: disjoint IDs and inferred groups',
              not ids[left] & ids[right] and not groups[left] & groups[right])

    a_root = ROOT / 'A_food_segmentation'
    protocol = a_root / 'method_comparison/protocol'
    for filename, split in [('train_development.csv', 'train'), ('dev_tuning.csv', 'dev')]:
        subset = rows(protocol / filename)
        canonical = {r['dish_id']: r for r in splits[split]}
        check(f'A {filename}: matches B IDs and groups', all(
            r['dish_id'] in canonical and
            r['plate_group'] == canonical[r['dish_id']]['plate_group'] for r in subset))

    info = json.loads((a_root / 'model_info.json').read_text())
    decoder = a_root / 'model_weights/food_decoder_no_prompt.pt'
    check('A selected decoder SHA256', hashlib.sha256(decoder.read_bytes()).hexdigest()
          == info['reference_decoder_sha256'])
    a_results = rows(a_root / 'method_comparison/results/03_finetuned_no_prompt.csv')
    check('A result IDs match tuning manifest',
          len(a_results) == len(rows(protocol / 'dev_tuning.csv')) and
          {r['dish_id'] for r in a_results} ==
          {r['dish_id'] for r in rows(protocol / 'dev_tuning.csv')})
    ious = []
    for row in a_results:
        tp, fp, fn = (int(row[key]) for key in ('tp', 'fp', 'fn'))
        ious.append(tp / (tp + fp + fn))
    check('A IoU recomputed from TP/FP/FN', all(
        math.isclose(value, float(row['iou']), abs_tol=1e-10)
        for value, row in zip(ious, a_results)))
    mean_iou = sum(ious) / len(ious)
    check('A aggregate IoU matches model metadata',
          math.isclose(mean_iou, info['reference_dev_mean_iou'], abs_tol=1e-10))

    means = {t: sum(float(r[t]) for r in splits['train']) / len(splits['train'])
             for t in TARGETS}
    dev = {r['dish_id']: r for r in splits['dev']}
    prediction_files = sorted((ROOT / 'experiments').glob('*/**/dev_predictions*.csv'))
    for path in prediction_files:
        predictions = rows(path)
        label = str(path.relative_to(ROOT))
        check(f'{label}: complete unique dev IDs',
              len(predictions) == len(dev) and {r['dish_id'] for r in predictions} == set(dev))
        check(f'{label}: groups and rounded labels match', all(
            r['dish_id'] in dev and r['plate_group'] == dev[r['dish_id']]['plate_group'] and
            all(abs(float(r[f'true_{t}']) - float(dev[r['dish_id']][t])) <= 0.000501
                for t in TARGETS) for r in predictions))
        check(f'{label}: finite nonnegative predictions', all(
            math.isfinite(float(r[f'pred_{t}'])) and float(r[f'pred_{t}']) >= 0
            for r in predictions for t in TARGETS))
        metrics = json.loads((path.parent / 'metrics.json').read_text())
        expected = (metrics['per_seed'][path.stem.split('seed')[-1]]['dev']
                    if 'seed' in path.stem else metrics['dev'])
        maes = {t: sum(abs(float(r[f'true_{t}']) - float(r[f'pred_{t}']))
                       for r in predictions) / len(predictions) for t in TARGETS}
        score = sum(maes[t] / means[t] for t in TARGETS[1:]) / 4
        check(f'{label}: MAE and normalized score reproduce',
              all(abs(maes[t] - expected['mae'][t]) < 0.002 for t in TARGETS) and
              abs(score - expected['normalized_nutrient_mae']) < 0.0002)

    source_files = (list(a_root.rglob('*.py')) + list((ROOT / 'src').glob('*.py'))
                    + list((ROOT / 'B_nutrition_estimation').glob('*.py')))
    for path in source_files:
        ast.parse(path.read_text(), filename=str(path))
    check('Merged Python sources parse', True)
    report = {
        'scope': 'Static/data/artifact preflight only; no neural inference or retraining',
        'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'split_counts': {k: len(v) for k, v in splits.items()},
        'a_recomputed_dev_iou': mean_iou,
        'b_prediction_files_checked': len(prediction_files),
        'python_files_parsed': len(source_files),
        'checks': checks,
        'passed': all(c['passed'] for c in checks),
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report['passed'] else 1)


if __name__ == '__main__':
    main()
