"""Real-image A/B smoke test, NOT a nutrition quality benchmark.

Downloads nine deterministic train/dev images, runs the selected A decoder,
extracts original/masked CLIP features, and trains/reloads tiny B trial heads.
All large outputs remain in ignored artifacts/. No test images are used.
"""
import argparse
import csv
import gc
import hashlib
import json
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace

import numpy as np
import open_clip
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'A_food_segmentation'))
from sam_runtime import infer_rgb, save_prediction
from download_photos import download
from extract_clip_features import pad_to_square
from train_head import build_model, predict, train_one_seed
from evaluate import TARGETS


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select(split, count):
    with (ROOT / 'data/splits' / f'{split}.csv').open() as stream:
        records = list(csv.DictReader(stream))
    random.Random(539).shuffle(records)
    selected, groups = [], set()
    for record in records:
        if record['plate_group'] not in groups:
            selected.append(record)
            groups.add(record['plate_group'])
        if len(selected) == count:
            break
    if len(selected) != count:
        raise ValueError('Not enough distinct plate groups')
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sam-checkpoint', type=Path, required=True)
    parser.add_argument('--train-count', type=int, default=6)
    parser.add_argument('--dev-count', type=int, default=3)
    parser.add_argument('--reuse-preprocessing', type=Path,
                        help='Reuse a prior run after validating images/masks and feature IDs')
    args = parser.parse_args()
    if args.train_count < 2 or args.dev_count < 1:
        parser.error('Need at least two training images and one dev image')
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(0)
    started = time.perf_counter()
    decoder = ROOT / 'A_food_segmentation/model_weights/food_decoder_no_prompt.pt'
    info = json.loads((ROOT / 'A_food_segmentation/model_info.json').read_text())
    assert sha(args.sam_checkpoint) == info['official_sam_sha256']
    assert sha(decoder) == info['reference_decoder_sha256']
    train, dev = select('train', args.train_count), select('dev', args.dev_count)
    records = train + dev
    (args.output / 'selected.json').write_text(json.dumps(records, indent=2))
    source = args.reuse_preprocessing or args.output
    if args.reuse_preprocessing:
        assert json.loads((source / 'selected.json').read_text()) == records
    original_root = ROOT / 'artifacts/ab-integration/images'
    image_records = []
    for row in records:
        dish = row['dish_id']
        _, status = download(dish, original_root)
        if status.startswith('failed'):
            raise RuntimeError(f'{dish}: {status}')
        image_path = original_root / dish / 'rgb.png'
        rgb = np.asarray(Image.open(image_path).convert('RGB'))
        t0 = time.perf_counter()
        output = source / 'masked' / dish
        if args.reuse_preprocessing:
            raw_mask = np.asarray(Image.open(output / 'mask.png'))
            assert set(np.unique(raw_mask)).issubset({0, 255})
            mask = raw_mask > 0
            metadata = json.loads((output / 'prediction.json').read_text())
        else:
            mask, metadata = infer_rgb(rgb, args.sam_checkpoint, decoder, device='cpu')
        assert mask.shape == rgb.shape[:2] and mask.dtype == np.bool_
        assert metadata['prompt'] is None
        assert mask.any() and not mask.all(), 'Degenerate mask requires review'
        if not args.reuse_preprocessing:
            save_prediction(rgb, mask, output, metadata)
        masked = np.asarray(Image.open(output / 'masked_rgb.png'))
        assert np.array_equal(masked[mask], rgb[mask])
        assert np.all(masked[~mask] == 128)
        image_records.append({'dish_id': dish, 'split': row['split'],
                              'image_sha256': sha(image_path),
                              'masked_sha256': sha(output / 'masked_rgb.png'),
                              'foreground_fraction': float(mask.mean()),
                              'a_seconds_including_model_load':
                              None if args.reuse_preprocessing else time.perf_counter() - t0})
        print(f'A {"cache validated" if args.reuse_preprocessing else "completed"} {dish}', flush=True)
        gc.collect()

    model = None
    if not args.reuse_preprocessing:
        model, _, preprocess = open_clip.create_model_and_transforms(
            'ViT-B-32', pretrained='openai',
            cache_dir=str(ROOT / 'artifacts/ab-integration/checkpoints'))
        model.eval()
    features = {}
    with torch.inference_mode():
        for condition in ('original', 'masked'):
            if args.reuse_preprocessing:
                cached = np.load(source / f'{condition}.npz')
                assert list(cached['dish_ids']) == [r['dish_id'] for r in records]
                features[condition] = cached['features']
                assert features[condition].shape == (len(records), 512)
                assert np.isfinite(features[condition]).all()
                continue
            vectors = []
            for row in records:
                dish = row['dish_id']
                path = (original_root / dish / 'rgb.png' if condition == 'original'
                        else args.output / 'masked' / dish / 'masked_rgb.png')
                with Image.open(path) as image:
                    x = preprocess(pad_to_square(image.convert('RGB')))[None]
                vectors.append(model.encode_image(x).float().numpy()[0])
            features[condition] = np.stack(vectors)
            assert features[condition].shape == (len(records), 512)
            assert np.isfinite(features[condition]).all()
            np.savez(args.output / f'{condition}.npz',
                     dish_ids=np.array([r['dish_id'] for r in records]),
                     features=features[condition])
            print(f'CLIP completed {condition}', flush=True)
    del model
    gc.collect()

    n = len(train)
    y = np.array([[float(r[t]) for t in TARGETS] for r in records], dtype=np.float32)
    target_scale = y[:n].mean(axis=0)
    config = SimpleNamespace(model='mlp', loss='l1', hidden=128, dropout=0.5, lr=0.001,
                             weight_decay=0.001, epochs=3, batch_size=32)
    trials = {}
    for condition, values in features.items():
        mu, sigma = values[:n].mean(axis=0), values[:n].std(axis=0) + 1e-6
        x = torch.tensor((values - mu) / sigma)
        torch.manual_seed(0)
        initial = build_model('mlp', 512, 128, 0.5)
        initial_weights = {key: value.clone() for key, value in initial.state_dict().items()}
        del initial
        head, best, metrics, log = train_one_seed(
            0, config, x[:n], torch.tensor(y[:n] / target_scale), x[n:], y[n:],
            [r['plate_group'] for r in dev], torch.tensor(target_scale), target_scale)
        assert any(not torch.equal(value, initial_weights[key])
                   for key, value in head.state_dict().items()), 'Training did not update weights'
        prediction = predict(head, x[n:], torch.tensor(target_scale))
        assert prediction.shape == (len(dev), 5) and np.isfinite(prediction).all()
        assert (prediction >= 0).all()
        checkpoint = args.output / f'{condition}_smoke_head.pt'
        # Store plain lists for portable weights_only checkpoint loading.
        torch.save({'state_dict': head.state_dict(), 'input_mean': mu.tolist(),
                    'input_std': sigma.tolist(), 'target_scale': target_scale.tolist(),
                    'targets': TARGETS, 'args': vars(config)}, checkpoint)
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        reloaded = build_model('mlp', 512, 128, 0.5)
        reloaded.load_state_dict(saved['state_dict'])
        restored_x = torch.tensor((values[n:] - np.array(saved['input_mean'], dtype=np.float32))
                                 / np.array(saved['input_std'], dtype=np.float32))
        restored_prediction = predict(reloaded, restored_x, torch.tensor(saved['target_scale']))
        assert np.allclose(prediction, restored_prediction, rtol=1e-6, atol=1e-5)
        trials[condition] = {'best_epoch': best['epoch'], 'weights_changed': True,
                             'checkpoint_roundtrip_passed': True,
                             'prediction_shape': list(prediction.shape),
                             'training_log': log, 'diagnostic_metrics_only': metrics}
        print(f'B train/reload completed {condition}', flush=True)
    report = {'status': 'passed', 'scope': 'smoke only, not a quality comparison',
              'test_images_accessed': False, 'train_count': len(train), 'dev_count': len(dev),
              'a_method': 'finetuned_no_prompt', 'sam_sha256': sha(args.sam_checkpoint),
              'decoder_sha256': sha(decoder), 'images': image_records,
              'preprocessing_source': str(source),
              'reused_preprocessing': bool(args.reuse_preprocessing),
              'feature_sha256': {name: sha(source / f'{name}.npz')
                                 for name in ('original', 'masked')},
              'trials': trials, 'seconds': time.perf_counter() - started,
              'versions': {'torch': torch.__version__, 'numpy': np.__version__}}
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'status': report['status'], 'seconds': report['seconds']}), flush=True)


if __name__ == '__main__':
    main()
