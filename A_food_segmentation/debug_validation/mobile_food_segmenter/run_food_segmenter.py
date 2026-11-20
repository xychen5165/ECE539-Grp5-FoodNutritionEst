"""Google's pretrained food semantic segmenter -> binary RGB food mask."""
import argparse
import csv
import hashlib
import json
import os
import time
from pathlib import Path

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')


def main():
    import numpy as np
    import tensorflow as tf
    from PIL import Image, ImageFilter
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--threshold', type=float, default=.5)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error('Use a fresh output directory')
    root = args.project_root.resolve()
    base = root / 'A_food_segmentation/debug_validation'
    model_folder = root / 'checkpoints/mobile_food_segmenter_v1'
    model = tf.saved_model.load(str(model_folder))
    name = 'default' if 'default' in model.signatures else 'serving_default'
    infer = model.signatures[name]
    specs = infer.structured_input_signature[1]
    if len(specs) != 1:
        raise RuntimeError('Unexpected model inputs: ' + str(specs))
    input_name, spec = next(iter(specs.items()))
    print('Input:', input_name, spec, 'Outputs:', infer.structured_outputs, flush=True)
    # The official map includes containers and tools; nonzero != food.
    with (model_folder / 'labelmap.csv').open(newline='') as handle:
        labelmap = {int(row['id']): row['name'] for row in csv.DictReader(handle)}
    excluded = [i for i, label in labelmap.items() if label in ['background', 'food_containers', 'dining_tools']]
    assert sorted(excluded) == [0, 23, 24]
    food_ids = [i for i in sorted(labelmap) if i not in excluded]
    with args.manifest.open(newline='') as handle:
        ids = [row['dish_id'] for row in csv.DictReader(handle)]
    args.output_dir.mkdir(parents=True)
    records = []
    for dish_id in ids:
        started = time.perf_counter()
        folder = args.output_dir / dish_id
        folder.mkdir()
        record = {'dish_id': dish_id, 'status': 'failed'}
        try:
            path = base / 'local_data/nutrition5k/imagery/realsense_overhead' / dish_id / 'rgb.png'
            image = Image.open(path).convert('RGB')
            width, height = image.size
            scale = min(513 / width, 513 / height)
            nw, nh = round(width*scale), round(height*scale)
            left, top = (513-nw)//2, (513-nh)//2
            canvas = Image.new('RGB', (513, 513), (127,127,127))
            canvas.paste(image.resize((nw, nh), Image.Resampling.BILINEAR), (left,top))
            tensor = tf.convert_to_tensor(np.asarray(canvas).astype(np.float32)/255)
            if spec.shape.rank == 4:
                tensor = tensor[None]
            output = infer(**{input_name: tensor})
            probs = output['food_group_segmenter:semantic_probabilities'].numpy()
            classes = output['food_group_segmenter:semantic_predictions'].numpy()
            if probs.ndim == 4:
                probs = probs[0]
            if classes.ndim == 3:
                classes = classes[0]
            assert probs.shape == (513,513,26), probs.shape
            assert np.isfinite(probs).all() and probs.min() >= -1e-5 and probs.max() <= 1.00001
            assert np.allclose(probs.sum(axis=-1), 1, atol=.01)
            food_probability = probs[...,food_ids].sum(axis=-1)[top:top+nh,left:left+nw]
            food_probability = np.asarray(Image.fromarray(food_probability).resize(
                (width,height), Image.Resampling.BILINEAR))
            mask = food_probability >= args.threshold
            binary = Image.fromarray(mask.astype(np.uint8)*255)
            binary.save(folder/'mask.png')
            Image.fromarray(np.round(food_probability*255).astype(np.uint8)).save(folder/'food_probability.png')
            semantic = classes[top:top+nh,left:left+nw].astype(np.uint8)
            Image.fromarray(semantic).resize((width,height), Image.Resampling.NEAREST).save(folder/'semantic_classes.png')
            rgb = np.asarray(image)
            masked = np.where(mask[...,None],rgb,127).astype(np.uint8)
            Image.fromarray(masked).save(folder/'masked_rgb.png')
            overlay = rgb.astype(np.float32)
            overlay[mask] = .55*overlay[mask] + .45*np.array([0,220,80])
            edge = np.asarray(binary.filter(ImageFilter.MaxFilter(3))) != np.asarray(binary.filter(ImageFilter.MinFilter(3)))
            overlay[edge] = 255
            Image.fromarray(overlay.astype(np.uint8)).save(folder/'overlay.png')
            image.save(folder/'original.png')
            record.update(status='completed', food_area_ratio=float(mask.mean()),
                          input_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        except Exception as error:
            record.update(error=str(error), error_type=type(error).__name__)
        record['seconds'] = round(time.perf_counter()-started,3)
        records.append(record)
        (folder/'result.json').write_text(json.dumps(record,indent=2))
        summary = {'method':'Google Mobile Food Segmenter V1 -> binary', 'tensorflow':tf.__version__,
                   'signature':name,'input_name':input_name,'input_spec':str(spec),
                   'preprocessing':'aspect-preserving resize with gray letterbox to 513x513; no crop',
                   'output_mapping':'sum probabilities of food classes; exclude 0,23,24',
                   'food_ids':food_ids,'threshold':args.threshold,'device':'CPU',
                   'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   'model_sha256':hashlib.sha256((model_folder/'saved_model.pb').read_bytes()).hexdigest(),
                   'labelmap':labelmap,'images':records}
        (args.output_dir/'summary.json').write_text(json.dumps(summary,indent=2))
        print(record,flush=True)


if __name__ == '__main__':
    main()
