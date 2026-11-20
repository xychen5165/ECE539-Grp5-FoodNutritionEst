import csv
import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path


def main():
    import numpy as np
    from PIL import Image
    folder = Path(__file__).resolve().parent
    base = folder.parent
    output = base / 'outputs/mobile_food_dev12_v1'
    summary = json.loads((output / 'summary.json').read_text())
    assert summary['script_sha256'] == hashlib.sha256((folder/'run_food_segmenter.py').read_bytes()).hexdigest()
    with (folder/'review.csv').open(newline='') as handle:
        review = {r['dish_id']: r for r in csv.DictReader(handle)}
    with (base/'manifests/fresh_dev12_selection.csv').open(newline='') as handle:
        selected = {r['dish_id']: r for r in csv.DictReader(handle)}
    assert len(summary['images']) == len(review) == len(selected) == 12
    assert {r['dish_id'] for r in summary['images']} == review.keys() == selected.keys()
    for r in summary['images']:
        assert r['status'] == review[r['dish_id']]['program_status'] == 'completed'
        p = output/r['dish_id']
        with Image.open(p/'original.png') as image, Image.open(p/'mask.png') as mask:
            assert image.size == mask.size
            assert set(np.unique(np.asarray(mask))) <= {0,255}
    summary['visual_counts'] = dict(Counter(r['manual_rating'] for r in review.values()))
    summary['visual_review'] = list(review.values())
    summary['median_seconds'] = statistics.median(r['seconds'] for r in summary['images'])
    summary['review_method'] = 'Assistant qualitative visual inspection; teammate confirmation pending; no IoU.'
    summary['manifest_sha256'] = hashlib.sha256((base/'manifests/fresh_dev12_selection.csv').read_bytes()).hexdigest()
    summary['model_source'] = 'https://tfhub.dev/google/seefood/segmenter/mobile_food_segmenter_V1/1'
    summary['not_paper_replication'] = True
    checkpoint = folder.parents[2] / 'checkpoints/mobile_food_segmenter_v1'
    provenance = {'source':summary['model_source'],
                  'archive_url':summary['model_source']+'?tf-hub-format=compressed',
                  'files':[{'name':p.name,'bytes':p.stat().st_size,
                            'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
                           for p in sorted(checkpoint.iterdir()) if p.is_file()]}
    (folder/'model_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    (folder/'results.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(summary['visual_counts'], 'median seconds:',summary['median_seconds'])


if __name__ == '__main__':
    main()
