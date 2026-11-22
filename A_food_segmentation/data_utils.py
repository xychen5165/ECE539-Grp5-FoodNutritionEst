"""Pinned train/dev inputs and metrics shared by the four Part A methods."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

ANNOTATION_COMMIT = '0997586b5c177fa8f2c0a61eb7f958ab18605362'
SPLIT_COMMIT = 'aec9a6c0f6ef6826185529602a7c6af72b638a80'
CHECKPOINT_SHA = 'ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path):
    with Path(path).open(newline='') as handle:
        return list(csv.DictReader(handle))


def inputs(root):
    base = Path(root)/'A_food_segmentation'
    protocol_dir = base/'method_comparison/protocol'
    protocol = json.loads((protocol_dir/'protocol.json').read_text())
    assert protocol['project_split_commit']==SPLIT_COMMIT
    assert protocol['annotation_commit']==ANNOTATION_COMMIT
    for name in ['assignment.csv','train_development.csv','dev_tuning.csv']:
        assert sha(protocol_dir/name)==protocol['generated_csv_sha256'][name], name
    assignment = read_csv(protocol_dir/'assignment.csv')
    test_groups = {r['plate_group'] for r in assignment if r['project_split']=='test'}
    rows = {}
    for split,name,role,count in [('train','train_development.csv','train_development',454),
                                  ('dev','dev_tuning.csv','dev_tuning',103)]:
        rows[split]=read_csv(protocol_dir/name)
        assert len(rows[split])==count
        assert all(r['project_split']==split and r['role']==role for r in rows[split])
        assert all(r['plate_group'] not in test_groups for r in rows[split])
        assert len({r['dish_id'] for r in rows[split]})==count
    assert not {r['dish_id'] for r in rows['train']} & {r['dish_id'] for r in rows['dev']}
    assert not {r['plate_group'] for r in rows['train']} & {r['plate_group'] for r in rows['dev']}
    annotations = base/'_local/data/hand_annotations'
    assert sha(annotations/'ingredient_nutrition.csv')==protocol['annotation_csv_sha256']
    by_dish = {}
    for r in read_csv(annotations/'ingredient_nutrition.csv'):
        by_dish.setdefault(r['dish_id'],[]).append(r)
    return base, protocol, rows, annotations, by_dish


def paths(base, annotations, row):
    assert row['project_split'] in {'train','dev'}
    return base/'_local/data/nutrition5k'/row['image_path'], annotations/row['annotation_mask']


def load_pair(base, annotations, row, annotation_rows):
    rgb_path, mask_path = paths(base,annotations,row)
    rgb = np.asarray(Image.open(rgb_path).convert('RGB'))
    labels = np.asarray(Image.open(mask_path))
    assert labels.ndim==2 and labels.shape==rgb.shape[:2], row['dish_id']
    assert set(np.unique(labels))=={0}|{int(a['label_value']) for a in annotation_rows}
    for a in annotation_rows:
        assert int((labels==int(a['label_value'])).sum())==int(a['mask_pixels'])
    return rgb,labels


def metrics(pred,gt):
    assert pred.shape==gt.shape and gt.any()
    tp=int((pred & gt).sum()); fp=int((pred & ~gt).sum()); fn=int((~pred & gt).sum())
    return {'tp':tp,'fp':fp,'fn':fn,'iou':tp/(tp+fp+fn),
            'precision':tp/(tp+fp) if tp+fp else 0.0,'recall':tp/(tp+fn),
            'dice':2*tp/(2*tp+fp+fn)}


def rating(m):
    if m['iou']>=.85 and m['precision']>=.90 and m['recall']>=.90:
        return 'pass'
    return 'fail' if m['iou']<.50 else 'partial'


def write_csv(path,rows):
    if not rows:
        return
    with Path(path).open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)
