"""Validate supplied B5 evidence without loading or running its checkpoints."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys
import zipfile
import numpy as np

p=argparse.ArgumentParser()
p.add_argument('--archive',type=Path,required=True)
p.add_argument('--project',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
args=p.parse_args()
sys.path.insert(0,str(args.project))
from src.evaluate import TARGETS,load_split,normalized_nutrient_mae

train_ids,_,train=load_split(args.project/'data/splits/train.csv')
dev_ids,groups,labels=load_split(args.project/'data/splits/dev.csv')
required=['metrics.json','training_log.csv']+[f'dev_predictions_seed{s}.csv' for s in (0,1,2)]
if args.archive.is_dir():
    data={name:(args.archive/name).read_bytes() for name in required}
    source_kind='already_extracted_directory'
    source_archive_sha256=None
else:
    with zipfile.ZipFile(args.archive) as z:
        bad=z.testzip()
        if bad:raise ValueError('Archive CRC failure: '+bad)
        candidates=[n for n in z.namelist() if n=='metrics.json' or n.endswith('/B5/metrics.json') or n=='B5/metrics.json']
        assert len(candidates)==1,candidates
        prefix=PurePosixPath(candidates[0]).parent
        data={name:z.read(str(prefix/name) if str(prefix)!='.' else name) for name in required}
    source_kind='zip_archive'
    source_archive_sha256=hashlib.sha256(args.archive.read_bytes()).hexdigest()
m=json.loads(data['metrics.json'])
assert m['status']=='completed' and m['condition']=='B5'
assert m['n_train']==2135 and m['n_dev']==538
assert m['target_order']==TARGETS
assert set(m['per_seed'])=={'0','1','2'}
assert m['inputs_sha256']=={
    'original':'307f2e7cdad43bb635d88e1e0e49b237631e76d0ff32d16fdf8a1098e2aabaa2',
    'crop':'3e7d4a418445c3914f29fd2eb4b13115fc124b94f8c6ebebb26460e748ea55fd'}
names=m['selected_visual_names']
assert any(n.startswith('transformer.resblocks.11.') for n in names)
assert all(n.startswith(('transformer.resblocks.11.','ln_post.')) or n=='proj' for n in names)
assert m['parameter_counts']['selected_visual']==7482624
settings=m['settings']
for name,value in {'epochs':30,'warmup_epochs':3,'batch_size':8,'accumulation_steps':4,
                   'head_lr':.001,'clip_lr':.00001,'weight_decay':.001}.items():
    assert settings[name]==value,(name,settings[name])
log=list(csv.DictReader(io.StringIO(data['training_log.csv'].decode('utf-8-sig'))))
assert len(log)==90
recomputed={}
for seed in (0,1,2):
    series=[r for r in log if int(r['seed'])==seed]
    assert [int(r['epoch']) for r in series]==list(range(1,31))
    assert np.isfinite([float(r['train_loss']) for r in series]).all()
    assert np.isfinite([float(r['dev_normalized_nutrient_mae']) for r in series]).all()
    rows=list(csv.DictReader(io.StringIO(data[f'dev_predictions_seed{seed}.csv'].decode('utf-8-sig'))))
    assert [r['dish_id'] for r in rows]==dev_ids
    assert [r['plate_group'] for r in rows]==groups
    truth=np.array([[float(r['true_'+name]) for name in TARGETS] for r in rows])
    values=np.array([[float(r['pred_'+name]) for name in TARGETS] for r in rows])
    np.testing.assert_allclose(truth,labels,atol=.00051,rtol=0)
    assert np.isfinite(values).all() and (values>=0).all()
    score=normalized_nutrient_mae(labels,values,train.mean(0))
    stored=m['per_seed'][str(seed)]
    assert abs(score-stored['dev']['normalized_nutrient_mae'])<.0001
    assert 1<=stored['best_epoch']<=30
    recomputed[str(seed)]={'normalized_nutrient_mae_from_csv':score,'reported':stored['dev']['normalized_nutrient_mae'],'best_epoch':stored['best_epoch']}
core=['B_nutrition_estimation/train_clip_finetune.py','B_nutrition_estimation/compare_finetune_condition.py','B_nutrition_estimation/package_finetune_results.py']
diff=subprocess.run(['git','diff','--exit-code','20a0bc8c80f47ccd154f8d5982d1833cd01032ef',m['code_commit'],'--',*core],cwd=args.project,capture_output=True,text=True)
assert diff.returncode==0,'B5 source does not match the frozen implementation: '+diff.stderr
out=args.output/'B5';out.mkdir(parents=True,exist_ok=True)
for name,raw in data.items():(out/name).write_bytes(raw)
report={'status':'passed','method':'B5 last visual block, not obsolete gated-fusion numbering',
        'source_name':args.archive.name,'source_kind':source_kind,'source_archive_sha256':source_archive_sha256,
        'source_code_commit':m['code_commit'],'frozen_B5_implementation_matches':True,
        'n_train':2135,'n_dev':538,'dev_plate_groups':len(set(groups)),'training_log_rows':90,
        'seeds':recomputed,'feature_hashes_match_handoff':True,'canonical_labels_and_ids_match':True,
        'checkpoint_weights_loaded_or_executed':False,'test_images_or_labels_read':False,
        'small_file_sha256':{name:hashlib.sha256(raw).hexdigest() for name,raw in data.items()}}
(args.output/'b5-handoff-check.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
