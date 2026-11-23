"""Recompute B6 numeric evidence from canonical labels and retained predictions."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

parser=argparse.ArgumentParser()
parser.add_argument('--results',type=Path,required=True)
parser.add_argument('--project',type=Path,required=True)
parser.add_argument('--environment',type=Path,required=True)
parser.add_argument('--comparison',type=Path)
parser.add_argument('--b5',type=Path)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
sys.path.insert(0,str(args.project))
from src.evaluate import TARGETS,load_split,normalized_nutrient_mae,plate_bootstrap_ci
from B_nutrition_estimation.compare_input_conditions import paired_interval

env=json.loads(args.environment.read_text())
m=json.loads((args.results/'metrics.json').read_text())
assert m['status']=='completed' and m['condition']=='B6'
assert m['code_commit']==env['git_head'] and env['git_status']==''
assert m['n_train']==2135 and m['n_dev']==538 and m['target_order']==TARGETS
assert set(m['per_seed'])=={'0','1','2'}
assert m['parameter_counts']=={'head':66309,'selected_visual':14570496}
names=m['selected_visual_names']
assert all(any(n.startswith(f'transformer.resblocks.{block}.') for n in names) for block in (10,11))
assert all(n.startswith(('transformer.resblocks.10.','transformer.resblocks.11.','ln_post.')) or n=='proj' for n in names)
for name,digest in env['source_sha256'].items():
    assert hashlib.sha256((args.project/'B_nutrition_estimation'/name).read_bytes()).hexdigest()==digest
for key,value in {'epochs':30,'warmup_epochs':3,'batch_size':8,'accumulation_steps':4,
                  'head_lr':.001,'clip_lr':.00001,'weight_decay':.001}.items():
    assert m['settings'][key]==value
assert m['settings']['seeds']==[0,1,2]
_,_,train=load_split(args.project/'data/splits/train.csv')
ids,groups,truth=load_split(args.project/'data/splits/dev.csv')
mean=train.mean(0)
with (args.results/'training_log.csv').open() as handle:
    log=list(csv.DictReader(handle))
assert len(log)==90
checks={}
candidate_predictions={}
for seed in (0,1,2):
    series=[r for r in log if int(r['seed'])==seed]
    assert [int(r['epoch']) for r in series]==list(range(1,31))
    assert all(r['phase']==('head_warmup' if int(r['epoch'])<=3 else 'partial_clip') for r in series)
    assert np.isfinite([float(r['train_loss']) for r in series]).all()
    with (args.results/f'dev_predictions_seed{seed}.csv').open() as handle:
        rows=list(csv.DictReader(handle))
    assert [r['dish_id'] for r in rows]==ids and [r['plate_group'] for r in rows]==groups
    labels=np.array([[float(r['true_'+key]) for key in TARGETS] for r in rows])
    pred=np.array([[float(r['pred_'+key]) for key in TARGETS] for r in rows])
    np.testing.assert_allclose(labels,truth,rtol=0,atol=.00051)
    assert pred.shape==(538,5) and np.isfinite(pred).all() and (pred>=0).all()
    candidate_predictions[seed]=pred
    score=normalized_nutrient_mae(truth,pred,mean)
    stored=m['per_seed'][str(seed)]
    assert abs(score-stored['dev']['normalized_nutrient_mae'])<.0001
    mae=np.abs(pred-truth).mean(0)
    np.testing.assert_allclose(mae,[stored['dev']['mae'][key] for key in TARGETS],atol=.002,rtol=0)
    ci=plate_bootstrap_ci(truth,pred,groups,mean)
    np.testing.assert_allclose(ci,stored['dev']['normalized_nutrient_mae_95ci'],atol=.0002,rtol=0)
    assert stored['dev']['n_plates']==182 and stored['checkpoint_reload_verified'] is True
    for key in ['frozen_visual_unchanged','text_tower_unchanged','selected_visual_updated']:
        assert stored['parameter_checks'][key] is True
    best=stored['best_epoch']
    scores=[float(r['dev_normalized_nutrient_mae']) for r in series]
    assert 1<=best<=30 and scores[best-1]<=min(scores)+.000001
    checks[str(seed)]={'normalized_nutrient_mae_from_csv':score,'mae_from_csv':dict(zip(TARGETS,map(float,mae))),
                      'bootstrap_95ci_from_csv':list(ci),'best_epoch':best,
                      'prediction_rows':len(rows),'parameter_and_reload_checks':True}
reported=np.array([m['per_seed'][str(s)]['dev']['normalized_nutrient_mae'] for s in (0,1,2)])
assert abs(float(reported.mean())-m['summary']['normalized_nutrient_mae_mean'])<1e-12
assert abs(float(reported.std(ddof=1))-m['summary']['normalized_nutrient_mae_std'])<1e-12
paired_checks=[]
if args.comparison:
    assert args.b5 is not None
    comparison=json.loads(args.comparison.read_text())
    assert comparison['n_dev']==538 and comparison['n_dev_groups']==182
    assert comparison['bootstrap_draws']==1000 and comparison['bootstrap_seed']==0
    for name,folder in [('B4',args.project/'experiments/006-b-natural-crop/results/B4'),('B5',args.b5)]:
        for seed in (0,1,2):
            with (folder/f'dev_predictions_seed{seed}.csv').open() as handle:
                rows=list(csv.DictReader(handle))
            assert [r['dish_id'] for r in rows]==ids
            baseline=np.array([[float(r['pred_'+key]) for key in TARGETS] for r in rows])
            candidate=candidate_predictions[seed]
            expected=comparison['comparisons']['B6_minus_'+name][str(seed)]
            delta=np.abs(candidate-truth).mean(0)-np.abs(baseline-truth).mean(0)
            np.testing.assert_allclose(delta,[expected['mae_delta'][key] for key in TARGETS],atol=1e-10,rtol=0)
            difference=normalized_nutrient_mae(truth,candidate,mean)-normalized_nutrient_mae(truth,baseline,mean)
            assert abs(difference-expected['normalized_nutrient_mae_delta'])<1e-10
            ci=paired_interval(truth,baseline,candidate,groups,mean)
            np.testing.assert_allclose(ci,expected['paired_plate_bootstrap_95ci'],atol=1e-10,rtol=0)
            paired_checks.append({'comparison':'B6_minus_'+name,'seed':seed,'passed':True})
report={'status':'passed','execution_commit':m['code_commit'],'training_log_rows':90,
        'canonical_dev_dishes':538,'dev_plate_groups':182,'all_three_seeds_verified':True,
        'source_matches_clean_execution_snapshot':True,
        'csv_rounding_tolerances':{'normalized_mae':.0001,'physical_mae':.002,'bootstrap_endpoints':.0002},
        'seeds':checks,'paired_comparisons_recomputed':paired_checks,'test_images_or_labels_read':False}
args.output.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
