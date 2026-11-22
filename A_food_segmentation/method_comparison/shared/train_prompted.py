"""Decoder fine-tuning with frozen RGB-derived whole-dish spatial prompts."""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
from PIL import Image,ImageDraw
import torch
import torch.nn.functional as F
from segment_anything import SamAutomaticMaskGenerator,sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide

from automatic_prompts import PARAMETERS,derive_prompts


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root',type=Path,default=Path(__file__).resolve().parents[3])
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--no-prompt-run',type=Path)
    parser.add_argument('--sam-checkpoint',type=Path)
    args=parser.parse_args()
    base=args.project_root/'A_food_segmentation'
    sys.path.insert(0,str(base))
    from data_utils import inputs,paths,load_pair,metrics,rating,sha,write_csv
    from train_decoder import module_sha,state_cpu
    base,protocol,parts,annotations,by_dish=inputs(args.project_root)
    out=args.output_dir
    if out.exists():
        parser.error('Use a new run directory; do not overwrite earlier results')
    previous=args.no_prompt_run or base/'_local/outputs/no_prompt_trial'
    prior=json.loads((previous/'results.json').read_text())
    downloads={r['dish_id']:r for r in json.loads((previous/'downloads.json').read_text())['pairs']}
    cache_provenance={r['dish_id']:r for r in json.loads((previous/'cache_provenance.json').read_text())}
    cache=previous/'embedding_cache'
    selected=parts['train']+parts['dev']
    checkpoint=args.sam_checkpoint or base/'_local/checkpoints/sam_vit_b_01ec64.pth'
    assert sha(checkpoint)==prior['config']['checkpoint_sha256']
    for name,value in prior['config']['manifest_sha256'].items():
        assert value==protocol['generated_csv_sha256'][name]
    torch.manual_seed(539);random.seed(539);np.random.seed(539)
    model=sam_model_registry['vit_b'](checkpoint=str(checkpoint)).to('cuda').eval()
    for p in model.parameters():
        p.requires_grad=False
    frozen_before={name:module_sha(getattr(model,name)) for name in ['image_encoder','prompt_encoder']}
    original_decoder_sha=module_sha(model.mask_decoder)
    original_decoder=state_cpu(model.mask_decoder)
    config={'method':'decoder_only_finetuning_with_automatic_whole_dish_box_and_signed_points',
        'initialization':'original official SAM ViT-B, not the previously fine-tuned decoder',
        'seed':539,'device':'cuda:0','torch_version':str(torch.__version__),
        'gpu':torch.cuda.get_device_name(0),'prompt_parameters':PARAMETERS,
        'prompt_source':'original frozen SAM candidates + RGB-only plate/seed geometry',
        'target':'union of all visible annotated food regions for the whole dish',
        'prompt_encoders_frozen':True,'image_encoder_frozen':True,
        'prompt_generation_uses_reference_pixels':False,
        'final_mask_postprocessing':'none; original-resolution logits > 0',
        'multimask_output':False,'max_epochs':12,'patience':4,'learning_rate':1e-4,
        'weight_decay':.01,'gradient_accumulation':4,'gradient_clip_norm':1.0,
        'loss':'full-resolution BCEWithLogits + soft Dice',
        'checkpoint_selection':'highest development mean image IoU, earliest epoch for ties',
        'train_count':454,'dev_count':103,'test_accessed':False,
        'project_split_commit':protocol['project_split_commit'],
        'annotation_commit':protocol['annotation_commit'],
        'thresholds':protocol['thresholds'],
        'manifest_sha256':prior['config']['manifest_sha256'],
        'official_checkpoint_sha256':sha(checkpoint),
        'previous_no_prompt_checkpoint_sha256':sha(previous/'best_decoder.pt'),
        'previous_no_prompt_results_sha256':sha(previous/'results.json'),
        'visual_examples_fixed_before_evaluation':prior['config']['visual_examples_fixed_before_evaluation'],
        'code_sha256':{p.name:sha(p) for p in Path(__file__).parent.glob('*.py')},
        'shared_source_sha256':{name:sha(base/name) for name in ['data_utils.py','train_decoder.py']}}
    out.mkdir(parents=True)
    (out/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    prompt_dir=out/'prompts';prompt_dir.mkdir()
    generator=SamAutomaticMaskGenerator(model,points_per_side=16,points_per_batch=64,
        pred_iou_thresh=.86,stability_score_thresh=.92,crop_n_layers=0,min_mask_region_area=100)
    prompts={};prompt_records=[];started=time.perf_counter()
    for i,row in enumerate(selected,1):
        rgb_path,_=paths(base,annotations,row)
        assert sha(rgb_path)==downloads[row['dish_id']]['rgb_sha256']
        # No human mask is opened during prompt generation. This function's
        # inputs are the unmasked RGB and ORIGINAL pretrained SAM candidates.
        rgb=np.asarray(Image.open(rgb_path).convert('RGB'))
        candidates=generator.generate(rgb)
        prompt=derive_prompts(rgb,candidates)
        prompt.update(dish_id=row['dish_id'],rgb_sha256=sha(rgb_path),
                      generator_decoder_sha256=original_decoder_sha)
        target=prompt_dir/(row['dish_id']+'.json')
        target.write_text(json.dumps(prompt,indent=2)+'\n')
        prompts[row['dish_id']]=prompt
        prompt_records.append({'dish_id':row['dish_id'],'prompt_sha256':sha(target)})
        if row['dish_id'] in config['visual_examples_fixed_before_evaluation']:
            image=Image.fromarray(rgb);draw=ImageDraw.Draw(image)
            draw.rectangle(prompt['box_xyxy'],outline='orange',width=3)
            for (x,y),label in zip(prompt['points_xy'],prompt['point_labels']):
                draw.ellipse((x-4,y-4,x+4,y+4),fill='lime' if label else 'red',outline='black')
            image.save(prompt_dir/(row['dish_id']+'_prompts.png'))
        if i%25==0 or i==557:
            print(f'RGB-only prompts fixed for {i}/557 images; elapsed {time.perf_counter()-started:.1f}s',flush=True)
    torch.cuda.synchronize()
    prompt_seconds=time.perf_counter()-started
    assert module_sha(model.mask_decoder)==original_decoder_sha
    (out/'frozen_prompt_manifest.json').write_text(json.dumps(prompt_records,indent=2)+'\n')
    # ONLY after all 557 prompt files are frozen, open labels to diagnose their
    # correctness. Diagnostics never modify or filter prompts or training rows.
    point_diagnostics=[]
    for row in selected:
        assert sha(cache/(row['dish_id']+'.npz'))==cache_provenance[row['dish_id']]['embedding_cache_sha256']
        rgb,labels=load_pair(base,annotations,row,by_dish[row['dish_id']]);gt=labels>0
        p=prompts[row['dish_id']]
        positive=[(x,y) for (x,y),label in zip(p['points_xy'],p['point_labels']) if label==1]
        negative=[(x,y) for (x,y),label in zip(p['points_xy'],p['point_labels']) if label==0]
        x0,y0,x1,y1=p['box_xyxy'];box=np.zeros_like(gt);box[y0:y1+1,x0:x1+1]=True
        point_diagnostics.append({'dish_id':row['dish_id'],'split':row['project_split'],
            'positive_count':len(positive),'positive_on_food':sum(int(gt[y,x]) for x,y in positive),
            'negative_count':len(negative),'negative_on_background':sum(int(not gt[y,x]) for x,y in negative),
            'box_food_coverage':float((box & gt).sum()/gt.sum()),'fallback_reason':p['fallback_reason'] or ''})
    write_csv(out/'prompt_diagnostics.csv',point_diagnostics)
    transform=ResizeLongestSide(model.image_encoder.img_size)
    encoded={}
    with torch.no_grad():
        pe=model.prompt_encoder.get_dense_pe()
        for row in selected:
            p=prompts[row['dish_id']]
            with np.load(cache/(row['dish_id']+'.npz'),allow_pickle=False) as data:
                shape=tuple(map(int,data['original_size']))
            box=torch.as_tensor(transform.apply_boxes(np.asarray(p['box_xyxy'],np.float32)[None],shape),device='cuda')
            points=None
            if p['points_xy']:
                coords=transform.apply_coords(np.asarray(p['points_xy'],np.float32),shape)
                points=(torch.as_tensor(coords,device='cuda')[None],torch.as_tensor(p['point_labels'],device='cuda')[None])
            encoded[row['dish_id']]=model.prompt_encoder(points=points,boxes=box,masks=None)

    def sample(row):
        with np.load(cache/(row['dish_id']+'.npz'),allow_pickle=False) as data:
            feature=torch.as_tensor(data['embedding'].astype(np.float32),device='cuda')
            target=torch.as_tensor(data['gt'].copy(),device='cuda',dtype=torch.float32)[None,None]
            input_size=tuple(map(int,data['input_size']));original_size=tuple(map(int,data['original_size']))
        sparse,dense=encoded[row['dish_id']]
        low,_=model.mask_decoder(image_embeddings=feature,image_pe=pe,
            sparse_prompt_embeddings=sparse,dense_prompt_embeddings=dense,multimask_output=False)
        return model.postprocess_masks(low,input_size,original_size),target

    def evaluate(rows,name,save=False):
        model.mask_decoder.eval();records=[];regions=[]
        with torch.no_grad():
            for row in rows:
                logits,target=sample(row)
                pred=(logits[0,0]>0).cpu().numpy();gt=target[0,0].bool().cpu().numpy()
                m=metrics(pred,gt)
                _,labels=load_pair(base,annotations,row,by_dish[row['dish_id']])
                values=[]
                for a in by_dish[row['dish_id']]:
                    region=labels==int(a['label_value']);recall=float((pred & region).sum()/region.sum())
                    values.append(recall)
                    regions.append({'dish_id':row['dish_id'],'label_value':a['label_value'],
                        'annotation_name':a['annotation_name'],'pixels':int(region.sum()),'recall':recall})
                records.append({'dish_id':row['dish_id'],'plate_group':row['plate_group'],**m,
                    'rating':rating(m),'minimum_annotation_recall':min(values)})
                if save:
                    folder=out/name/row['dish_id'];folder.mkdir(parents=True,exist_ok=True)
                    Image.fromarray(pred.astype(np.uint8)*255).save(folder/'mask.png')
        grouped=defaultdict(list)
        for r in records:grouped[r['plate_group']].append(r['iou'])
        summary={'count':len(records),'plate_group_count':len(grouped),
            **{'mean_'+key:float(np.mean([r[key] for r in records])) for key in ['iou','precision','recall','dice']},
            'plate_macro_mean_iou':float(np.mean([np.mean(v) for v in grouped.values()])),
            'rating_counts':dict(Counter(r['rating'] for r in records)),
            'annotation_regions_below_90pct_recall':sum(r['recall']<.9 for r in regions),
            'images_with_region_below_90pct_recall':len({r['dish_id'] for r in regions if r['recall']<.9})}
        return records,regions,summary

    original_rows,_,original_prompted=evaluate(parts['dev'],'original_prompted',save=True)
    write_csv(out/'original_prompted_dev_metrics.csv',original_rows)
    print('ORIGINAL + AUTOMATIC PROMPTS',json.dumps(original_prompted),flush=True)
    no_prompt_state=torch.load(previous/'best_decoder.pt',map_location='cpu',weights_only=True)
    model.mask_decoder.load_state_dict(no_prompt_state['decoder'])
    added_rows,_,added_prompts=evaluate(parts['dev'],'existing_decoder_added_prompts',save=True)
    write_csv(out/'existing_decoder_added_prompts_dev_metrics.csv',added_rows)
    print('PREVIOUS DECODER + ADDED AUTOMATIC PROMPTS',json.dumps(added_prompts),flush=True)
    model.mask_decoder.load_state_dict(original_decoder)
    assert module_sha(model.mask_decoder)==original_decoder_sha
    for p in model.mask_decoder.parameters():p.requires_grad=True
    optimizer=torch.optim.AdamW(model.mask_decoder.parameters(),lr=1e-4,weight_decay=.01)
    history=[];best_score=-1;best_epoch=None;best_state=None;stale=0;first_gradient=None
    began=time.perf_counter()
    for epoch in range(1,13):
        model.mask_decoder.train();order=list(parts['train']);random.Random(539+epoch).shuffle(order)
        loss_sum=0;start=time.perf_counter()
        for offset in range(0,len(order),4):
            batch=order[offset:offset+4];optimizer.zero_grad(set_to_none=True)
            for row in batch:
                logits,target=sample(row);prob=logits.sigmoid()
                dice=1-(2*(prob*target).sum()+1)/(prob.sum()+target.sum()+1)
                loss=F.binary_cross_entropy_with_logits(logits,target)+dice
                assert torch.isfinite(loss)
                (loss/len(batch)).backward();loss_sum+=float(loss.detach().cpu())
            norm=torch.nn.utils.clip_grad_norm_(model.mask_decoder.parameters(),1)
            assert torch.isfinite(norm)
            if first_gradient is None:first_gradient=float(norm.detach().cpu())
            optimizer.step()
        _,_,dev=evaluate(parts['dev'],f'epoch_{epoch}')
        record={'epoch':epoch,'train_loss':loss_sum/454,'dev_iou':dev['mean_iou'],
            'dev_precision':dev['mean_precision'],'dev_recall':dev['mean_recall'],
            'dev_dice':dev['mean_dice'],'seconds':time.perf_counter()-start}
        history.append(record);write_csv(out/'history.csv',history)
        print('EPOCH',json.dumps(record),flush=True)
        if dev['mean_iou']>best_score+1e-6:
            best_score=dev['mean_iou'];best_epoch=epoch;best_state=state_cpu(model.mask_decoder);stale=0
            torch.save({'decoder':best_state,'epoch':epoch,'dev_mean_iou':best_score,'config':config},out/'best_prompted_decoder.pt')
        else:stale+=1
        if stale>=4:break
    train_seconds=time.perf_counter()-began
    assert first_gradient>0 and module_sha(model.mask_decoder)!=original_decoder_sha
    model.mask_decoder.load_state_dict(best_state)
    final_rows,regions,final=evaluate(parts['dev'],'finetuned_prompted',save=True)
    assert abs(final['mean_iou']-best_score)<1e-6
    _,_,training=evaluate(parts['train'],'best_training')
    write_csv(out/'finetuned_prompted_dev_metrics.csv',final_rows)
    write_csv(out/'per_annotation_recall.csv',regions)
    for name,digest in frozen_before.items():assert module_sha(getattr(model,name))==digest
    for record in prompt_records:assert sha(prompt_dir/(record['dish_id']+'.json'))==record['prompt_sha256']
    summary={'config':config,'previous_no_prompt_development':prior['finetuned_development'],
        'original_prompted_development':original_prompted,'existing_decoder_added_prompts_development':added_prompts,
        'finetuned_prompted_development':final,'finetuned_prompted_training':training,
        'best_epoch':best_epoch,'epochs_completed':len(history),'first_decoder_gradient_norm':first_gradient,
        'decoder_changed':True,'image_encoder_unchanged':True,'prompt_encoder_unchanged':True,
        'all_prompt_files_unchanged_during_training':True,'prompt_generation_seconds':prompt_seconds,
        'training_and_epoch_validation_seconds':train_seconds,'test_accessed':False,
        'prompt_fallback_counts':dict(Counter(p['fallback_reason'] or 'none' for p in prompts.values())),
        'best_decoder_file_sha256':sha(out/'best_prompted_decoder.pt'),
        'warning':'Development-selected automatic-prompt experiment; no independent test or nutrient prediction evidence.'}
    (out/'results.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('FINAL',json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    main()
