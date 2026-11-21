"""Zero-shot no-prompt SAM vs decoder-only supervised no-prompt fine-tuning.

All evaluations use the approved development split. Never reads test pixels.
Image and prompt encoders are frozen; SAM's single-mask output is used in both
conditions. No automatic grid, bounding box, point, input mask or food metadata
enters either forward pass. Reference masks only enter losses and metrics.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import platform
import random
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn.functional as F
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide

from data_utils import CHECKPOINT_SHA, inputs, load_pair, metrics, paths, rating, sha, write_csv
from sam_runtime import resolve_device


def synchronize(device):
    if device=='mps':
        torch.mps.synchronize()
    elif device=='cuda':
        torch.cuda.synchronize()


def state_cpu(module):
    return {k:v.detach().cpu().clone() for k,v in module.state_dict().items()}


def module_sha(module):
    import hashlib
    h=hashlib.sha256()
    for name,value in module.state_dict().items():
        h.update(name.encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def choose_examples(rows):
    # Fix examples from metadata before any model result is available.
    chosen=[]; seen=set()
    for lower,upper in [(1,1),(2,4),(5,1000)]:
        count=0
        for row in rows:
            if lower<=int(row['annotation_regions'])<=upper and row['plate_group'] not in seen:
                chosen.append(row['dish_id']); seen.add(row['plate_group']); count+=1
                if count==2:
                    break
    return chosen


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root',type=Path,default=Path(__file__).resolve().parent.parent)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--device',choices=['auto','cpu','mps','cuda'],default='auto')
    parser.add_argument('--sam-checkpoint',type=Path)
    parser.add_argument('--data-record',type=Path)
    parser.add_argument('--max-epochs',type=int,default=12)
    parser.add_argument('--patience',type=int,default=4)
    parser.add_argument('--lr',type=float,default=1e-4)
    parser.add_argument('--skip-figures',action='store_true',help='Render figures later on a machine with the CJK font')
    args=parser.parse_args()
    args.device=resolve_device(args.device)
    base,protocol,rows,annotations,by_dish=inputs(args.project_root)
    out=args.output_dir
    out.mkdir(parents=True,exist_ok=True)
    if (out/'config.json').exists():
        parser.error('This run is already configured; use a new run folder')
    record=args.data_record or base/'_local/data/downloads.json'
    assert record.exists(), 'Run A_food_segmentation/prepare_data.py first'
    if record.resolve()!=(out/'downloads.json').resolve():
        (out/'downloads.json').write_bytes(record.read_bytes())
    download=json.loads((out/'downloads.json').read_text())
    download_records={r['dish_id']:r for r in download['pairs']}
    assert len(download_records)==557
    selected=rows['train']+rows['dev']
    assert set(download_records)=={r['dish_id'] for r in selected}
    checkpoint=args.sam_checkpoint or base/'_local/checkpoints/sam_vit_b_01ec64.pth'
    assert sha(checkpoint)==CHECKPOINT_SHA
    torch.manual_seed(539); np.random.seed(539); random.seed(539)
    model=sam_model_registry['vit_b'](checkpoint=str(checkpoint)).to(args.device).eval()
    for param in model.parameters():
        param.requires_grad=False
    for param in model.mask_decoder.parameters():
        param.requires_grad=True
    frozen_before={'image_encoder':module_sha(model.image_encoder),'prompt_encoder':module_sha(model.prompt_encoder)}
    decoder_before=module_sha(model.mask_decoder)
    transform=ResizeLongestSide(model.image_encoder.img_size)
    examples=choose_examples(rows['dev'])
    config={'method_1':'pretrained_vit_b_no_external_prompts',
        'method_2':'decoder_only_finetuned_vit_b_no_external_prompts',
        'seed':539,'device':args.device,'torch_version':str(torch.__version__),
        'python_version':platform.python_version(),'cuda_runtime':torch.version.cuda,
        'gpu_name':torch.cuda.get_device_name(0) if args.device=='cuda' else None,
        'max_epochs':args.max_epochs,'early_stopping_patience':args.patience,'learning_rate':args.lr,
        'weight_decay':.01,'gradient_accumulation':4,'gradient_clip_norm':1.0,
        'loss':'full-original-resolution BCEWithLogits + soft Dice; equal coefficients',
        'selection':'highest development macro image IoU; earliest epoch breaks ties',
        'logit_threshold':0.0,'external_points':None,'external_boxes':None,'external_mask_input':None,
        'multimask_output':False,'embedding_cache_dtype':'float16 storage, float32 decoder input',
        'image_encoder_frozen':True,'prompt_encoder_frozen':True,
        'train_count':454,'dev_count':103,'test_accessed':False,
        'project_split_commit':protocol['project_split_commit'],'annotation_commit':protocol['annotation_commit'],
        'checkpoint_sha256':CHECKPOINT_SHA,'thresholds':protocol['thresholds'],
        'manifest_sha256':{name:protocol['generated_csv_sha256'][name] for name in ['train_development.csv','dev_tuning.csv']},
        'code_sha256':{p.name:sha(p) for p in Path(__file__).parent.glob('*.py')},
        'visual_examples_fixed_before_evaluation':examples}
    (out/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    cache=out/'embedding_cache'; cache.mkdir(exist_ok=True)
    began=time.perf_counter()
    cache_records=[]
    for index,row in enumerate(selected,1):
        rgb_path,mask_path=paths(base,annotations,row)
        assert sha(rgb_path)==download_records[row['dish_id']]['rgb_sha256']
        assert sha(mask_path)==download_records[row['dish_id']]['reference_sha256']
        rgb,labels=load_pair(base,annotations,row,by_dish[row['dish_id']])
        cached=cache/(row['dish_id']+'.npz')
        # No GT mask is read by the image encoder or either prompt encoder call.
        transformed=transform.apply_image(rgb)
        tensor=torch.as_tensor(transformed.copy(),device=args.device).permute(2,0,1)[None]
        with torch.no_grad():
            embedding=model.image_encoder(model.preprocess(tensor))
        synchronize(args.device)
        np.savez(cached,embedding=embedding.detach().cpu().numpy().astype(np.float16),
                 gt=(labels>0).astype(np.uint8),input_size=np.array(transformed.shape[:2]),
                 original_size=np.array(rgb.shape[:2]))
        cache_records.append({'dish_id':row['dish_id'],'split':row['project_split'],
            'embedding_cache_sha256':sha(cached),'rgb_sha256':sha(rgb_path),'reference_sha256':sha(mask_path)})
        del tensor,embedding
        if index%25==0 or index==len(selected):
            print(f'Cached {index}/{len(selected)} frozen image embeddings; elapsed {time.perf_counter()-began:.1f}s',flush=True)
    cache_seconds=time.perf_counter()-began
    (out/'cache_provenance.json').write_text(json.dumps(cache_records,indent=2)+'\n')
    with torch.no_grad():
        sparse,dense=model.prompt_encoder(points=None,boxes=None,masks=None)
        pe=model.prompt_encoder.get_dense_pe()
    assert sparse.shape[1]==0, 'Unexpected sparse prompt tokens'
    optimizer=torch.optim.AdamW(model.mask_decoder.parameters(),lr=args.lr,weight_decay=.01)

    def predict(row):
        with np.load(cache/(row['dish_id']+'.npz'),allow_pickle=False) as record:
            feature=torch.as_tensor(record['embedding'].astype(np.float32),device=args.device)
            gt=torch.as_tensor(record['gt'].copy(),device=args.device,dtype=torch.float32)[None,None]
            input_size=tuple(map(int,record['input_size']))
            original_size=tuple(map(int,record['original_size']))
        low,_=model.mask_decoder(image_embeddings=feature,image_pe=pe,
            sparse_prompt_embeddings=sparse,dense_prompt_embeddings=dense,multimask_output=False)
        logits=model.postprocess_masks(low,input_size=input_size,original_size=original_size)
        return logits,gt

    def evaluate(selected_rows,name,save_masks=False):
        model.mask_decoder.eval()
        records=[]; started=time.perf_counter()
        with torch.no_grad():
            for row in selected_rows:
                logits,target=predict(row)
                pred=(logits[0,0]>0).cpu().numpy()
                gt=target[0,0].bool().cpu().numpy()
                m=metrics(pred,gt)
                # Region recall diagnostics use labels only AFTER prediction.
                _,labels=load_pair(base,annotations,row,by_dish[row['dish_id']])
                min_region_recall=min(float((pred & (labels==int(a['label_value']))).sum()/int(a['mask_pixels']))
                                      for a in by_dish[row['dish_id']])
                records.append({'dish_id':row['dish_id'],'plate_group':row['plate_group'],**m,
                    'rating':rating(m),'minimum_annotation_recall':min_region_recall})
                if save_masks:
                    folder=out/name/row['dish_id']; folder.mkdir(parents=True,exist_ok=True)
                    Image.fromarray(pred.astype(np.uint8)*255).save(folder/'mask.png')
        synchronize(args.device)
        grouped=defaultdict(list)
        for r in records:
            grouped[r['plate_group']].append(r['iou'])
        summary={'count':len(records),'plate_group_count':len(grouped),
            **{'mean_'+key:float(np.mean([r[key] for r in records])) for key in ['iou','precision','recall','dice']},
            'plate_macro_mean_iou':float(np.mean([np.mean(v) for v in grouped.values()])),
            'rating_counts':dict(Counter(r['rating'] for r in records)),
            'seconds_excluding_image_encoding':time.perf_counter()-started}
        return records,summary

    baseline_rows,baseline=evaluate(rows['dev'],'zero_shot',save_masks=True)
    write_csv(out/'zero_shot_dev_metrics.csv',baseline_rows)
    (out/'zero_shot_summary.json').write_text(json.dumps(baseline,indent=2)+'\n')
    print('ZERO-SHOT DEVELOPMENT',json.dumps(baseline),flush=True)
    # Select the best TRAINED epoch, even if every trained epoch is worse than baseline.
    best_score=float('-inf'); best_epoch=None; best_state=None
    history=[]; stale=0; trained=time.perf_counter(); first_gradient_norm=None
    for epoch in range(1,args.max_epochs+1):
        model.mask_decoder.train()
        order=list(rows['train']); random.Random(539+epoch).shuffle(order)
        loss_sum=0.0; epoch_started=time.perf_counter()
        for batch_start in range(0,len(order),4):
            batch=order[batch_start:batch_start+4]
            optimizer.zero_grad(set_to_none=True)
            for row in batch:
                logits,gt=predict(row)
                prob=logits.sigmoid()
                dice=1-(2*(prob*gt).sum()+1)/(prob.sum()+gt.sum()+1)
                loss=F.binary_cross_entropy_with_logits(logits,gt)+dice
                assert torch.isfinite(loss), 'Non-finite training loss'
                (loss/len(batch)).backward()
                loss_sum+=float(loss.detach().cpu())
            norm=torch.nn.utils.clip_grad_norm_(model.mask_decoder.parameters(),max_norm=1.0)
            assert torch.isfinite(norm), 'Non-finite gradients'
            if first_gradient_norm is None:
                first_gradient_norm=float(norm.detach().cpu())
                assert first_gradient_norm>0
            optimizer.step()
        _,development=evaluate(rows['dev'],'epoch_'+str(epoch))
        record={'epoch':epoch,'train_loss':loss_sum/len(order),
                'dev_iou':development['mean_iou'],'dev_precision':development['mean_precision'],
                'dev_recall':development['mean_recall'],'dev_dice':development['mean_dice'],
                'seconds':time.perf_counter()-epoch_started}
        history.append(record); write_csv(out/'history.csv',history)
        print('EPOCH',json.dumps(record),flush=True)
        if development['mean_iou']>best_score+1e-6:
            best_score=development['mean_iou']; best_epoch=epoch
            best_state=state_cpu(model.mask_decoder); stale=0
            torch.save({'decoder':best_state,'epoch':epoch,'dev_mean_iou':best_score,'config':config},out/'best_decoder.pt')
        else:
            stale+=1
        if stale>=args.patience:
            print('Early stopping: no development IoU improvement.',flush=True)
            break
    training_seconds=time.perf_counter()-trained
    trained_decoder_sha=module_sha(model.mask_decoder)
    assert trained_decoder_sha!=decoder_before, 'Decoder parameters never changed'
    model.mask_decoder.load_state_dict(best_state)
    final_rows,finetuned=evaluate(rows['dev'],'finetuned',save_masks=True)
    assert abs(finetuned['mean_iou']-best_score)<1e-6
    _,train_summary=evaluate(rows['train'],'best_training')
    write_csv(out/'finetuned_dev_metrics.csv',final_rows)
    assert module_sha(model.image_encoder)==frozen_before['image_encoder']
    assert module_sha(model.prompt_encoder)==frozen_before['prompt_encoder']
    summary={'config':config,'zero_shot_development':baseline,'finetuned_development':finetuned,
        'finetuned_training':train_summary,'best_epoch':best_epoch,'epochs_completed':len(history),
        'first_decoder_gradient_norm':first_gradient_norm,
        'decoder_parameters_updated':True,'image_encoder_unchanged':True,'prompt_encoder_unchanged':True,
        'initial_decoder_sha256':decoder_before,'last_trained_decoder_sha256':trained_decoder_sha,
        'best_decoder_sha256':module_sha(model.mask_decoder),'cache_seconds':cache_seconds,
        'training_and_epoch_validation_seconds':training_seconds,'test_accessed':False,
        'warning':'Development-selected experiment, not independent final test performance; no nutrient-regression evaluation.'}
    (out/'results.json').write_text(json.dumps(summary,indent=2)+'\n')
    if not args.skip_figures:
        make_figures(base,annotations,by_dish,rows['dev'],out,examples,baseline_rows,final_rows)
    print('FINAL',json.dumps(summary,indent=2),flush=True)


def make_figures(base,annotations,by_dish,dev_rows,out,examples,baseline_rows,final_rows):
    candidates=['/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf','/System/Library/Fonts/Supplemental/Arial.ttf']
    font_path=next((p for p in candidates if Path(p).exists()),None)
    font=ImageFont.truetype(font_path,18) if font_path else ImageFont.load_default(size=18)
    small=ImageFont.truetype(font_path,15) if font_path else ImageFont.load_default(size=15)
    lookup={r['dish_id']:r for r in dev_rows}
    scores=[{r['dish_id']:r for r in rs} for rs in [baseline_rows,final_rows]]
    for page in range((len(examples)+2)//3):
        ids=examples[page*3:(page+1)*3]
        canvas=Image.new('RGB',(1308,1030),'#f4f6fa'); d=ImageDraw.Draw(canvas)
        d.text((12,8),'No external prompts: pretrained vs decoder fine-tuning (development)',font=font,fill='#172033')
        d.text((12,38),'Red = false positive; blue = false negative; examples fixed before evaluation.',font=small,fill='#405066')
        for i,title in enumerate(['Original RGB','Human food reference','Pretrained SAM','Fine-tuned decoder']):
            d.text((12+326*i,67),title,font=small,fill='#172033')
        for i,dish in enumerate(ids):
            y=100+i*295
            rgb,labels=load_pair(base,annotations,lookup[dish],by_dish[dish]); gt=labels>0
            reference=np.full_like(rgb,128); reference[gt]=rgb[gt]
            panels=[rgb,reference]
            for name in ['zero_shot','finetuned']:
                pred=np.asarray(Image.open(out/name/dish/'mask.png'))>0
                color=np.full_like(rgb,128); color[pred & gt]=rgb[pred & gt]
                color[pred & ~gt]=[240,55,65]; color[~pred & gt]=[45,115,255]
                panels.append(color)
            d.text((12,y),dish,font=small,fill='#172033')
            for col,array in enumerate(panels):
                canvas.paste(Image.fromarray(array).resize((320,240),Image.Resampling.LANCZOS),(12+326*col,y+25))
            for col,score in enumerate(scores,2):
                r=score[dish]
                d.text((12+326*col,y+268),f"IoU {r['iou']:.3f}  P {r['precision']:.3f}  R {r['recall']:.3f}  {r['rating']}",font=small,fill='#172033')
        d.text((12,1000),'Annotations: Hand et al. / Nutrition5K, CC BY 4.0. Development evidence; final test untouched.',font=small,fill='#526176')
        canvas.save(out/f'comparison_{page+1}.png')


if __name__=='__main__':
    main()
