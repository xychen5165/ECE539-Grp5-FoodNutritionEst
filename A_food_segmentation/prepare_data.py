"""Download only protocol-approved train/dev RGBs and pinned human masks."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import io
import json
from pathlib import Path
import time
import urllib.request

from PIL import Image
from data_utils import ANNOTATION_COMMIT, inputs, paths, load_pair, sha


def download(url,target):
    if target.exists():
        with Image.open(target) as image:
            image.verify()
        return 0
    for attempt in range(3):
        try:
            request=urllib.request.Request(url,headers={'User-Agent':'ECE539-PartA-research'})
            data=urllib.request.urlopen(request,timeout=45).read()
            with Image.open(io.BytesIO(data)) as image:
                image.verify()
            target.parent.mkdir(parents=True,exist_ok=True)
            partial=target.with_suffix(target.suffix+'.part')
            partial.write_bytes(data)
            partial.replace(target)
            return len(data)
        except Exception:
            if attempt==2:
                raise
            time.sleep(attempt+1)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root',type=Path,default=Path(__file__).resolve().parent.parent)
    parser.add_argument('--record',type=Path)
    args=parser.parse_args()
    base=args.project_root/'A_food_segmentation'
    annotations=base/'_local/data/hand_annotations'
    annotations.mkdir(parents=True,exist_ok=True)
    index=annotations/'ingredient_nutrition.csv'
    if not index.exists():
        url=f'https://raw.githubusercontent.com/liamhand/ingredient-level-nutrition-regression/{ANNOTATION_COMMIT}/data/ingredient_nutrition.csv'
        index.write_bytes(urllib.request.urlopen(url,timeout=45).read())
    base,protocol,rows,annotations,by_dish=inputs(args.project_root)
    args.record=args.record or base/'_local/data/downloads.json'
    selected=rows['train']+rows['dev']
    def one(row):
        rgb_path,mask_path=paths(base,annotations,row)
        rgb_url='https://storage.googleapis.com/nutrition5k_dataset/nutrition5k_dataset/'+row['image_path']
        mask_url=f'https://raw.githubusercontent.com/liamhand/ingredient-level-nutrition-regression/{ANNOTATION_COMMIT}/data/'+row['annotation_mask']
        added=download(rgb_url,rgb_path)+download(mask_url,mask_path)
        rgb,labels=load_pair(base,annotations,row,by_dish[row['dish_id']])
        return {'dish_id':row['dish_id'],'split':row['project_split'],
                'rgb_sha256':sha(rgb_path),'reference_sha256':sha(mask_path),
                'rgb_url':rgb_url,'mask_url':mask_url,'width':rgb.shape[1],
                'height':rgb.shape[0],'downloaded_bytes':added}
    records=[]; started=time.perf_counter()
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures={pool.submit(one,row):row['dish_id'] for row in selected}
        for future in as_completed(futures):
            records.append(future.result())
            if len(records)%25==0 or len(records)==len(selected):
                print(f'Downloaded/verified {len(records)}/{len(selected)} pairs in {time.perf_counter()-started:.1f}s',flush=True)
    assert len(records)==557
    args.record.parent.mkdir(parents=True,exist_ok=True)
    args.record.write_text(json.dumps({'test_accessed':False,'pairs':sorted(records,key=lambda r:r['dish_id']),
                           'source_commit':ANNOTATION_COMMIT,'seconds':time.perf_counter()-started},indent=2)+'\n')
    print('All 454 train and 103 dev pairs ready. No test images or masks downloaded.',flush=True)


if __name__=='__main__':
    main()
