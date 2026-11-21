"""Shared image-only SAM inference; no annotation files are read here."""
import argparse
import hashlib
import json
from pathlib import Path
import uuid

import numpy as np
from PIL import Image
import torch
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide

COMPONENT_ROOT = Path(__file__).resolve().parent


def resolve_device(requested='auto'):
    if requested!='auto':
        return requested
    if torch.cuda.is_available():
        return 'cuda'
    if hasattr(torch.backends,'mps') and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_sam(checkpoint,device):
    model=sam_model_registry['vit_b']()
    model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True))
    return model.to(device).eval()


def infer_rgb(rgb,sam_checkpoint,decoder_checkpoint=None,device='auto',prompt_factory=None):
    """A factory, if supplied, sees original SAM + RGB before decoder loading."""
    device=resolve_device(device)
    model=load_sam(sam_checkpoint,device)
    prompt=None
    if prompt_factory is not None:
        prompt=prompt_factory(model,rgb)
    selected_epoch=None
    if decoder_checkpoint is not None:
        checkpoint=torch.load(decoder_checkpoint,map_location='cpu',weights_only=True)
        if 'decoder' not in checkpoint:
            raise ValueError('Expected a decoder checkpoint produced by train_decoder.py')
        model.mask_decoder.load_state_dict(checkpoint['decoder'])
        selected_epoch=checkpoint.get('epoch')
    transform=ResizeLongestSide(model.image_encoder.img_size)
    resized=transform.apply_image(rgb)
    image=torch.as_tensor(resized.copy(),device=device).permute(2,0,1)[None]
    with torch.inference_mode():
        # Match the measured trials' float16 cache -> float32 decoder input.
        features=model.image_encoder(model.preprocess(image)).to(torch.float16).to(torch.float32)
        points=None;box=None
        if prompt is not None:
            coordinate_dtype=torch.float32 if torch.device(device).type=='mps' else None
            box=torch.as_tensor(transform.apply_boxes(np.asarray(prompt['box_xyxy'],np.float32)[None],rgb.shape[:2]),device=device,dtype=coordinate_dtype)
            if prompt['points_xy']:
                coords=transform.apply_coords(np.asarray(prompt['points_xy'],np.float32),rgb.shape[:2])
                points=(torch.as_tensor(coords,device=device,dtype=coordinate_dtype)[None],torch.as_tensor(prompt['point_labels'],device=device)[None])
        sparse,dense=model.prompt_encoder(points=points,boxes=box,masks=None)
        logits,_=model.mask_decoder(image_embeddings=features,image_pe=model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,dense_prompt_embeddings=dense,multimask_output=False)
        logits=model.postprocess_masks(logits,resized.shape[:2],rgb.shape[:2])
        mask=(logits[0,0]>0).cpu().numpy()
    return mask,{'device':device,'selected_epoch':selected_epoch,'prompt':prompt,
        'annotation_files_read':False,'feature_cache_read':False,'multimask_output':False,
        'logit_threshold':0.0,'embedding_storage_roundtrip':'float16 -> float32'}


def save_prediction(rgb,mask,output_dir,record):
    output_dir=Path(output_dir)
    output_dir.mkdir(parents=True,exist_ok=False)
    assert mask.shape==rgb.shape[:2]
    Image.fromarray(mask.astype(np.uint8)*255).save(output_dir/'mask.png')
    masked=np.full_like(rgb,128);masked[mask]=rgb[mask]
    Image.fromarray(masked).save(output_dir/'masked_rgb.png')
    overlay=rgb.copy()
    overlay[mask]=(rgb[mask].astype(np.float32)*.6+np.array([0,200,80])*.4).astype(np.uint8)
    Image.fromarray(overlay).save(output_dir/'overlay.png')
    record.update(width=rgb.shape[1],height=rgb.shape[0],foreground_pixels=int(mask.sum()))
    (output_dir/'prediction.json').write_text(json.dumps(record,indent=2)+'\n')


def run_cli(mode='finetuned_no_prompt',prompt_factory=None,argv=None):
    parser=argparse.ArgumentParser(description=f'SAM food segmentation: {mode}')
    parser.add_argument('--image',type=Path,required=True)
    parser.add_argument('--sam-checkpoint',type=Path,default=COMPONENT_ROOT/'_local/checkpoints/sam_vit_b_01ec64.pth')
    default_decoder=None
    if mode=='finetuned_no_prompt':default_decoder=COMPONENT_ROOT/'_local/checkpoints/food_decoder_no_prompt.pt'
    if mode=='finetuned_auto_prompt':default_decoder=COMPONENT_ROOT/'_local/checkpoints/food_decoder_prompted.pt'
    parser.add_argument('--decoder-checkpoint',type=Path,default=default_decoder)
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--device',choices=['auto','cpu','cuda','mps'],default='auto')
    args=parser.parse_args(argv)
    if mode.startswith('pretrained') and args.decoder_checkpoint is not None:
        parser.error('Pretrained controls do not accept a fine-tuned decoder checkpoint')
    for path in [args.image,args.sam_checkpoint]+([args.decoder_checkpoint] if args.decoder_checkpoint else []):
        if not path.is_file():parser.error(f'Missing input: {path}. See A_food_segmentation/README.md for setup.')
    output=args.output_dir or COMPONENT_ROOT/'_local/outputs'/f'{mode}_{uuid.uuid4().hex[:8]}'
    if output.exists():parser.error('Choose a new output directory; existing results are never overwritten')
    rgb=np.asarray(Image.open(args.image).convert('RGB'))
    mask,record=infer_rgb(rgb,args.sam_checkpoint,args.decoder_checkpoint,args.device,prompt_factory)
    record.update(method=mode,image_sha256=file_sha256(args.image),
        sam_checkpoint_sha256=file_sha256(args.sam_checkpoint),
        decoder_checkpoint_sha256=file_sha256(args.decoder_checkpoint) if args.decoder_checkpoint else None)
    save_prediction(rgb,mask,output,record)
    print(json.dumps({'output_dir':str(output.resolve()),**record},indent=2))
