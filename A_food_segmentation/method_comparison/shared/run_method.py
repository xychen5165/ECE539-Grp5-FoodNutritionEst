"""Route all four public experiment entries through the same inference code."""
from pathlib import Path
import sys

COMPONENT_ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(COMPONENT_ROOT))


def automatic_prompt_factory(model,rgb):
    from segment_anything import SamAutomaticMaskGenerator
    from automatic_prompts import derive_prompts
    device=next(model.parameters()).device
    # Original SAM's automatic grid constructs float64 tensors, unsupported by
    # MPS. Keep the official generator unchanged and run that stage on CPU;
    # move the original model back before the final decoder pass.
    generation_device='cpu' if device.type=='mps' else str(device)
    if device.type=='mps':model.to('cpu')
    try:
        generator=SamAutomaticMaskGenerator(model,points_per_side=16,points_per_batch=64,
            pred_iou_thresh=.86,stability_score_thresh=.92,crop_n_layers=0,min_mask_region_area=100)
        prompt=derive_prompts(rgb,generator.generate(rgb))
    finally:
        if device.type=='mps':model.to(device)
    prompt['generation_device']=generation_device
    return prompt


def main(mode):
    from sam_runtime import run_cli
    factory=automatic_prompt_factory if mode.endswith('auto_prompt') else None
    run_cli(mode,prompt_factory=factory)
