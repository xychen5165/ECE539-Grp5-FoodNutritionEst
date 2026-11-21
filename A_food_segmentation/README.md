# A - Food segmentation

The primary implementation is **SAM ViT-B with a fine-tuned mask decoder and no
external point, box or mask prompts**. The image and prompt encoders are frozen.
This is a supervised adaptation of the proposal's frozen-SAM starting point.
Four controlled method variants are kept in [method_comparison/](method_comparison/README.md).

The [contribution-guide assessment](method_comparison/CONTRIBUTING_REVIEW.md)
records technical checks, the requested component-local layout exception and
remaining teammate/course-policy review items.

## Primary method: run one image

From the repository root, using an environment with [requirements.txt](requirements.txt):

```bash
python A_food_segmentation/segment_food.py \
  --image /path/to/meal.png \
  --output-dir A_food_segmentation/_local/outputs/my_meal
```

The default checkpoints are local-only:

```text
A_food_segmentation/_local/checkpoints/sam_vit_b_01ec64.pth
A_food_segmentation/_local/checkpoints/food_decoder_no_prompt.pt
```

The current workspace has these weights available. A fresh checkout needs the
official SAM ViT-B checkpoint and the team's trained decoder, or a newly trained
decoder. See [artifact access](method_comparison/ARTIFACTS.md) for authorized
copy/reproduction steps. Weight files are not committed. [model_info.json](model_info.json)
records the reference hashes and training configuration. Different retraining
runs can legitimately produce a different decoder hash.

Use `--sam-checkpoint` / `--decoder-checkpoint` to supply other locations.
`--device auto` selects CUDA, then MPS, then CPU; an explicit device is also
accepted. Existing output directories are rejected to preserve earlier runs.

Outputs are `mask.png` (0/255), `masked_rgb.png` (background RGB 128),
`overlay.png`, and `prediction.json`. Dimensions match the original image, and
food pixels retain their original RGB values. This is the image interface for
Part B. No annotation, cached feature, nutrient label or human click is needed
at inference time. `sam_runtime.infer_rgb` exposes the same logic as a Python API.

## Setup and training

For a new environment:

```bash
python -m pip install -r A_food_segmentation/requirements.txt
mkdir -p A_food_segmentation/_local/checkpoints
curl -fL https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth \
  -o A_food_segmentation/_local/checkpoints/sam_vit_b_01ec64.pth
```

Prepare only the approved train/dev RGBs and human masks:

```bash
python A_food_segmentation/prepare_data.py
```

Train the primary method (use a new output directory):

```bash
python A_food_segmentation/train_decoder.py \
  --output-dir A_food_segmentation/_local/outputs/no_prompt_trial \
  --device cuda --max-epochs 12 --patience 4 --skip-figures
```

The best decoder is saved as `best_decoder.pt` inside that run. To use it,
pass it explicitly to `segment_food.py --decoder-checkpoint ...`; a training run
does not silently replace the default selected checkpoint.

The 454 train / 103 dev manifests are pinned under
[method_comparison/protocol/](method_comparison/protocol/README.md). Test and
test-linked plates are excluded from training/tuning. `prepare_data.py` verifies
the public annotation index and does not download test pixels.

## Method comparison and status

[COMPARISON.md](method_comparison/COMPARISON.md) documents the four methods,
measured development results, training settings and limitations. Method 3's
test entry calls this same primary implementation rather than maintaining a
second copy of the model logic.

Reported scores are **development** evidence, not final test or nutrient
prediction results. A whole-image pass can still hide small ingredient omissions.
Data, model weights, caches and generated images belong under `_local/`, which
is ignored by Git. The previous exploration data remains local-only and is not
part of this branch's public implementation.
