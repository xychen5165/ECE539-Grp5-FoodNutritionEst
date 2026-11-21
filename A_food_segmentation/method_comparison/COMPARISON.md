# SAM food segmentation: four controlled methods

## Question and data

Compare original vs decoder-fine-tuned SAM ViT-B, each with and without external
spatial prompts. All conditions use the same 103 development dishes (81 inferred
plate groups), original RGB preprocessing and single-mask head. Decoder training
uses 454 dishes (357 groups). The final test is reserved.

The original Nutrition5K RGBs are paired with Hand et al.'s manually annotated
visible ingredient masks. Merge every nonzero label into a single food mask,
including regions without nutrient labels. Reference masks supervise losses and
metrics only; they are never inputs to image inference or automatic cue generation.

## Measured development results

| Method | Mean IoU | Precision | Recall | Pass / partial / fail |
|---|---:|---:|---:|---|
| 1. Pretrained SAM, no external prompts | approximately 0% | 0.97% | approximately 0% | 0 / 0 / 103 |
| 2. Pretrained SAM + automatic prompts | 71.90% | 78.98% | 79.15% | 62 / 18 / 23 |
| 3. Decoder fine-tuning, no prompts | **97.17%** | 98.63% | 98.50% | 103 / 0 / 0 |
| 4. Decoder fine-tuning + automatic prompts | **97.10%** | 98.56% | 98.49% | 103 / 0 / 0 |

These are unweighted per-image means. Exact values and per-image records are
under `results/`. Method 1's near-empty output is scoped to the measured
no-external-prompt, single-mask configuration, not every possible SAM head or
adaptation. The internal learned no-mask embedding remains part of SAM.

For this fixed recipe, method 4 does not improve overall dev IoU over method 3
(difference -0.0726 percentage points). Thirty-eight images improve and 65
worsen. The user selected method 3 for the Part A root implementation. This
is a development-based working choice, not a final generalization claim.

## What the automatic prompts contain

Original, frozen SAM generates candidates using a 16 x 16 point grid, predicted
IoU threshold .86, stability .92, no crop layers, minimum region 100 pixels.
Geometry and neutral-surface appearance estimate a plate. Structurally plausible
interior candidates and plate holes supply up to 16 deep positive points;
inferred bare plate supplies up to four negatives. One estimated plate box
covers the whole-dish target. No food-color gate or food-specific rule is added.

If plate estimation fails, use the full-image box without points. All 557 dishes
remain in the experiment. There were 56 such fallbacks in train and 13 in dev.
The final decoder mask is not clipped or vetoed by the geometry rules.

All prompt files were generated and hashed before opening human mask pixels,
and remained fixed during training. Afterwards, 95.66% of dev positives fell
on annotated food and 98.54% of dev negatives fell on annotated background.
These cues can still be wrong; diagnostics did not repair or remove them.

## Training and evaluation settings

Both fine-tuning conditions start anew from official SAM ViT-B. Freeze image
and prompt encoders; update the mask decoder only. Use full-resolution BCE plus
soft Dice, AdamW lr 1e-4, weight decay .01, accumulation 4, gradient clipping 1,
seed 539, up to 12 epochs, early stopping patience 4. Select the best trained
epoch by dev mean IoU. Method 3 selected epoch 12/12; method 4 selected epoch 5
and stopped after epoch 9. Both use float16-stored frozen features cast back to
float32 and logits > 0 for a binary prediction.

IoU = TP/(TP+FP+FN), precision = TP/(TP+FP), recall = TP/(TP+FN).
The provisional project rule is pass iff IoU >= .85 and precision/recall >= .90;
fail iff IoU < .50; otherwise partial. This is not a criterion from the paper.

Whole-image pass does not guarantee every small ingredient is preserved.
Method 3 has 4 of 282 annotated dev regions below 90% recall; method 4 has 5,
across 4 images. On the preselected dish `dish_1559590005`, a small
`alium_vegetable` region improves from 13.26% to 85.30% recall with method 4.
No downstream nutrient-prediction improvement has been measured.

## Extra inference-only ablation

Adding the automatic prompts directly to the already no-prompt-trained decoder
(without retraining) gives IoU 96.67%, pass 103/103, and 10 annotated regions
below 90% recall. This is retained in `results/extra_added_prompts.csv` and the
summary, but is not one of the four main train/prompt combinations.

## Evidence and limitations

The executed trials verified nonzero decoder gradients, changed decoder weights,
unchanged frozen encoder fingerprints, and unchanged prompt hashes. All saved
prediction metrics were independently recomputed. RGB-only fresh inference
reproduced the selected no-prompt and prompted predictions pixel for pixel.
No exact RGB hashes overlap between train and dev.

This is one-seed, development-selected evidence. Plate groups are inferred and
can miss physical links. Visible-food annotations do not cover invisible oils
or quantify mass. Test dishes and test-linked plates are excluded from tuning;
no final test scores are reported. Cached component timings are not a fair
end-to-end latency benchmark.

The historical server environment was Python 3.9.25, PyTorch 2.8.0+cu128,
torchvision 0.23.0+cu128, NumPy 2.0.2, Pillow 11.3.0, OpenCV 4.11.0, and one
RTX PRO 6000 Blackwell GPU. Exact source/checkpoint fingerprints are retained
in `results/summary.json`; packaging changes and checks are listed separately.

## Sources

- SAM: https://github.com/facebookresearch/segment-anything , source commit
  `dca509fe793f601edb92606367a655c15ac00fdf`.
- Nutrition5K: https://github.com/google-research-datasets/Nutrition5k , Thames
  et al., CC BY 4.0.
- Hand et al.: https://doi.org/10.3389/fnut.2026.1927530 ; annotation release
  https://github.com/liamhand/ingredient-level-nutrition-regression at
  `0997586b5c177fa8f2c0a61eb7f958ab18605362`, CC BY 4.0.
- Team split/grouping snapshot: `aec9a6c0f6ef6826185529602a7c6af72b638a80`
  from `xin-workspace`. The paper authors' split never overrides this assignment.
