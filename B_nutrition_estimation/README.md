# B - Nutrition estimation

Component workspace for frozen CLIP features and a trainable nutrient regression head.

## Status
- Done: median baseline ([001](../experiments/001-median-baseline/README.md), 0.737), CLIP features for the original photos, regression head on those features ([002](../experiments/002-clip-head-original/README.md), 0.360; [003](../experiments/003-regularization/README.md), 0.354 with dropout).
- Next: food-only (masked) photos from part A through the same steps, for the original-vs-masked comparison.

Models are compared by the **normalized nutrient error**: for energy, protein, carbohydrate and fat, the average error divided by a typical dish's amount (the training average), averaged over the four. Read it as a percentage of a typical dish: the median baseline scores 0.737, about 74% off; lower is better, 0 is perfect. Results are also reported per nutrient in real units (kcal, g).

## CLIP features
[B_nutrition_estimation/extract_clip_features.py](../B_nutrition_estimation/extract_clip_features.py) turns each photo into 512 numbers that describe what it shows. CLIP ViT-B/32 (OpenAI weights, via OpenCLIP) is frozen: it is run once per photo and never trained. The regression head then learns from these numbers instead of raw pixels, so training takes seconds and needs far fewer examples (see d2l.ai 14.2 on pretrained models).

| | |
|---|---|
| Input | `<nutrition5k>/imagery/realsense_overhead/<dish_id>/rgb.png` (640 × 480), all 3,262 Cafe 1 dishes with an overhead photo |
| Preprocessing | Pad to a square with CLIP's mean colour (nothing is cropped, so portion size is kept), resize to 224 × 224, CLIP normalization |
| Output | `<nutrition5k>/features/clip_vitb32_<name>.npz`: `dish_ids` (3,262), `features` (3,262 × 512), `settings` (model, weights, preprocessing). Not in Git. |
| Run | `python B_nutrition_estimation/extract_clip_features.py` (original photos); `--image-dir <folder> --name masked` for food-only photos from part A |
| Time | About 5 seconds per 64 photos on a laptop processor (about 4–5 minutes in total if the computer stays awake) |

Features are extracted for every dish, so they stay valid if the split files change; training selects dishes using the split files.

## Deliverables
- Extract 512-dimensional features with frozen CLIP ViT-B/32 for original and SAM-masked image experiments.
- Train a 512 -> 128 (ReLU) -> 5 (Softplus) regression head using Nutrition5k labels.
- Predict mass, energy, protein, carbohydrate, and fat.
- Record the median baseline, original-versus-masked comparison, three training seeds, checkpoints and nutrient error metrics.

## Interface
Input: one RGB image, masked by component A in the proposed pipeline or original in the controlled comparison.
Output: estimated mass (g), energy (kcal), protein (g), carbohydrate (g) and fat (g) for component C.

Component B implementation lives here: `extract_clip_features.py`, `train_head.py`,
and `baseline_median.py`. This component-local layout follows the project owner's
request. Shared data preparation and evaluation remain in `../src/`, and run
records remain in `../experiments/`. Use shared group-separated splits. Do not
commit datasets, model weights or private identity mappings.
