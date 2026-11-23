# A-03-finetuned-no-prompt - Decoder fine-tuning without spatial prompts

Status: Completed (historical measured execution; code subsequently reorganized)

## Question and setup

- Contributor (GitHub handle or shorthand): `YifeiDorthy`.
- Execution date and time zone: 2026-11-20, America/Chicago. Exact start time was not captured.
- Hypothesis and technical change: compare this condition against the same SAM architecture and development subset. No external point, box or mask prompt; SAM retains its internal learned no-mask representation.
- Baseline/source and matched conditions: official SAM ViT-B; same image preprocessing, single-mask output head, float16-stored features cast to float32 and logit threshold 0.
- Code commit SHA: the experiment code was uncommitted at execution. Original per-file SHA256 values are retained in [summary.json](../results/summary.json). The commit publishing these reorganized sources is not claimed to be the historical execution commit.
- Environment and dependency versions: Python 3.9.25; torch 2.8.0+cu128; torchvision 0.23.0+cu128; NumPy 2.0.2; Pillow 11.3.0; OpenCV 4.11.0; SAM source `dca509fe793f601edb92606367a655c15ac00fdf`.
- Hardware and time budget: one RTX PRO 6000 Blackwell GPU. Selected epoch 12 after 12 completed epochs; maximum 12, early-stopping patience 4. Cached component timings are not end-to-end inference latency.
- Exact historical command: shown below, with machine/account paths replaced by placeholders. This record covers the **training and selected-checkpoint phase of the combined no-prompt trial**.
- Configuration path/settings: [comparison](../COMPARISON.md) and [exact result summary](../results/summary.json). Training, when applicable: AdamW lr 1e-4, weight decay .01, accumulation 4, clipping 1, BCE plus soft Dice, best development mean IoU.
- Dataset/subset: Nutrition5K overhead RGBs and Hand et al. visible-ingredient masks, annotation commit `0997586b5c177fa8f2c0a61eb7f958ab18605362`. All nonzero labels form the binary food target.
- Split/sample version: team commit `aec9a6c0f6ef6826185529602a7c6af72b638a80`; 454 train / 103 dev, with 103 evaluated dishes in this condition. Train manifest SHA256 `6912e399315c44f966215704ce179790c454d0d43697457a758cad635ccba77c`; dev manifest SHA256 `fa5fccc8164a587fa8ec411b46fed431bedaa330751cf0bced33990b178fc6c0`. See [protocol](../protocol/README.md).
- Random seed: 539. No stratification or threshold was fitted using final-test results.
- Preprocessing and fitted transformations: original aspect-preserving SAM resize/padding and original-size mask restoration; no training of an image normalization transform. Prompts, where applicable, are generated for all approved samples before any human mask pixels are opened and remain fixed.
- Components and interfaces: RGB in; same-size binary food mask and gray-background RGB out. Public official weights and team decoder access are documented in [ARTIFACTS.md](../ARTIFACTS.md).
- Loss, targets, updated/frozen parameters: all visible food is foreground; Only the mask decoder is updated; image and prompt encoders are frozen. Selected epoch 12 after 12 completed epochs; maximum 12, early-stopping patience 4.
- Metrics: IoU, precision, recall and Dice are ratios in [0,1]; image-macro means. Pass requires IoU >= .85 AND precision/recall >= .90; fail means IoU < .50; otherwise partial. This is a project convention, not a paper criterion.

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="<repository>/python_deps" \
  <gpu-python> -u -B A_food_segmentation/debug_validation/prompt_free_sam/run_experiment.py \
  --project-root <repository> \
  --output-dir <repository>/A_food_segmentation/debug_validation/outputs/prompt_free_sam_v1 \
  --device cuda --max-epochs 12 --patience 4 --skip-figures
```

Current reproducible command (run data preparation first; use a new output directory):

```bash
python A_food_segmentation/train_decoder.py \
  --output-dir A_food_segmentation/_local/outputs/no_prompt_trial \
  --device cuda --max-epochs 12 --patience 4 --skip-figures
```

This driver evaluates the original no-prompt control before decoder training. The current command is supplied for reproduction, not described as a new full training run performed during reorganization.

## Evidence

- Training logs and curves: [Epoch log and learning curves as CSV](../results/history_no_prompt.csv). Complete raw logs remain with the external run artifacts.
- Metrics: [03_finetuned_no_prompt.csv](../results/03_finetuned_no_prompt.csv); [summary](../results/summary.json).
- Predictions and labels: per-dish TP/FP/FN counts and metrics are in the CSV; full predicted masks and the public reference masks remain external/local-only. No raw RGB is published in this change.
- End-to-end/intermediate example: previously used dev dish `dish_1559590005`; [packaging checks](../results/packaging_checks.json). The selected fine-tuned no-prompt and prompted fresh RGB-only checks reproduce their saved predictions without labels or saved cues.
- External checkpoints/artifacts and access instructions: [ARTIFACTS.md](../ARTIFACTS.md), with model hashes in [summary.json](../results/summary.json).

## Outcome and interpretation

Mean IoU 0.971716384, precision 0.986275913, recall 0.985025725, Dice 0.985604820. Ratings: {'pass': 103}.

These are development-selected results. Whole-image pass can hide small ingredient omissions; inferred plate groups may miss physical links; no final test or nutrient-prediction improvement is claimed. The fixed four-method comparison and extra control are interpreted in [COMPARISON.md](../COMPARISON.md). Next: teammate review and a separately frozen final-evaluation/integration protocol.
