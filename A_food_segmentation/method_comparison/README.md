# Four-method SAM comparison

The four method directories are separate runnable experiment entries. They
share `../sam_runtime.py` and `shared/` so preprocessing, output format and
thresholds stay matched. The selected primary method is also exposed directly
as `../segment_food.py` and `../train_decoder.py`.

| Directory | Weights | Spatial prompts |
|---|---|---|
| `01_pretrained_no_prompt/` | Original SAM | None |
| `02_pretrained_auto_prompt/` | Original SAM | Automatically generated box and signed points |
| `03_finetuned_no_prompt/` | Decoder fine-tuned without prompts | None |
| `04_finetuned_auto_prompt/` | Decoder fine-tuned with automatic prompts | Same frozen automatic prompt recipe |

## Predict with any variant

After following the parent README's setup, from the repository root:

```bash
python A_food_segmentation/method_comparison/01_pretrained_no_prompt/predict.py --image /path/to/meal.png
python A_food_segmentation/method_comparison/02_pretrained_auto_prompt/predict.py --image /path/to/meal.png
python A_food_segmentation/method_comparison/03_finetuned_no_prompt/predict.py --image /path/to/meal.png
python A_food_segmentation/method_comparison/04_finetuned_auto_prompt/predict.py --image /path/to/meal.png
```

Each writes a new, separately named run under `../_local/outputs/`. All accept
`--device`, `--sam-checkpoint`, `--output-dir`; fine-tuned variants also accept
`--decoder-checkpoint`. Method 4's default local decoder filename is
`food_decoder_prompted.pt`; method 3 uses `food_decoder_no_prompt.pt`.
Methods 1 and 2 require only original SAM weights and do not train anything.

For prompted inference, original SAM generates candidates and cues BEFORE
loading a fine-tuned decoder. Human annotations never generate or repair cues.

## Train the decoder variants

Prepare the approved data with `python A_food_segmentation/prepare_data.py`.
Method 3 delegates to the primary training implementation:

```bash
python A_food_segmentation/method_comparison/03_finetuned_no_prompt/train.py \
  --output-dir A_food_segmentation/_local/outputs/no_prompt_trial \
  --device cuda --skip-figures
```

Method 4 reuses that run's verified frozen encoder cache and starts its decoder
from the original official weights:

```bash
python A_food_segmentation/method_comparison/04_finetuned_auto_prompt/train.py \
  --no-prompt-run A_food_segmentation/_local/outputs/no_prompt_trial \
  --output-dir A_food_segmentation/_local/outputs/prompted_trial
```

The prompted training implementation currently targets CUDA, as in the measured
experiment. Inference for all methods supports CPU/CUDA/MPS. On MPS, automatic
candidate generation falls back to CPU because original SAM constructs float64
grid tensors; the final decoder still runs on MPS with float32 coordinates.
CPU/CUDA use the historical generation recipe unchanged. The model outputs
remain whole-dish food/background masks, not ingredient-category predictions.

## Evidence

- [COMPARISON.md](COMPARISON.md): settings, results, caveats and sources.
- `results/01_...csv` through `results/04_...csv`: one row per development image.
- `results/summary.json`: exact measured aggregates and provenance.
- `results/history_*.csv`: training histories.
- `results/extra_added_prompts.csv`: an additional inference-only ablation,
  retained as an appendix rather than a fifth core method directory.
- `protocol/`: fixed sample assignments and integrity checks; no raw images.
- [run_records/](run_records/): separate template-based records for all measured conditions.
- [development-log.md](development-log.md): attempts, failures, outcomes and next actions.
- [ARTIFACTS.md](ARTIFACTS.md): external models/logs and authorized access instructions.
- [CONTRIBUTING_REVIEW.md](CONTRIBUTING_REVIEW.md): publication checks and pending review items.

These are the previously measured runs carried into the organized implementation.
Relocating code does not constitute another full training run. Packaging checks
and golden inference comparisons are recorded separately in `results/packaging_checks.json`.
