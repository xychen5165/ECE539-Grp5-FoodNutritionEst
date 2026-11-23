# 005 - Original, masked, fused and geometry-aware nutrition regression

Status: Completed on November 22, 2026 (America/Chicago); final test remained sealed.

## Question

Does the selected Part A segmentation improve Part B nutrient regression, and
are original-image context, masked-food appearance and mask geometry
complementary?

## Frozen conditions

| Run | Input | Trainable head | Planned comparison |
|---|---|---|---|
| [B0](run-records/B0.md) | Original CLIP 512 | 512→128→5 | Same-environment original baseline |
| [B1](run-records/B1.md) | Masked CLIP 512 | 512→128→5 | B1−B0: effect of replacing the image input |
| [B2](run-records/B2.md) | Original + masked CLIP | Two 512→64 branches, fused 128→5 | B2−B0 and B2−B1: complementary views at matched parameter count |
| [B3](run-records/B3.md) | B2 + eight mask geometry values | B2 fusion plus 8 raw standardized inputs | B3−B2: incremental geometry information |

## Measured result

| Run | Development normalized nutrient MAE, mean ± sample SD (3 seeds) |
|---|---:|
| B0 | 0.3423 ± 0.0014 |
| B1 | 0.3615 ± 0.0025 |
| B2 | 0.3390 ± 0.0008 |
| B3 | **0.3384 ± 0.0022** |

B1's masked-only input was worse than the same-environment original-image
baseline. B2 recovered that loss and beat B1 in every seed with paired
plate-bootstrap intervals below zero, but neither B2 nor B3 established an
improvement over B0: every paired B2−B0 and B3−B0 interval crossed zero. B3's
small numerical advantage over B2 was also inconclusive. Therefore the
selected conclusion is that original context must be retained; this experiment
does not demonstrate added value from the current gray-background mask or
geometry representation.

All heads use L1, AdamW, learning rate 0.001, weight decay 0.001, dropout
0.5, batch size 32, 30 epochs and seeds 0/1/2. The lowest development
normalized nutrient MAE epoch is retained independently for each seed. See
[protocol.json](protocol.json) for immutable hashes and exact definitions.

## Evidence layout

- `run-records/`: one template-based record per condition, updated without
  erasing failures or earlier attempts.
- `results/<condition>/metrics.json`: settings, hashes and per-seed metrics.
- `results/<condition>/training_log.csv`: every epoch and seed.
- `results/<condition>/dev_predictions_seed<N>.csv`: dish/group IDs, labels and predictions.
- `comparison.json`: paired plate-group bootstrap differences after all conditions finish.
- `environment.json` and `artifact-manifest.json`: captured Colab runtime and
  hashes for excluded large artifacts.
- Large RGBs, masks, CLIP arrays and checkpoints: ignored
  `artifacts/b-input-comparison/`, with hashes copied into small result records.

Final-test images and labels are excluded from preparation, training and this
development comparison. Any later final evaluation requires a separately frozen
record after method selection.

The uploaded small-results archive had SHA-256
`a05d28c3a2039ebc4f16a4ef87150f52373705c9fe4dfeb2fcd7e39503c64026`.
It contained 26 files and excluded raw RGBs, masks, CLIP arrays and model
checkpoints. The repository retains the reviewable metrics, epoch logs,
ID-keyed development predictions, comparison, environment and artifact hashes.

For the selected Google Colab Pro execution path, follow [COLAB.md](COLAB.md).

