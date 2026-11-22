# 005 - Original, masked, fused and geometry-aware nutrition regression

Status: Planned; protocol and implementation validation completed, model runs not yet started.

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
- Large RGBs, masks, CLIP arrays and checkpoints: ignored
  `artifacts/b-input-comparison/`, with hashes copied into small result records.

Final-test images and labels are excluded from preparation, training and this
development comparison. Any later final evaluation requires a separately frozen
record after method selection.

For the selected Google Colab Pro execution path, follow [COLAB.md](COLAB.md).

