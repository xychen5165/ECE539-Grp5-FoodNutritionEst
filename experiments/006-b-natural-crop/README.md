# 006 - Natural crop versus original/masked input

Status: Planned; protocol frozen before execution.

## Question

Does replacing the artificial gray-background masked view with a natural RGB
crop improve nutrition regression while retaining the original-image context?

Experiment 005 found that masked-only B1 was worse than original B0, while
original+masked B2 recovered the loss but did not significantly beat B0. B4
changes only B2's second view: it uses the predicted-mask bounding box plus 10%
RGB context. The original branch, CLIP encoder, head capacity, loss, optimizer,
budget, seeds, splits and metrics remain matched.

| Condition | Inputs | Trainable head | Parameters |
|---|---|---|---:|
| B0 (retained 005) | Original CLIP | 512→128→5 | 66,309 |
| B2 (retained 005) | Original + gray-mask CLIP | two 512→64 branches→5 | 66,309 |
| B4 (new) | Original + natural-crop CLIP | two 512→64 branches→5 | 66,309 |

Primary comparisons are paired B4−B0 and B4−B2 normalized nutrient MAE on the
same 538 development dishes/182 plate groups. Negative deltas favor B4. The
canonical final test remains sealed.

See [protocol.json](protocol.json), [B4 run record](run-records/B4.md), and the
[Colab guide](COLAB.md). Large images, masks, CLIP arrays and checkpoints remain
in the existing ignored Drive run root from experiment 005.
