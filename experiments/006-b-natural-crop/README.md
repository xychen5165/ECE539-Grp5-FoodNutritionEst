# 006 - Natural crop versus original/masked input

Status: Completed on November 22, 2026 (America/Chicago); final test remained sealed.

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

## Measured result

| Condition | Development normalized nutrient MAE, mean ± sample SD (3 seeds) |
|---|---:|
| B0 (005) | 0.3423 ± 0.0014 |
| B2 (005) | 0.3390 ± 0.0008 |
| B4 | **0.3255 ± 0.0020** |

B4 scores were 0.3254, 0.3275 and 0.3236. Against B0, paired deltas were
−0.0180, −0.0133 and −0.0193, and all three plate-bootstrap 95% intervals were
strictly below zero. This is consistent evidence that the original plus natural
crop improves over original alone under the frozen development protocol.

Against B2, deltas were −0.0145, −0.0110 and −0.0152. All favored B4, but seed
0 and 1 intervals crossed zero; only seed 2 excluded zero. The natural crop is
the better numerical representation, but the per-seed evidence versus the gray
dual view is not uniformly conclusive.

The uploaded `006-small-results.zip` SHA-256 was
`de08af59cc475a3820990c65ce3bb07845f6f2e47ec51ec2fa51649d1c354584`.
All 2,673 train/dev crops were generated at the frozen 10% margin with unique
dish IDs. Reviewable crop metadata, feature hashes, logs, metrics, predictions,
environment and paired comparisons are retained here; images, feature arrays
and checkpoints remain external.

See [protocol.json](protocol.json), [B4 run record](run-records/B4.md), and the
[Colab guide](COLAB.md). Large images, masks, CLIP arrays and checkpoints remain
in the existing ignored Drive run root from experiment 005.
