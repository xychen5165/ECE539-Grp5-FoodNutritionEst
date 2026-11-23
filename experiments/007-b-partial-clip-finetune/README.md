# 007 - Partial CLIP fine-tuning on original plus natural crop

Status: Completed on November 22, 2026 (America/Chicago); final test remained sealed.

## Question

Does adapting the last visual Transformer block of the shared CLIP ViT-B/32
improve B4's original-plus-natural-crop nutrition regression?

B5 keeps experiment 006's images, crop definition, dual-view head, targets,
loss, optimizer family, effective batch size, seeds and development metric. The
only modeling change is allowing CLIP visual block 11, post-layer normalization
and visual projection to update after a three-epoch head-only warmup.

| Condition | Visual encoder | Input | Head |
|---|---|---|---|
| B4 (retained 006) | Fully frozen CLIP | Original + natural crop | two 512→64 branches→5 |
| B5 (new) | Shared CLIP last block + post-LN/projection trainable | Same | Same |

The primary comparison is paired B5−B4 normalized nutrient MAE on the same 538
development dishes and 182 plate groups. Negative deltas favor B5. Final-test
images and labels remain sealed.

## Measured result

| Condition | Development normalized nutrient MAE, mean ± sample SD (3 seeds) |
|---|---:|
| B4 (006, frozen CLIP) | 0.3255 ± 0.0020 |
| B5 (partial CLIP adaptation) | **0.3169 ± 0.0048** |

B5 scores were 0.3162, 0.3125 and 0.3220, selected at epochs 30, 29 and 29.
The three-seed mean improved numerically by approximately 0.0086 over B4.
Mean B5 raw MAEs were 41.91 g mass, 65.45 kcal energy, 5.82 g protein,
5.99 g carbohydrate and 4.88 g fat.

Seed-matched B5−B4 normalized nutrient MAE deltas were −0.0092, −0.0149 and
−0.0016. Their paired plate-bootstrap 95% intervals were [−0.0204, 0.0023],
[−0.0264, −0.0032] and [−0.0134, 0.0124]. All point estimates favor B5, but
only seed 1 excludes zero. B5 is therefore the best numerical development
condition so far, while the evidence does not establish a consistent
improvement across seeds. By itself B5 does not satisfy the handoff gate for
full-visual B7, which also requires at least two of three paired intervals to
be strictly below zero.

The uploaded `007-small-results.zip` SHA-256 was
`394b88baf355c54e19516e578c077ec272b1899a6eaaef415c7d6b176554079e`.
The comparison was independently reproduced from the retained ID-keyed
predictions. The frozen protocol's uncertainty description mentions seed 0,
but the already frozen comparison implementation and executed artifact report
all three seeds; this reporting clarification does not change the experiment.

See [protocol.json](protocol.json), [B5 run record](run-records/B5.md) and
[COLAB.md](COLAB.md). Large images, feature arrays and checkpoints remain in
external Google Drive storage.
