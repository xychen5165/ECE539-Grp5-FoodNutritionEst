# 007 - Partial CLIP fine-tuning on original plus natural crop

Status: Planned; protocol frozen before execution.

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

See [protocol.json](protocol.json), [B5 run record](run-records/B5.md) and
[COLAB.md](COLAB.md). Large images, feature arrays and checkpoints remain in
external Google Drive storage.
