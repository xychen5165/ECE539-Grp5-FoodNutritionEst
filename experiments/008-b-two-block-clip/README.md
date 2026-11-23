# 008 - Two-block CLIP adaptation (B6)

Status: Preparing. Protocol and implementation are committed; nine unit tests passed on the server. Full input verification and smoke training are pending. No measured B6 result yet.

B6 adds visual block 10 to B5's trainable block 11, post-LN and projection.
The original RGB and natural crop share one CLIP visual tower. SAM remains fixed.
All other training settings follow the frozen B5 recipe.

Primary comparison: B6-B5. Secondary: B6-B4. Three seeds 0/1/2; canonical
2135 train and 538 dev dishes / 182 dev plate groups. Final test stays sealed.

See [protocol.json](protocol.json) and [run record](run-records/B6.md).
Images, feature arrays and checkpoints remain external. This task stops after
B6 and does not start B7-B10.

## Preparation notes

- Original/crop feature hashes match the handoff, including all 2673 dish IDs.
- The selected CUDA host provides Python 3.9.25 / PyTorch 2.8.0+cu128, unlike
  the historical Colab B4 runtime (Python 3.13 / PyTorch 2.14). These environment
  differences will be recorded; a cross-environment delta is not a pure
  estimate of the extra-block effect. OpenCLIP stays at 2.24.0.
- Raw/crop files are accepted only when their hashes match the acquired handoff.
  Public RGBs and the existing predicted crop boxes can recover identical files;
  no SAM inference or model fitting is involved in that restoration.
- The primary B5 comparison is pending receipt of completed B5 results.
