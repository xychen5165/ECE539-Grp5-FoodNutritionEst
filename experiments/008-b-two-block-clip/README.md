# 008 - Two-block CLIP adaptation (B6)

Status: Planned. No measured B6 result yet.

B6 adds visual block 10 to B5's trainable block 11, post-LN and projection.
The original RGB and natural crop share one CLIP visual tower. SAM remains fixed.
All other training settings follow the frozen B5 recipe.

Primary comparison: B6-B5. Secondary: B6-B4. Three seeds 0/1/2; canonical
2135 train and 538 dev dishes / 182 dev plate groups. Final test stays sealed.

See [protocol.json](protocol.json) and [run record](run-records/B6.md).
Images, feature arrays and checkpoints remain external. This task stops after
B6 and does not start B7-B10.
