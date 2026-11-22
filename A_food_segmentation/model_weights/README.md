# Selected trained decoder weights

These are the two selected decoder checkpoints from the recorded experiments.
Their inclusion is explicitly requested by the project owner, as a scoped
exception to the contribution guide's checkpoint exclusion.

| File | Method | Best epoch | SHA256 |
|---|---|---|---|
| `food_decoder_no_prompt.pt` | Fine-tuned decoder, no external prompts | 12 | `456c7afdc12dcb8e6ec009526178f5bc542f35493492904f09bc3a1d6af2c4ba` |
| `food_decoder_prompted.pt` | Fine-tuned decoder, automatic box and points | 5 | `8333f1a5b333f60a707191284e7ad787a06945c0554698407b268d49ae562413` |

Each file is about 16 MiB and contains decoder parameters and training metadata.
The frozen SAM image and prompt encoders are not included. Obtain the official
SAM ViT-B checkpoint using the [component setup](../README.md).

Copy both files into the default local runtime directory:

```bash
mkdir -p A_food_segmentation/_local/checkpoints
cp A_food_segmentation/model_weights/food_decoder_no_prompt.pt A_food_segmentation/_local/checkpoints/
cp A_food_segmentation/model_weights/food_decoder_prompted.pt A_food_segmentation/_local/checkpoints/
```

Alternatively, use `--decoder-checkpoint A_food_segmentation/model_weights/food_decoder_no_prompt.pt`
for the primary no-prompt method. The prompted checkpoint must be used with
method 4's automatic-prompt pipeline. See the [method comparison](../method_comparison/COMPARISON.md)
and [artifact provenance](../method_comparison/ARTIFACTS.md).
