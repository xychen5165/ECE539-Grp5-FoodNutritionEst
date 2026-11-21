# Checkpoints, datasets and larger run artifacts

Raw images, human masks, model weights, generated images and feature caches are
not stored in Git. `_local/` is ignored. The public Git records contain only
code, curated manifests, small metric tables and provenance.

## Official model and raw inputs

The public SAM ViT-B checkpoint is available from:
https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

Its SHA256 is
`ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912`.
The component README documents the download command. `prepare_data.py` retrieves
only approved train/dev RGBs and human masks from the pinned public sources.
Nutrition5K and the annotation release are CC BY 4.0; attribution is recorded
in [COMPARISON.md](COMPARISON.md). No dataset photos are included in this change.

## Team-trained decoder weights

The current external copies are on the team's authorized GPU server, under the
following paths relative to the authorized account's home directory:

| Artifact | Home-relative server path | Local filename |
|---|---|---|
| No-prompt decoder | `cs539/sam_prompt_free_20261120/A_food_segmentation/debug_validation/outputs/prompt_free_sam_v1/best_decoder.pt` | `food_decoder_no_prompt.pt` |
| Automatic-prompt decoder | `cs539/sam_prompt_free_20261120/A_food_segmentation/debug_validation/outputs/prompted_sam_v1/best_prompted_decoder.pt` | `food_decoder_prompted.pt` |

The same run directories retain logs, development predictions, prompt records,
input checksums and configuration. The no-prompt run also retains frozen image
feature caches. Server addresses, account credentials and private identity
mappings are intentionally not published; obtain authorized team access privately.

With that access, replace the placeholders in these commands:

```bash
mkdir -p A_food_segmentation/_local/checkpoints
scp '<team-user>@<team-server>:~/cs539/sam_prompt_free_20261120/A_food_segmentation/debug_validation/outputs/prompt_free_sam_v1/best_decoder.pt' \
  A_food_segmentation/_local/checkpoints/food_decoder_no_prompt.pt
scp '<team-user>@<team-server>:~/cs539/sam_prompt_free_20261120/A_food_segmentation/debug_validation/outputs/prompted_sam_v1/best_prompted_decoder.pt' \
  A_food_segmentation/_local/checkpoints/food_decoder_prompted.pt
```

Check the copied files with `shasum -a 256` (macOS) or `sha256sum` (Linux).
Expected hashes are in [results/summary.json](results/summary.json):

- No-prompt decoder: `456c7afdc12dcb8e6ec009526178f5bc542f35493492904f09bc3a1d6af2c4ba`.
- Prompted decoder: `8333f1a5b333f60a707191284e7ad787a06945c0554698407b268d49ae562413`.

A collaborator without server access can reproduce training using the public
inputs and the commands in [README.md](README.md). Retraining can legitimately
produce a different checkpoint hash. Never add the copied weights to Git.
