# Pretrained Mobile Food Segmenter binary-mask experiment

This experiment tests Google's pretrained Mobile Food Segmenter V1, linked as
a RELATED model by the official Nutrition5k repository. It is not established
that this is the binary model used in Nutrition5k paper section 4.4. We do not
implement the depth-volume calculation or train a U-Net. SAM is not used in this
experiment. The pretrained model is DeepLab-V3 with a MobileNet-V2 backbone.

## Fixed processing

The official model accepts normalized float32 RGB tensors of size 513 x 513
and returns a 26-class semantic prediction and per-class probabilities.
The official label map includes background (0), food containers (23), and
dining tools (24). We exclude ALL three, sum probabilities of the other 23
classes, and threshold at 0.5. Nonzero semantic class does not necessarily mean
food. There are no category-specific rules, SAM prompts, plate heuristics,
connected-component filtering, or per-image threshold adjustments.

We resize with aspect ratio preserved and gray-pad to a square. After inference
we remove padding and resize probabilities back to the original image dimensions
before thresholding. No center crop is applied, so the experiment does not
deliberately discard food at image boundaries. This preprocessing differs from
the Nutrition5k paper's side-video RGB regression setup. The model card expects
well-cropped dishes; the full-frame input used here is one explicit development
configuration rather than a claimed exact implementation of its training setup.

## Same-image development comparison

All 12 images in `../manifests/fresh_dev12_selection.csv` completed. Assistant
visual review using the earlier rubric, pending teammate confirmation:

| Method | Pass | Partial | Fail |
|---|---:|---:|---:|
| Plate appearance v1 | 7 | 4 | 1 |
| Grounded-SAM fixed prompt | 8 | 2 | 2 |
| Two-round SAM | 5 | 6 | 1 |
| Mobile Food Segmenter, food probability >=0.5 | 8 | 4 | 0 |

Both apples exclude the whole plate. The complex meal previously failed by
plate appearance now retains the principal foods, with a small plate patch and
boundary uncertainty. Remaining partial cases include tiny plate specks,
reflection and edge artifacts. No per-pixel ground truth is available; these
counts are not segmentation accuracy or IoU and are not downstream nutrient
prediction results. These reused development images cannot establish a general
performance ranking. The model was trained on a Google-internal dataset; its
overlap with Nutrition5k is not documented in the model card.

Median measured per-image preprocessing/inference/output time is about 0.29 s
on this CPU run, excluding TensorFlow import/model loading. This timing is not
a standardized hardware benchmark against SAM.

## Files and reproduction

The TensorFlow environment lives in the ignored `data/cache/food_segmenter_env`,
so the existing PyTorch/SAM environment is unchanged. The official compressed
model is about 10 MB and is stored under ignored `checkpoints/`.

From the repository root:

```bash
.venv/bin/python -m venv data/cache/food_segmenter_env
data/cache/food_segmenter_env/bin/python -m pip install \
  -r A_food_segmentation/debug_validation/mobile_food_segmenter/requirements.txt
mkdir -p checkpoints/mobile_food_segmenter_v1
curl -fL 'https://tfhub.dev/google/seefood/segmenter/mobile_food_segmenter_V1/1?tf-hub-format=compressed' \
  -o data/cache/mobile_food_segmenter_v1.tar.gz
tar -xzf data/cache/mobile_food_segmenter_v1.tar.gz -C checkpoints/mobile_food_segmenter_v1
curl -fL 'https://www.gstatic.com/aihub/tfhub/labelmaps/seefood_mobile_food_segmenter_V1_labelmap.csv' \
  -o checkpoints/mobile_food_segmenter_v1/labelmap.csv
data/cache/food_segmenter_env/bin/python -u \
  A_food_segmentation/debug_validation/mobile_food_segmenter/run_food_segmenter.py \
  --manifest A_food_segmentation/debug_validation/manifests/fresh_dev12_selection.csv \
  --output-dir A_food_segmentation/debug_validation/outputs/mobile_food_dev12_rerun
```

The recorded local run is in `../outputs/mobile_food_dev12_v1/`. It stores
`mask.png`, `masked_rgb.png`, `overlay.png`, `original.png`,
`food_probability.png` and `semantic_classes.png`. The last file contains raw
class IDs 0-25, not human-friendly display colors. Raw images and masks stay
ignored. `results.json` and `review.csv` archive small metrics and judgments;
`labelmap.csv` and `model_provenance.json` record the class mapping and model
fingerprint. `finalize_results.py` checks shape, binary output values, all 12
IDs and the script fingerprint before archiving results.

## Primary sources

- [Nutrition5k paper sections 4.2-4.4](https://arxiv.org/html/2103.03375)
- [Nutrition5k repository related-model link](https://github.com/google-research-datasets/Nutrition5k)
- [Official archived model card](https://github.com/tensorflow/tfhub.dev/blob/master/assets/docs/google/models/seefood/segmenter/mobile_food_segmenter_V1/1.md)
- [Official label map](https://www.gstatic.com/aihub/tfhub/labelmaps/seefood_mobile_food_segmenter_V1_labelmap.csv)
