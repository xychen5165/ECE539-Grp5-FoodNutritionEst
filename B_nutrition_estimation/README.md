# B - Nutrition estimation

Component workspace for frozen CLIP features and a trainable nutrient regression head. Implementation is planned.

## Deliverables
- Extract frozen CLIP embeddings of the original and component A masked views.
- Train a 1024 -> 128 -> 5 regression head using public Nutrition5k labels.
- Predict mass, calories, protein, carbohydrate and fat; do not infer unlabeled fiber or micronutrients.
- Record the median baseline, masked-feature ablation, three training seeds, checkpoints and nutrient error metrics.

## Interface
Input: original RGB image plus component A mask/masked view.
Output: estimated mass (g), energy (kcal), protein (g), carbohydrate (g) and fat (g) for component C.

Keep component experiments and notes here; reusable code belongs in ../src/ and run records in ../experiments/. Use shared group-separated splits. Do not commit datasets, model weights or private identity mappings.
