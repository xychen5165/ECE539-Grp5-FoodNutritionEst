# B - Nutrition estimation

Component workspace for frozen CLIP features and a trainable nutrient regression head. Implementation is planned.

## Deliverables
- Extract 512-dimensional features with frozen CLIP ViT-B/32 for original and SAM-masked image experiments.
- Train a 512 -> 128 (ReLU) -> 5 (Softplus) regression head using Nutrition5k labels.
- Predict mass, energy, protein, carbohydrate, and fat.
- Record the median baseline, original-versus-masked comparison, three training seeds, checkpoints and nutrient error metrics.

## Interface
Input: one RGB image, masked by component A in the proposed pipeline or original in the controlled comparison.
Output: estimated mass (g), energy (kcal), protein (g), carbohydrate (g) and fat (g) for component C.

Keep component experiments and notes here; reusable code belongs in ../src/ and run records in ../experiments/. Use shared group-separated splits. Do not commit datasets, model weights or private identity mappings.
