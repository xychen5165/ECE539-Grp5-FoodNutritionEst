# A - Food isolation

Component workspace for public-data preparation and frozen SAM integration. Implementation is planned.

## Deliverables
- Match Nutrition5k RGB images to nutrient labels and audit related plate groups.
- Freeze shared train/development/final-evaluation manifests before model comparisons.
- Generate food masks from user point or box prompts with frozen SAM.
- Document prompt handling, mask quality checks and the portion-confirmation workflow.

## Interface
Input: RGB meal image and user prompts.
Output: union food mask and masked image for component B.

Keep component experiments and notes here; place reusable pipeline code in ../src/ and shared data instructions/manifests in ../data/. Follow ../CONTRIBUTING.md. Do not commit raw datasets, checkpoints or private identity mappings.
