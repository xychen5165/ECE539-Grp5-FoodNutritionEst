# A - Food segmentation

Component workspace for public-data preparation and frozen SAM integration.

The initial metadata audit, exploratory manifests, three local RGB examples,
and frozen-SAM feasibility run are isolated under
[`debug_validation/`](debug_validation/README.md). That folder documents a
small test of whether the data and SAM pipeline work; it is not the team's
canonical train/development/test split.

## Deliverables
- Match Nutrition5k RGB images to nutrient labels and audit related plate groups.
- Freeze shared train/development/final-evaluation manifests before model comparisons.
- Generate candidate masks with frozen SAM ViT-B using an internal point grid; select the food region using development-set area, location, and overlap rules.
- Document mask selection, segmentation quality checks, and the portion-confirmation workflow.

## Interface
Input: RGB meal image.
Output: selected food-region mask and RGB image with a neutral background, preserving image scale, for component B.

Keep component experiments and notes here; place reusable pipeline code in ../src/ and shared data instructions/manifests in ../data/. Follow ../CONTRIBUTING.md. Do not commit raw datasets, checkpoints or private identity mappings.
