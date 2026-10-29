# Data

## Proposed source

[Nutrition5k](https://github.com/google-research-datasets/Nutrition5k) is the proposed source of overhead RGB meal images and dish-level nutrition labels.

Intended input: one available overhead RGB image. Intended targets: calories (kcal), protein (g), carbohydrates (g), and fat (g). Dish ID is the matching and split unit. Total mass and ingredient information may support analysis; their role as training targets or inputs is not decided.

## Access checklist

- [ ] Verify source terms, download instructions, and overhead image availability.
- [ ] Select a manageable subset and record selection criteria and size.
- [ ] Verify image-label pairs, units, missing values, and duplicate dish IDs.
- [ ] Record exact acquisition commands and storage requirements.
- [ ] Save train/development/final-evaluation manifests and selection method.

Keep raw data under ignored `data/raw/` or an external location. Use configurable data paths so teammates can run the same code on their machines. Keep credentials outside Git and document the source and sharing permissions of published examples.

## Splits

Store small versioned manifests in `splits/`. Keep all images of one dish in the same partition. Record dataset/subset version, split seed and method, counts, and manifest identifier/hash. Partition proportions remain to be decided. Use the same manifests for baseline and controlled comparison.
