# Data

## Proposed source

[Nutrition5k](https://github.com/google-research-datasets/Nutrition5k) is the proposed source of overhead RGB meal images and dish-level nutrition labels.

Intended input: one available overhead RGB image. Regression targets: mass (g), energy (kcal), protein (g), carbohydrate (g), and fat (g). Dish IDs match images to labels; related incremental scans must be grouped by physical plate for splitting. Mass is evaluated separately from the four nutrients used for tracking and planning.

## Access checklist

- [ ] Verify source terms, download instructions, and overhead image availability.
- [ ] Select a manageable subset and record selection criteria and size.
- [ ] Verify image-label pairs, units, missing values, and duplicate dish IDs.
- [ ] Record exact acquisition commands and storage requirements.
- [ ] Save train/development/final-evaluation manifests and selection method.

Keep raw data under ignored `data/raw/` or an external location. Use configurable data paths so teammates can run the same code on their machines. Keep credentials outside Git and document the source and sharing permissions of published examples.

## Splits

Store small versioned manifests in `splits/`. Keep all images and related incremental scans of one physical plate in the same partition. Record dataset/subset version, split seed and method, counts, and manifest identifier/hash. We plan to use up to 800 eligible official training records, group-split into approximately 640 training and 160 development records, plus 200 official test records. Counts depend on verified eligibility and grouping. Use the same manifests for baseline and controlled comparison.
