# Development log

Record meaningful attempts as they occur, including failures. Identify contributors by GitHub handle or member shorthand. Link each entry to a commit, PR, run record, or supporting file.

Use the following format for each entry:

### YYYY-MM-DD - Attempt title

- Contributor:
- Attempt:
- Outcome:
- Lesson and next action:
- Evidence:

### 2026-11-18 - Nutrition5k metadata and RGB-path audit

- Contributor: YifeiDorthy
- Attempt: Download only the official dish metadata and RGB split lists, parse the five dish-level regression targets, check split integrity, and match a few public overhead RGB files to their labels without downloading the 181 GB archive.
- Outcome: Added a standard-library audit/manifest script and unit tests. The observed files contain 5,006 valid unique metadata rows, 4,059 official RGB train IDs, 709 official RGB test IDs, and zero train/test ID overlap. Three downloaded overhead examples matched metadata and official-train entries. The script records missing-label counts, local image availability, deterministic sampling settings, and input hashes.
- Lesson and next action: Metadata and selective public-image access are feasible, but a final internal train/development split still needs a verified physical-plate grouping rule. Next, expand the verified local image subset and run frozen SAM on a few examples while keeping the original-versus-masked comparison reproducible.
- Evidence: `A_food_segmentation/audit_nutrition5k.py`, `tests/test_audit_nutrition5k.py`, `data/splits/nutrition5k_debug_images_seed539.csv`, and its summary JSON.
