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
- Evidence: `A_food_segmentation/debug_validation/audit_nutrition5k.py`, its tests, and the exploratory manifests under `A_food_segmentation/debug_validation/manifests/`.

### 2026-11-19 - Frozen SAM debug segmentation

- Contributor: YifeiDorthy
- Attempt: Loaded the official SAM ViT-B checkpoint and ran automatic mask generation on three 640 x 480 Nutrition5k overhead RGB images using a 16 x 16 point grid on CPU. Developed a deterministic first-pass selector based on candidate size, position, predicted IoU, stability, and containment within a filled plate silhouette.
- Outcome: All three images completed in about 14 seconds each and generated `mask.png`, `masked_rgb.png`, `overlay.png`, and selection metrics. The first rule selected the plate instead of the olives; a later rule selected white paper below the plate. Filling the plate's outer contour before filtering small candidates corrected both failure modes. Final visual review captured the main olives in two images and the main salad cluster in the third, although thin onion strands were partly missed.
- Lesson and next action: A single largest centered mask is not a reliable food selector. Disconnected foods need a union of candidates constrained by the plate silhouette, while a connected salad can use a centered cluster fallback. Next, keep these rules fixed and evaluate them on additional images selected from the team's canonical split.
- Evidence: `A_food_segmentation/debug_validation/run_sam_debug.py`; local ignored outputs under `A_food_segmentation/debug_validation/outputs/sam_debug/`.
