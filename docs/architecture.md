# Architecture and evaluation plan

This plan is based on our November 1, 2026 proposal, *Photo-Based Daily Nutrition and Gordon Meal Planning*. We have independently tested Gordon menu retrieval and a basic recommendation prototype. The following sections describe the planned complete system. Sources are listed in [References](references.md).

## User flow

Users enter their profile, activity, goals, and dietary preferences, then confirm editable daily targets. They upload a meal photo, review nutrient estimates and consumed amounts, and confirm intake; manual entry is also supported. The system updates daily totals and remaining needs. Users select a Gordon menu date and meal, review suggested foods and portions, and confirm consumption to update the log.

## Components and interfaces

### A - Food segmentation

Workspace: [A_food_segmentation](../A_food_segmentation/README.md).

Frozen SAM ViT-B [2] generates candidate masks from an internal point grid. Area, location, and overlap rules selected on development data identify the food region. A replaces the background with a neutral color while preserving image scale, then passes the masked RGB image to B.

Segmentation is intended to reduce background influence. Mask errors may remove food or retain distracting regions; the original-versus-masked comparison will measure its effect on nutrition prediction.

### B - Nutrition estimation

Workspace: [B_nutrition_estimation](../B_nutrition_estimation/README.md).

Frozen CLIP ViT-B/32 [3] maps an RGB image to 512 features. A trained MLP maps those features to mass (g), energy (kcal), protein (g), carbohydrate (g), and fat (g). The planned head is 512 → 128 (ReLU) → 5 (Softplus). B passes its estimates to C for user review.

This adapts Nutrition5k's RGB regression task [4]. Freezing the encoder reduces trainable parameters and allows feature caching, while limiting adaptation of the visual representation to meal imagery. Inference uses RGB images; measured mass and depth are excluded from model inputs.

### C - Daily tracking and menu access

Workspace: [C_daily_tracking](../C_daily_tracking/README.md).

C calculates editable daily energy and macronutrient targets using references [5–7]. For generally healthy adults aged 19+, the planned starting values use 2023 DRI EER equations, carbohydrate and fat ranges of 45–65% and 20–35% of energy, and protein at 0.8 g/kg/day. Users confirm exercise-related protein and goal-specific energy adjustments. These are planning targets; implementation and calculation tests remain pending.

C applies user-confirmed consumption fractions to estimates, records intake under unique IDs, and replaces entries when corrected. Energy and macronutrients drive daily totals and remaining needs; predicted mass is evaluated separately. Profiles and logs are planned to be stored locally.

The menu client retrieves Gordon items for the selected date and meal, timestamps and caches responses, and validates units. C supplies remaining needs and menu data to D. Menu completeness and freshness affect which candidates D can evaluate.

### D - Meal planning

Workspace: [D_meal_planning](../D_meal_planning/README.md).

D enumerates combinations of 1–3 Gordon foods at 0.5, 1, or 1.5 listed servings. Candidates require complete energy, protein, carbohydrate, and fat values. D applies dietary exclusions and an energy ceiling, then ranks projected daily totals using an equally weighted sum of normalized energy-target deviation, protein shortfall, and carbohydrate/fat distance from their planning ranges. Range penalties are zero within range; normalization uses the target or range midpoint. Ties favor fewer foods.

The output includes food choices, serving amounts, stations, and projected daily totals. User-confirmed consumption is returned to C for logging. Tests will cover infeasible, already-met, and incomplete-log scenarios.

## Training and evaluation

SAM and CLIP remain frozen; the team trains the MLP. Targets are normalized by their training-set means to balance regression losses. The planned configuration uses AdamW, learning rate 0.001, weight decay 0.0001, batch size 32, up to 30 epochs, and seeds 0/1/2. Checkpoints are selected using development-set normalized nutrient MAE. The exact training loss function remains to be specified.

- **Photo baseline:** each target's training-set median, which minimizes absolute error for a constant predictor.
- **Controlled comparison:** original versus SAM-masked RGB, with matched preprocessing, encoder, MLP, splits, seeds, and optimization.
- **Photo metrics:** MAE in original units, four-nutrient MAE normalized by training means, separate mass MAE, three-seed results, and paired plate-group bootstrap intervals.
- **Segmentation check:** planned IoU evaluation against published food-region annotations [8], subject to matching development images.
- **Planner comparison:** joint ranking versus protein-only ranking with the same menus, portions, histories, and constraints. Measure composite goal deviation, nutrient gaps, constraint violations, and menu membership.
- **Error propagation:** compare recommendations based on reference versus predicted intake. Select settings on development data and evaluate held-out plates and scenario families.

## Data and feasibility

The initial plan uses up to 1,000 matched Nutrition5k overhead records: up to 800 official training records, group-split into approximately 640 training and 160 development records, and 200 official test records. Related incremental scans stay together. Access, licensing, image-label matching, missing values, duplicates, and physical-plate grouping must be checked before fixing manifests. Test labels are reserved for final evaluation. Synthetic profile/intake scenarios are split by family, with referenced plate IDs disjoint across partitions.

We budget 4–6 hours per member per week, 16 GB RAM, up to 10 GB data/cache, and up to 10 accelerator-hours on an 8–16 GB GPU or local accelerator. A/B will verify resources by November 8. Main risks are data access/grouping, portion estimation, and differences between training and user photos.

By November 10, the team will decide whether to use the fallback: approximately 300 grouped records (200/50/50), 32 MLP hidden units, and 15 epochs, preserving official test separation and both neural stages.
