# ECE539-Grp5-FoodNutritionEst

ECE 539 Fall 2026, Group 5: Photo-Based Daily Nutrition and Gordon Meal Planning.

## Goal and current status

Based on our November 1, 2026 proposal, we plan to build a system that estimates meal nutrients from photos, tracks user-confirmed intake against editable daily targets, and recommends Gordon meals based on remaining needs and dietary preferences. Nutrition5k supplies photo-regression data; Gordon's published menu supplies serving and nutrition data for recommendations.

The planned pipeline connects four components:

1. **A - Food segmentation:** frozen SAM ViT-B generates food-region masks and passes a masked RGB image to B.
2. **B - Nutrition estimation:** frozen CLIP ViT-B/32 produces 512-dimensional features. A team-trained MLP predicts mass, energy, protein, carbohydrate, and fat.
3. **C - Daily tracking:** calculates daily targets, records confirmed consumption and corrections, and supplies remaining needs and dated Gordon menu data.
4. **D - Meal planning:** ranks food and serving combinations against daily goals and dietary constraints. Users confirm consumption before a recommendation enters their intake log.

The training plan uses a 512 → 128 (ReLU) → 5 (Softplus) MLP with frozen SAM and CLIP. Evaluation includes a training-set median baseline, original-versus-masked image comparison, and joint meal ranking versus protein-only ranking. See the [architecture and evaluation plan](docs/architecture.md) for interfaces, settings, rationale, and remaining uncertainties.

We have tested Gordon menu retrieval and a basic recommendation prototype independently. Photo inference, target calculation, joint planning, and end-to-end integration are planned. Data access, image-label matching, plate grouping, and compute availability still require verification.

Repository: https://github.com/xychen5165/ECE539-Grp5-FoodNutritionEst

## Start here

- [Contribution guide](CONTRIBUTING.md): branches, reviews, experiments, and public-repository rules.
- [Architecture](docs/architecture.md): component and interface planning.
- [Work plan](docs/work-plan.md): responsibilities, milestones, and open decisions.
- [Development log](docs/development-log.md): attempts, outcomes, and lessons.
- [Data guide](data/README.md): access and split requirements.
- [Experiments](experiments/README.md): run records and evidence.
- [References](docs/references.md): sources supporting the design.

## Layout

```text
A_food_segmentation/       Component A: food segmentation and data preparation
B_nutrition_estimation/    Component B: nutrition estimation
C_daily_tracking/         Component C: daily nutrition tracking
D_meal_planning/           Component D: meal planning
final/                    Integrated team deliverable
.github/                  Task, experiment, and pull request templates
src/                      Shared implementation
data/                     Data access documentation and split manifests
experiments/              Experiment settings, notebooks, and small results
docs/                     Architecture, work plan, development log, and references
```

Add `app/` when product development starts and `tests/` alongside implementation checks. Include dependency versions and setup/run commands with the first runnable implementation.

## Reproducibility and delivery

Use common physical-plate-group splits and shared evaluation code across comparable experiments. Keep configurations, small metrics, and run records in Git; document external access for datasets and model weights. See the [run template](experiments/run-template.md).

For final submission, record the repository commit or release used to produce the reported results. Keep Canvas-name-to-GitHub identity mappings and private submission files outside this public repository.
