# ECE539-Grp5-FoodNutritionEst

ECE 539 Fall 2026, Group 5: food image-based nutrition estimation.

## Goal and current status

Help college students estimate meal calories, protein, carbohydrates, and fat from one overhead RGB image. Nutrition5k is the proposed data source. The team brief is complete; architecture, training methods, dataset subset, compute resources, and an additional differentiating feature remain under discussion.

The repository is in the planning stage and currently contains documentation and collaboration templates. Implementation and experiment results will be added as development proceeds. The product interface and backend scope remain to be decided.

Repository: https://github.com/xychen5165/ECE539-Grp5-FoodNutritionEst

## Start here

- [Contribution guide](CONTRIBUTING.md): branches, reviews, experiments, and public-repository rules.
- [Architecture](docs/architecture.md): component and interface planning.
- [Work plan](docs/work-plan.md): responsibilities, milestones, and open decisions.
- [Development log](docs/development-log.md): attempts, outcomes, and lessons.
- [Data guide](data/README.md): access and split requirements.
- [Experiments](experiments/README.md): run records and evidence.
- [References](docs/references.md): sources from the brief and verification status.

## Layout

```text
.github/       Task, experiment, and pull request templates
src/           Shared implementation; currently a scope guide only
data/          Data access documentation and versioned split manifests
experiments/   Experiment-specific settings, notebooks, and small results
docs/          Architecture, work plan, development log, and references
```

Add `app/` when product development starts and `tests/` alongside implementation checks. Include dependency versions and setup/run commands with the first runnable implementation.

## Reproducibility and delivery

Use common dish-level splits and shared evaluation code across comparable experiments. Keep configurations, small metrics, and run records in Git; document external access for datasets and model weights. See the [run template](experiments/run-template.md).

For final submission, record the repository commit or release used to produce the reported results. Keep Canvas-name-to-GitHub identity mappings and private submission files outside this public repository.
