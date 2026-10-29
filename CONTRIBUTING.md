# Contributing

## Getting started

1. Accept the repository invitation and read the README and work plan.
2. Claim a task in an issue or the work plan using your GitHub handle or member shorthand.
3. Develop on a focused branch and open a pull request into `main`.
4. Request review from another teammate and address feedback before merging.

Use English for shared documentation and reports; discussion may use the team's preferred language. Keep the Canvas-name-to-GitHub identity mapping only in the private Canvas PDF, not in public files, issues, or PRs.

## Branches and review

Use descriptive branches such as `data/verify-access`, `experiment/baseline`, or `app/upload-flow`. Browser edits should also create a new branch and PR. Teammates review PRs before merging. GitHub enforcement depends on repository rules, which have not been checked during this setup.

Keep PRs focused. Describe the purpose, changes, verification, and related issue or experiment. Identify changes to component interfaces, data splits, dependencies, or evaluation behavior. Preserve others' work during conflict resolution. Update the relevant documentation when a change affects setup, usage, interfaces, or project status.

## Code organization

- Put reusable data, model, training, evaluation, and inference logic in `src/`.
- Keep experiment-specific configuration, optional notebooks, and small results together under `experiments/<id>-<name>/`.
- Have experiments import reusable pipeline code from `src/`.
- Add `app/` and `tests/` when needed. Product code should call shared inference logic.
- Add dependency versions and exact commands alongside the first runnable implementation.

## Experiments and evaluation

Agree on a starter subset and save train/development/final-evaluation manifests before comparing methods. All views of the same dish must stay in one partition. Fit data-dependent transformations on training data, use development data for tuning and selection, and reserve final evaluation for the agreed final protocol.

Use common splits, metric definitions, and evaluation code. For a controlled comparison, specify the technical change and keep other conditions matched; document unavoidable differences, including training budget and preprocessing.

Create a separate record for each reported run using the [run template](experiments/run-template.md). Record its settings and evidence, and retain earlier runs for comparison. Label planned work and measured results clearly.

Update the development log after meaningful attempts, including failures, diagnosis, and what changed next. Link results to a run record and supporting metrics or predictions.

## Files and privacy

Do not commit raw datasets, large generated arrays, model checkpoints, credentials, or private Canvas submissions. Document an agreed external storage location and access steps for large artifacts. Check dataset permissions before publishing images. Keep small reviewable split manifests, metrics, and permitted figures in Git.

Clear unnecessary notebook outputs and avoid embedding raw datasets. Review staged files before committing. Follow the course AI policy; its exact requirements must be confirmed before submission.

## Ready to merge

- Relevant commands or checks pass, or a documentation-only check is stated.
- Interfaces, dependencies, and user instructions reflect the change.
- Reported numbers link to actual evidence.
- No private information or unintended large files are included.
- Another teammate has reviewed the change.
