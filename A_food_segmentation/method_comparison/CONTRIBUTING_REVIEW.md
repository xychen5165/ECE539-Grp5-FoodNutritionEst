# Contribution-guide assessment

Scope: the authorized Part A commit and push on
`yifei/part-a-sam-method-comparison`, based on `main`.
Reference: [CONTRIBUTING.md](../../CONTRIBUTING.md).

| Requirement | Assessment | Evidence or remaining action |
|---|---|---|
| Focused descriptive branch and contributor ownership | Checked | Branch scoped to Part A; owner recorded by GitHub handle in the component log and run records. |
| English shared documentation | Checked | Component README, comparison, run records and log are English. |
| Setup, exact commands and interfaces documented | Checked | Parent README, four method guides, pinned SAM/OpenCV dependency, measured environment and run records. RGB in; original-size binary mask and gray-background RGB out. |
| Reusable code in `src/`, experiments in `experiments/` | Intentional layout exception | The user explicitly requested the selected method directly under Part A and the four-method tests beneath it. Shared inference/training code remains component-local; this is not represented as literal compliance with the default directory guidance. |
| Fixed partitions, consistent metrics, reserved test | Checked | Pinned manifest hashes, unique/group-disjoint train/dev rows, excluded test-linked groups, common metrics; reported scores are development only. Grouping remains inferred. |
| Separate run record for each reported condition | Completed | Five records follow [the run template](../../experiments/run-template.md): four main conditions and the additional inference-only control. |
| Development log includes attempts and failures | Completed locally | [development-log.md](development-log.md), including original near-empty masks and the MPS float64 failure/fix. The repository-wide log remains unchanged to preserve the requested Part A-only scope. |
| Results linked to evidence; pending work marked | Checked | Exact summary, four per-image CSVs, histories, region diagnostics and packaging checks. The old execution source hashes are distinguished from reorganized code; no new full-training claim is made. |
| External artifacts have location/access instructions | Completed | [ARTIFACTS.md](ARTIFACTS.md): team-server home-relative locations, authorized-copy commands, public input sources and expected model hashes. Private endpoints are not published. |
| No raw datasets, weights, private mappings or credentials | Checked before staging/commit | Only code, curated assignment manifests and small text results are intended for Git. `_local/`, legacy exploration assets, caches and model files are ignored. |
| Relevant implementation checks | Checked | Nine CLI help checks; all 557 local pairs validated; four variants and root inference exercised on one old dev fixture; both epoch loops AST-identical to the executed trials; overwrite/mixed-control guards exercised. See packaging checks. |
| PR into main and another teammate's review | Pending | This request authorizes a branch commit/push. No teammate review has occurred and no merge readiness is claimed. Open the PR, request a teammate and resolve feedback before merging. |
| Course AI policy confirmed | Pending | The exact course rules are not present in this repository. Confirm them before course submission; this review does not certify that policy. |

The change is suitable for publishing as a **reviewable branch**, subject to the
final staged-file checks. It is **not yet certified ready to merge** because a
teammate's review is outstanding. The layout exception is deliberate and based
on the user's requested organization, not a hidden change to team conventions.

Historical JSON checks record the state when verification occurred, before the
commit/push; their `commit_created` and `remote_push_performed` flags are not a
live statement about the branch after publication.
