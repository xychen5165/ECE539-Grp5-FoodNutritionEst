# Experiment record audit

Audit date: November 22, 2026 (America/Chicago).

Auditor: `Tardirisu`.

Scope: every measured experiment record present on `merge-A&B` before the
planned B0–B3 comparison. Standard: [CONTRIBUTING.md](../CONTRIBUTING.md) and
[run-template.md](run-template.md). This audit checks record completeness and
traceability; it does not reinterpret the reported scientific results.

## Method

The audit checked each record for: explicit status; contributor and date; a
question and controlled technical change; code provenance; environment,
hardware and command; dataset and split fingerprints; seeds; preprocessing and
training-only fitted transformations; frozen and updated components; metric
definitions; logs, metrics and ID-keyed predictions; artifact access; measured
outcome, limitations and next action. It also inspected tracked file types and
sizes, searched for test-evaluation claims, and used the existing A/B preflight
to reproduce stored B metrics from predictions.

Ratings mean:

- **Conforms**: template fields and linked evidence are present, or an
  inapplicable field is explicitly explained.
- **Partial**: the measured result is traceable, but historical evidence or
  template fields are missing. Missing historical facts are not reconstructed.
- **Exception**: a known project-owner-approved departure from the default
  repository rule is disclosed.

## Findings

| Record | Rating | Evidence present | Gaps or exceptions |
|---|---|---|---|
| Part A runs A-01 through A-05 | Conforms with disclosed exceptions | Five separate template-based records; contributor/date, historical command, source hashes, environment, hardware, manifests, seed, metrics, per-dish TP/FP/FN, histories where applicable, limitations and artifact instructions | Historical execution code was uncommitted, so source hashes replace a false commit claim. Component-local implementation and the two tracked decoder checkpoints are explicit owner-requested exceptions. Teammate review and course AI-policy confirmation remain pending. |
| 001 median baseline | Partial | Status, purpose, exact current command, contributor/date, split hashes, procedure, metric definition, bootstrap seed, metrics and ID-keyed predictions | Historical environment versions, hardware/time budget and a clean execution commit were not fully captured. The implementation was later moved; commit `dcde4c8` used the old `src/` path. No checkpoint is applicable. |
| 002 CLIP heads | Partial | Status, question, three distinct result directories, exact commands, contributor/date, feature and split hashes, settings, three seeds, logs, metrics and ID-keyed predictions | The three configurations lack separate template-shaped Markdown run records. Hardware is only “laptop processor”; complete package versions are indirect. Checkpoints were local-only and no durable team access location is recorded. Code provenance says the training script was committed with the record rather than naming its resulting clean commit. |
| 003 regularization grid | Partial | Status, question, 11 distinct new result directories plus the reused 002 cell, complete settings in JSON, three seeds, logs and aggregate metrics; winning predictions retained | It lacks a separate template record for each grid cell. Contributor, execution date, clean code commit, hardware/time and environment are inherited informally from 002 rather than repeated. Ten new non-winning cells do not retain ID-keyed predictions, and local checkpoints have no durable access location. These artifacts cannot be recreated from Git alone. |
| 004 A/B integration | Conforms after metadata/index repair | Explicit status and provenance, exact commands, environment lock, hashes, preflight, baseline rerun, real-image smoke evidence, negative checks, limitations and next action | The first smoke attempt failed after feature generation because the harness omitted `loss`; the fixed run reused hash-validated artifacts. Raw images, model weights and temporary heads are intentionally ignored. It is a software smoke test, not a predictive comparison. |

## Repository-level checks

- Shared documentation is in English; private identity mappings and credentials
  were not found in experiment records.
- No raw Nutrition5k images, feature arrays or temporary B checkpoints are
  tracked. Large local downloads remain under ignored `.venv/` and `artifacts/`.
- The only tracked model files are the two Part A decoder checkpoints. Their
  hashes and explicit owner-requested exception are documented in Part A.
- B records use the same `train.csv` / `dev.csv` fingerprints and common metric
  implementation. Existing preflight checks reproduce all 13 retained B
  prediction files and their reported metrics.
- Records state that final test was not used for model selection. The A protocol
  records one previously accessed test annotation and does not claim the entire
  test set is unseen to every teammate.
- No record is ready to merge solely on this audit: `CONTRIBUTING.md` still
  requires another teammate's review, and the exact course AI policy remains to
  be confirmed before submission.

## Requirements for B0–B3

Each condition will receive its own run record before execution. A shared frozen
protocol will record the clean code commit, configuration file, environment,
hardware, time budget, data/split hashes, pretrained-weight hashes, seeds,
training-only transformations, exact command and metrics. Every executed seed
must retain a log and ID-keyed development predictions; failures remain recorded
instead of being overwritten. Large images, masks, features and checkpoints stay
outside Git with hashes and local/team access instructions. The experiment index
will distinguish planned, running, failed and completed states, and the repository
development log will link every meaningful attempt.

Historical partial ratings remain partial. This audit improves discoverability
and honesty; it does not invent missing artifacts or retroactively make those
runs template-complete.
