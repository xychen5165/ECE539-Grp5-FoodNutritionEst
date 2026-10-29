# Architecture plan

Status: pending team agreement.

## Task

Input: one overhead RGB meal image. Output: estimated calories, protein, carbohydrates, and fat. Target users: college students seeking approximate meal logging.

## Required design decisions

The proposal must diagram at least three substantive connected task-relevant components, including at least two neural components. At least one used neural component must receive parameter updates by the team. Explain how each component contributes to the nutrition-estimation task.

For each component, document method/source, owner shorthand, input/output form or dimensions and units, next consumer, taught/pretrained/later-topic status, and trained/adapted/frozen status. The actual components and diagram remain undecided.

## Product integration

Proposed integration: the product interface passes a meal image to the shared inference pipeline and displays its nutrition estimates. If a backend is needed, it will handle requests and invoke that pipeline. Define input validation, output fields, model loading, and failure handling when the team chooses the product scope.

## Open questions

- Which two neural stages and third substantive component serve the task?
- Which parameters will be trained, with what targets and loss?
- What differentiating feature is feasible and how will it be evaluated?
- What baseline and single controlled technical change will be compared?
- What compute, subset size, and fallback are available?
