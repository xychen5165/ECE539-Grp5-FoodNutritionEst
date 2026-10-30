# Architecture plan

Status: pending team agreement.

## Task

Input: one overhead RGB meal image. Output: estimated calories, protein, carbohydrates, and fat. Target users: college students seeking approximate meal logging.

## Design decisions and open questions

The proposal must diagram at least three substantive, connected, task-relevant components, including at least two neural components, and explain how each contributes to nutrition estimation. The team must decide which components to use and ensure that at least one neural component receives parameter updates.

For each component, document:

- Method and source
- Owner shorthand
- Input/output form, dimensions, and units
- Next consumer
- Taught, pretrained, or later-topic status
- Trained, adapted, or frozen status

The team must also decide:

- Which parameters to train, along with their targets and loss functions
- What feasible differentiating feature to implement and how to evaluate it
- What baseline to use and what single controlled technical change to compare against it
- What compute resources and dataset subset size are available, and what fallback plan to use

## Product integration

Proposed integration: the product interface passes a meal image to the shared inference pipeline and displays its nutrition estimates. If a backend is needed, it will handle requests and invoke that pipeline. Define input validation, output fields, model loading, and failure handling when the team chooses the product scope. 
