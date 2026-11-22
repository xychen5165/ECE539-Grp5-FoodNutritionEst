# Shared implementation

This directory contains shared data preparation (`build_splits.py`,
`download_photos.py`, `html_plate_review.py`), nutrition evaluation (`evaluate.py`),
and A/B integration checks (`audit_ab_integration.py`, `smoke_ab_pipeline.py`).

Component B's feature extraction, regression training, and median baseline are
in [B_nutrition_estimation](../B_nutrition_estimation/README.md). Import them using
the `B_nutrition_estimation` package, or run their scripts from the repository root.

Experiments and the future product interface should call the same shared logic. Document component input/output contracts before integration. Add dependencies, run instructions, and appropriate interface checks with implementation.
