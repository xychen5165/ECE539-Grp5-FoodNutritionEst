# Shared implementation

This directory is reserved for shared data loading, neural components, training, evaluation, and inference code. Implementation will follow the agreed architecture.

Experiments and the future product interface should call the same shared logic. Document component input/output contracts before integration. Add dependencies, run instructions, and appropriate interface checks with implementation.

## Implemented packages

- `daily_tracking`: Component C adult reference targets, correction-safe local intake tracking, Chicago-date daily summaries, and normalized Gordon Nutrislice menu access. It uses only the Python standard library; see [`C_daily_tracking/README.md`](../C_daily_tracking/README.md).
