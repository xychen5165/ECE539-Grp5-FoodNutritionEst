# Shared implementation

This directory is reserved for shared data loading, neural components, training, evaluation, and inference code. Implementation will follow the agreed architecture.

Experiments and the future product interface should call the same shared logic. Document component input/output contracts before integration. Add dependencies, run instructions, and appropriate interface checks with implementation.

## Implemented packages

- Component C and D implementations live in their component directories: [`C_daily_tracking`](../C_daily_tracking/) and [`D_meal_planning`](../D_meal_planning/).
