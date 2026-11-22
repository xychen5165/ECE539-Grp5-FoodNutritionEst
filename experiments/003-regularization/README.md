# 003 - Weight decay and dropout

Status: Completed

**Result:** stronger regularization barely changes the error. The best setting, **weight decay 0.001 + dropout 0.5**, scores **0.354** (energy error 72.9 kcal) against 0.360 for the plan's settings, a difference well inside the uncertainty. Dropout mainly makes results steadier across seeds.

## Question
The network in [002](../002-clip-head-original/README.md) overfits after about 10 epochs. Does stronger weight decay (d2l.ai 3.7) or dropout (5.6) improve the development error?

## Results (network, absolute loss, development split, mean ± standard deviation over seeds 0, 1, 2)

| Weight decay ↓ / Dropout → | 0 | 0.2 | 0.5 |
|---|---|---|---|
| 0.0001 (plan) | 0.360 ± 0.009 | 0.355 ± 0.002 | 0.357 ± 0.001 |
| 0.001 | 0.361 ± 0.005 | 0.356 ± 0.005 | **0.354 ± 0.002** |
| 0.01 | 0.359 ± 0.006 | 0.356 ± 0.001 | 0.357 ± 0.001 |
| 0.1 | 0.356 ± 0.010 | 0.357 ± 0.001 | 0.357 ± 0.004 |

The weight decay 0.0001, dropout 0 cell is the network from 002.

## Findings
- All settings fall between 0.354 and 0.361; the best gain (0.006) is far smaller than the 95% interval (about ± 0.045).
- Dropout cuts the spread across seeds from ± 0.009 to about ± 0.002.
- Regularization does reduce overfitting (seed 0, dropout 0.5 vs. none: training loss at epoch 30 is 0.24 vs. 0.10, and the development error no longer rises late), but the development error stays near 0.36. The limit is more likely what frozen CLIP features capture than overfitting.
- Recommended default for later runs: weight decay 0.001, dropout 0.5.

## Setup
Same data, features and settings as 002 except weight decay and dropout. Command for each cell:
```
python src/train_head.py --model mlp --loss l1 --weight-decay <wd> --dropout <d> --name 003-regularization/wd<wd>-do<d>
```

## Evidence
Each subfolder has `metrics.json` and `training_log.csv`; the best setting (`wd0.001-do0.5`) also has `dev_predictions_seed<N>.csv`. Checkpoints stay local.

## Limitation
Choosing among 12 settings on development makes the best score slightly more optimistic.
