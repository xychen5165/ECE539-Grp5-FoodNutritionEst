# 002 - CLIP features + regression head (original photos)

Status: Completed

**Result:** the planned network (512 → 128 → 5, absolute loss) has normalized nutrient error **0.360** on the development split, about **36% of a typical dish's amount**, against **0.737** (74%) for the median baseline. Energy error falls from 172 kcal to 75 kcal per dish.

## Question
Can a regression head on frozen CLIP features beat the median baseline, and does a hidden layer or the choice of loss matter?

## Results (development split, 538 dishes, mean of seeds 0, 1, 2)

| Average error | Baseline | Linear, absolute loss | **Network, absolute loss** | Network, squared loss |
|---|---|---|---|---|
| Mass | 120.3 g | 59.4 g | **50.8 g** | 51.2 g |
| Energy | 172.1 kcal | 82.8 kcal | **74.7 kcal** | 76.8 kcal |
| Protein | 15.1 g | 8.1 g | **6.7 g** | 6.9 g |
| Carbohydrate | 12.0 g | 7.0 g | **6.9 g** | 7.0 g |
| Fat | 10.6 g | 5.8 g | **5.4 g** | 5.7 g |
| **Normalized nutrient error** | 0.737 | 0.397 ± 0.002 | **0.360 ± 0.009** | 0.372 ± 0.006 |
| 95% interval (seed 0) | 0.672–0.812 | 0.348–0.451 | 0.326–0.419 | 0.337–0.423 |
| Best epoch (seeds 0, 1, 2) | — | 18, 20, 18 | 10, 12, 18 | 12, 11, 26 |

± is the standard deviation across seeds. The normalized nutrient error is the average error divided by a typical dish's amount, averaged over energy, protein, carbohydrate and fat (see [001](../001-median-baseline/README.md)).

## Findings
- All three models halve the baseline error, with intervals far from the baseline's.
- The hidden layer helps (0.360 vs. 0.397 for the linear model).
- Absolute loss is slightly better than squared loss (0.360 vs. 0.372), as it matches the reported error.
- The network overfits after about 10 epochs: training loss keeps falling (0.17 at epoch 10, 0.10 at epoch 30) while the development error stays near 0.37. Keeping the best development epoch handles this ([003](../003-regularization/README.md) tests regularization).
- Fat and protein are hardest (about 36–40% of a typical amount); hidden oil and sauces are not visible in a photo.

## Setup
- Contributor: xychen5165; 2026-11-21 (America/Chicago)
- Code: [src/train_head.py](../../src/train_head.py), scored with [src/evaluate.py](../../src/evaluate.py); repository at `aec9a6c` plus the training script (committed with this record)
- Features: frozen CLIP ViT-B/32 (OpenAI weights), original photos padded to square ([B_nutrition_estimation/README.md](../../B_nutrition_estimation/README.md#clip-features)); file fingerprint `49225c72281ef059`
- Data: `train.csv` (2,135 dishes, `b4713f8b5125b1f7`), `dev.csv` (538 dishes, `e3025e1a9f47e963`). Test not used.
- Settings (plan): inputs standardized and targets divided by training averages; AdamW, learning rate 0.001, weight decay 0.0001, batch size 32, 30 epochs, best development epoch kept; seeds 0, 1, 2. Linear predictions clipped at 0; the network uses Softplus.
- Environment: [requirements.txt](../../requirements.txt), laptop processor, about 15 seconds per model (3 seeds)

Commands:
```
python src/train_head.py --model linear --loss l1 --name 002-clip-head-original/linear-l1
python src/train_head.py --model mlp --loss l1 --name 002-clip-head-original/mlp-l1
python src/train_head.py --model mlp --loss mse --name 002-clip-head-original/mlp-mse
```

## Evidence
Each subfolder has `metrics.json` (settings, best epoch and metrics per seed), `training_log.csv` (training loss and development error per epoch) and `dev_predictions_seed<N>.csv`. Model weights (`checkpoints/`) stay local and are not in Git.

## Limitation
Development was used to choose the best epoch and compare models, so these scores are slightly optimistic. Final numbers come from the test split once, at the end.
