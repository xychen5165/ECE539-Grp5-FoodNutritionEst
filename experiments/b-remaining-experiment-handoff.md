# Part B remaining experiment handoff (B6-B10)

This is the execution and reporting contract for the remaining Part B work.
It is written for a teammate using a terminal or job scheduler, not a notebook.
Discussion can use the team's preferred language; shared code, records and
reports must be in English per [`CONTRIBUTING.md`](../CONTRIBUTING.md).

## 1. Numbering and current evidence

The earlier draft reused B5-B7 for two different idea families. The definitive
numbering is now:

| Condition | Technical change | Required baseline |
|---|---|---|
| B4 | Frozen CLIP, original RGB + natural predicted-mask crop | Completed selected input baseline |
| B5 | B4 + shared CLIP visual block 11 fine-tuning | Running under experiment 007 |
| B6 | B5 + also unfreeze visual block 10 | B5 |
| B7 | Fine-tune the full CLIP **visual** tower; text tower stays frozen | Best of B5/B6; conditional |
| B8 | B4 + gated residual fusion; CLIP stays frozen | B4 |
| B9 | B8 + log1p target transform and Huber loss | B8 |
| B10 | B9 + energy-consistency soft penalty | B9 |

The three former draft labels are therefore renamed B8, B9 and B10. Do not
create another condition called B5, B6 or B7 for gated fusion or loss changes.

Measured development evidence available now:

| Condition | Normalized nutrient MAE, mean ± sample SD (3 seeds) | Record |
|---|---:|---|
| Median constant | 0.737 | [`001`](001-median-baseline/README.md) |
| B0 original frozen CLIP | 0.3423 ± 0.0014 | [`005`](005-b-input-comparison/README.md) |
| B1 gray-mask only | 0.3615 ± 0.0025 | [`005`](005-b-input-comparison/README.md) |
| B2 original + gray mask | 0.3390 ± 0.0008 | [`005`](005-b-input-comparison/README.md) |
| B3 B2 + geometry | 0.3384 ± 0.0022 | [`005`](005-b-input-comparison/README.md) |
| B4 original + natural crop | **0.3255 ± 0.0020** | [`006`](006-b-natural-crop/README.md) |
| B5 last-block CLIP adaptation | Running; do not quote an interim result | [`007`](007-b-partial-clip-finetune/README.md) |

All values are development evidence. The canonical final test remains sealed.

## 2. Authoritative branches and commits

Use `origin/merge-A&B` as the integration source. It contains the selected Part
A implementation, all B0-B4 evidence and the frozen B5 workflow.

Important historical anchors:

- Part A selected no-prompt fine-tuned decoder and its comparison are also on
  `origin/yifei/part-a-sam-method-comparison`; use the integrated copy on
  `origin/merge-A&B` for new B work.
- Completed B4 evidence: commit `be59203423cec763d198b2603dd878460eafedbf`.
- Frozen B5 implementation: commit
  `20a0bc8c80f47ccd154f8d5982d1833cd01032ef`.
- B5 terminal/Colab workflow: commit
  `a71cdd71d333b8ea46be18514388cfd58a4cdbe4`.
- C/D branches are out of scope for these model comparisons. Do not base B
  experiments on `merge-C&D`.

Before starting each condition:

```bash
git fetch origin --prune
git switch -c experiment/b6-two-block-clip 'origin/merge-A&B'
# Use the corresponding focused name for B7-B10.
git status --short
git rev-parse HEAD
```

Suggested branches:

```text
experiment/b6-two-block-clip
experiment/b7-full-visual-clip
experiment/b8-gated-residual
experiment/b9-log-huber
experiment/b10-energy-consistency
```

Never place two conditions in one result directory or silently overwrite a
failed attempt. Preserve a failed directory with an attempt suffix and record
the failure.

## 3. Shared data, inputs and non-notebook setup

The existing external run root contains the large inputs. On the original
Colab/Drive setup it is:

```text
/content/drive/MyDrive/ECE539-FoodNutritionEst/005-b-input-comparison
```

Required external artifacts:

```text
images/<dish_id>/rgb.png
crops/<dish_id>/crop_rgb.png
clip_original.npz
clip_crop.npz
results/B4/dev_predictions_seed{0,1,2}.csv
results/B4/metrics.json
```

Verified feature hashes:

```text
clip_original.npz  307f2e7cdad43bb635d88e1e0e49b237631e76d0ff32d16fdf8a1098e2aabaa2
clip_crop.npz      3e7d4a418445c3914f29fd2eb4b13115fc124b94f8c6ebebb26460e748ea55fd
```

For B6/B7, copy images and crops to node-local storage before training. Repeated
small-file reads from Google Drive or a network filesystem can dominate runtime:

```bash
mkdir -p /scratch/$USER/b-nutrition/images /scratch/$USER/b-nutrition/crops
rsync -a "$SOURCE_ROOT/images/" /scratch/$USER/b-nutrition/images/
rsync -a "$SOURCE_ROOT/crops/" /scratch/$USER/b-nutrition/crops/
test "$(find /scratch/$USER/b-nutrition/images -name rgb.png | wc -l)" -eq 2673
test "$(find /scratch/$USER/b-nutrition/crops -name crop_rgb.png | wc -l)" -eq 2673
```

Use a project-specific environment, preferably Python 3.11+:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install \
  'numpy>=1.26,<3' 'Pillow>=10,<13' \
  open_clip_torch==2.24.0 opencv-python-headless==4.11.0.86 \
  'segment-anything @ git+https://github.com/facebookresearch/segment-anything.git@dca509fe793f601edb92606367a655c15ac00fdf'
python -m unittest discover -s tests -v
```

Record the actual Python, PyTorch, CUDA, GPU and dependency versions. Do not
claim environments are matched when they are not.

## 4. Shared evaluation contract

Unless a condition explicitly changes one item below, preserve:

- Canonical 2,135 train / 538 development split and plate groups.
- Train split SHA-256
  `0fe38b0f6ff50e7031dea091da19675a76adfba533e44e2bff3276cd9e5b9fbd`.
- Development split SHA-256
  `c6462b3444676502915de5606891b4fbe162e248e186c3df7d7bd4d7ee179a31`.
- Targets: mass, energy, protein, carbohydrate and fat.
- Seeds 0, 1 and 2; report all seeds, including failures.
- Development checkpoint selection independently per seed.
- Primary metric: mean of energy/protein/carbohydrate/fat MAE divided by the
  corresponding training-set means; mass is reported separately.
- Per-target MAE in g/kcal and MAE as percent of the development mean.
- Paired 1,000-draw plate-group bootstrap, seed 0, using the same dish IDs and
  same training seed for baseline/candidate. Negative candidate-minus-baseline
  deltas favor the candidate.
- No final-test image, label, prediction or aggregate is opened until the team
  chooses one final method and freezes a separate final-evaluation protocol.

Do not choose a method from mean scores alone. Report the three seed values,
paired confidence intervals, best epochs, train/dev curves and per-nutrient
deltas.

## 5. B6 — last two visual blocks

### Question

Does increasing CLIP adaptation from one to two visual Transformer blocks
improve B5 without the instability of full visual fine-tuning?

### Exact intended change

Start from `B_nutrition_estimation/train_clip_finetune.py`. Keep both views and
one shared CLIP. Change the selected trainable visual parameters from:

```text
transformer.resblocks.11.*, ln_post.*, proj
```

to:

```text
transformer.resblocks.10.*,
transformer.resblocks.11.*,
ln_post.*,
proj
```

Everything else stays at the B5 values:

```text
head LR                 1e-3
selected CLIP LR        1e-5
weight decay            1e-3
head-only warmup        3 epochs
total epochs            30
physical batch          8
gradient accumulation   4
effective batch         32
dropout                 0.5
precision               CUDA float16 autocast + gradient scaling
seeds                   0, 1, 2
```

Add a unit test proving that blocks 10 and 11 are trainable and block 9 is not.
Record selected parameter names and counts in `metrics.json`.

### Comparisons

- Primary: B6−B5.
- Secondary: B6−B4, to determine whether either fine-tuned model beats the
  strongest frozen model.

### Repository location

```text
experiments/008-b-two-block-clip/
  README.md
  protocol.json
  run-records/B6.md
  environment.json
  comparison.json
  results/B6/metrics.json
  results/B6/training_log.csv
  results/B6/dev_predictions_seed0.csv
  results/B6/dev_predictions_seed1.csv
  results/B6/dev_predictions_seed2.csv
```

Large checkpoints remain external; retain their SHA-256 and access path.

## 6. B7 — full visual CLIP fine-tuning (conditional)

“Full CLIP” here means the entire **visual tower**. The unused text tower stays
frozen and must not be added to the optimizer.

### Run gate

Run B7 only if B5 or B6 satisfies both predeclared criteria versus B4:

1. Mean normalized nutrient MAE improves by at least 0.005 absolute; and
2. At least two of three seed-matched paired 95% intervals have upper bounds
   below zero.

If the gate fails, create the B7 run record with status `Not run — prespecified
gate failed`; this is a valid outcome, not missing work.

### Frozen settings if the gate passes

- Start from the better of B5/B6.
- Make every `model.visual` parameter trainable after the same three-epoch
  head-only warmup.
- Text encoder stays frozen.
- Head LR: `1e-3`.
- Visual LR: `1e-6` (lower than B5/B6 because many more parameters update).
- Weight decay: `1e-3`; epochs: 30; seeds: 0/1/2.
- First perform a one-batch memory preflight without reading development labels.
  If batch 8 fits, use batch 8 / accumulation 4. If it OOMs, freeze batch 4 /
  accumulation 8 **before** the real run and record the preflight. Do not change
  it after seeing a development score.

### Comparisons and location

- Primary: B7−the selected B5/B6 source.
- Secondary: B7−B4.

Store under:

```text
experiments/009-b-full-visual-clip/
```

Use the same evidence layout as B6, replacing the condition with B7.

## 7. B8 — gated residual fusion with frozen CLIP

B8 is independent of B5-B7 and can run on cached `clip_original.npz` and
`clip_crop.npz`, so no GPU image encoder is required.

### Exact architecture

Use B4's separately train-standardized 512-dimensional frozen features:

```text
h_original = ReLU(W_original z_original)       # 64
h_crop     = ReLU(W_crop z_crop)               # 64
gate       = sigmoid(W_gate [h_original; h_crop] + b_gate)  # 64
h_fused    = [h_original; h_original + gate * h_crop]       # 128
prediction = Softplus(W_out Dropout(h_fused) + b_out)       # 5
```

Initialize `W_gate` to zero and `b_gate` to `-2`, so training begins with a
small crop correction rather than an uncontrolled gate. Preserve B4's L1 loss,
target mean scaling, AdamW `1e-3`, weight decay `1e-3`, dropout 0.5, batch 32,
30 epochs and seeds 0/1/2. Record the increased parameter count; do not describe
this as parameter matched.

### Comparison and location

- Primary: B8−B4.
- Optional diagnostic: gate mean/std and per-feature saturation fraction below
  0.05 or above 0.95 on development; do not use these to select a checkpoint.

Store under:

```text
experiments/010-b-gated-residual/
```

## 8. B9 — log1p targets and Huber loss

B9 is cumulative from B8. Do not change the image features or gated fusion.

### Exact target/loss change

For each of the five targets, fit on training only:

```text
u = log1p(y)
mean_u = mean(u_train)
std_u = std(u_train) + 1e-6
z = (u - mean_u) / std_u
```

Replace the final Softplus with a linear five-output layer and optimize equally
weighted Huber/SmoothL1 loss on `z` with `delta=1.0`. Convert predictions back:

```text
y_pred = clamp(expm1(z_pred * std_u + mean_u), min=0)
```

All normalization statistics must come from train only and be stored in each
checkpoint. Preserve B8's optimizer, training budget and seeds.

### Comparisons and location

- Primary: B9−B8, isolating the target/loss package.
- Secondary: B9−B4.
- Report tail diagnostics by target (at minimum median, 90th and 95th absolute
  error percentiles) in addition to the shared metrics.

Store under:

```text
experiments/011-b-log-huber/
```

Log1p and Huber are intentionally treated as one predeclared package here. Do
not claim which of the two caused an improvement without a separate ablation.

## 9. B10 — energy-consistency soft constraint

B10 is cumulative from B9. Use the same outputs, inverse transform and primary
Huber loss. Add only:

```text
macro_energy = 4 * protein_g + 4 * carbohydrate_g + 9 * fat_g
consistency = SmoothL1(log1p(pred_energy_kcal), log1p(macro_energy), delta=1.0)
total_loss = B9_huber_loss + 0.1 * consistency
```

The coefficient `0.1` is frozen before execution and must not be tuned on the
same development results. This is a soft relation because fibre, alcohol,
rounding and label noise can prevent exact equality. Never overwrite reported
energy with the macro-derived value.

### Comparisons and location

- Primary: B10−B9.
- Secondary: B10−B4.
- Also report mean/median absolute energy-consistency gap in kcal before and
  after the penalty, plus all ordinary nutrient MAEs. A smaller consistency gap
  is not sufficient if predictive MAE worsens.

Store under:

```text
experiments/012-b-energy-consistency/
```

## 10. Required implementation and preflight tests

Before a full run, each branch must add tests appropriate to its change:

- Exact trainable CLIP parameter names and count for B6/B7.
- Frozen parameters have `requires_grad=False` and do not change after one
  optimizer step.
- Both views share one visual encoder object, not two copies.
- Train-only normalization; perturbing dev values cannot change statistics.
- Output shape `(batch, 5)`, finite inverse-transformed predictions and
  nonnegative physical outputs.
- B8 gate initialization and residual equation.
- B9 log1p round trip and Huber loss on a hand-computed example.
- B10 consistency loss on exact-consistent and inconsistent examples.
- A one-epoch, one-seed smoke run writes a reloadable checkpoint, metrics,
  training log and 538 ID-aligned development predictions. Mark smoke outputs
  as validation only; never report them as experimental results.

Run:

```bash
python -m unittest discover -s tests -v
python -m compileall -q B_nutrition_estimation src tests
git diff --check
```

## 11. Experiment record and report standard

Before training, create the experiment directory and copy
[`run-template.md`](run-template.md). Commit a `Planned` run record containing:

- Contributor and date/time zone.
- Precise hypothesis and the one intended technical change.
- Baseline and every matched/unmatched condition.
- Frozen implementation commit SHA and clean-tree check.
- Exact commands, configuration and environment.
- Hardware/time budget and expected failure policy.
- Train/dev counts and hashes; explicit final-test prohibition.
- Seeds, preprocessing, train-only fitted transforms, loss, optimizer,
  trainable/frozen parameters and metric definitions.

After execution, change status to `Completed` or `Failed` and retain:

```text
README.md                         concise question, table and honest conclusion
protocol.json                    settings frozen before the run
run-records/B<condition>.md      full template-based record
environment.json                 runtime and dependency versions
artifact-manifest.json           hashes/locations for excluded large artifacts
comparison.json                  paired baseline/candidate deltas and intervals
results/B<condition>/metrics.json
results/B<condition>/training_log.csv
results/B<condition>/dev_predictions_seed0.csv
results/B<condition>/dev_predictions_seed1.csv
results/B<condition>/dev_predictions_seed2.csv
```

Checkpoints, raw images, crops and feature arrays stay outside Git. Never commit
credentials, private mappings, final-test predictions or large model files.

The written conclusion must distinguish:

- Numerical improvement from statistically supported improvement.
- Development selection from final-test evidence.
- A failed method from an execution failure.
- The intended change from unavoidable hardware/batch differences.
- Mean score from per-nutrient effects and seed variability.

Update [`docs/development-log.md`](../docs/development-log.md) after every
meaningful attempt, including OOMs and negative results. Update
[`experiments/README.md`](README.md) with status and a one-line verified result.

## 12. Git and review handoff

For each condition:

1. Start from the specified baseline branch/commit and create a focused branch.
2. Commit the planned protocol before the full run.
3. Push implementation/protocol so the exact execution commit exists remotely.
4. Run from a clean checkout and capture `git rev-parse HEAD` plus `git status`.
5. Commit only small reviewable evidence and English records.
6. Push the branch and open a PR into `merge-A&B` (not directly into `main`).
7. PR description must state purpose, technical change, commands, environment,
   results, uncertainty, files kept externally, interface changes and final-test
   access status.
8. Request review from a teammate who did not run the condition.

Do not merge B9 before B8's implementation/result source is explicit, or B10
before B9's source is explicit. Preserve negative results; do not rewrite an
earlier record to make a later method appear cleaner.

## 13. Completion checklist

A condition is complete only when all are true:

- [ ] Protocol and run record were committed before the measured run.
- [ ] Train/dev hashes match and final test was not accessed.
- [ ] All three seeds completed or every failure is documented.
- [ ] Metrics, logs and ID-keyed predictions are retained.
- [ ] Baseline/candidate paired comparison uses 182 plate groups.
- [ ] Large artifacts have hashes and external access instructions.
- [ ] Unit/interface tests and `git diff --check` pass.
- [ ] README, run record, experiment index and development log agree.
- [ ] Branch is pushed and another teammate is requested for review.

