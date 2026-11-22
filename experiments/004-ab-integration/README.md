# A/B integration: test protocol and execution record

Date: November 22, 2026 (America/Chicago).

Merged `origin/yifei/part-a-sam-method-comparison` and `origin/xin-workspace`
into `merge-A&B`, with merge commits `ff82a6b` and `c4803e3` respectively.
Neither branch needed conflict resolution. Nothing has been pushed.

Execution status: preflight, real-image smoke, baseline rerun and two negative
checks passed. Full-data controlled training and final test remain pending.

## Measured execution results

- `preflight.json`: 65 checks passed, including 13 historical B prediction files.
- `baseline-recheck.json`: reexecuted B's baseline over all 538 dev dishes;
  metrics and bootstrap interval exactly match historical results (normalized
  nutrient MAE 0.737, energy MAE 172.089 kcal).
- `smoke-result.json`: nine real images processed by A; original and masked
  features each `(9, 512)`; two trial heads trained with actual weight updates;
  each produced finite nonnegative `(3, 5)` outputs; both checkpoint reloads
  reproduced predictions. No test images downloaded or evaluated.
- `negative-checks.json`: missing image and existing output directory rejected.
- `environment-macos-intel.txt`: complete installed dependency versions;
  `pip check` and imports of torch/OpenCLIP/SAM passed.

The first smoke attempt completed all A/CLIP computation but stopped because
the test harness omitted `loss` from B's configuration. After adding `loss='l1'`,
the second attempt validated and reused the first run's image/mask/feature
artifacts and completed B training and reload checks. The 4.82 seconds in the
result measures only the second attempt, NOT end-to-end image processing.
The report retains diagnostic losses/MAE for debugging; the six-image trial
heads cannot support a claim that masking improves nutrition prediction.

Resume command used (the first run remains intact):

```bash
.venv/bin/python src/smoke_ab_pipeline.py \
  --sam-checkpoint artifacts/ab-integration/checkpoints/sam_vit_b_01ec64.pth \
  --output artifacts/ab-integration/smoke-002 \
  --reuse-preprocessing artifacts/ab-integration/smoke-001
```

## Executed preflight

`python3 src/audit_ab_integration.py --output experiments/004-ab-integration/preflight.json`

Checks canonical split uniqueness and inferred plate-group separation; maps A's
454 train and 103 dev records into B's splits; verifies the selected decoder
SHA256; recomputes A IoU from recorded TP/FP/FN; checks all 13 tracked B prediction
files against canonical dev labels and recomputes MAE/normalized nutrient MAE;
parses merged Python source files. CSV rounding has explicit numeric tolerances.
This does not rerun neural inference or prove that inferred groups equal physical
plates. Test CSV is used only to check membership/group separation.

## Real-image smoke test

Purpose: verify the operational connection, not estimate predictive quality.
Deterministically select six training and three development dishes, each from a
distinct group within its split, using seed 539. Download only their RGB images.
Use the committed no-prompt decoder and verify both decoder and official SAM
checkpoint hashes. A is fixed throughout; no segmentation training occurs.

The runner calls A's real `infer_rgb` and `save_prediction`, B's exact
`pad_to_square`, OpenCLIP encoding, and B's `train_one_seed`/`predict` functions.
It reads A's `masked_rgb.png` directly, resolving the filename mismatch without
changing either component's historical entry point. Original and masked heads
are independently fitted on the same six images, with seed 0 and three epochs.
These temporary heads are not the teammate's trained B models and are not for
user-facing nutrition estimation.

Acceptance checks:

- RGB/mask dimensions agree; mask is boolean, nonempty and not the entire frame.
- No external prompts; food pixels unchanged; background exactly RGB 128.
- Original and masked features each have shape `(9, 512)`, with finite values.
- Normalization uses only the selected training images.
- Each B path trains and returns finite nonnegative `(3, 5)` predictions in
  `mass_g, energy_kcal, protein_g, carbohydrate_g, fat_g` order.
- Save/load of each head and its normalization parameters reproduces predictions
  within `rtol=1e-6, atol=1e-5`.
- Raw images, masked images, feature arrays and trial checkpoints stay ignored.

Reproduction (new output directory on each attempt):

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r experiments/004-ab-integration/requirements-macos-intel.txt
mkdir -p artifacts/ab-integration/checkpoints
curl -fL https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth \
  -o artifacts/ab-integration/checkpoints/sam_vit_b_01ec64.pth
.venv/bin/python src/smoke_ab_pipeline.py \
  --sam-checkpoint artifacts/ab-integration/checkpoints/sam_vit_b_01ec64.pth \
  --output artifacts/ab-integration/smoke-001
```

Environment is deliberately separate from B's original Windows CPU lock. This
Intel Mac uses Python 3.11 / torch 2.2.2 / OpenCLIP 2.24.0. Do not mix new results
with the teammate's torch 2.15 / OpenCLIP 4.2.2 results as if environments matched.
A requires Pillow <13 whereas B pins 13.0.0; the joint environment uses 11.3.0.
An initial installation failed building sentencepiece 0.2.3; pinning 0.2.0 avoids
that source build. Full environment provenance should be saved after validation.

## Planned full controlled experiment (not the smoke test)

Question: does the selected segmentation improve nutrient prediction over the
same CLIP/head recipe on original RGB images?

The masked condition includes A's prior supervised mask-decoder training on
454 annotated training dishes. Any gain is evidence for this complete trained
front end, not for unsupervised masking alone. Keep A fixed during the B runs.

| Condition | Image input | CLIP | Head |
|---|---|---|---|
| Reference baseline | No image | None | Training-target medians |
| Original | Original RGB | Frozen ViT-B/32, OpenAI weights | MLP, separately trained |
| Masked | A method 3 masked RGB | Same frozen CLIP | Same MLP recipe, separately trained |

1. Freeze current train/dev manifests (2,135 / 538 images), A decoder fingerprint,
   CLIP weights, preprocessing and software versions. Obtain only train/dev RGBs.
   Batch selection must use manifests: B's existing extractor enumerates all
   overhead images, so it needs a manifest-aware entry point for this experiment.
2. Generate masks for all 2,673 images without human annotations as inference
   inputs. Preserve image size/food pixels and record image/mask hashes, timing,
   foreground fraction and failure status. Reuse a loaded SAM model for the batch.
   Do not silently exclude failures or selectively switch to original RGB.
   A failure stops the experiment for a documented policy decision and rerun.
3. Cache original and masked CLIP vectors under separate names, with identical
   dish ordering and image coverage. Fit feature mean/std separately for each
   condition on its training features; use identical training-label means.
4. Freeze head settings before comparing: 512→128→5, ReLU, Softplus, L1,
   AdamW lr 0.001, weight decay 0.001, dropout 0.5, batch 32, 30 epochs,
   seeds 0/1/2. Use the minimum dev normalized nutrient MAE epoch independently
   in each run. The older default recipe may be a separately labeled secondary
   comparison, not an extra search for whichever setting favors segmentation.
5. Rerun original and masked heads in the SAME environment. Historical 0.3541
   is context, not the new original control. Save six checkpoints, predictions,
   training logs, input fingerprints and the full package lock.
6. Report per-nutrient MAE (kcal/g), separate mass MAE, four-nutrient normalized
   MAE and seed mean/std. Compute paired deltas (masked minus original).
   Bootstrap the SAME dev plate-group draws in both conditions (1,000 draws,
   seed 0), reporting per-seed paired intervals and the seed-averaged delta.
   Negative delta favors masking; an interval crossing zero is inconclusive.
   Separate runtime correctness from predictive benefit: a higher masked error
   is a valid negative experimental result, not a failed software test.
7. Review predeclared dev strata: ingredient counts 1, 2–4, >=5; retain all
   examples in aggregate results. The 103 annotated A-dev images support
   segmentation/nutrient error diagnostics. The remainder must exclude whole
   A-tuning groups for an auxiliary less-A-tuned view (257 images / 101 groups
   in the merged manifests); it is still B development
   data, not independent final evaluation.
8. Freeze all choices before final test. Use the 506 canonical test images once,
   evaluate original and masked pipelines with paired plate-group intervals,
   and document prior exposure from A's protocol (one test annotation previously
   accessed). Do not claim that all test images are wholly unseen to the team.

## Additional robustness checks before product integration

Test missing/corrupt images, missing/incompatible checkpoints, empty/full masks,
duplicate dish IDs, missing features, NaN/Inf inputs, incorrect target ordering,
and output directory reuse. Ensure each failure is explicit and cannot emit a
plausible nutrition result. Test single-image inference against cached-feature
inference, including identical padding and checkpoint normalization. Confirm
repeat inference with dropout disabled is stable. These checks remain planned
unless explicitly listed in an execution result.

## Resources and reporting

The local host is a six-core Intel Mac with 16 GB RAM and initially about 11 GiB
free disk. Smoke timing includes per-image SAM model loading and must not be
presented as a warmed batch throughput benchmark. Measure batch throughput and
disk use before estimating full-run time. Store downloads/cache under ignored
`artifacts/ab-integration/`; do not duplicate datasets or use test labels to tune.
Only small manifests, numeric results and run logs belong in version control.

Full-data training, robustness matrix and final test are separate from the smoke
run and must be labeled pending until actually executed.
