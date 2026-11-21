# Part A development log

Contributor/task owner: `YifeiDorthy` (GitHub handle). Component scope: food
segmentation, automatic-prompt experiments and decoder adaptation. This is the
component-local log; the shared work plan and other component workspaces are not
changed by this branch.

## 2026-11-20 - No-prompt supervised decoder experiment

- Attempt: compare original SAM without external cues with decoder-only
  adaptation on 454 training dishes; evaluate the same 103 development dishes.
- Outcome: original single-mask no-prompt output is almost empty. The trained
  decoder reaches development IoU 0.971716, best epoch 12. Frozen encoder
  fingerprints remain unchanged. No final test was used.
- Lesson and next action: a no-prompt adaptation is feasible, but whole-image
  pass can hide small ingredient omissions. Retain region-level diagnostics.
- Evidence: [run 01](run_records/01-pretrained-no-prompt.md),
  [run 03](run_records/03-finetuned-no-prompt.md),
  [training history](results/history_no_prompt.csv).

## 2026-11-20 - Automatic cues and matched prompted adaptation

- Attempt: freeze RGB-derived box/point cues for all 557 approved train/dev
  images, then compare original SAM, inference-only cue addition, and a freshly
  trained decoder with those cues. No reference mask supplies a cue position.
- Outcome: original-plus-cues IoU 0.718993; old trained decoder plus cues
  0.966668; prompted retraining 0.970991, best epoch 5, stopped after epoch 9.
  The no-prompt model remains slightly higher in mean IoU; some small regions
  improve. Cue fingerprints remain fixed. No final test was used.
- Lesson and next action: additional cues are not an automatic improvement.
  Keep all comparisons, and evaluate nutrient prediction before claiming a
  downstream benefit.
- Evidence: [run 02](run_records/02-pretrained-auto-prompt.md),
  [run 04](run_records/04-finetuned-auto-prompt.md),
  [additional control](run_records/05-existing-decoder-added-prompts.md),
  [paired comparison](results/paired_comparison.csv).

## 2026-11-20 - Four-method packaging and local compatibility

- Attempt: create a focused branch from `main`, expose the selected no-prompt
  method at the Part A root, and retain four separate experiment entries.
- Outcome: nine CLI help checks pass; all four prediction entries and the root
  method run on the same previously used dev image. Root/method-3 predictions
  match exactly. Original SAM's float64 grid fails on MPS; its automatic-cue
  stage now uses CPU on that backend, with float32 prompt coordinates for the
  MPS decoder. CUDA generation is unchanged. The prompted pretrained control
  differs by six pixels from the historical CUDA mask on that fixture; both
  fine-tuned predictions match the CUDA fixture exactly.
- Lesson and next action: keep inference shared, retain the measured results as
  historical evidence, and seek teammate review before merge. Full model
  training was not repeated for this packaging change.
- Evidence: [packaging checks](results/packaging_checks.json),
  [comparison](COMPARISON.md), [contribution review](CONTRIBUTING_REVIEW.md).

## 2026-11-20 - Pre-publication review

- Attempt: check the change against the repository contribution guide before
  the authorized commit and branch push.
- Outcome: add template-based run records, artifact access instructions and this
  log. Keep all changes within Part A as explicitly requested; document the
  component-local layout exception. Raw data, weights, private machine details
  and generated images remain excluded.
- Lesson and next action: technical checks support publishing a reviewable
  branch; teammate review and confirmation of the course AI policy remain
  pending before merge/submission as applicable.
- Evidence: [contribution review](CONTRIBUTING_REVIEW.md),
  [artifact access](ARTIFACTS.md).
