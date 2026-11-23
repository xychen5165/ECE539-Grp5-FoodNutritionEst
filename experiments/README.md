# Experiments

Keep each experiment's settings, optional notebook, and small results together. Use names such as `001-baseline` and `002-<technical-change>` once the experiment is defined.

Copy [run-template.md](run-template.md) into the experiment directory as a run record. For repeated runs, use distinct run IDs and separate result paths. Update this index with links, status, and a short verified conclusion.

Shared implementation belongs in `src/`. Notebooks and configuration files are optional; exact reproducible settings and commands are required for reported results. Store large artifacts externally and document access.

The evaluation plan includes a training-set median baseline, an original-versus-SAM-masked image comparison, and joint meal ranking versus protein-only ranking. See the [evaluation plan](../docs/architecture.md#training-and-evaluation). Add run records and measured results here as experiments are executed.

## Index

| Run | Status | Result (development split) |
|---|---|---|
| [001-median-baseline](001-median-baseline/README.md) | Completed | Normalized nutrient error 0.737; energy error 172.1 kcal |
| [002-clip-head-original](002-clip-head-original/README.md) | Completed | Network on CLIP features: 0.360; energy error 74.7 kcal |
| [003-regularization](003-regularization/README.md) | Completed | Best (weight decay 0.001, dropout 0.5): 0.354; gain within noise |
| [004-ab-integration](004-ab-integration/README.md) | Completed smoke / full comparison pending | Real-image A→B smoke passed; no predictive-benefit claim |
| [005-b-input-comparison](005-b-input-comparison/README.md) | Completed | B3 0.3384; gray-mask-only B1 worsened, and B2/B3 did not establish improvement over B0 |
| [006-b-natural-crop](006-b-natural-crop/README.md) | Completed | B4 0.3255; all three paired B4−B0 intervals favored natural-crop dual view |
| [007-b-partial-clip-finetune](007-b-partial-clip-finetune/README.md) | Running | B5 adapts shared CLIP visual block 11; no final result until all seeds finish |
| [008-b-two-block-clip](008-b-two-block-clip/README.md) | Preparing | B6 protocol and implementation committed; tests passed, input verification pending; no measured result |
| [Part A method comparison](../A_food_segmentation/method_comparison/README.md) | Completed on development | Five template-based run records; selected no-prompt fine-tuned decoder reached 0.9717 mean IoU |

Record-format and evidence compliance for all entries above is reviewed in
[record-audit.md](record-audit.md). “Completed” describes the recorded development
run, not final-test evaluation or teammate approval.

The remaining B6–B10 numbering, controlled settings, branch workflow and report
requirements are frozen in the [Part B experiment handoff](b-remaining-experiment-handoff.md).
