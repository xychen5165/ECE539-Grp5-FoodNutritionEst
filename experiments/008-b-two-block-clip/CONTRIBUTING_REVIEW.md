# B6 contribution-guide assessment

- Focused shared supplementary branch; protocol committed before implementation
  and measured execution. Training checkout clean at the recorded source SHA.
- Core B5 source preserved. Only B6 was run; other conditions and final test sealed.
- Nine unit tests, GPU/smoke checks, full 2673-pair verification, checkpoint reload,
  compileall and whitespace checks passed. All three seeds and 90 log rows retained.
- Three numeric results and six paired comparisons were independently recomputed
  from canonical labels and saved predictions with explicit CSV-rounding tolerances.
- Common split hashes, evaluation definitions and plate-group resampling retained.
- Runtime differences from historical Colab are disclosed; no isolated causal or
  final-test claim. B5 is supplied evidence rather than a run performed here.
- Component-local training code follows the existing Part B layout; records/results
  are separate under this experiment. No component C/D interface was changed.
- Raw images, feature matrices, credentials and model checkpoints excluded from Git.
  Selected checkpoint/input hashes and approved-access instructions are retained.
- Teammate review and any later PR/merge remain pending. No merge performed.
