# 4. Decoder fine-tuning with automatic prompts

`train.py` uses `../shared/train_prompted.py`, starting anew from original SAM.
It freezes RGB-derived cues before training and reuses a verified encoder cache
from a no-prompt trial. `predict.py` derives cues with ORIGINAL SAM before loading
the trained decoder. Human masks are never used to supply inference positions.
