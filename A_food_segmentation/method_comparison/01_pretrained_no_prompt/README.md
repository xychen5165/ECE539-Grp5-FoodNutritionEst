# 1. Pretrained SAM, no external prompts

`predict.py` runs original SAM ViT-B with points/boxes/mask input all absent,
single-mask output, and no fine-tuned decoder. It calls the shared Part A
runtime. No training is needed. See `../COMPARISON.md` for measured results.
