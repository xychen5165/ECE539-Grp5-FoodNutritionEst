# 2. Pretrained SAM with automatic prompts

`predict.py` uses original SAM for both candidate generation and final output.
`../shared/automatic_prompts.py` derives one box and signed points from RGB and
the candidates only. No human mask is input. No training is needed.
