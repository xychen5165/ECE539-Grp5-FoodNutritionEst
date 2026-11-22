# 3. Decoder fine-tuning without external prompts

`predict.py` runs the exact primary method also exposed as
`../../segment_food.py`. `train.py` delegates to `../../train_decoder.py`.
The canonical implementations live directly in Part A; they are not duplicated.
Image/prompt encoders stay frozen; only the decoder is adapted to all visible food.
