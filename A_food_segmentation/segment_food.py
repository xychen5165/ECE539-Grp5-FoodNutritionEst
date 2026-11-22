"""Primary Part A method: decoder-fine-tuned SAM without external prompts."""
from sam_runtime import run_cli


if __name__=='__main__':
    run_cli('finetuned_no_prompt')
