"""Method 1: original SAM ViT-B, no external spatial prompts."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'shared'))
from run_method import main

if __name__=='__main__':
    main('pretrained_no_prompt')
