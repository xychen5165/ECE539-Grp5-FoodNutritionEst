"""Method 3: the same no-prompt food method exposed at the Part A root."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'shared'))
from run_method import main

if __name__=='__main__':
    main('finetuned_no_prompt')
