"""Train method 3 through the canonical Part A training implementation."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from train_decoder import main

if __name__=='__main__':
    main()
