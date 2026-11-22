"""Validate and capture the Colab runtime before experiment 005 execution."""
from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import shutil
import subprocess

import cv2
import numpy as np
import PIL
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
FROZEN_IMPLEMENTATION = "79a3e117b44de42983551984da2350542da6a7da"
EXPECTED = {
    "data/splits/train.csv": "0fe38b0f6ff50e7031dea091da19675a76adfba533e44e2bff3276cd9e5b9fbd",
    "data/splits/dev.csv": "c6462b3444676502915de5606891b4fbe162e248e186c3df7d7bd4d7ee179a31",
    "A_food_segmentation/model_weights/food_decoder_no_prompt.pt":
        "456c7afdc12dcb8e6ec009526178f5bc542f35493492904f09bc3a1d6af2c4ba",
}
CORE_PATHS = [
    "A_food_segmentation/sam_runtime.py",
    "B_nutrition_estimation/prepare_input_comparison.py",
    "B_nutrition_estimation/train_input_comparison.py",
    "B_nutrition_estimation/compare_input_conditions.py",
    "tests/test_b_input_comparison.py",
]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def command(*args):
    return subprocess.check_output(args, cwd=ROOT, text=True).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable: choose a Colab GPU runtime before execution")
    subprocess.run(["git", "diff", "--exit-code", FROZEN_IMPLEMENTATION, "HEAD", "--", *CORE_PATHS],
                   cwd=ROOT, check=True)
    subprocess.run(["git", "diff", "--exit-code", "--", *CORE_PATHS],
                   cwd=ROOT, check=True)
    subprocess.run(["git", "diff", "--cached", "--exit-code", "--", *CORE_PATHS],
                   cwd=ROOT, check=True)
    actual = {name: sha256(ROOT / name) for name in EXPECTED}
    if actual != EXPECTED:
        raise SystemExit(f"Input fingerprint mismatch: {actual}")
    args.run_root.mkdir(parents=True, exist_ok=True)
    disk = shutil.disk_usage(args.run_root)
    gpu = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
        text=True).strip()
    report = {
        "status": "validated_before_execution",
        "git_head": command("git", "rev-parse", "HEAD"),
        "frozen_implementation": FROZEN_IMPLEMENTATION,
        "core_diff_empty": True,
        "git_status": command("git", "status", "--short"),
        "gpu": gpu,
        "cuda_device": torch.cuda.get_device_name(0),
        "cuda_runtime": torch.version.cuda,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "dependencies": {
            "torch": torch.__version__, "torchvision": version("torchvision"),
            "numpy": np.__version__, "Pillow": PIL.__version__,
            "open_clip_torch": version("open_clip_torch"), "opencv": cv2.__version__,
            "segment_anything_source_commit": "dca509fe793f601edb92606367a655c15ac00fdf",
        },
        "verified_sha256": actual,
        "run_root": str(args.run_root),
        "drive_free_bytes": disk.free,
        "test_images_or_labels_accessed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
