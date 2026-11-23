"""Validate and capture the Colab runtime for experiment 007/B5."""
from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import shutil
import subprocess

import numpy as np
import PIL
import torch

ROOT = Path(__file__).resolve().parents[1]
FROZEN_IMPLEMENTATION = "20a0bc8c80f47ccd154f8d5982d1833cd01032ef"
CORE_PATHS = [
    "B_nutrition_estimation/train_clip_finetune.py",
    "B_nutrition_estimation/compare_finetune_condition.py",
    "B_nutrition_estimation/package_finetune_results.py",
]
EXPECTED = {
    "data/splits/train.csv": "0fe38b0f6ff50e7031dea091da19675a76adfba533e44e2bff3276cd9e5b9fbd",
    "data/splits/dev.csv": "c6462b3444676502915de5606891b4fbe162e248e186c3df7d7bd4d7ee179a31",
}


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
        raise SystemExit("CUDA unavailable: choose a Colab GPU runtime")
    subprocess.run(["git", "diff", "--exit-code", FROZEN_IMPLEMENTATION,
                    "HEAD", "--", *CORE_PATHS], cwd=ROOT, check=True)
    subprocess.run(["git", "diff", "--exit-code", "--", *CORE_PATHS],
                   cwd=ROOT, check=True)
    actual = {name: sha256(ROOT / name) for name in EXPECTED}
    if actual != EXPECTED:
        raise SystemExit(f"Split fingerprint mismatch: {actual}")
    required = [args.run_root / path for path in (
        "clip_original.npz", "clip_crop.npz", "results/B4/metrics.json")]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit(f"Missing retained B4 evidence: {missing}")
    report = {
        "status": "validated_before_execution",
        "experiment": "007-b-partial-clip-finetune",
        "git_head": command("git", "rev-parse", "HEAD"),
        "frozen_implementation": FROZEN_IMPLEMENTATION,
        "core_diff_empty": True,
        "git_status": command("git", "status", "--short"),
        "gpu": subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
             "--format=csv,noheader"], text=True).strip(),
        "cuda_device": torch.cuda.get_device_name(0),
        "cuda_runtime": torch.version.cuda,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "dependencies": {"torch": torch.__version__, "torchvision": version("torchvision"),
                         "numpy": np.__version__, "Pillow": PIL.__version__,
                         "open_clip_torch": version("open_clip_torch")},
        "verified_sha256": actual,
        "run_root": str(args.run_root),
        "drive_free_bytes": shutil.disk_usage(args.run_root).free,
        "test_images_or_labels_accessed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
