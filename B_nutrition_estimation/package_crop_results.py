"""Package reviewable experiment-006 evidence without images/features/checkpoints."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    required = [args.run_root / name for name in (
        "environment-006.json", "crop_metadata.json", "clip_original.npz",
        "clip_crop.npz", "comparison-006.json", "results/B4/metrics.json")]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit(f"Cannot package incomplete run; missing: {missing}")
    large = [args.run_root / "clip_original.npz", args.run_root / "clip_crop.npz"]
    manifest = {str(path.relative_to(args.run_root)): {
        "sha256": sha256(path), "bytes": path.stat().st_size} for path in large}
    include = [args.run_root / name for name in (
        "environment-006.json", "crop_metadata.json", "comparison-006.json")]
    result = args.run_root / "results/B4"
    include.extend(sorted(path for path in result.iterdir()
                          if path.is_file() and path.suffix in (".json", ".csv")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("artifact-manifest-006.json", json.dumps(manifest, indent=2) + "\n")
        for path in include:
            archive.write(path, str(path.relative_to(args.run_root)))
    print(json.dumps({"output": str(args.output), "files": len(include) + 1,
                      "bytes": args.output.stat().st_size}, indent=2))


if __name__ == "__main__":
    main()
