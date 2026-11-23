"""Package small B5 evidence while excluding images/features/checkpoints."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    required = [args.run_root / name for name in (
        "environment-007.json", "clip_original.npz", "clip_crop.npz",
        "comparison-007.json", "results/B5/metrics.json")]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit(f"Cannot package incomplete run: {missing}")
    manifest = {name: {"sha256": sha256(args.run_root / name),
                       "bytes": (args.run_root / name).stat().st_size}
                for name in ("clip_original.npz", "clip_crop.npz")}
    include = [args.run_root / "environment-007.json",
               args.run_root / "comparison-007.json"]
    result = args.run_root / "results/B5"
    include.extend(sorted(path for path in result.iterdir()
                          if path.is_file() and path.suffix in (".json", ".csv")))
    with zipfile.ZipFile(args.output, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("artifact-manifest-007.json", json.dumps(manifest, indent=2) + "\n")
        for path in include:
            archive.write(path, str(path.relative_to(args.run_root)))
    print(json.dumps({"output": str(args.output), "files": len(include) + 1,
                      "bytes": args.output.stat().st_size}, indent=2))


if __name__ == "__main__":
    main()
