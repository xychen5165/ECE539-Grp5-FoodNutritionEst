"""Package small experiment-005 evidence while excluding large/private artifacts."""
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
    required = [args.run_root / name for name in
                ("environment.json", "downloads.json", "segmentation.json",
                 "selection.json", "clip_original.npz", "clip_masked.npz",
                 "geometry.npz", "comparison.json")]
    required += [args.run_root / "results" / condition / "metrics.json"
                 for condition in ("B0", "B1", "B2", "B3")]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit(f"Cannot package incomplete run; missing: {missing}")
    large = [args.run_root / name for name in
             ("clip_original.npz", "clip_masked.npz", "geometry.npz")]
    checkpoint = args.run_root / "checkpoints/sam_vit_b_01ec64.pth"
    if checkpoint.is_file():
        large.append(checkpoint)
    manifest = {str(path.relative_to(args.run_root)): {"sha256": sha256(path),
                                                       "bytes": path.stat().st_size}
                for path in large}
    include = [args.run_root / name for name in
               ("environment.json", "downloads.json", "segmentation.json",
                "selection.json", "comparison.json")]
    for condition in ("B0", "B1", "B2", "B3"):
        result = args.run_root / "results" / condition
        include.extend(sorted(path for path in result.iterdir()
                              if path.is_file() and path.suffix in (".json", ".csv")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("artifact-manifest.json", json.dumps(manifest, indent=2) + "\n")
        for path in include:
            archive.write(path, str(path.relative_to(args.run_root)))
    print(json.dumps({"output": str(args.output), "files": len(include) + 1,
                      "bytes": args.output.stat().st_size}, indent=2))


if __name__ == "__main__":
    main()
