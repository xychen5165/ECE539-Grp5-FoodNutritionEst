"""Run one fixed selector on the recorded nine-image development subset."""
import argparse
import csv
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selector', required=True,
                        choices=['candidate_union', 'plate_complement', 'plate_appearance'])
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, default=Path('checkpoints/sam_vit_b_01ec64.pth'))
    parser.add_argument('--manifest', type=Path, help='Alternative recorded subset CSV; default: dev9_selection.csv')
    args = parser.parse_args()
    base = Path(__file__).resolve().parent
    with (args.manifest or base / 'manifests/dev9_selection.csv').open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    command = [sys.executable, '-u', str(base / 'run_sam_debug.py'),
               '--selector', args.selector, '--output-dir', str(args.output_dir),
               '--checkpoint', str(args.checkpoint), '--device', 'cpu', '--points-per-side', '16']
    command += [str(base / 'local_data/nutrition5k/imagery/realsense_overhead' /
                    row['dish_id'] / 'rgb.png') for row in rows]
    subprocess.run(command, check=True)


if __name__ == '__main__':
    main()
