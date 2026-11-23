"""Run verified B6 smoke, three-seed training and comparisons; stop after B6."""
from __future__ import annotations
import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from B_nutrition_estimation.verify_b6_inputs import sha


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root', type=Path, required=True)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--clip-cache', type=Path, required=True)
    parser.add_argument('--b5-results', type=Path)
    args = parser.parse_args()
    run = args.run_root
    if run.exists():
        parser.error('Use a fresh run root; preserve previous attempts')
    if git('status', '--porcelain'):
        parser.error('Measured source checkout must be clean')
    if git('branch', '--show-current') != 'ab-supplementary-experiments':
        parser.error('Wrong experiment branch')
    run.mkdir(parents=True)
    (run/'logs').mkdir()
    commit = git('rev-parse', 'HEAD')
    status = {'experiment': 'B6', 'git_head': commit, 'stage': 'preflight',
              'measured_training_started': False, 'scope': 'B6 only; no B7-B10',
              'start_unix': time.time()}

    def save_status():
        (run/'status.json').write_text(json.dumps(status, indent=2)+'\n')

    def execute(name, command):
        status['stage'] = name
        save_status()
        print(name, flush=True)
        with (run/'logs'/(name+'.log')).open('w') as log:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)

    save_status()
    try:
        execute('verify_inputs', [sys.executable, '-u', '-B',
            str(ROOT/'B_nutrition_estimation/verify_b6_inputs.py'),
            '--input-root', str(args.input_root), '--output', str(run/'data-preflight.json')])
        weight_files = [p for p in args.clip_cache.rglob('*.pt') if
                        sha(p) == '40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af']
        if len(weight_files) != 1:
            raise ValueError('Expected exactly one verified official CLIP checkpoint')
        environment = {
            'git_head': commit, 'git_status': git('status', '--porcelain'),
            'python': platform.python_version(), 'platform': platform.platform(),
            'cpu_count': os.cpu_count(), 'cuda_runtime': torch.version.cuda,
            'cuda_device': torch.cuda.get_device_name(0),
            'dependencies': {name: version(name) for name in
                ['torch','torchvision','open_clip_torch','numpy','Pillow','timm',
                 'ftfy','regex','sentencepiece','protobuf','opencv-python-headless']},
            'gpu': subprocess.check_output(['nvidia-smi',
                '--query-gpu=name,driver_version,memory.total','--format=csv,noheader'],text=True).strip(),
            'clip_checkpoint_sha256': sha(weight_files[0]),
            'source_sha256': {name: sha(ROOT/'B_nutrition_estimation'/name) for name in
                ['train_two_block_clip.py','train_clip_finetune.py','compare_two_block_clip.py',
                 'verify_b6_inputs.py','run_b6_pipeline.py']},
            'difference_from_historical_colab': 'Python/PyTorch/hardware differ; retain this limitation in comparisons.',
            'final_test_images_or_labels_accessed': False}
        (run/'environment.json').write_text(json.dumps(environment,indent=2)+'\n')
        execute('unit_tests',[sys.executable,'-B','-m','unittest','discover','-s','tests','-v'])
        common = [sys.executable,'-u','-B',str(ROOT/'B_nutrition_estimation/train_two_block_clip.py'),
            '--image-root',str(args.input_root),'--artifact-root',str(args.input_root),
            '--clip-cache',str(args.clip_cache),'--batch-size','8','--accumulation-steps','4']
        execute('smoke', common + ['--output-root',str(run/'smoke'),'--epochs','1',
                                   '--warmup-epochs','0','--seeds','0'])
        smoke=json.loads((run/'smoke/B6/metrics.json').read_text())
        if smoke['n_dev'] != 538 or not smoke['per_seed']['0']['checkpoint_reload_verified']:
            raise ValueError('B6 smoke interface/reload failed')
        if git('rev-parse','HEAD') != commit or git('status','--porcelain'):
            raise ValueError('Source changed during preflight')
        status['measured_training_started'] = True
        execute('train', common + ['--output-root',str(run/'results'),'--epochs','30',
                                   '--warmup-epochs','3','--seeds','0','1','2'])
        metrics=json.loads((run/'results/B6/metrics.json').read_text())
        if set(metrics['per_seed']) != {'0','1','2'} or metrics['n_train'] != 2135 or metrics['n_dev'] != 538:
            raise ValueError('Incomplete measured result')
        if not all(row['checkpoint_reload_verified'] and all(row['parameter_checks'].values())
                   for row in metrics['per_seed'].values()):
            raise ValueError('Parameter/reload evidence failed')
        compare=[sys.executable,'-B',str(ROOT/'B_nutrition_estimation/compare_two_block_clip.py'),
                 '--b6-results',str(run/'results/B6'),'--output',str(run/'comparison.json')]
        if args.b5_results:
            compare.extend(['--b5-results',str(args.b5_results)])
        execute('compare',compare)
        manifest={}
        for path in sorted((run/'results/B6/checkpoints').glob('*.pt')):
            manifest[str(path.relative_to(run))]={'sha256':sha(path),'bytes':path.stat().st_size,
                'location':'$B6_RUN_ROOT/'+str(path.relative_to(run))}
        (run/'artifact-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        status.update(stage='B6_finished',primary_B5_comparison='completed' if args.b5_results else 'pending_B5_results',
                      stop_after_B6=True,elapsed_seconds=time.time()-status['start_unix'])
        save_status()
        print(json.dumps(status,indent=2),flush=True)
    except BaseException as error:
        status.update(stage_failed=status['stage'],stage='failed',error=str(error),
                      elapsed_seconds=time.time()-status['start_unix'])
        save_status()
        raise


if __name__ == '__main__':
    main()
