"""Print (default) or execute training commands based on archived run arguments."""
import argparse
import ast
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

def commands(args):
    selection = json.loads((ROOT / 'results/run_selection.json').read_text())
    tree = ast.parse((ROOT / 'src/train.py').read_text())
    options = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'add_argument':
            options.update(v.value[2:] for v in node.args if isinstance(v, ast.Constant)
                           and isinstance(v.value, str) and v.value.startswith('--'))
    data_root = Path(args.data_root).resolve()
    if args.execute:
        if not (data_root / 'imgs/train').is_dir() or not (data_root / 'condition_image_level.csv').is_file():
            raise SystemExit('Training data/condition manifest missing: ' + str(data_root))
    selected = 0
    for run in selection[args.cohort]:
        record = json.loads((ROOT / 'results' / args.cohort / run / 'final_metrics.json').read_text())
        saved = record['args']
        if args.model and saved['model'] != args.model:
            continue
        if args.seed is not None and saved['seed'] != args.seed:
            continue
        selected += 1
        command = [sys.executable, str(ROOT / 'src/train.py'), 'train_eval',
                   '--root', str(data_root), '--out_dir', str(Path(args.out_dir).resolve()),
                   '--exp_name', f'reproduce_{args.cohort}']
        for key, value in saved.items():
            if key not in options or key in {'root', 'out_dir', 'exp_name', 'run_script', 'dry_run'}:
                continue
            if key == 'condition_csv':
                value = str(data_root / 'condition_image_level.csv')
            if key == 'pidnet_pretrained' and value:
                value = str(Path(args.pidnet_pretrained).resolve())
                if args.execute and not Path(value).is_file():
                    raise SystemExit('PIDNet pretrained checkpoint missing: ' + value)
            if isinstance(value, bool):
                command.append('--' + ('' if value else 'no-') + key)
            elif isinstance(value, list):
                command += ['--' + key] + [str(v) for v in value]
            elif value is not None:
                command += ['--' + key, str(value)]
        print(f'# {run}', flush=True)
        print(shlex.join(command), flush=True)
        if args.execute:
            subprocess.run(command, cwd=ROOT, check=True)
    if not selected:
        raise SystemExit('No matching recorded runs')

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cohort', choices=['main', 'supplementary'], default='main')
    p.add_argument('--model')
    p.add_argument('--seed', type=int, choices=[42, 3407, 2026])
    p.add_argument('--data-root', default='data')
    p.add_argument('--out-dir', default='outputs')
    p.add_argument('--pidnet-pretrained', default='checkpoints/PIDNet_S_ImageNet.pth.tar')
    p.add_argument('--execute', action='store_true', help='Run training commands; the default prints them.')
    commands(p.parse_args())
