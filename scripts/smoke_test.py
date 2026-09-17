"""CPU smoke: real checkpoint, real sample, metrics, and timestamped CLI outputs."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

def run(args):
    import numpy as np
    import torch
    from framework.data import LiquidDataset
    from framework.metrics import binary_metrics, boundary_f1, liquid_level
    from models.factory import build_model_from_checkpoint

    torch.set_num_threads(1)
    mask = np.zeros((8, 8), dtype=bool)
    mask[4:, 2:6] = True
    assert binary_metrics(mask, mask)['dice'] == 1.0
    assert abs(liquid_level(mask) - 4/7) < 1e-12
    assert boundary_f1(np.zeros_like(mask), np.zeros_like(mask)) == 1.0
    ds = LiquidDataset(args.data_root, 'test', train=False)
    sample = ds[0]
    assert len(ds) == 1354
    assert tuple(sample['image'].shape) == (3, 256, 256)
    model, checkpoint = build_model_from_checkpoint(args.checkpoint, torch.device('cpu'))
    with torch.inference_mode():
        logits = model(sample['image'].unsqueeze(0))
    assert tuple(logits.shape) == (1, 1, 256, 256)
    assert torch.isfinite(logits).all()
    # The direct forward covers numerical shape/loading; CLI verifies packaging and output paths.
    del model, logits, checkpoint
    image = next(p for p in sorted((args.data_root / 'imgs/test').rglob('*')) if p.suffix.lower() in {'.jpg', '.png', '.jpeg'})
    with tempfile.TemporaryDirectory() as temp:
        result = subprocess.run([sys.executable, str(ROOT / 'src/predict.py'),
                                 '--input', str(image), '--checkpoint', str(args.checkpoint),
                                 '--out_dir', temp, '--device', 'cpu', '--no-amp'],
                                cwd=ROOT, text=True, capture_output=True, timeout=180)
        assert result.returncode == 0, result.stdout + result.stderr
        assert list(Path(temp).rglob('predictions.csv'))
        assert list(Path(temp).glob('**/masks/*.png'))
        assert list(Path(temp).glob('**/overlays/*.png'))
    print(json.dumps({'status': 'passed', 'test_frames': len(ds), 'output_shape': [1, 1, 256, 256],
                      'device': 'cpu', 'cli_prediction': 'passed'}, indent=2))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    run(parser.parse_args())
