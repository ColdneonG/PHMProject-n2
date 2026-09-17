# Release checkpoint validation

The three PHMProject-n2 paper checkpoints were tested on an independent RTX 4090 D host on 2026-09-17. A new Python 3.10 virtual environment was installed from the pinned direct dependencies. The repository snapshot and weight package were copied without the original project, private dataset, or model caches.

## Results

| Seed | CPU FP32 | CUDA AMP |
|---|---|---|
| 42 | Passed | Passed |
| 3407 | Passed | Passed |
| 2026 | Passed | Passed |

Each command processed three synthetic RGB images with landscape, portrait, and square dimensions. All six commands produced three masks, three overlays, and a three-row CSV. Masks and overlays had the expected 256 × 256 dimensions, masks contained only 0 and 255, and predicted foreground fractions were finite and within [0, 1]. Separate forward passes verified finite outputs of shape 1 × 1 × 256 × 256 for every seed and device.

Inference used initially empty, isolated Hugging Face and PyTorch cache directories, offline flags, and a Python socket guard against outbound IP connections. No pretrained model files were downloaded. Package and checkpoint SHA-256 checks passed. pip check reported no broken requirements.

## Environment and installation

Python: 3.10.21; PyTorch: 2.13.0+cu126; CUDA runtime: 12.6; GPU: NVIDIA GeForce RTX 4090 D.
The complete installed package list is in [release-validation-requirements.txt](../environment/release-validation-requirements.txt).

The host default HTTP package mirror was slow. Installation used an HTTPS mirror, upgraded pip, and curl-assisted wheel transport while retaining pip dependency resolution and wheel hash verification. The direct dependency versions were unchanged. This validates a fresh installed environment; the original README installation command did not complete unassisted on this host.

## Repeating inference

After extracting the primary weight package into the repository root and checking SHA256SUMS, run the following with a directory of RGB images:

```bash
for SEED in 42 3407 2026; do
  python src/predict.py --input /path/to/images --checkpoint checkpoints/phm_project_n2_seed${SEED}.pt --device cuda --amp --out_dir predictions/gpu
  python src/predict.py --input /path/to/images --checkpoint checkpoints/phm_project_n2_seed${SEED}.pt --device cpu --no-amp --out_dir predictions/cpu
done
```

The synthetic-image tests verify checkpoint loading and inference outputs. They do not measure segmentation accuracy or recompute the paper test-set metrics.
