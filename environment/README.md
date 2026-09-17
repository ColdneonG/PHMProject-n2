# Software environment

The following versions were used for the CPU checkpoint tests on 2026-09-17. Direct dependencies are pinned in `requirements.txt`; the complete package inventory is in [environment.json](environment.json).

| Component | Version |
| --- | --- |
| Python | 3.10.20 |
| PyTorch | 2.13.0+cu126 |
| torchvision | 0.28.0+cu126 |
| segmentation-models-pytorch | 0.5.0 |
| NumPy | 2.2.6 |
| pandas | 2.3.3 |
| Pillow | 12.2.0 |
| matplotlib | 3.10.9 |
| transformers | 5.14.1 |

```bash
python -m pip install --upgrade pip
python -m pip install --extra-index-url https://download.pytorch.org/whl/cu126 -r requirements.txt
```

The recorded training configuration uses CUDA AMP. Installed CUDA wheels also support CPU inference with `--device cpu --no-amp`. OpenCV is absent in this environment, so Boundary F1 uses the NumPy implementation. Adding OpenCV changes the metric backend.

This snapshot describes the tested software version. Historical experiment provenance is documented in [docs/REPRODUCIBILITY.md](../docs/REPRODUCIBILITY.md).
