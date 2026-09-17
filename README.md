# PHMProject-n2

PyTorch implementation of **Industrial Liquid-Level Segmentation via Local Scale Selection and Structured Hypercomplex Fusion**.

Mingyang Gao, Zhiheng Zheng, Wenzhu Dai, and Leiquan Wang
China University of Petroleum (East China)

[中文说明](README_zh.md) · [Experimental protocol](docs/REPRODUCIBILITY.md) · [Results](results/TABLES.md)

PHMProject-n2 is an encoder–decoder network for binary liquid-region segmentation in industrial compressor images. It combines a ResNet18 encoder, adaptive local scale selection, an independent global context branch, and an order-2 parameterized hypercomplex projection. The segmentation output is used to estimate the liquid level from the upper foreground boundary.

## Results

Test-set mean ± sample standard deviation over seeds 42, 3407, and 2026:

| Model | Dice ↑ | IoU ↑ | Small Dice ↑ | Boundary F1 ↑ | Level MAE ↓ |
| --- | --- | --- | --- | --- | --- |
| PHMProject-n2 | **0.7864 ± 0.0433** | **0.6700 ± 0.0558** | **0.6663 ± 0.0186** | 0.2665 ± 0.0205 | **0.0977 ± 0.0113** |
| DeepLabV3+ | 0.7515 ± 0.0473 | 0.6335 ± 0.0583 | 0.5129 ± 0.0591 | 0.2909 ± 0.0462 | 0.1057 ± 0.0126 |
| SegFormer-B0 | 0.7068 ± 0.0379 | 0.5935 ± 0.0432 | 0.5453 ± 0.1643 | 0.2878 ± 0.0319 | 0.1383 ± 0.0374 |
| U-Net | 0.6846 ± 0.0421 | 0.5948 ± 0.0412 | 0.4524 ± 0.0813 | **0.2945 ± 0.0176** | 0.1190 ± 0.0404 |
| PIDNet-S | 0.6739 ± 0.0371 | 0.5492 ± 0.0332 | 0.3844 ± 0.1070 | 0.2366 ± 0.0264 | 0.1505 ± 0.0235 |
| PSPNet | 0.6436 ± 0.0317 | 0.5454 ± 0.0330 | 0.3586 ± 0.0099 | 0.2322 ± 0.0168 | 0.1254 ± 0.0095 |

PHMProject-n2 obtains a blurred-frame Dice of **0.7747 ± 0.0240**. Main experiments and the subsequent supplementary experiments are reported separately in [results/TABLES.md](results/TABLES.md). Per-run aggregate metrics and training logs are under `results/main/` and `results/supplementary/`.

Regenerate the tables from the included numerical records:

```bash
python scripts/aggregate_results.py
```

## Installation

The tested environment uses Python 3.10 and CUDA 12.6 PyTorch wheels.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --extra-index-url https://download.pytorch.org/whl/cu126 -r requirements.txt
```

Package versions and evaluation-backend details are listed in [environment/README.md](environment/README.md).

## Data

The industrial dataset remains private. The three PHMProject-n2 paper checkpoints form the primary weight release; all 27 main-table and ablation checkpoints form the complete reproducibility release. Both packages are prepared locally and await publication; see [checkpoint packages](checkpoints/README.md). Dataset structure, annotation conventions, and split statistics are described in [data/README.md](data/README.md). Training and evaluation require local data in this format; checkpoint-based evaluation also requires the corresponding trained weights.

The commands below use `data/` as the dataset root. Set `--data-root` in the experiment launcher, or `--root` in the training/evaluation entry points, to use another location. Validate the study's fixed split with:

```bash
python scripts/validate_data.py --root data
```

## Training

Train PHMProject-n2 with the study configuration:

```bash
python scripts/reproduce.py --cohort main --model phm_project_n2 --seed 42 --data-root data --execute
```

Omitting `--execute` prints the command. Omitting the model and seed filters selects all 27 main runs; `--cohort supplementary` selects the separate 33-run supplementary matrix. PIDNet training uses `--pidnet-pretrained checkpoints/PIDNet_S_ImageNet.pth.tar`. Initialization sources are listed in the [experimental protocol](docs/REPRODUCIBILITY.md).

Individual experiments can also be launched directly:

```bash
python src/train.py train_eval --root data --model phm_project_n2 --seed 42 --epochs 50 --patience 0 --batch_size 16 --lr 0.0003 --weight_decay 0.0001 --global_branch avg --phm_n 2 --rates 12 24 36 --device cuda --amp --out_dir outputs
```

Each run writes to a timestamped directory under `outputs/`. The best checkpoint is selected using the validation score defined in [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md). YAML files in `configs/` document the settings; command-line arguments control execution.

## Evaluation and prediction

Set `CHECKPOINT` to a paper checkpoint from the release package or a `best_model.pt` produced by training:

```bash
CHECKPOINT=/path/to/best_model.pt
python src/evaluate.py --root data --checkpoint "$CHECKPOINT" --split test --device cuda --amp --out_dir evaluations
python src/predict.py --input data/imgs/test --checkpoint "$CHECKPOINT" --device cuda --amp --out_dir predictions
```

For CPU inference, replace `--device cuda --amp` with `--device cpu --no-amp`. Evaluation writes aggregate metrics and detail tables. Prediction writes binary masks, overlays, and a CSV index. See [checkpoints/README.md](checkpoints/README.md) for checkpoint metadata.

## Tests

```bash
python -m unittest discover -s tests -p 'test_imports.py'
python scripts/smoke_test.py --data-root data --checkpoint "$CHECKPOINT"
```

The smoke test checks checkpoint loading, metric edge cases, a CPU forward pass, and prediction outputs. Test coverage and recorded results are summarized in [docs/VALIDATION.md](docs/VALIDATION.md).

## Repository structure

| Directory | Contents |
| --- | --- |
| `src/` | Models, datasets, training, evaluation, and prediction |
| `scripts/` | Experiment launcher, result aggregation, and validation tools |
| `configs/` | Experiment settings |
| `results/` | Per-run aggregate metrics, training logs, and summary tables |
| `data/` | Dataset format and split statistics |
| `environment/` | Software versions |
| `docs/` | Experimental protocol and tests |

## Citation

Author and software citation metadata are provided in [CITATION.cff](CITATION.cff).

## License

The project code is released under the [MIT License](LICENSE). Incorporated PIDNet and DDRNet components retain their upstream MIT notices. Source attribution and dependency licenses are listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
