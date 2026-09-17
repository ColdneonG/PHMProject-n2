# Paper checkpoints

Two checkpoint packages are prepared for the paper:

| Package | Contents | Manifest |
|---|---|---|
| `phmproject-n2-paper-seeds.tar` | PHMProject-n2, seeds 42, 3407, 2026 (3 checkpoints); primary release | [Seed manifest](phmproject-n2-paper-seeds.json) |
| `phmproject-n2-paper-reproducibility.tar` | All 27 checkpoints for Tables 1 and 2; complete reproducibility release | [Full manifest](phmproject-n2-paper-reproducibility.json) |

The packages are staged locally and have not yet been published. Each contains checkpoint files, a manifest, SHA-256 checksums, and the repository license notices. The full package includes the three PHMProject-n2 checkpoints; downloading both is unnecessary.

## Selection and paper results

Each checkpoint is the best validation-score checkpoint from its recorded training run. The paper reports the mean and sample standard deviation across three seeds, rather than an ensemble. PHMProject-n2 is also A4 in Table 2. Figure 2 uses seed 42. Supplementary experiments are separate from these 27 main runs.

| Seed | Recorded epoch | Test Dice | Filename |
|---|---:|---:|---|
| 42 | 5 | 0.737245 | `phm_project_n2_seed42.pt` |
| 3407 | 3 | 0.819055 | `phm_project_n2_seed3407.pt` |
| 2026 | 3 | 0.803008 | `phm_project_n2_seed2026.pt` |

## Evaluation

Extract a package into the repository root, then verify its files:

```bash
tar -xf phmproject-n2-paper-seeds.tar
sha256sum -c SHA256SUMS
CHECKPOINT=checkpoints/phm_project_n2_seed42.pt
python src/evaluate.py --root data --checkpoint "$CHECKPOINT" --split test --device cuda --amp
python src/predict.py --input /path/to/images --checkpoint "$CHECKPOINT" --device cpu --no-amp --out_dir predictions
```

The industrial dataset remains private. Evaluation of the reported metrics uses the original test split and the [experimental protocol](../docs/REPRODUCIBILITY.md). The manifests record the original run identifiers, model settings identifiers, seeds, epochs, validation scores, test metrics, and file hashes. Per-run training arguments are in `results/main/<run_id>/final_metrics.json`.

## PIDNet initialization

Retraining PIDNet-S uses ImageNet-pretrained weights from the [official PIDNet repository](https://github.com/XuJiacong/PIDNet), supplied through `--pidnet-pretrained`. These initialization weights are distinct from the trained PIDNet-S checkpoints in the complete paper package.

## Validation

All three paper checkpoints passed CPU FP32 and CUDA AMP inference on an independent host. See [release validation](../docs/RELEASE_VALIDATION.md) for the environment, installation observations, and test scope.
