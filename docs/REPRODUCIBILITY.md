# Experimental protocol

## Data and preprocessing

The train, validation, and test splits contain 4,593, 588, and 1,354 frames from 13, 5, and 6 physical videos, respectively. Fragments from the same recording share one split; for example, `video18_*` folders belong to one physical video. The dataset includes clear, blur, and small conditions. Empty reference masks are excluded.

Images are resized with their aspect ratio preserved and longest side set to 256 pixels, then center-padded to 256 × 256. Images use edge-replication padding and masks use zero padding. A valid-region mask excludes padding from the loss and evaluation. Images use ImageNet normalization.

Training augmentation consists of horizontal flipping (probability 0.5), brightness and contrast jitter (probability 0.25 each; factors 0.85–1.15), and Gaussian blur (probability 0.15; radius 0.2–0.8). Replacement sampling combines inverse physical-video and condition frequencies, each raised to 0.5, with a weight of 2 for small foregrounds.

## Optimization and model selection

| Setting | Value |
| --- | --- |
| Input resolution | 256 × 256 |
| Epochs | 50 |
| Batch size | 16 |
| Optimizer | AdamW |
| Initial learning rate | 0.0003 |
| Weight decay | 0.0001 |
| Learning-rate schedule | Cosine annealing |
| Early stopping | Disabled |
| Precision | CUDA AMP |
| Prediction threshold | 0.5 |
| Random seeds | 42, 3407, 2026 |
| Loss | 0.5 valid-region BCE + 0.5 valid-region Dice loss |

The best checkpoint maximizes the following validation score:

```text
0.50 × Dice + 0.20 × Small Dice + 0.15 × Blur Dice
+ 0.10 × Boundary F1 − 0.05 × Level MAE
```

PHMProject-n2 uses ResNet18 at output stride 8, local dilation rates 12/24/36, an average-pooling global branch, and an order-2 PHM projection. Detailed parameters are in the per-run `args` objects and `configs/phm_project_n2.yaml`.

## Initialization

- PHMProject-n2, LocalSK variants, U-Net, DeepLabV3+, and PSPNet use ImageNet-initialized ResNet18 encoders.
- SegFormer-B0 uses `nvidia/segformer-b0-finetuned-ade-512-512` through Hugging Face Transformers.
- PIDNet-S uses its ImageNet-pretrained checkpoint. The main comparison comprises the three pretrained runs identified in `results/run_selection.json`.

Training obtains pretrained parameters from their upstream sources. Baseline checkpoint construction also instantiates pretrained models. Cached weights or network access are required for those paths; PHMProject-n2 checkpoint loading initializes the encoder without downloading pretrained parameters.

## Metrics

Dice and IoU are averaged over frames. Small Dice uses the union of frames labeled condition-small and frames with an original foreground ratio at most 0.10. Condition-small Dice is recorded separately.

Liquid level is the mean top foreground coordinate over columns containing foreground, normalized by valid image height minus one. Level MAE averages the absolute difference between defined predicted and reference levels. Empty predictions produce an undefined level and are excluded from this mean. `nonempty_miss_rate` reports the fraction of predictions with foreground area below 0.01.

Boundary F1 uses a 3 × 3 morphological gradient and 5 × 5 dilation when OpenCV is available. The NumPy backend uses label transitions and two four-neighbor dilations. The tested environment uses the NumPy backend. The two implementations can yield different scores and validation checkpoint selections.

## Experiment records

`results/main/` contains 27 runs: six comparison models and three intermediate ablations, each evaluated with three seeds. `results/supplementary/` contains 33 subsequent runs across 11 configurations, including repeated A3/A4 anchors. Each experimental series is summarized independently. Run identifiers, seed assignments, and pilot exclusions are listed in `results/run_selection.json`.

`scripts/aggregate_results.py` computes means and sample standard deviations (`ddof=1`) and checks the main table entries against `results/manuscript_expected.json`. Each `final_metrics.json` contains validation/test aggregates and run arguments; `training_log.csv` records epoch-level measurements. Undefined JSON metric values are represented as `null`.

## Reproducibility notes

The dataset and sample-level records remain private. The [checkpoint packages](../checkpoints/README.md) cover the three PHMProject-n2 runs and the complete 27-run main experiment cohort. These packages are staged locally for publication. The original video-extraction and split-search scripts are unavailable; evaluation uses the fixed split manifest. The environment snapshot describes the version tested on 2026-09-17. Early main runs lack contemporaneous source hashes, complete environment locks, and pretrained-model revision identifiers. The current code also includes metric-code revisions relative to earlier scripts, so exact retraining agreement with the early results has not been established.

Validation covers result aggregation, dataset consistency, command parsing, and CPU checkpoint inference, as detailed in [VALIDATION.md](VALIDATION.md). Historical `fps` entries are runtime measurements from the evaluation loop; no common latency-benchmark protocol is recorded for those entries.
