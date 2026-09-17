# Tests

## Import and CLI tests

```bash
python -m unittest discover -s tests -p 'test_imports.py'
```

The tests import the training, evaluation, prediction, dataset, and model modules and build the command-line parsers.

## Checkpoint smoke test

```bash
python scripts/smoke_test.py --data-root data --checkpoint /path/to/best_model.pt
```

This test checks metric edge cases, loads a PHMProject-n2 checkpoint, performs a CPU forward pass on one test image, and verifies prediction CSV, mask, and overlay outputs.

## Recorded checks

Checks completed on 2026-09-17:

| Check | Result |
| --- | --- |
| Main-table mean and sample-standard-deviation pairs | 42 matched |
| Main and supplementary training-command parsing | 60 commands passed |
| Import and parser unit tests | 2 passed |
| Fixed dataset counts and condition mappings | 6535 image–mask pairs matched |
| Physical-video split overlap | 0 of 24 videos |
| Identical image-file hashes across splits | 0 |
| Seed-42 PHMProject-n2 CPU output | Finite logits, shape 1 × 1 × 256 × 256 |
| Prediction CLI outputs | CSV, masks, and overlays generated |

The checkpoint test used CPU FP32. GPU retraining and full-test-set re-evaluation were outside the test scope. A subsequent independent-host installation and all three paper checkpoints were tested on CPU and GPU; see [release checkpoint validation](RELEASE_VALIDATION.md).
