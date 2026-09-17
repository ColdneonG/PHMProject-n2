# Dataset

The study uses 6,535 frames from 24 physical videos of industrial compressor liquid-level observations. Binary masks annotate the visible liquid region. The images, masks, and sample-level manifests are private.

## Directory structure

```text
data/
  imgs/{train,val,test}/<video>/<frame>.jpg
  masks/{train,val,test}/<video>/<frame>.png
  condition_image_level.csv
  split_manifest.csv
  group_assignment.csv
```

Images and masks share the same frame stem. Mask values greater than 127 denote foreground.

`condition_image_level.csv` has columns `split,video,frame,condition`, where condition is `clear`, `blur`, or `small`. The split validator uses `new_split,video,group,new_frame,condition,dst_image,dst_mask` from `split_manifest.csv`. Destination paths follow the form `data/imgs/train/<video>/<frame>.jpg` and `data/masks/train/<video>/<frame>.png`.

## Split statistics

| Split | Frames | Physical videos | Clear | Blur | Small condition |
| --- | --- | --- | --- | --- | --- |
| Train | 4593 | 13 | 3320 | 879 | 394 |
| Validation | 588 | 5 | 404 | 130 | 54 |
| Test | 1354 | 6 | 965 | 267 | 122 |

All fragments of a physical video share one split. Empty reference masks are excluded. Condition-specific evaluation uses the condition CSV; without that file, the loader falls back to foreground-area heuristics.

## Validation

```bash
python scripts/validate_data.py --root data
```

This command checks the study's split counts, image–mask correspondence, condition mappings, physical-video disjointness, and identical image-file hashes across splits. It is intended for the fixed study dataset. Training on another dataset requires the same image/mask directory structure and an appropriate condition CSV.
