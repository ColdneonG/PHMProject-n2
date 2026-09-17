# Third-party notices

## Incorporated source code

| Local source | Upstream | License and copyright |
| --- | --- | --- |
| `src/models/pidnet_official/pidnet.py` | [PIDNet](https://github.com/XuJiacong/PIDNet/blob/4c158cf24ce432f0a8cb43364fae38d93cee0dc3/models/pidnet.py) | [MIT](src/models/pidnet_official/LICENSE), © 2022 Jiacong Xu |
| `src/models/pidnet_official/model_utils.py` | [PIDNet](https://github.com/XuJiacong/PIDNet/blob/4c158cf24ce432f0a8cb43364fae38d93cee0dc3/models/model_utils.py) | [MIT](src/models/pidnet_official/LICENSE), © 2022 Jiacong Xu |
| `BasicBlock`, `Bottleneck`, `segmenthead`, and `DAPPM` in `model_utils.py` | [DDRNet model components](https://github.com/ydhongHIT/DDRNet/blob/de0db317c5af4b0946231f475cf74ca4f51b8ac2/segmentation/DDRNet_23_slim.py), included through PIDNet | [MIT](licenses/DDRNet.txt), © 2020 xingkong |

The two PIDNet files match the upstream source at revision `4c158cf24ce432f0a8cb43364fae38d93cee0dc3`, apart from the additional attribution comments. Local binary-segmentation adaptation is implemented in `src/models/baselines.py`; `pidnet_official/__init__.py` provides the package exports.

PIDNet and DDRNet identify [HRNet-Semantic-Segmentation](https://github.com/HRNet/HRNet-Semantic-Segmentation/tree/0bbb2880446ddff2d78f8dd7e8c4c610151d5a51) as their base code. Its Microsoft MIT notice is retained in [licenses/HRNet-MIT.txt](licenses/HRNet-MIT.txt) for that upstream lineage. HRNet's separate `syncbn` implementation is not included here.

## Coordinate-attention reference

`CoordBranch` in `src/models/phm_project.py` uses the coordinate-pooling, concatenation, split, and two-axis gating structure of [CoordAttention](https://github.com/houqb/CoordAttention/blob/7619bea9acbe260b3793833cc78cef3f124c8112/coordatt.py). The local implementation uses tensor means, GroupNorm/ReLU, additional input/output projections, and a different bottleneck size. Its relation is recorded as a structural reference; an exact file-copy origin has not been established. The reference implementation's [MIT notice](licenses/CoordAttention.txt), © 2021 Qibin (Andrew) Hou, is retained with the attribution.

## External packages

These packages are installed as dependencies; their implementations are not vendored in this repository.

| Package | Use | Upstream license |
| --- | --- | --- |
| [segmentation_models_pytorch](https://github.com/qubvel-org/segmentation_models.pytorch) | ResNet18 encoders, U-Net, DeepLabV3+, PSPNet | [MIT](https://github.com/qubvel-org/segmentation_models.pytorch/blob/v0.5.0/LICENSE) |
| [torchvision](https://github.com/pytorch/vision) | Encoder and pretrained-model support | [BSD-3-Clause](https://github.com/pytorch/vision/blob/7b0e250acf82aac5a2389f54c6855da17bfeace9/LICENSE) |
| [Transformers](https://github.com/huggingface/transformers) | SegFormer-B0 | [Apache-2.0](https://github.com/huggingface/transformers/blob/main/LICENSE) |

SegFormer initialization uses [nvidia/segformer-b0-finetuned-ade-512-512](https://huggingface.co/nvidia/segformer-b0-finetuned-ade-512-512). Pretrained weights retain their separate distribution terms.

## Method references

The PHM formulation follows *Beyond Fully-Connected Layers with Quaternions: Parameterization of Hypercomplex Multiplications with 1/n Parameters* (ICLR 2021), with the [authors' reference implementation](https://github.com/astonzhang/Parameterization-of-Hypercomplex-Multiplications). Local scale selection follows [Selective Kernel Networks](https://openaccess.thecvf.com/content_CVPR_2019/html/Li_Selective_Kernel_Networks_CVPR_2019_paper.html). These are method references, separate from the source-code incorporations listed above.

Reference revisions, file hashes, and attribution categories are recorded in [licenses/provenance.json](licenses/provenance.json).
