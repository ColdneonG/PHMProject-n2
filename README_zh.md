# PHMProject-n2

论文 **Industrial Liquid-Level Segmentation via Local Scale Selection and Structured Hypercomplex Fusion** 的 PyTorch 实现。

作者：Mingyang Gao、Zhiheng Zheng、Wenzhu Dai、Leiquan Wang
单位：中国石油大学（华东）

[English](README.md) · [实验设置](docs/REPRODUCIBILITY.md) · [完整结果](results/TABLES.md)

PHMProject-n2 面向工业压缩机图像中的液体区域分割，结合 ResNet18 编码器、自适应局部尺度选择、独立全局分支和二阶参数化超复数投影，并根据预测区域上边界估计液位。

## 实验结果

种子 42、3407、2026 的测试集均值及样本标准差：

| 指标 | PHMProject-n2 |
| --- | --- |
| Dice | 0.7864 ± 0.0433 |
| IoU | 0.6700 ± 0.0558 |
| Small Dice | 0.6663 ± 0.0186 |
| Boundary F1 | 0.2665 ± 0.0205 |
| Level MAE | 0.0977 ± 0.0113 |
| Blur Dice | 0.7747 ± 0.0240 |

主实验包含 27 次运行，后续补充实验包含 33 次运行，两组结果分别汇总。各次运行的指标与训练日志位于 `results/`。执行 `python scripts/aggregate_results.py` 可生成统计表。

## 环境与数据

使用 Python 3.10。安装命令及软件版本见 [英文首页](README.md) 和 [环境说明](environment/README.md)。

工业数据集暂不公开。权重首批发布范围为 PHMProject-n2 的 3 个正式种子 checkpoint；完整复现发布包包含正文主表与消融实验的 27 个 checkpoint。两组发布包已在本地准备，尚未上线，文件清单与使用方法见 [checkpoints/README.md](checkpoints/README.md)。数据目录结构、标注规则和划分统计见 [data/README.md](data/README.md)。以下命令以本地 `data/` 为数据目录；可通过 `--data-root` 或训练/评估入口的 `--root` 指定其他位置。

## 训练

```bash
python scripts/validate_data.py --root data
python scripts/reproduce.py --cohort main --model phm_project_n2 --seed 42 --data-root data --execute
```

省略 `--execute` 时打印命令；省略模型和种子筛选时选择完整实验矩阵。`--cohort supplementary` 选择补充实验。训练输出保存到 `outputs/` 下带时间戳的目录，依据验证集综合分数选择最佳权重。

## 评估与预测

将 `CHECKPOINT` 设置为训练输出的 `best_model.pt`：

```bash
CHECKPOINT=/path/to/best_model.pt
python src/evaluate.py --root data --checkpoint "$CHECKPOINT" --split test --device cuda --amp --out_dir evaluations
python src/predict.py --input data/imgs/test --checkpoint "$CHECKPOINT" --device cuda --amp --out_dir predictions
```

CPU 推理使用 `--device cpu --no-amp`。完整训练参数、指标定义及实验记录见 [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md)。

## 引用与许可

引用信息见 [CITATION.cff](CITATION.cff)。项目代码采用 [MIT 许可证](LICENSE)，引入的 PIDNet、DDRNet 组件保留上游 MIT 声明。代码来源与依赖许可见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
