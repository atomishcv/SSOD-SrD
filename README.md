# SSOD-SrD

基于 [DEIMv2](https://github.com/Intellindust-AI-Lab/DEIMv2) 与 DINOv3 的半监督目标检测实验代码。仓库包含监督预训练、Teacher–Student 半监督训练、伪标签与一致性相关模块，以及数据转换、评估和可视化脚本。

> 本仓库仅发布代码、配置和文档。数据集、标注、训练日志、可视化输出和模型权重均需自行准备，不随仓库提供。

## 目录

| 路径 | 内容 |
| --- | --- |
| `engine/` | 检测模型、数据集、训练器与半监督训练实现 |
| `configs/` | DEIMv2、SSOD 和数据集配置 |
| `dinov3/` | 项目使用的 DINOv3 代码；相关许可见其 `LICENSE.md` |
| `tools/` | 数据转换、评估及可视化工具 |
| `ew_detr_reproduce/` | 相关 DETR 实验代码 |
| 根目录脚本 | 训练、推理、分析和数据准备入口 |

## 环境准备

建议使用 Python 环境并根据 CUDA 版本安装 PyTorch。基础依赖在 `requirements.txt`，完整环境记录见 `deimv2_environment.yml` 和 `deimv2_requirements.txt`。

```bash
pip install -r requirements.txt
```

部分实验脚本依赖额外的图像处理与分析库，请按脚本导入和运行环境补充安装。

## 数据与权重

1. 自行准备符合对应数据集配置的图像与 COCO 格式标注。
2. 修改 `configs/dataset/` 中的图像、标注和未标注样本路径。现有 DJI 配置保留了原实验环境的绝对路径，运行前必须修改。
3. 自行获取所需 DINOv3 预训练权重，并修改配置中的 `weights_path`。权重文件建议放在本地 `ckpts/`，该目录已被忽略。
4. 训练产物默认写入 `outputs/`；该目录不会上传。

## 训练

监督预训练示例：

```bash
python supervised_pretrain.py -c configs/deimv2/deimv2_dinov3_s_coco.yml
```

半监督训练示例：

```bash
python train_ssod.py -c configs/ssod/deim_ssod_dji.yml
```

其他实验入口包括 `ssod_train.py` 和 `train.py`。运行参数及配置请查看对应脚本的命令行帮助和 `configs/`。不同实验配置对数据、显存和权重的要求不同；上面的命令只是启动示例。

## 许可与致谢

根目录 `LICENSE` 保留了 DEIMv2 项目的 Apache 2.0 许可。`dinov3/` 中保留了其独立许可和说明。请在使用或再发布相关代码时遵守各自许可。感谢 DEIMv2、DINOv3 和相关开源项目的作者。

