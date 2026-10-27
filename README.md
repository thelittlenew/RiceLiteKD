# RiceLiteKD

论文使用的原始训练、模型、实验和测速代码归档。

本仓库按模块整理已有代码与配置，**未修改原始代码内容**。原始绝对路径、导入方式和实验依赖保持原样；这是代码归档，不是重新封装的一键复现工程。数据集、图片、模型权重、训练日志和论文文件不随仓库提供。

## 目录

| 目录 | 内容 |
| --- | --- |
| `pseudo_masks/` | 无标签聚类、自动种子拟合、三路投票生成与独立评分原始代码 |
| `stage1/` | DINOv3 源域自监督适配 |
| `stage2/` | 伪标签教师训练，以及跳过 Stage 1 的教师训练 |
| `stage3_lite_20260926/` | 最终 CGNet-Refine7 模型、完整源蒸馏、损失函数与源数据准备代码 |
| `stage3_comparison_20261006/` | 最终 RiceLiteKD 下游微调、评估和随机初始化学生对照 |
| `cea_public_20260915/` | RiceSEG 教师与基础模型配置，以及被 Refine7 构建器引用的历史配方 |
| `cea_cvrp_full_20260923/` | CVRP_full 对应配置与评估代码 |
| `mmseg_extensions/` | 原有 MMSegmentation 自定义骨干扩展 |
| `segmentation_comparisons_v2_20261011/` | 对比模型实现、训练入口和配置快照 |
| `random_teacher_20261016/` | 教师完全随机初始化消融 |
| `limited_annotations/` | 20%、30%、40%、50% 标注实验的原始代码版本与 48 组配置 |
| `flops_fps_20261014/` | 论文使用的服务器 FLOPs / FPS 测量和自定义算子计数 |

## 主要入口

- 无标签聚类：`pseudo_masks/unsupervised_only/global_unsupervised.py`
- 伪掩膜生成：`pseudo_masks/unsupervised_seed_refinement/auto_seed_refinement.py`，论文对应 `--methods global_c0_vote5`
- Stage 1：`stage1/stage1_train_dinov3_small_rawpth.py`
- Stage 2：`stage2/stage2_train_dinov3_mask2former_pseudolabel.py`
- Stage 3：`stage3_lite_20260926/train_full_refine7_distill.py`
- 最终学生结构：`stage3_lite_20260926/matched_cgnet_refine7_recipe.py`
- 蒸馏损失：`stage3_lite_20260926/full_distill_components.py`
- 最终学生下游训练：`stage3_comparison_20261006/run_segmentation.py`
- 对比模型训练：`segmentation_comparisons_v2_20261011/run_job.py`
- 少量标注实验：各实验目录内的 `train_downstream.py` 和 `train_table4.py`
- 服务器测速：`flops_fps_20261014/benchmark.py`

`cea_public_20260915` 和 `cea_cvrp_full_20260923` 中的旧 Stage 3 配置作为配方依赖保留；最终 RiceLiteKD 使用 `stage3_comparison_20261006/configs` 中的 Refine7 配置。对比工程的公共模块保留了原有其他模型实现，以保持源文件完整。

## 依赖与来源

代码涉及 PyTorch、TorchVision、MMSegmentation、MMCV、MMEngine、NumPy、Pillow、OpenCV、scikit-learn、Albumentations、timm、einops 和部分模型的 mamba-ssm。具体依赖以相应源文件为准，本次整理未重新安装训练环境或运行训练。

对比模型的原始仓库地址、提交版本与适配记录保留在 `segmentation_comparisons_v2_20261011/source_provenance.json`；各源文件中的作者与版权声明原样保留。

`SHA256SUMS.txt` 记录归档代码与配置的校验值。本次整理已逐文件确认与本地原始文件一致。
