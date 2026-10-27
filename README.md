# RiceLiteKD

论文使用的原始训练、模型、实验和测速代码归档。



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

