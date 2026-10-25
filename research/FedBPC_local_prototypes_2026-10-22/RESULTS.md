# 合成诊断完整结果

所有数值来自实际 CPU PyTorch 运行；四个全局均衡类别、八客户端、每轮两客户端、20 轮、E=3。每类训练 128、测试 256 个 8 维高斯输入，模型为小型 MLP。它们不是 CIFAR 结果。

每组三个训练 seed，标准差为样本标准差。每组方法共享训练/测试数据、划分、初始化、参与序列和训练 batch 顺序。同一组的三个 seed 只改变初始化与训练顺序，不能视为三个独立数据划分。测试集从未用于梯度或原型生成。

按实验发生顺序保留全部结果；第二组设计根据第一组观察修订，因此属于探索性对照。第三组在修订版确定后使用新数据/新划分；第五组再使用第三套新数据/划分，只有部分方法。重复基线不是新增独立证据。所有指标均按百分比显示，末 5 轮为每 seed 的五轮平均后再跨 seed 平均。

特别注意：LDA 个别 seed 出现严重退化，平均值差距很大；不应将其当作现实任务的提升幅度。需要同时查看每 seed、逐轮曲线、末段均值，以及原 FedBPC 对照，不能只看最后一轮。

## 初版：真实样本梯度与分类头一起降权

原始目录：[20261022_161208_e1d2c9e0](synthetic/20261022_161208_e1d2c9e0/aggregate.json)。训练 seeds=[7, 11, 19]；数据 seeds=[71001, 71002]；partition_seed=19940817。

| 划分 | 方法 | 最终 Top-1 ± std | 末5轮 Top-1 | 最终 Macro-F1 | 最差类 Recall |
|---|---|---:|---:|---:|---:|
| sharding | fedavg | 94.76 ± 0.65 | 95.87 | 94.76 | 89.06 |
| sharding | original_fedbpc | 94.27 ± 0.89 | 95.31 | 94.23 | 86.46 |
| sharding | complete | 94.11 ± 0.96 | 95.32 | 94.11 | 87.11 |
| sharding | mean_only | 94.11 ± 1.04 | 95.57 | 94.11 | 87.63 |
| sharding | no_alignment | 94.14 ± 0.96 | 95.41 | 94.12 | 86.72 |
| sharding | alignment_only | 94.86 ± 0.54 | 95.99 | 94.86 | 89.45 |
| lda | fedavg | 92.45 ± 2.60 | 86.97 | 92.34 | 81.51 |
| lda | original_fedbpc | 92.77 ± 3.64 | 88.52 | 92.77 | 88.02 |
| lda | complete | 94.66 ± 0.31 | 86.84 | 94.65 | 90.89 |
| lda | mean_only | 93.95 ± 0.26 | 90.54 | 93.92 | 88.41 |
| lda | no_alignment | 94.37 ± 0.88 | 87.19 | 94.36 | 89.84 |
| lda | alignment_only | 92.42 ± 2.18 | 85.62 | 92.32 | 81.77 |

此处只有四类，worst20_recall 取 ceil(4×.2)=1 类，等于最差类 Recall。

## 修订版：保留骨干真实 CE 梯度，同数据配对复查

原始目录：[20261022_161502_200f7749](synthetic/20261022_161502_200f7749/aggregate.json)。训练 seeds=[7, 11, 19]；数据 seeds=[71001, 71002]；partition_seed=19940817。

| 划分 | 方法 | 最终 Top-1 ± std | 末5轮 Top-1 | 最终 Macro-F1 | 最差类 Recall |
|---|---|---:|---:|---:|---:|
| sharding | fedavg | 94.76 ± 0.65 | 95.87 | 94.76 | 89.06 |
| sharding | original_fedbpc | 94.27 ± 0.89 | 95.31 | 94.23 | 86.46 |
| sharding | complete | 94.24 ± 0.64 | 95.46 | 94.24 | 86.85 |
| sharding | mean_only | 94.43 ± 0.93 | 95.70 | 94.44 | 88.41 |
| sharding | no_alignment | 94.47 ± 0.78 | 95.64 | 94.46 | 87.76 |
| sharding | alignment_only | 94.86 ± 0.54 | 95.99 | 94.86 | 89.45 |
| sharding | coupled_backbone | 94.11 ± 0.96 | 95.32 | 94.11 | 87.11 |
| lda | fedavg | 92.45 ± 2.60 | 86.97 | 92.34 | 81.51 |
| lda | original_fedbpc | 92.77 ± 3.64 | 88.52 | 92.77 | 88.02 |
| lda | complete | 94.82 ± 0.52 | 88.17 | 94.82 | 91.80 |
| lda | mean_only | 89.52 ± 6.83 | 87.10 | 89.22 | 73.05 |
| lda | no_alignment | 94.99 ± 0.49 | 88.33 | 94.98 | 92.19 |
| lda | alignment_only | 92.42 ± 2.18 | 85.62 | 92.32 | 81.77 |
| lda | coupled_backbone | 94.66 ± 0.31 | 86.84 | 94.65 | 90.89 |

此处只有四类，worst20_recall 取 ceil(4×.2)=1 类，等于最差类 Recall。

## 修订版：更换数据、划分与训练种子

原始目录：[20261022_161635_9d4bd12b](synthetic/20261022_161635_9d4bd12b/aggregate.json)。训练 seeds=[31, 43, 59]；数据 seeds=[81001, 81002]；partition_seed=20010410。

| 划分 | 方法 | 最终 Top-1 ± std | 末5轮 Top-1 | 最终 Macro-F1 | 最差类 Recall |
|---|---|---:|---:|---:|---:|
| sharding | fedavg | 95.21 ± 0.29 | 95.27 | 95.20 | 89.97 |
| sharding | original_fedbpc | 94.76 ± 0.44 | 95.23 | 94.75 | 89.45 |
| sharding | complete | 95.02 ± 0.35 | 95.31 | 95.00 | 89.45 |
| sharding | mean_only | 94.95 ± 0.44 | 95.25 | 94.94 | 89.71 |
| sharding | no_alignment | 94.99 ± 0.30 | 95.32 | 94.97 | 89.06 |
| sharding | alignment_only | 95.15 ± 0.34 | 95.40 | 95.13 | 89.84 |
| sharding | coupled_backbone | 95.12 ± 0.49 | 95.54 | 95.11 | 89.71 |
| lda | fedavg | 62.57 ± 28.70 | 62.59 | 55.09 | 30.86 |
| lda | original_fedbpc | 94.73 ± 1.35 | 85.12 | 94.75 | 90.89 |
| lda | complete | 92.81 ± 1.59 | 86.80 | 92.79 | 85.94 |
| lda | mean_only | 92.25 ± 1.26 | 79.72 | 92.27 | 85.29 |
| lda | no_alignment | 91.34 ± 2.48 | 86.07 | 91.28 | 82.42 |
| lda | alignment_only | 75.33 ± 29.85 | 71.20 | 72.04 | 57.81 |
| lda | coupled_backbone | 94.11 ± 0.15 | 90.15 | 94.10 | 88.28 |

此处只有四类，worst20_recall 取 ceil(4×.2)=1 类，等于最差类 Recall。

逐 seed 的 LDA 主对照：

| seed | FedAvg | 原 FedBPC | 完整候选 |
|---|---:|---:|---:|
| 31 | 45.90 | 96.29 | 94.53 |
| 43 | 46.09 | 93.95 | 92.48 |
| 59 | 95.70 | 93.95 | 91.41 |

## 固定梯度修正分支：初始设置，已拒绝

原始目录：[20261022_162021_dc8d2fe2](synthetic/20261022_162021_dc8d2fe2/aggregate.json)。训练 seeds=[7, 11, 19]；数据 seeds=[71001, 71002]；partition_seed=19940817。

| 划分 | 方法 | 最终 Top-1 ± std | 末5轮 Top-1 | 最终 Macro-F1 | 最差类 Recall |
|---|---|---:|---:|---:|---:|
| sharding | fedavg | 94.76 ± 0.65 | 95.87 | 94.76 | 89.06 |
| sharding | complete | 94.24 ± 0.64 | 95.46 | 94.24 | 86.85 |
| sharding | control_variate | 88.48 ± 10.52 | 94.27 | 87.28 | 66.67 |
| lda | fedavg | 92.45 ± 2.60 | 86.97 | 92.34 | 81.51 |
| lda | complete | 94.82 ± 0.52 | 88.17 | 94.82 | 91.80 |
| lda | control_variate | 37.92 ± 12.04 | 38.34 | 24.42 | 0.00 |

此处只有四类，worst20_recall 取 ceil(4×.2)=1 类，等于最差类 Recall。

## 固定梯度修正分支：第三组独立设置，已拒绝

原始目录：[20261022_162122_dc64df6b](synthetic/20261022_162122_dc64df6b/aggregate.json)。训练 seeds=[71, 83, 97]；数据 seeds=[91001, 91002]；partition_seed=20130917。

| 划分 | 方法 | 最终 Top-1 ± std | 末5轮 Top-1 | 最终 Macro-F1 | 最差类 Recall |
|---|---|---:|---:|---:|---:|
| sharding | fedavg | 96.42 ± 0.11 | 95.22 | 96.42 | 94.27 |
| sharding | complete | 95.83 ± 0.25 | 94.93 | 95.84 | 93.23 |
| sharding | control_variate | 94.95 ± 0.63 | 94.25 | 94.94 | 89.97 |
| lda | fedavg | 84.28 ± 11.38 | 83.09 | 80.90 | 50.26 |
| lda | complete | 93.07 ± 1.27 | 93.30 | 93.03 | 83.20 |
| lda | control_variate | 26.14 ± 1.97 | 31.63 | 12.02 | 0.00 |

此处只有四类，worst20_recall 取 ceil(4×.2)=1 类，等于最差类 Recall。

## 决策

保留分布原型补全作为可运行候选。方差版本在已记录 LDA 中较均值版更稳，但 sharding 没有一致优势；方向对齐与骨干路由的消融结果混合。因此目前不推荐全面替换原 FedBPC，也不把全部模块称为必要。

固定梯度修正分支在 LDA 明显失败，已从正式代码删除。其快照为 [rejected_control_variate_snapshot.tar.gz](rejected_control_variate_snapshot.tar.gz)，仅供审计；如需复现，在独立仓库副本中解压，不要覆盖当前实现。快照依赖本仓库公共模块与测试环境。

共 156 次短程运行，其中包含重复配对基线；记录的训练及逐轮评估时间之和约 174.6 秒，不含各进程导入与部分构造开销。没有进行真实图像数据训练。

由 summarize_diagnostics.py 从原始 JSON 重建。
