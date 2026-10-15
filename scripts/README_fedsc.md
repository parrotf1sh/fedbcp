# FedSC 批量对比实验

在项目根目录、已安装项目依赖的环境中运行：

```bash
python scripts/run_fedsc_experiments.py
```

单 GPU 顺序运行，默认 300 轮，训练种子 `[2022, 2023]`，只改变训练种子。
划分种子固定 `19940817`，客户端抽样沿用逐轮 `numpy.seed(round_index)`。
参数统一放在脚本头部。缺失数据集自动下载；划分复用与 FedAvg、MOON、FedProc、FedNH 相同的缓存。
LDA 最多尝试 128 次，随后按公共规则从充足客户端转移样本补足不足 10 个样本的客户端；不复制样本。

| 实验 | 数据集 | 变化参数 | 训练次数 |
| --- | --- | --- | --- |
| 主实验：LDA | CIFAR10、CIFAR100、Tiny-ImageNet | α=0.03/0.1/0.3 | 18 |
| 主实验：分片 | 同上 | 每客户端分片数 2/4/6 | 18 |
| 补充：参与率 | CIFAR100，LDA α=0.1 | q=0.05/0.2 | 4 |
| 补充：本地更新 | CIFAR100，LDA α=0.1 | E=1/10 | 4 |

共 44 次训练、22 个条件，每条件两个种子。补充实验的 q=0.1、E=5 基准直接复用主实验。
默认 N=100、q=0.1、E=5、batch=50；SGD lr=0.01、momentum=0.9、weight_decay=1e-5；
逐轮 StepLR gamma=0.99。CIFAR 使用 `fedavg_cifar`，Tiny-ImageNet 使用 `fedavg_tiny`，无投影头。

## 保留源码行为

算法温度 `CL_TAU=0.05`；首轮 CE，后续 CE+RPCL+CPDR，系数均为 1。
每次参与重新建立 SGD 并清空状态。模型聚合保留先乘样本量、求和、再除总样本量的操作顺序。
原型仅来自最后一个本地 epoch 已有训练前向，不另跑提取过程。

本队列明确标为 `provided_source_behavior_v1`，名称含 `source`，W&B tags 含 `source-behavior`。
保留当前移植的正原型列表异常回退、关系原型临时修正未写回、按类别列表位置使用权重、
原地改写关系原型等行为。有限特征下 CPDR 为零，RPCL 仍参与训练。
不能把结果描述成修复这些行为后的 FedSC；详细边界见 [移植说明](../src/algorithms/fedsc/README.md)。
空负池、分布距离总和为零等错误保持原失败语义，单任务失败后队列继续下一项。

## 日志与通信

每次独立训练对应一个 W&B run，项目默认 `Point1`，自动填写名称、tags 和完整 config。
每轮记录全局 Top-1、CE、Macro-F1、最差 20% 类别平均召回、各类别召回；
训练在线准确率、按实际处理样本数加权的 CE/RPCL/CPDR/总损失，以及时间、通信量。
训练指标覆盖所有本地 epoch；上传的原型与类别计数仅覆盖最后 epoch。
批量模式不进行额外客户端测试、训练集评估或训练前评估。

FedSC 特有指标包括回退次数/查询次数/回退率、每 batch 共享负池大小均值及最大值、
更新后的类别覆盖、关系/一致性原型个数、原型涉及样本数、额外提取前向次数（0）、
本地原型均值计算时间和服务器原型更新时间。首轮查询与回退均为 0；其回退率按 0 记录。
负池统计保留源码的重复项，均值按 batch 等权；首轮负池为 0。

通信估算在公共模型上下传和优化器张量下传基础上，增加每客户端：

- 上传本地原型张量、每个原型的 int64 类别 ID、长度为类别总数的 int64 样本计数。
- 下载关系及一致性原型张量、各字典条目的 int64 类别 ID；列表值另计一个 int64 列表长度。

首轮实际复制的占位张量仅按形状和 dtype 计字节，不读取其未初始化值。
下行量在本轮原型更新前统计，上行量按各实际上传原型统计。
这是逻辑载荷估算，不包括传输协议、Python 容器、tensor 形状元数据或本机 CPU/GPU 拷贝开销。
原型计算时间属于每轮训练时间的子项，不应再次相加。局部计时只覆盖均值归约，
不单独计最后 epoch 特征收集的 Python 开销；该开销已经包含在整轮训练时间中。

输出目录默认 `fedsc_experiments/`，保存任务清单、config、日志、状态、metrics.csv、
clients.jsonl、划分统计、summary.json 和跨种子的 aggregate.csv。
模型、优化器、原型仅驻留内存，不保存 checkpoint、权重或模型 artifact。
划分缓存 NPZ 仅含样本索引。异常重跑使用新 attempt 和新 run，默认跳过已成功任务。
Ctrl+C 停止当前训练与队列，之后重启可继续未成功任务。

```bash
# 可选：8 核 CPU 预生成 18 个不同划分，训练队列仍为串行
python scripts/prepare_partitions.py --source fedsc --workers 8

# 仅生成清单，不训练、不下载、不连接 W&B
python scripts/run_fedsc_experiments.py --dry-run

# 数据与结果目录可自定义
python scripts/run_fedsc_experiments.py --data-root /data --output-root /results/fedsc
```

## 验证范围

使用无训练的队列测试、NumPy 张量替身验证指标与调用流程，并检查 Python 语法。
覆盖任务去重、两种子配对、18 个共享划分、异常继续、成功跳过、损失分项、
最后 epoch 原型收集、源码回退与失败分支、通信统计和统一指标输出。
没有运行小规模训练、真实 PyTorch/GPU 前反向、数据下载或 W&B 联网；
这些检查不替代实验机上的完整运行验证。
