# FedProc 两种子串行实验

在实验机项目根目录、已有 PyTorch/torchvision/W&B 环境中执行：

```bash
wandb login
python scripts/run_fedproc_experiments.py --data-root /absolute/path/to/data
```

只查看任务、不下载或训练可加 `--dry-run`。可选的 8 核划分预生成：

```bash
python scripts/prepare_partitions.py --source fedproc --data-root /absolute/path/to/data --workers 8
```

已经准备过相同设置的 FedAvg/MOON 划分时直接复用缓存，无需再次预生成。
请更新完整项目，包括公共调度器、报告模块和 FedProc 适配代码，不要仅复制新入口文件。

## 已确认的算法设置

| 参数 | 默认值 |
| --- | --- |
| `AGGREGATION` | `sampled`，按实际参与客户端的样本量归一化聚合 |
| `USE_PROJECT_HEAD` | `True` |
| `OUT_DIM` | 256 |
| `ALPHA_ROUNDS` | 100 |
| `SERVER_MOMENTUM` | 0.0 |
| `TRAIN_SEEDS` | `[2022, 2023]` |
| `N_ROUNDS` | 300 |
| W&B 项目 | `Point1` |

这些参数及公共超参数均集中在脚本头部。旧的 `src/config/fedproc.json` 仍保留 source/无投影头设置，
批量脚本独立生成自己的完整配置，不读取该 JSON 作为默认配置。

投影头使用项目已有 `ModelWithProjection`：骨干特征 → Linear(512,512) → ReLU →
Linear(512,256) → L2 normalize → Linear(256,类别数)。默认两个 CNN 骨干均输出 512 维特征。
该结构在创建优化器前安装，投影层和分类头参与本地训练、广播及聚合；原型处于 256 维投影空间。
全局测试仍使用分类 logits，不改成最近原型分类。

FedAvg/MOON 使用相同骨干，但没有该投影头，因此本组属于“相同骨干、FedProc 带投影头”的比较，
不能表述为全部算法模型结构与参数数量完全相同。实际参数总量记录在运行 summary 中。
`sampled` 与项目保留的原始 `source` 聚合行为不同，论文应明确这两项设置。

## 实验矩阵

| 类别 | 设置 | 次数 |
| --- | --- | ---: |
| LDA 主实验 | CIFAR10/CIFAR100/Tiny-ImageNet × α={0.03,0.1,0.3} × 2 种子 | 18 |
| 分片主实验 | 三个数据集 × 每客户端分片数={2,4,6} × 2 种子 | 18 |
| 参与率补充 | CIFAR100、LDA α=0.1、q={0.05,0.2} × 2 种子 | 4 |
| 本地 epoch 补充 | CIFAR100、LDA α=0.1、E={1,10} × 2 种子 | 4 |
| 合计 | 36 个主实验 + 8 个补充实验 | 44 |

默认 N=100、q=0.1、E=5、batch size=50；q=0.1、E=5 的补充对照复用主实验。
CIFAR 使用 `fedavg_cifar`，Tiny 使用 `fedavg_tiny`；SGD lr=0.01、momentum=0.9、weight decay=1e-5；
StepLR step_size=1、gamma=0.99。每个任务 300 轮，默认使用 cuda:0。

仅训练种子变化，划分种子固定 19940817，客户端采样仍按从零开始的轮次编号设种子。
共复用 18 份[公共划分](README_partitions.md)，包括统一的最少 10 样本补足规则。
用索引指纹及 `clients.jsonl` 核对配对实验。与 FedAvg 按种子配对时选择其 2022/2023 两组，
MOON 已使用相同两组种子。论文报告 n=2，标准差仅描述固定划分与采样下的训练随机性。

## 损失与原型更新

LDA alpha 与 `ALPHA_ROUNDS` 无关。保留当前 FedProc 的有效损失权重（显示轮次从 1 开始）：

- 第 1 轮：CE 权重 1，原型损失权重 0。
- 第 2～100 轮：CE 权重为 `(round - 1) / 100`，原型权重为其补数。
- 第 101～300 轮：CE 权重 1，原型权重 0。

即使权重为零，也保留现有原型损失计算和原型刷新流程。损失定义、首轮初始化顺序、
训练模式下提取原型、缺失类别沿用旧原型/首次置零等行为均保持不变。
批量模式省去原有逐客户端的额外训练/测试评估，改为在线训练统计及每轮一次聚合后全局评估；
因此不宣称与旧入口的评估随机数消耗完全一致。非有限损失会使当前任务报错，队列继续下一个任务。

## 指标与通信口径

公共指标沿用 FedAvg/MOON：每轮全局 Top-1、测试 CE、Macro-F1、逐类 Recall、最差 20% 类别 Recall，
在线训练准确率、学习率、处理样本数、训练/评估耗时、通信字节估计；总结最后一轮及最后 10 轮平均 Top-1。
精度/F1/Recall 为 0～1 小数。全局结果可跨算法比较，本地总损失不应直接与 FedAvg CE 比较。

FedProc 专有指标：

| 字段 | 含义 |
| --- | --- |
| `train/ce_loss`、`train/prototype_loss` | 未加权损失，按实际处理样本数加权统计 |
| `train/weighted_ce_loss`、`train/weighted_prototype_loss`、`train/total_loss` | 有效加权分量与总损失 |
| `fedproc/ce_weight`、`fedproc/prototype_weight` | 含首轮特例的实际生效权重 |
| `fedproc/prototype_initialization_classes` | 首轮初始化覆盖类别数；后续轮为 0 |
| `fedproc/prototype_refresh_classes` | 本轮训练后刷新覆盖类别数 |
| `fedproc/prototype_training_known_classes` | 本轮训练开始时已有观测原型的类别数 |
| `fedproc/prototype_preserved_classes` | 刷新时未观测、沿用历史原型的类别数 |
| `fedproc/prototype_uninitialized_classes` | 刷新后仍从未观测、使用零原型的类别数 |
| `fedproc/prototype_phase_seconds`、`prototype_processed_samples` | 本轮原型阶段时间与额外前向样本数 |
| `fedproc/prototype_upload_bytes`、`prototype_download_bytes` | 本轮原型通信；上传另分 initialization/refresh 字段 |
| `fedproc/prototype_cache_bytes` | 当前全局原型张量字节数，不含 Python 容器 |
| `fedproc/prototype_cumulative_bytes/samples/seconds` | 累计原型通信、处理样本数和阶段时间 |

`time/train_round_seconds` 已包含原型提取和聚合阶段，原型阶段时间作为其子项，不能再次相加。
`train/processed_samples` 仅统计反向训练批次（含重复 epoch），原型前向样本数单独记录。

总通信字节 = 公共的模型上传 + 模型下载 + 优化器张量下载，加上原型通信。
原型上传按每个实际出现类别的特征总和张量 + int64 类别编号 + int64 样本数估计；
下载按发送给每个参与客户端的完整类别原型列表估计。首轮初始化与训练后的刷新上传均计入。
假设客户端在初始化、训练、刷新之间保留本地模型；模拟器重新载入本地状态不算额外网络传输。
这是声明了编码口径的逻辑载荷估计，不是实测网络流量，不含协议、Python 容器或 CPU/GPU 搬运开销。

## 输出与恢复

默认输出目录 `fedproc_experiments/`：队列 manifest/status、逐轮 metrics、参与客户端名单、
划分统计/元数据、每次尝试的 config/train.log/summary，以及跨种子 `aggregate.csv`。
**不保存模型 checkpoint、模型/优化器权重或原型张量文件，不上传模型 artifact。**
划分索引缓存保存在数据目录，与训练输出分开。

每次训练尝试对应一个在线 W&B run，名称包含 sampled、proj256、alpha_rounds、server_momentum 及训练种子。
tags 标明算法、聚合、投影设置、数据集、种子、实验类别；config 记录全部超参数与划分信息。
例如 `fedproc_cifar10_lda-a0.03_n100_q0.1_e5_r300_sampled_proj256_ar100_sm0_seed2022`。

失败时打印错误并继续下个任务；Ctrl+C 停止队列。再次执行相同命令跳过成功任务，失败任务从头训练，
没有权重断点恢复。修改算法参数会改变任务标识；更换代码后希望强制重跑，可用新的输出目录或 `--rerun-successful`。
各输出目录有队列锁，但单 GPU 上仍应依次运行不同算法的队列。

## 验证范围

```bash
python -m unittest discover -s tests -p 'test_*scheduler.py' -v
python -m unittest discover -s tests -p 'test_partition_cache.py' -v
```

覆盖任务矩阵、参数标识/校验、失败恢复、共享划分、损失权重边界、不同 batch 大小的指标加权、
首轮/后续轮原型顺序、缺失类别回退、通信总量及报告落盘。
使用模拟张量/对象和标签索引，不执行真实模型训练、数据下载或 W&B 联网；未运行训练冒烟测试。
