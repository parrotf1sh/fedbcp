# MOON 串行对比实验

在实验机项目根目录运行，使用已有 PyTorch / torchvision / W&B 环境：

```bash
wandb login
python scripts/run_moon_experiments.py --dry-run
python scripts/run_moon_experiments.py --data-root /absolute/path/to/data
```

所有可配置超参数集中在 `run_moon_experiments.py` 头部。W&B 默认项目是 `Point1`。
调度和下载功能复用 FedAvg；请保留完整项目，不要仅复制 MOON 入口文件。

## 实验矩阵与公平比较

默认 **44 次训练（36 个主实验 + 8 个补充实验），每次 300 轮**，
与 FedAvg 中训练种子为 2022/2023 的任务对应：

| 实验 | 设置 | 次数 |
| --- | --- | ---: |
| LDA 主实验 | CIFAR10/CIFAR100/Tiny-ImageNet × α={0.03,0.1,0.3} × 2 种子 | 18 |
| 分片主实验 | 同上三个数据集 × 每客户端分片数={2,4,6} × 2 种子 | 18 |
| 参与率补充 | CIFAR100、LDA α=0.1、q={0.05,0.2} × 2 种子 | 4 |
| 本地 epoch 补充 | CIFAR100、LDA α=0.1、E={1,10} × 2 种子 | 4 |

默认 N=100、q=0.1、E=5、batch size=50。
补充实验中的 q=0.1 和 E=5 复用主实验，不重复训练。
只改变训练种子 **2022/2023**；划分固定为 19940817，客户端采样仍按从 0 开始的轮次编号设种子。
因此重复实验的标准差只体现固定数据划分/采样下的训练随机性。
两个种子的均值/标准差估计比三个更不稳定，论文中应明确报告 n=2。FedAvg 默认仍保留三个种子。
减少种子不会改变保留任务的配置哈希；相同配置和输出目录下，已成功的 2022/2023 任务仍会跳过。
旧的 2024 结果保留在磁盘上，但不纳入本次队列及重新生成的汇总。

CIFAR 使用 `fedavg_cifar`，Tiny-ImageNet 使用 `fedavg_tiny`。
SGD lr=0.01、momentum=0.9、weight decay=1e-5；StepLR step_size=1、gamma=0.99。
每轮只在聚合后做一次全局测试，数据增强、学习率调度和公共指标与 FedAvg 一致。
使用相同数据根目录运行，以便配对核对 `partition_stats.json` 和 `clients.jsonl`。
LDA 最小样本补足与索引缓存采用[公共划分协议](README_partitions.md)，与 FedAvg 完全共用；
`partition_metadata.json` 的 `indices_sha256` 可用于确认两个算法使用完全相同的客户端索引。

MOON 使用项目已有的 backbone features（无额外投影头），默认 μ=0.1、τ=0.5：

```text
total_loss = cross_entropy + mu * contrastive_loss
```

这些是项目默认参数，不是经验证集搜索得到的最优值。本队列不额外搜索 μ/τ。
如果手动修改，任务标识、W&B 名称和 config 会同步变化，避免误跳过其他参数的结果。

## 历史模型与资源

- 每客户端保留它最近一次参与后的本地模型，未参与客户端的历史不变。
- 初始历史沿用初始全局模型。未参与过的客户端共享只读 CPU 初始快照，首次更新后获得独立快照。
- 所有历史快照均为 detached CPU 副本，客户端加载时不会修改快照。
- 训练时只将当前客户端需要的全局/历史参考模型放到 GPU；参考模型冻结、eval、no_grad。
- 每个客户端完成后释放其参考模型对象，历史更新不依赖 checkpoint。
- **不保存任何模型权重、优化器状态或 checkpoint 文件，不上传模型 artifact。**
- CPU 历史内存随已参与客户端数量增加；若几乎所有客户端都已参与，约需 N 份模型权重加一份初始快照。
  额外参考前向、CPU 历史拷贝的实际耗时均包含在训练耗时内。
- 通信估计将历史视为客户端本地保留状态，包含当前模型上传/下载及优化器张量下载；
  模拟器的 CPU/GPU 搬运不计作联邦网络流量。该口径写入 config 和 summary。

## 指标与 W&B

公共指标保持与 FedAvg 一致：全局 Top-1、测试 CE、Macro-F1、逐类 Recall、最差 20% 类别 Recall，
本地 online Top-1、训练/评估耗时、累计处理样本数和通信载荷估计。
精度、F1、Recall 都是 0–1 的小数。

MOON 另外记录：

| 字段 | 含义 |
| --- | --- |
| `train/ce_loss` | 本地分类交叉熵 |
| `train/contrastive_loss` | 未乘 μ 的对比损失 |
| `train/weighted_contrastive_loss` | μ 加权后的对比损失 |
| `train/total_loss` / `train/loss` | 本地训练总损失 |
| `moon/history_cache_bytes` | CPU 历史张量的逻辑字节数，不含 Python 对象开销 |
| `moon/history_updated_clients` | 已有独立历史的客户端数 |
| `moon/first_participations` | 本轮首次参与的客户端数 |
| `moon/history_age_count` | 本轮有历史更新时间的客户端数 |
| `moon/history_age_mean/max` | 这些客户端距上次参与的轮数差；连续两轮参与时为 1 |

历史年龄不包含首次参与客户端；当 count=0 时，mean/max 以 0 占位。
所有本地损失按实际处理样本数（含重复 epoch）加权。
不要直接比较 MOON 总损失与 FedAvg CE 的大小；跨算法比较应使用相同的全局测试指标。

每任务对应独立 W&B run，例如：

```text
moon_cifar100_lda-a0.1_n100_q0.1_e5_r300_mu0.1_tau0.5_seed2022
```

同一条件两个种子归入同一个 MOON group；算法、数据集、种子、实验组写入 tags，
完整参数、特征类型、历史缓存/初始化/通信策略写入 config。

## 调度、输出与恢复

数据集不存在时自动下载，沿用 [FedAvg 数据目录布局](README_fedavg.md)。
`--dry-run` 只生成 manifest 并打印队列，不下载、不训练、不创建 W&B run。
任务串行，单个任务失败后打印错误并继续；Ctrl+C 则停止整个队列。
每个输出目录有互斥锁。**同一 GPU 上不要同时启动 FedAvg 与 MOON 两个队列**，应依次运行。

默认输出到 `moon_experiments/`，结构与 FedAvg 相同：
`manifest.json`、`queue_summary.json`、`aggregate.csv`，每任务独立的 config、metrics、summary、日志、
划分统计和参与名单。默认跳过已成功任务；失败任务从第 1 轮重训，并创建新的 W&B run。
`--rerun-successful` 强制重跑。修改代码或 W&B 目标后希望重新开始时，使用新的 `--output-root`。
`aggregate.csv` 对公共结果按两种子计算 mean / sample std，并注明成功种子数。
MOON 专有损失与缓存指标保存在逐轮 CSV 和每次运行 summary 中。

## 非训练验证

```bash
python -m unittest discover -s tests -p 'test_*scheduler.py' -v
```

这些检查覆盖队列、配对配置、参数标识、失败续跑、历史快照隔离与年龄统计等。
历史测试使用模拟张量验证缓存操作，不执行任何模型训练，不需要下载数据或连接 W&B。
