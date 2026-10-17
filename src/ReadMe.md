## How to Run Codes?

FedHPro 的双梯度匹配、元原型分类头重训及已确认的独立权重快照修正见
[FedHPro 移植说明](algorithms/fedhpro/README.md)。在 `src` 目录运行
`python main.py --config_path ./config/fedhpro.json`，公共数据、模型和训练配置与 FedBPC 一致。
本次移植仅进行静态验证。

FedNH 的两训练种子、44 任务队列见 [FedNH 批量运行说明](../scripts/README_fednh.md)。
入口：`python scripts/run_fednh_experiments.py`；每次 300 轮，使用等权客户端更新和归一化原型分类头。

所有使用标准 LDA/分片数据入口的算法共享[最小样本补足与划分缓存](../scripts/README_partitions.md)。
8 核实验机可先运行 `python scripts/prepare_partitions.py --workers 8` 预生成 18 份公共划分。

FedAvg 的 66 任务串行实验队列（300 轮、三训练种子、自动下载、仅指标）见
[批量运行说明](../scripts/README_fedavg.md)。入口：`python scripts/run_fedavg_experiments.py`（项目根目录）。

MOON 的两训练种子、44 任务队列见 [MOON 批量运行说明](../scripts/README_moon.md)。
入口：`python scripts/run_moon_experiments.py`（项目根目录），默认 μ=0.1、τ=0.5，历史模型只保存在 CPU 内存。

FedProc 的两训练种子、44 任务队列见 [FedProc 批量运行说明](../scripts/README_fedproc.md)。
入口：`python scripts/run_fedproc_experiments.py`，采用 sampled 聚合和 256 维投影头，每次 300 轮。

FedProto 的移植说明、总体测试性能定义和输出字段见 [FedProto](algorithms/fedproto/README.md)。
在 `src` 目录运行 `python main.py --config_path ./config/fedproto.json`。
主指标是所有客户端在完整测试集上的原型预测准确率均值，实验结果仅保存到 W&B 的 history 和 summary。

FedRCL 的模型层级映射、原始损失分支、等权聚合及学习率适配见
[FedRCL 移植说明](algorithms/fedrcl/README.md)。在 `src` 目录运行
`python main.py --config_path ./config/fedrcl.json`，默认数据、模型及公共训练配置与 FedBPC 一致，
保留本地学习率衰减 `0.998`。本次移植仅进行静态验证。

FedBTR 的新算法、独立长尾划分、配对消融和运行说明见 [FedBTR](../docs/FedBTR.md)，实现自审见 [对抗式审查记录](../docs/FedBTR_adversarial_review.md)。

FedSC 的源码行为、原型与损失流程及实验设置适配见
[FedSC 移植说明](algorithms/fedsc/README.md)。在 `src` 目录运行
`python main.py --config_path ./config/fedsc.json`，公共数据、模型与训练配置和 FedBPC 一致。
保留原源码的正原型异常回退（正常有限特征下 CPDR 为零）、原型修正未写回及位置权重行为。
本次移植仅进行静态验证。

The configuration skeleton for each algorithm is in `./config/*.json`. 
- `python ./main.py --config_path ./config/algorithm_name.json` conducts the experiment with the default setups.

There are two ways to change the configurations:
1. Change (or Write a new one) the configuration file in `./config` directory with the above command.
2. Use parser arguments to overload the configuration file.
- `--dataset_name`: name of the datasets (e.g., `mnist`, `cifar10`, `cifar100` or `cinic10`).
  - for cinic-10 datasets, the data should be downloaded first using `./data/cinic10/download.sh`.
- `--n_clients`: the number of total clients (default: 100).
- `--batch_size`: the size of batch to be used for local training. (default: 50)
- `--partition_method`: non-IID partition strategy (e.g. `sharding`, `lda`).
- `--partition_s`: shard per user (only for `sharding`).
- `--partition_alpha`: concentration parameter alpha for latent Dirichlet Allocation (only for `lda`).
- `--model_name`: model architecture to be used (e.g., `fedavg_mnist`, `fedavg_cifar`, or `mobile`).
- `--n_rounds`: the number of total communication rounds. (default: `200`)
- `--sample_ratio`: fraction of clients to be ramdonly sampled at each round (default: `0.1`)
- `--local_epochs`: the number of local epochs (default: `5`).
- `--lr`: the initial learning rate for local training (default: `0.01`)
- `--momentum`: the momentum for SGD (default: `0.9`).
- `--wd`: weight decay for optimization (default: `1e-5`)
- `--algo_name`: algorithm name of the experiment (e.g., `fedavg`, `fedntd`)
- `--seed`: random seed

FedSC 的两训练种子、44 任务批量队列见 [FedSC 批量运行说明](../scripts/README_fedsc.md)。
入口：`python scripts/run_fedsc_experiments.py`，每次 300 轮，保留所提供源码行为，温度 0.05，无投影头。

FedRCL 两训练种子、44 任务批量队列见 [FedRCL 批量运行说明](../scripts/README_fedrcl.md)。
入口：`python scripts/run_fedrcl_experiments.py`，每次 300 轮，保留等权聚合、本地学习率衰减 0.998 和默认 AMP。
