## How to Run Codes?

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

FedBTR 的新算法、独立长尾划分、配对消融和运行说明见 [FedBTR](../docs/FedBTR.md)，实现自审见 [对抗式审查记录](../docs/FedBTR_adversarial_review.md)。

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
