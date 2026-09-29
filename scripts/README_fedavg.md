# FedAvg 串行实验队列

在实验机的项目根目录执行（先激活已有 PyTorch / torchvision / W&B 环境）：

```bash
wandb login
python scripts/run_fedavg_experiments.py --dry-run
python scripts/run_fedavg_experiments.py --data-root /absolute/path/to/data
```

脚本头部集中配置超参数、GPU、数据路径、模型、W&B project/entity 和实验开关。
`--dry-run` 仅检查和输出队列，写入 manifest，不下载数据、不连接 W&B、不启动训练。
调度器采用 Python 标准库；训练需要项目现有依赖。实验机须能访问 W&B 和数据下载源。
只需设置数据集的父目录，不要传某个数据集的 train 目录。

## 默认实验

共 **66 个独立训练任务，每个任务 300 轮**，每个任务对应一个 W&B run：

| 组别 | 设置 | 任务数 |
| --- | --- | ---: |
| LDA 主实验 | 3 数据集 × α={0.03,0.1,0.3} × 3 训练种子 | 27 |
| 分片主实验 | 3 数据集 × 每客户端分片数={2,4,6} × 3 训练种子 | 27 |
| 参与率补充 | CIFAR100 / LDA α=0.1 / q={0.05,0.2} × 3 训练种子 | 6 |
| 本地 epoch 补充 | CIFAR100 / LDA α=0.1 / E={1,10} × 3 训练种子 | 6 |

默认 N=100、q=0.1、E=5、batch size=50；q=0.1 与 E=5 的补充对照复用主实验。
只改变训练种子 2022/2023/2024：数据划分仍采用项目原来的 19940817；
客户端采样仍采用 `numpy.seed(round_index)`（轮次从 0 开始），不随训练种子变化。
所以跨种子标准差描述**固定划分和固定客户端采样下的训练随机性**，不是划分不确定性。

运行次序：全部 LDA → 全部分片 → 参与率补充 → 本地 epoch 补充。
每次等待子进程完全退出，再启动下一任务。默认只使用 `cuda:0`。
同一输出目录有排他锁，防止重复启动同一个队列；请勿另开不同输出目录的并行队列抢占同一 GPU。
设置 `INCLUDE_SUPPLEMENTARY=False` 可只跑 54 个主实验。

## 数据集与模型

- CIFAR10/CIFAR100 使用 `fedavg_cifar`，Tiny-ImageNet 使用 `fedavg_tiny`。
- `tiny-imagenet` 会映射到项目内部名称 `tinyimagenet`。
- CIFAR 数据缺失时由 torchvision 下载并校验。
- Tiny-ImageNet 数据缺失时从 Stanford 下载 ZIP 并解压；自动检查训练/验证图像数量。
- Tiny-ImageNet 使用有标签的官方 val 集作为评测集，不使用无标签 test 集。
- 不修改现有划分算法；分片数仍是每客户端分片数，不保证每客户端类别数恰好等于该值。
- 改 N 后会校验 `N * shards` 是否能被类别数整除，避免实际分片数与配置不一致。

默认准备为：

```text
data/
  cifar10/cifar-10-batches-py/...
  cifar100/cifar-100-python/...
  tinyimagenet/wnids.txt
  tinyimagenet/train/...
  tinyimagenet/val/val_annotations.txt
  tinyimagenet/val/images/...
```

也支持 `data/tinyimagenet/tiny-imagenet-200/` 的嵌套解压布局。
Tiny-ImageNet 下载文件保留在数据父目录，便于失败后再次解压；它是数据，不是模型文件。

## 记录与输出

**不保存任何模型 checkpoint、模型权重或优化器状态文件，不上传模型 artifact。**
内存中的模型聚合不受影响；模型参数仍在内存中正常上传/下载。

- 每轮全局 Top-1、交叉熵、Macro-F1、逐类 Recall、最差 20% 类别平均 Recall。
- 本地训练损失按本轮实际处理样本数加权；online Top-1 是训练过程中、增强数据上的准确率。
- 本轮使用的学习率、累计处理样本数、训练/评估耗时分别记录。
- 通信载荷估计包括每个参与客户端的模型上传、模型下载及优化器张量下载；不包含网络协议开销。
- 结束时记录最后一轮、最后 10 轮平均准确率；不按测试集挑最佳模型。
- 准确率、F1 和 Recall 均为 0–1 的小数，需要百分数时乘 100。
- 同一轮只做一次全局测试评估。批量路径不执行旧路径中的逐客户端全局测试，避免重复开销。
  新路径的评估随机数消耗与旧路径不同，后续其他算法要严格配对时应采用相同评估协议。

W&B config 记录完整数据/模型/优化器/调度器配置、固定随机性协议、训练种子、运行库版本。
三个种子共享同一 W&B group，tags 标注算法、数据集、划分、种子和实验类别。
使用在线模式且要求已登录；初始化失败时该任务失败，不静默切换为离线训练。

本地默认输出：

```text
fedavg_experiments/
  manifest.json                 # 全部任务配置
  queue_summary.json            # 本次队列成功/失败/跳过数
  aggregate.csv                 # 按实验条件汇总训练种子的 mean / sample std
  <配置哈希>/
    status.json                 # 最近一次尝试的状态
    summary.json                # 最近成功尝试的结果；以 status.json 为准
    attempt-<唯一编号>/
      config.json
      train.log
      metrics.csv
      clients.jsonl             # 每轮实际参与客户端名单
      partition_stats.json      # 每客户端类别计数、标签熵等
      summary.json
      wandb/...
```

所有任务串行；遇到实验报错会打印 traceback 并继续。队列全部结束后，若有失败任务，
调度器返回退出码 1；手动 Ctrl+C 会停止当前进程组并停止整个队列，返回 130。
失败阶段可能尚未生成全部指标文件；W&B 初始化失败也不会产生有效远端 run。

重新执行相同命令会跳过已成功任务，只重跑失败或未完成任务；失败任务从第 1 轮重训，
没有权重断点续训。每次尝试均保留自己的日志，并使用新的 W&B run。
修改训练参数会生成新的配置哈希。若更换代码版本或 W&B 目标项目后希望全部重跑，
使用新的 `--output-root`，或加 `--rerun-successful`。
跨种子汇总注明 `n_success` / `n_expected`；不足两个成功种子时 std 留空，不伪造三种子结果。

## 非训练验证

```bash
python -m unittest discover -s tests -p 'test_fedavg_scheduler.py' -v
```

仅验证任务矩阵、去重、配置隔离、失败续跑、中断、进程退出码和下载归档安全；
不运行任何训练冒烟实验，不下载真实数据集。
