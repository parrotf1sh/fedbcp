# FedRCL 批量实验

在实验机项目根目录、已安装项目依赖的环境中运行：

```bash
python scripts/run_fedrcl_experiments.py
```

默认单 GPU 串行执行 44 次训练，每次 300 轮，训练种子 `[2022, 2023]`。
超参数集中在脚本头部；只改变训练种子，划分种子固定 `19940817`，
客户端抽样保持逐轮 `numpy.seed(round_index)`。未找到数据集时自动下载。

| 实验 | 数据集 | 变化参数 | 次数（包含两个种子） |
| --- | --- | --- | --- |
| 主实验：LDA | CIFAR10、CIFAR100、Tiny-ImageNet | α=0.03/0.1/0.3 | 18 |
| 主实验：分片 | 同上 | 每客户端分片数 2/4/6 | 18 |
| 补充：参与率 | CIFAR100，LDA α=0.1 | q=0.05/0.2 | 4 |
| 补充：本地更新 | CIFAR100，LDA α=0.1 | E=1/10 | 4 |

共 22 个条件、44 次训练。q=0.1、E=5 的共同基准复用主实验；不额外加入算法消融或搜索。
N=100、默认 q=0.1、E=5、batch=50。SGD lr=0.01、momentum=0.9、weight_decay=1e-5；
轮间 StepLR 每轮乘 0.99。CIFAR 使用 `fedavg_cifar`，Tiny-ImageNet 使用 `fedavg_tiny`。

## FedRCL 配置与比较口径

保留当前 [FedRCL 移植实现](../src/algorithms/fedrcl/README.md)：

- 参与客户端完整模型 state dict 等权平均，不按客户端样本量加权。
- 不添加投影头。默认对比层为 `[0,1,2]`，对应两个卷积层和第一个全连接隐藏层，
  读取 ReLU 前特征；卷积特征沿用已有池化规则，分类输出不参与默认 RCL。
- 目标为 `CE + mean_layers(SCL) + mean_layers(penalty)`，三项系数均为 1。
  温度 0.05、topk_pos=1、topk_neg=2、penalty 阈值 0.7。
- 近端项和特征对齐项权重均为 0，仍保留原计算流程、全局参考模型前向和原损失归约。
  参考模型冻结参数，但处于训练模式，前向在 `no_grad()` 内执行。
- 每次参与重新建立 SGD，清空动量状态。本地 epoch 结束后学习率乘 0.998；
  零起始轮次 r、epoch e 对应 `0.01 * 0.99**r * 0.998**e`。
- 梯度裁剪范数 10，默认 `USE_AMP=True`。CUDA 上 GradScaler 独立于 autocast 开关启用，
  CPU 禁用两者，保持当前实现。默认 AMP 与普通精度基线不同，时间比较必须注明精度模式。

任务 config 记录解析后的完整算法参数和层级映射，tags 标记等权聚合、AMP、本地衰减等。
改变参数会改变任务身份，不会错误跳过为旧成功结果。单次 `src/config/fedrcl.json` 默认配置保持不变。
不改变正负样本 top-k 方向、阈值哨兵处理或 NaN/Inf 元素的原有处理。
原损失参数的特殊覆盖语义仍以移植说明为准。默认 topk_pos=1 支持非空尾批次，
手动提高它可能在小尾批次上报错；队列不会静默跳过该 batch 或改写公式。

## 数据划分、W&B 和输出

复用公共划分缓存：LDA 最多生成 128 个候选，再按公共规则转移样本补足不足 10 个样本的客户端。
不重复、删除样本。不同算法和训练种子复用同一份最终索引；比较时核对 `indices_sha256`。
44 个任务只有 18 个不同划分，已有缓存直接验证并复用。

每次独立训练创建一个 W&B run，项目默认 `Point1`，自动设置 name、tags、config 和条件分组。
逐轮记录全局 Top-1、CE、Macro-F1、各类召回、最差 20% 类别平均召回、
训练在线准确率、按处理样本数加权的损失、训练/评估时间、累计处理样本数和通信量。

`train/cls_loss` 是源码的 CE 名称，`train/ce_loss` 是相同值的公共别名。
`scl_loss`、`penalty_loss`、`prox_loss`、`cossim_loss` 均为乘目标权重前的原始分项；
即使 prox/cossim 非零，默认它们的目标贡献仍为零。`train/total_loss` 才是加权总损失。
另记录 `fedrcl/local_first_epoch_lr`、`fedrcl/local_last_epoch_lr`，后者对应最后一个 epoch
实际使用的学习率，而不是该 epoch 结束后的 scheduler 值。

通信量统计模型上下传及优化器张量下传。多层特征留在客户端；参考模型由已下载模型本地复制，
不重复计下载量。不含传输协议、Python 容器开销或本机 CPU/GPU 拷贝。
训练时间包含本地参考前向及多层损失计算，不能据此假定与 FedAvg 单轮算力开销相同。

每轮模型更新后评估一次全局模型；批量模式跳过训练前及额外客户端评估。
最终记录最后一轮和最后 10 轮平均准确率、最终损失/分类指标、总时间、通信量及算法设置。
`aggregate.csv` 汇总两个种子的均值和标准差，同时记录预期/成功种子数，部分成功不能当成完整两种子结果。
与三种子 FedAvg 作配对比较时使用共同的 2022、2023 两个种子。

输出目录默认 `fedrcl_experiments/`，保存任务清单、配置、日志、状态、metrics.csv、
clients.jsonl、划分统计、summary.json 和 aggregate.csv。
不保存模型 checkpoint、权重、优化器文件或模型 artifact。参考模型只在内存中存在；
划分缓存 NPZ 只含索引和元数据。

单任务失败记录错误后继续下一任务；默认跳过已成功任务。
失败任务重跑创建新 attempt 和 W&B run，从头训练；不从中间轮次恢复。
Ctrl+C 停止当前训练和队列。仅在当前队列目录内加锁，其他算法队列请勿同时启动争抢单 GPU。

```bash
# 可选：在 8 核 CPU 上预生成/检查共享划分，不启动训练
python scripts/prepare_partitions.py --source fedrcl --workers 8

# 仅生成任务清单，不训练、不下载、不连接 W&B
python scripts/run_fedrcl_experiments.py --dry-run

# 自定义目录；也可通过 --python 指定实验环境解释器
python scripts/run_fedrcl_experiments.py --data-root /data --output-root /results/fedrcl
```

## 非训练验证

验证任务配对、去重、缓存复用、异常继续、成功跳过、等权聚合、指标汇总和本地学习率重置。
配置回归使用项目实际 ConfLoader；运行流程检查使用替身，不执行 PyTorch/GPU 前反向。
未进行数据下载、W&B 联网、完整训练或小规模冒烟训练；这些检查不证明训练精度或数值等价。
