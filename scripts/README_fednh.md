# FedNH 两种子串行实验

SwanLab 安装、历史结果迁移与继续训练见 [SwanLab 使用说明](README_swanlab.md)。

在实验机项目根目录、已激活的 PyTorch/torchvision/SwanLab 环境中运行：

```bash
swanlab login
python scripts/run_fednh_experiments.py --data-root /absolute/path/to/data
```

`--dry-run` 仅列出任务并写入 manifest，不下载数据、不创建 SwanLab run、不启动训练。
应更新完整项目，包括 FedNH 模块和公共调度器，不要仅复制新入口脚本。

## 实验矩阵

| 类别 | 设置 | 次数 |
| --- | --- | ---: |
| LDA 主实验 | CIFAR10/CIFAR100/Tiny-ImageNet × α={0.03,0.1,0.3} × 2 种子 | 18 |
| 分片主实验 | 三个数据集 × 每客户端分片数={2,4,6} × 2 种子 | 18 |
| 参与率补充 | CIFAR100、LDA α=0.1、q={0.05,0.2} × 2 种子 | 4 |
| 本地 epoch 补充 | CIFAR100、LDA α=0.1、E={1,10} × 2 种子 | 4 |
| 合计 | 36 个主实验 + 8 个补充实验 | 44 |

默认 100 客户端、参与率 0.1、本地 epoch=5、batch size=50，每任务 300 轮。
q=0.1、E=5 的补充对照复用主实验。训练种子为 2022/2023；FedAvg 配对时使用对应的两组结果。
只改变训练种子，划分种子固定 19940817，客户端采样仍按从零开始的轮次编号设种子。
结果报告 n=2 的均值/样本标准差，体现固定数据划分和客户端抽样下的训练随机性。

骨干配置：CIFAR 使用 `fedavg_cifar`，Tiny 使用 `fedavg_tiny`。SGD lr=0.01、momentum=0.9、
weight decay=1e-5；StepLR step_size=1、gamma=0.99。默认设备为 cuda:0。

## FedNH 设置与比较口径

| 脚本头部参数 | 默认值 |
| --- | --- |
| `HEAD_INIT` | `orthogonal` |
| `SMOOTHING` | 0.9 |
| `CLIENT_ADV_PROTOTYPE_AGG` / `SERVER_ADV_PROTOTYPE_AGG` | False / False |
| `FIX_SCALING` / `SCALING_INIT` | False / None；当前 CNN 自动初始化为 1 |
| `FIXED_SCALING` | 30.0，仅启用固定 scaling 时生效 |
| `MAX_GRAD_NORM` | 10.0 |
| `SERVER_LR` / `SERVER_LR_DECAY` | 1.0 / 1.0 |
| `CLIENT_LR_SCHEDULER` | `project` |
| `CLIENT_LR_DECAY` | 0.99，仅旧入口的 diminishing 模式使用 |
| `EXCLUDE` | [] |

使用 FedNH 的归一化原型分类头，不增加 FedProc 的投影头。默认骨干特征为 512 维，
原型头形状分别为 10×512、100×512、200×512。分类 logits 为归一化特征与原型的点积乘 scaling。
原型不参与梯度更新，scaling 默认可训练；原型由服务器聚合和平滑更新。
这是“相同骨干、算法专属分类头”的比较，不能称为所有算法完整模型结构相同。

模型按实际参与客户端等权更新，保留服务器步长；不改成按样本量加权的 FedAvg 聚合。
普通原型分支仍按类别计数聚合、归一化并进行 smoothing 更新。缺失类别不另加新回退规则。
本地只用 CE 损失，保留梯度裁剪；每次参与重建优化器并清空动量。
本地训练后在 eval/no_grad 下再次遍历原训练 loader 估计原型，保留训练数据增强。

批量入口要求 `EXCLUDE=[]`、`CLIENT_LR_SCHEDULER="project"`，避免公共全局评估与私有参数排除、
另一套客户端学习率调度混用。当前批量图像骨干要求 orthogonal 初始化，uniform 分支仅适用于 2D 特征。
高级客户端/服务器开关必须一致；若自行启用，配置和任务标识会变化，不自动增加新的实验组合。
非有限训练损失、原型 payload 或聚合后的原型/scaling 会使当前任务报错，队列继续下一任务；不改变公式来隐藏异常。

旧的 `src/config/fednh.json` 不受队列默认轮数影响，仍使用旧评估路径；新脚本独立生成完整配置。

## 公共数据划分

复用现有 [公共划分缓存与最小样本规则](README_partitions.md)，44 个任务对应 18 份独立划分。
已有相同数据与划分配置的 FedAvg/MOON/FedProc 缓存时会直接命中。可选预生成：

```bash
python scripts/prepare_partitions.py --source fednh --data-root /absolute/path/to/data --workers 8
```

缺失数据自动下载；Tiny-ImageNet 使用有标签的 val 集作全局评测。
划分索引指纹、补足数量及比例写入元数据；各算法核对 `indices_sha256` 和 `clients.jsonl`。
FedNH 类别计数来自公共 `data_map`，不使用客户端 Dataset 的 targets 属性重新推断。

## 指标与通信量

每轮聚合后进行一次全局测试，记录 Top-1、CE、Macro-F1、逐类 Recall、最差 20% 类别平均 Recall。
在线训练 CE/准确率按实际处理样本数加权，省去旧路径的逐客户端额外训练/测试评估。
另记录学习率、训练/评估时间、通信量、最终轮和最后 10 轮平均 Top-1。
准确率/F1/Recall 为 0～1 小数，不按测试集挑选最佳权重。

| 字段 | 含义 |
| --- | --- |
| `train/ce_loss` | 在线训练 CE，与 `train/loss` 相同 |
| `fednh/scaling` | 当轮聚合后的全局 scaling |
| `fednh/server_lr` | 当轮实际服务器更新步长 |
| `fednh/prototype_covered_classes` / `prototype_missing_classes` | 本轮参与客户端覆盖/缺失的类别数；依据真实类别计数，不把 1e-12 当作样本 |
| `fednh/prototype_norm_min/mean/max` | 聚合后原型行范数统计 |
| `fednh/prototype_estimation_seconds` | 本轮所有参与客户端的原型估计耗时之和 |
| `fednh/prototype_processed_samples` | 本轮额外原型前向样本数 |
| `fednh/prototype_upload_bytes` | 本轮独立原型 payload 的上传字节数 |
| `fednh/prototype_head_bytes` | 原型头本身的张量大小，已包含在模型通信中 |
| `fednh/prototype_cumulative_upload_bytes/samples/seconds` | 累计原型上传量、前向样本数和估计时间 |

训练总时间包含原型估计，子项不能再次相加。`train/processed_samples` 仅统计本地反向训练批次，
原型额外前向样本数单列。批量模式减少了旧路径的额外评估，因此不宣称与旧路径随机数消耗逐位一致。

总通信按每个参与客户端的完整模型状态上传/下载、优化器张量下载，加上独立原型 payload 上传估计。
完整模型状态已经包含 prototype 和 scaling，**不额外重复计算原型下发**。
普通 payload 包含完整 `scaled_prototype` 矩阵和 `count_by_class_full` 向量；高级分支则使用其实际 payload。
按实际张量元素数和 dtype 大小计费，保留现有完整模型上传中冻结原型头的载荷，不假设另有压缩/去重协议。
这是逻辑载荷估计，不是实测网络流量，不含网络协议、Python 容器或模拟器 CPU/GPU 搬运开销。

## 输出和恢复

SwanLab 项目默认为 `Point1`。每次训练尝试独立一个 run，名称/tags 标注 FedNH、平滑系数、
scaling 模式、高级聚合开关、数据集、训练种子和实验组；config 记录全部有效参数。
修改参数会产生新的任务 ID。同一实验条件的两个种子归入同一个 group。

输出到 `fednh_experiments/`，包含 manifest、队列汇总、两种子 aggregate.csv，
以及每任务的 config、status、逐轮 metrics、参与名单、划分统计/元数据、train.log 和 summary。
不保存模型 checkpoint、优化器权重或原型张量文件，不上传模型 artifact。数据索引缓存仍可保存并复用。

单任务报错继续；Ctrl+C 停止队列。重新运行相同命令跳过成功任务，失败任务从第 1 轮重训。
使用新输出目录或 `--rerun-successful` 可强制重跑；不存在模型权重断点恢复。
队列串行运行；同一 GPU 上不要同时启动其他算法队列。

## 非训练验证

```bash
python -m unittest discover -s tests -p 'test_*scheduler.py' -v
python -m unittest discover -s tests -p 'test_fednh_metrics.py' -v
python -m unittest discover -s tests -p 'test_partition_cache.py' -v
```

指标测试使用 NumPy 张量替身检查实际辅助公式和控制流程，包括等权更新、缺类平滑、
优化器状态清空、在线统计、原型提取、跨轮服务器状态和通信总量。
这些检查不执行真实 PyTorch 模型训练、数据下载或 SwanLab 联网，不替代实验机上的实际运行验证。
