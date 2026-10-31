# FedSA 两种子批量实验

SwanLab 安装、历史结果迁移与继续训练见 [SwanLab 使用说明](README_swanlab.md)。

在实验机项目根目录、已安装项目依赖的环境中运行：

```bash
python scripts/run_fedsa_experiments.py
```

单 GPU 串行执行 44 次训练，每次 300 轮。训练种子 `[2022, 2023]`；
划分种子固定 `19940817`，逐轮客户端抽样仍用 `numpy.seed(round_index)`，
FedSA 独立锚点初始化种子固定 `0`。只改变训练种子。
所有实验参数和评估间隔位于脚本头部。

| 实验 | 数据集 | 变化参数 | 次数（含两个种子） |
| --- | --- | --- | --- |
| 主实验：LDA | CIFAR10、CIFAR100、Tiny-ImageNet | α=0.03/0.1/0.3 | 18 |
| 主实验：分片 | 同上 | 每客户端分片数 2/4/6 | 18 |
| 补充：参与率 | CIFAR100，LDA α=0.1 | q=0.05/0.2 | 4 |
| 补充：本地更新 | CIFAR100，LDA α=0.1 | E=1/10 | 4 |

共 22 个条件、44 次训练。q=0.1、E=5 的共同基准复用主实验，无额外消融或搜索。
默认 N=100、q=0.1、E=5、batch=50；SGD lr=0.01、momentum=0.9、weight_decay=1e-5；
每通信轮 StepLR 乘 0.99，无额外本地衰减。
CIFAR 使用 `fedavg_cifar`，Tiny-ImageNet 使用 `fedavg_tiny`，不增加可训练投影头。

## 保留当前 FedSA 算法

详细计算语义见 [FedSA 移植说明](../src/algorithms/fedsa/README.md)。

- 每个客户端拥有持续保留的模型和历史本地原型；首次参与使用共同初始化。
  不聚合、广播全局模型参数，服务器模板模型不用于主指标评估。
- 三项权重 `lambda_anchor/lambda_mcl/lambda_cc` 均为 1；目标是样本 CE、
  锚点 MSE、间隔对比损失、分类器校准 CE 的原有加权和，保持计算顺序。
- 锚点旧值系数 0.9999，新原型系数保持源码字面值 0.0001，未覆盖类别锚点不变。
  初始化模块仅生成锚点，不属于客户端可训练投影头。
- 原型来自所有本地 epoch 的更新前特征；本轮按每类贡献客户端等权平均，
  缺失类不沿用旧全局原型。历史本地原型继续服务于该客户端下次参与时的间隔计算。
- 每次参与重建 SGD，不保留跨轮优化器动量。客户端模型和历史原型存放于 CPU 内存，
  当前计算按需转至 GPU。单 GPU 不代表只需单模型的 CPU 内存容量。
- 不修改距离公式，不添加 epsilon/clamp，也不静默跳过原计算异常。

## 40 个全量评估轮次

默认 `EVAL_INTERVAL=10`、`FINAL_EVAL_ROUNDS=10`：

```text
1, 10, 20, ..., 290, 291, 292, ..., 300
```

训练仍有 300 轮，每轮记录训练指标；只在上述 40 个轮次评估完整测试集。
每次评估使用全部 100 个客户端的最新模型，包括未参与本轮或从未参与的客户端。
未评估轮次不写入准确率、不沿用上次结果、不插值。
`last10_top1_mean` 明确使用训练轮次 291–300；不是任意最后 10 个稀疏评估点。
`completed_rounds=300`、`evaluated_rounds=40` 分别表示训练和评估次数。

改变 `EVAL_INTERVAL=1` 可恢复每轮评估；`FINAL_EVAL_ROUNDS` 必须至少为 10。
如果缩短总轮数至不足 10，最后窗口使用全部实际训练轮次。
每种评估配置参与任务身份哈希。原单次 JSON 入口默认仍逐轮评估。
原有最终验证集/本地测试报告保留（仅在对应 loader 存在时）；它们不属于这 40 次主评估，
其额外耗时和原型分发量另计。

## 指标口径

主指标为全部客户端在同一完整测试集上的原型预测准确率等权均值。
原型候选为当轮全局原型，不限制本地训练类别，不用锚点替代测试原型。
Macro-F1、逐类召回和最差 20% 类别平均召回先按客户端计算，再等权平均。
与单一全局模型方法比较时必须注明这个预测器口径。

分类头准确率和 CE 为辅助指标，使用 `test/classifier/*` 及
`final_classifier_test_accuracy/final_classifier_loss`，不冒充原型损失。
没有 `global/loss` 或 `final_loss` 的原型 CE 定义，公共 aggregate.csv 中该损失列留空。
客户端准确率标准差与两个训练种子间的标准差分别报告。

训练损失沿用“客户端内 batch 等权平均，再按参与客户端等权平均”。
记录四项原始分量和加权总目标、学习率、原型覆盖率、锚点间距、有效间隔、
local_gap 为负无穷的客户端比例及有限值计数。保留负无穷的算法语义，
只将标志及有限值均值写入日志，不输出非法 JSON 数值。

每次训练独立 SwanLab run，默认项目 `Point1`；自动记录配置、run 名称、tags、条件分组和划分指纹。
最终汇总包括最终 Top-1、最后 10 轮均值、Macro-F1、最差类召回、训练/评估时间和通信量。
aggregate.csv 对共同条件的两个种子统计均值及样本标准差，并列出成功/预期次数。
与三种子 FedAvg 配对比较时使用共同的 2022、2023 两个种子。

## 通信和时间统计

`communication_bytes` 只统计训练逻辑载荷：参与客户端的稠密语义锚点下行，
以及本地类别原型张量上行，每个稀疏原型加一个 int64 类别 ID。
没有每轮模型上传/下载费用；服务器存储客户端权重是模拟器状态恢复，非算法网络传输。

`evaluation_communication_bytes` 单独估算全局原型向所有实际评估客户端的分发，
包括最终额外报告；按每次评估调用分发一次计算，不假设跨评估调用缓存复用。
`initial_model_distribution_bytes_estimate` 为所有客户端各下发一份初始模型的单次估算，
单独报告，不加到训练总量；若部署时共同 seed 本地初始化，则不必发生这笔传输。
以上不含标量学习率等控制信息、协议/张量形状元数据以及本机 CPU/GPU 拷贝。
与 FedAvg 比较总通信时需统一是否包含初始化及评估成本。

逐轮训练和主评估分别计时，CUDA 使用同步边界。
`total_train_seconds`、`total_eval_seconds` 不含最终额外报告，后者写入
`final_additional_report_seconds`。轮总时间还包括间距诊断和调度开销，
不能把减少评估后的墙钟耗时直接解释为训练更快。

## 缓存、输出及异常

缺失数据集自动下载。复用公共 LDA/分片缓存，仍只有 18 个不同划分；
LDA 有限候选和不足 10 个样本的统一补足规则不变，比较时核对 `indices_sha256`。

默认输出 `fedsa_experiments/`：任务清单、config、日志、状态、summary、
evaluation_rounds.json、clients.jsonl、划分统计、metrics.jsonl、metrics.csv 和 aggregate.csv。
JSONL 每轮追加，成功结束后转换为 CSV，未评估的 CSV 指标单元格为空。
中途失败仍可读取已经写入的 JSONL 和错误日志。
模型、优化器、语义锚点、历史原型均不保存为文件，也不上传模型 artifact；
划分缓存 NPZ 只有索引与元数据。

新入口同时启用 `fedsa_batch` 和 `write_metrics_files`，让共享 SwanLab 生命周期写出
summary.json，供队列判定成功及跳过。原 FedSA 单次入口，以及 FedProto/TinyProto 的
默认 SwanLab-only 路径保持不变。
失败任务报错后继续下一项；重跑失败任务从头训练并建立新 attempt/run。
成功任务默认跳过，Ctrl+C 则停止队列。不要同时运行其他 GPU 队列。

```bash
# 可选：8 核 CPU 预生成或验证 18 份共享划分
python scripts/prepare_partitions.py --source fedsa --workers 8

# 仅生成清单，不训练、不下载、不连接 SwanLab
python scripts/run_fedsa_experiments.py --dry-run

# 可指定数据、结果目录和实验环境解释器
python scripts/run_fedsa_experiments.py --data-root /data --output-root /results/fedsa
```

## 验证范围

非训练测试覆盖 44 个任务、两个训练种子、固定锚点种子、缓存配对、失败继续、
成功跳过、客户端状态保留、原型等权聚合、全客户端评估、稀疏日志、
真实最后 10 轮均值、训练/评估通信分离、SwanLab-only 兼容和本地成功状态文件。
数值与流程测试使用 NumPy/接口替身，不运行 PyTorch/GPU 前反向。
未执行数据下载、SwanLab 联网、实际训练或小规模冒烟训练。
