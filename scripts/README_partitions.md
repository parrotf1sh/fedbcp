# 各算法共用的数据划分

标准数据入口 `src/train_tools/preprocessing/datasetter.py` 默认启用公共划分缓存。
FedAvg、MOON、FedProc、FedBPC 及其他通过该入口使用 LDA/分片的算法都会采用同一规则。
独立的 longtail pipeline 继续使用自己的划分协议，不与此处的 LDA 混用。

## LDA 最小样本规则

旧实现会反复丢弃整个划分，直到每个客户端至少有 10 个样本；没有重试上限。
小 alpha 下，尤其 CIFAR10 的 100 客户端设置，满足条件可能非常困难。
把 alpha 从 0.03 改成 0.05 也不能保证每个客户端都达到 10 个样本。

现在的统一规则是：

1. 固定划分种子为 19940817，最多生成 128 个候选划分；提前遇到全部达标的候选即采用。
2. 若均未达标，选取总缺口 `sum(max(10 - n_i, 0))` 最小的候选；并列时采用最先出现的候选。
3. 按客户端编号依次补足缺口，每次从当前样本最多的客户端随机转移一个样本，直到接收方达到 10 个。
4. 转出客户端也至少保留 10 个样本。不存在复制、删除样本或改变整个训练集类别总数的情况。
5. 固定种子控制采样、补足和客户端内部索引顺序。选定候选的转移次数恰好等于总缺口。

这是带最小样本约束和有限次采样的 Dirichlet 划分，不能表述为未经调整的 Dirichlet 样本。
论文应交代上述规则，并报告补足数量/比例。少量转移仍可能明显影响个别小客户端的类别构成；
公平性依靠各算法使用同一份最终索引，而不是只设置相同 alpha。

默认队列中只改变训练种子，划分种子和客户端采样规则保持不变。FedAvg 默认 66 次训练，MOON/FedProc 默认各两种子共 44 次训练，每次均为 300 轮。

## 8 核 CPU 预生成

在实验机的项目根目录、已有 PyTorch/torchvision/NumPy 环境中运行：

```bash
python scripts/prepare_partitions.py --data-root /absolute/path/to/data --workers 8
python scripts/run_fedavg_experiments.py --data-root /absolute/path/to/data
python scripts/run_moon_experiments.py --data-root /absolute/path/to/data
```

前一条训练命令结束后再执行后一条，避免争抢 GPU。
预生成命令先检查/下载缺失数据集，再用最多 8 个 CPU 进程并行生成不同实验条件的划分；
不启动训练、不使用 GPU、不创建 W&B run。FedAvg 的 66 个任务或 MOON/FedProc 的 44 个任务去重后都只需生成 18 份划分。
并行发生在不同划分之间，每份划分内部仍按固定顺序生成候选，因此 worker 数量不改变结果。
预生成可重复执行，已有缓存直接验证并复用。省略预生成时，训练入口也会自动生成/读取缓存。

预生成默认读取 FedAvg 脚本头部配置；如以 MOON 或 FedProc 配置为准，增加 `--source moon` 或 `--source fedproc`。
`--dry-run` 只列出划分条件，不读取或下载数据。准备结果写入数据根目录的 `partition_cache_manifest.json`。

## 缓存与复现

默认保存位置为 `<data-root>/<dataset>/.partition_cache/*.npz`，只包含客户端样本索引及 JSON 元数据。
没有图像副本、模型权重或 checkpoint。文件锁和原子写入防止多个进程同时生成半成品。
加载时验证索引校验和、完整覆盖、唯一性和最小样本数；损坏文件隔离后重新生成。

缓存键包含数据集名称、按顺序排列的标签摘要、样本数、客户端数、划分种子、方法、有效参数及实现版本。
LDA 参数包括 alpha、最小样本数、候选次数和补足策略；分片包含每客户端分片数及对应测试标签摘要。
算法名、训练种子、本地 epoch、参与率、batch size 和模型不参与缓存键。
因此配对算法和各训练种子复用同一份缓存；改变划分设置时自动生成另一份。
标签摘要不校验图像内容本身，跨机器复用仍需确保数据版本和样本顺序相同。

LDA 保持全局测试集不变；分片缓存包含配对的训练/客户端测试索引。
划分使用独立随机数生成器，缓存是否命中不会改变后续全局随机数状态。
初次生成所用 NumPy 版本记录在元数据中；跨环境严格复现优先复制已有缓存。

W&B 的 `resolved_partition` 记录完整划分元数据，summary 记录缓存键、索引指纹、补足数量和比例。
FedAvg/MOON/FedProc 每次运行另写 `partition_metadata.json`，并保留每客户端类别计数与标签熵。
比较算法时核对 `indices_sha256`。日志将标签加载、划分、构建客户端 DataLoader 分阶段打印；
缓存仅省去重复划分，不省去图像读取或 DataLoader 初始化。

## 配置项

FedAvg/MOON/FedProc 脚本头部提供：

```python
PARTITION_SEED = 19940817
PARTITION_CACHE_ENABLED = True
PARTITION_CACHE_DIR = None  # 默认数据集目录中的 .partition_cache
LDA_MIN_SAMPLES = 10
LDA_MAX_ATTEMPTS = 128
LDA_INSUFFICIENT_POLICY = "repair"
```

其他算法的 JSON 配置可在 `data_setups` 中指定 `partition_seed`、
`partition_cache: {"enabled": true, "directory": null}`，并在 `partition` 中指定
`min_samples`、`max_attempts`、`insufficient_policy`。未指定时采用上述公共默认值。
同组比较应保持这些设置和数据根目录一致。
将策略改为 `strict` 会在候选全部失败后报错，不补足；不要在同组算法之间混用两种策略。

本次更新会改变 FedAvg/MOON 队列的任务配置哈希。旧成功任务不会被自动当作新协议结果跳过。
旧结果只有在能验证最终索引和完整实验协议一致时才适合配对比较。
实验机上的旧进程不会热更新，需要更新项目代码后重新启动。

## 非训练验证

```bash
python -m unittest discover -s tests -p 'test_partition_cache.py' -v
python -m unittest discover -s tests -p 'test_*scheduler.py' -v
```

只检查标签/索引、并发缓存、队列和配置，不下载数据，不训练模型。
