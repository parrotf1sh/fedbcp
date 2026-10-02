# FedNH 项目移植

以本仓库 `FedNH-main` 为实现依据，接入项目的 `BaseServer` 和
`BaseClientTrainer`。数据与特征提取网络使用本项目实现；分类头、CE 损失、
原型估计及聚合保留 FedNH 计算。没有引入 FedBPC 的辅助损失或原型过滤。

## 模块与原源码对应

| 移植模块 | 原源码 |
| --- | --- |
| `model.py` | `src/flbase/models/CNN.py` 的 NH 分类头；`FedUH.py` 的初始化 |
| `ClientTrainer.py` | `FedUHClient.training`、`FedNHClient._estimate_prototype*` |
| `utils.py` | `FedNHServer.aggregate`、`flbase/utils.py` 的优化器与线性组合 |
| `Server.py` | FedAvg/FedUH/FedNH 的下载、训练、上传、聚合生命周期 |
| `config.py` | 原 client/server 配置的项目化命名及校验 |

## 数据、模型与实验设置

默认配置位于 `src/config/fednh.json`。数据、模型、场景、SGD、学习率调度、
训练种子与移植时的 `src/config/fedbpc.json` 相同。
通过现有入口运行，例如在 `src` 目录中执行：

```bash
python main.py --config_path config/fednh.json
```

此命令会启动实验，不属于本次静态验证。

数据直接使用统一入口构建的 `data_distributed`，支持现有 legacy 和 longtail
管线。legacy 的 LDA/sharding 复用公共划分缓存；longtail 复用 manifest。
配对比较时使用相同数据文件、配置和最终索引，并核对 `indices_sha256` 或
`split_manifest.sha256`。原型类计数来自 `data_map`，不重新划分数据。

`ModelWithNormalizedHead` 独立于公共模型文件。先由 `create_models()` 创建
项目模型，再把最终 Linear 替换为 Identity，以完整调用原骨干前向。
支持当前注册模型的 `classifier`、`linear`、`linear_2` 末层：
`fedavg_cifar`、`fedavg_tiny`、`fedavg_mnist`、`vgg11`、`res10`、`res18`。
这些接口经过静态检查，尚未执行前向验证。

默认 `fedavg_cifar` 保持项目 512 维特征；不复制原 FedNH 的 192 维 CNN。
ResNet 保留项目自身的归一化层，不切换为原仓库的 GroupNorm/no_norm 网络。
特征之后使用 FedNH 的归一化、无偏置原型点积与 scaling，而非项目普通线性头。
因此比较设置是“相同骨干、算法专属分类头”，并非完整分类器结构相同。

适配必须在构造 optimizer 之前完成。直接调用组件时按下面的顺序装配：

```python
model = ModelWithNormalizedHead(project_model, algo_params)
optimizer = create_optimizer(model, lr=0.01, momentum=0.9, weight_decay=1e-5)
# 此后创建项目 scheduler，再创建 Server。
```

## 保留的计算流程

1. 使用 `orthogonal_(torch.rand(C, D))` 初始化 prototype，冻结其梯度；不额外
   归一化初始化矩阵。`uniform` 分支只支持实际特征维度为 2，否则明确报错。
2. `z = f(x) / clamp(norm(f(x)), min=1e-12)`，
   `logits = scaling * (z @ prototype.T)`。冻结的 prototype 直接用于点积；
   保留原前向中可训练 prototype 的归一化分支。默认返回 logits，
   `get_features=True` 按项目接口返回 `(logits, z)`。
3. 本地使用普通交叉熵，反向传播后按原实现裁剪可训练参数的梯度，再更新优化器。
   每次本地训练清空优化器状态，客户端之间和轮次之间均不继承动量。
4. 本地训练后，在 `eval()`/`no_grad()` 下再次遍历同一训练 loader，
   保留训练数据增强及 loader 行为；不切换到 calibration/train_eval loader。
5. 普通分支：各类归一化特征求均值，再归一化，再乘该类样本数上传。
   缺失类原型为零，上传计数为 `1e-12`。服务器按各类总计数汇总。
6. 高级客户端分支：以真实类别 softmax 概率加权特征，除以概率之和，再归一化。
   高级服务器分支：以 `exp(sum(W * local_prototype))` 加权并除以权重总和。
   缺失类也参与原实现的权重分母，不增加有效类掩码。
7. 模型按参与客户端等权更新：
   `server + server_lr * server_lr_decay**round_idx / K * sum(local - server)`。
   不使用项目基类按数据量加权的聚合。可训练 scaling 一同聚合。
8. 原型均值先归一化，然后执行
   `normalize(smoothing * W + (1 - smoothing) * normalized_mean)`。
   即便所有参与客户端缺少某类，也保留此公式；不增加保留旧原型的特殊分支。

保留高级客户端概率分母未 clamp 的原行为；概率和为零仍可能产生非有限值。
零均值、零向量、`smoothing=0/1` 按原运算处理，不改写为新的兜底算法。
客户端和服务器高级开关必须一致；原源码中不匹配会造成 payload KeyError，
这里将该无效组合提前报告为配置错误。空训练集也明确报错，不跳过客户端。

## 参数

全部算法参数放在 `train_setups.algo.params`，未知参数会报错。

| 参数 | 默认值 | 对应含义 |
| --- | --- | --- |
| `head_init` | `orthogonal` | 原 `FedNH_head_init` |
| `smoothing` | `0.9` | 原 `FedNH_smoothing` |
| `client_adv_prototype_agg` | `false` | 原客户端高级估计开关 |
| `server_adv_prototype_agg` | `false` | 原服务器高级聚合开关 |
| `fix_scaling` | `false` | 原 `FedNH_fix_scaling` |
| `scaling_init` | `null` | 自动：ResNet 20，其他已支持骨干 1；可显式指定 |
| `fixed_scaling` | `30.0` | 启用固定 scaling 时的值 |
| `max_grad_norm` | `10.0` | 原梯度裁剪上限 |
| `server_lr` | `1.0` | 服务器模型更新步长 |
| `server_lr_decay` | `1.0` | 服务器逐轮步长衰减 |
| `client_lr_scheduler` | `project` | 使用项目 scheduler，或源码的 diminishing/stepwise |
| `client_lr_decay` | `0.99` | 源码 diminishing 模式的衰减率 |
| `exclude` | `[]` | 参数键子串，排除模型聚合和下载 |

`scaling_init=null` 对 VGG/MNIST 的默认 1 是本次新增骨干适配约定；
上游正式图像实现只有 Conv2CifarNH 与 ResNetModNH。不移入独立的合成二维 MLP。
scaling 保持原实现的直接可训练参数，不施加正值约束。

`train_setups.optimizer.name` 支持 `sgd`（默认）、`adam`、`rmsprop`。
参数来自同层 `params`；Adam 的默认 weight_decay 为 `1e-5`，RMSprop 默认 eps
为 `1e-8`。切换优化器时提供该优化器接受的参数，不沿用 SGD 专用参数。

`project` 模式直接使用公共 scheduler 每轮更新后的学习率；默认 StepLR 在第
1 轮使用 0.01、第 2 轮使用 0.0099，与原 diminishing 指数对应。
若选源码 `diminishing` 或 `stepwise`，需设置 `train_setups.scheduler.enabled=false`
以避免双重调度。源码 `stepwise` 严格保留 `round < n_rounds // 2`（round 从 1
开始），例如 200 轮实验在第 100 轮降至初始学习率的 0.1。

`exclude` 使用适配后 state_dict 的名字，例如 `features.conv2d_1`。
排除项按客户端 ID 保存私有状态，重用 trainer 不会串用私有参数。原型 EMA
仍独立执行，即使 `prototype` 被排除；服务器评估模型也按原实现跳过排除项下载。
默认不排除任何参数，建议对比实验保持该设置。

## 与原实验脚手架的差异

- 使用项目的数据、骨干、训练种子和按轮次确定的客户端抽样规则，不使用原仓库
  硬编码的初始化种子或每次本地训练重新设置全局随机种子的操作。
- 使用项目规定的参与集合，不额外引入原框架的 straggler/drop_ratio 模拟。
- 使用项目的评估指标与评估节奏，包括本地统计和可选 longtail 最终报告；
  不复刻原仓库的 split_testset 开关、best-checkpoint 文件或 wandb 脚本。
- 模型/原型的训练计算分支完整保留，评估接口返回顺序按项目统一为 logits 在前。

## 静态验证范围

仅检查 Python 语法/AST、JSON、模块导出与注册、配置对齐、依赖，以及逐段人工
核对源码公式、分支、状态生命周期和调度轮次。未导入算法运行，未执行训练、
模型前向/反向、数值测试、数据下载或 GPU 验证。静态检查不保证运行时数值正确。
