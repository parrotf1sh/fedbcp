# FedRCL 移植说明

本实现用于 FedBPC 对比实验，算法来源为项目根目录的 `FedRCL-main`：
`clients/rcl_client.py`、`utils/loss.py`、`servers/base.py`，以及
`models/resnet_base.py` 的多层特征语义。运行时不依赖该源码目录。

## 运行入口

在 `src` 目录使用项目统一入口：

```bash
python main.py --config_path ./config/fedrcl.json
```

配置结构、命令行覆盖方式、评估和 W&B 生命周期沿用项目现有实现。
默认配置的 `data_setups`、`scenario`、`model`、`optimizer`、`scheduler`
及训练种子与 `config/fedbpc.json` 一致。默认不是原论文的 ResNet-18 实验设置。

## 数据与模型

数据直接来自项目的 `data_distributer`，复用原有数据增强、DataLoader、
划分种子（默认 `19940817`）和算法无关的划分缓存。
不使用原仓库的 `client_data` 或 classwise sampler，不修改尾批次或重采样数据。
对照时应保持数据配置一致，并比较已有日志中的 `partition/indices_sha256`。
客户端抽样使用 `BaseServer._client_sampling`，与 FedBPC 使用同一轮次抽样规则。

模型仍由 `train_tools.utils.create_models` 创建。
`ModelWithLayerFeatures` 只包装该模型，训练时临时注册 hooks 读取特征；
分类前向始终调用原模型，不添加投影头、可训练参数或新的归一化层。
普通 `forward(data)` 和 `get_features=True` 保留项目模型的返回格式。
包装后的 state dict 键具有 `model.` 前缀，参数形状不变。

| 模型 | 默认 RCL 特征层 | 分类输出层 |
| --- | --- | --- |
| `fedavg_cifar` | 0：conv2d_1；1：conv2d_2；2：linear_1 | 3 |
| `fedavg_tiny` | 0：conv2d_1；1：conv2d_2；2：linear_1 | 3 |
| `vgg11` | 0–4：五个卷积阶段最后一次 ReLU；5–6：两个 FC 隐藏层 ReLU | 7 |

默认读取 ReLU 前特征；卷积特征取池化前的位置，随后由原 RCL 流程执行
`adaptive_avg_pool2d(..., 1)`。`feature_relu=true` 则读取同一位置的 ReLU 后特征。
VGG 原地 ReLU 前的特征使用 `clone()` 保留，不能 `detach()`，不修改原地激活本身。
不支持的模型明确报错，不退化为单层对比。

这是经确认的模型层级映射：原仓库默认 ResNet 有五个隐藏层级，当前 CNN 有三个，
VGG 有七个。保留多层损失及平均方式，不声称与原 ResNet 的特征数值相同。

## 保留的计算流程

每个客户端加载当轮全局参数，并建立冻结参数的全局参考模型。
本地模型和参考模型均处于训练模式；参考前向在 `no_grad()` 中执行，
不因冻结参数而改为 `eval()`。默认 `nn` 对比只使用本地特征相似度，
但仍保留参考前向、特征对齐及参数近端损失的计算，包括它们权重为零时。

默认目标为 `CE + mean_layers(SCL) + mean_layers(penalty)`：

- 正样本使用同类中相似度最小的 top-k，不统一去除对角线；
  unsupervised 正样本分支则以自身为正样本。
- 普通负样本分支屏蔽相应正样本，取相似度最大的 `min(topk_neg, batch_size)` 项。
- `intra_class_thresholding` 分支将异类和自身置为正无穷，按最小相似度取
  `min(topk_pos, batch_size)` 项，然后将低于阈值的项及正无穷项替换为 `-1`。
  不改为“先阈值筛选，再取最相似样本”。
- 保留特征 L2 归一化、逐正样本拼接负样本、温度缩放、可选 beta 加权、
  `log_softmax`、无穷/NaN 损失元素置零，以及正样本、batch、层级的平均顺序。
  不对最终损失或梯度额外执行 `nan_to_num`，不吞异常或跳过 batch。
- 特征对齐支持 L2/cosine，遍历所有层（包括分类输出），独立于 RCL 层级筛选。
  近端项保持逐参数平方差之和，不添加 `1/2` 或参数数量归一化。
- 更新顺序为 scale/backward → unscale → 梯度范数裁剪 → SGD step → scaler update；
  本地 epoch 结束后才更新本地 scheduler。

源码中不直观但保留的配置语义：

- pair 的覆盖规则是 `pair.get(key) or shared_value`，零值不会替换共享值。
- 客户端调用显式传入共享 `topk_neg`，覆盖 pair 内的同名值；惩罚分支实际使用
  `topk_pos`。不将它们重新解释为另一套采样参数。
- pair 的 `temp` 控制对比 logits 的缩放；内部 beta 权重项仍使用共享温度。
- `scl`、`penalty` 的遍历顺序及 pair 级 `branch_level` 对 `train_layer` 的顺序更新
  按源码保留。pair 的空列表/false/null 不覆盖当前层级选择。
- `rcl_loss.weight`、`pos_sample_type` 和 `eps` 在原有效训练路径中不参与目标计算，
  保留配置接口但不赋予新算法行为。目标权重由各 pair 的 `weight` 决定。
- 原仓库只实现 `nn` 相似度，不新增 `no/on/oo` 等特征配对。

## 聚合与学习率

服务器覆盖 `_aggregation`，对参与客户端的完整 state dict 执行等权平均：
`sum(client_weights) / number_of_clients`，不使用样本量加权。
不引入源仓库其他实验用的服务器动量或 Adam 变体。

每次客户端下载全局模型后重新建立 SGD，清除本地动量状态。
共享服务器 optimizer 只提供当轮学习率和公共 SGD 参数；本地更新不回写它。
本地 `LambdaLR` 的 `initial_lr` 显式重置为当轮学习率，防止继承服务器
scheduler 的初始值而把学习率重置到第一轮。

默认第 `r` 轮、第 `e` 个本地 epoch（均从零计数）使用：

```text
lr(r, e) = 0.01 * 0.99**r * 0.998**e
```

其中轮间 `0.99` 来自 FedBPC 的公共 StepLR，本地 `0.998` 来自原 FedRCL。
原仓库默认使用 `0.1 * 0.998**r * 0.998**e`；这一区别属于已确认的实验设置适配。
更换公共 scheduler 后，仍以它提供的当轮学习率乘 `local_lr_decay**e`。

## 参数与边界

算法参数位于 `train_setups.algo.params`：

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `ce_loss.weight` | 1 | 分类损失权重 |
| `prox_loss.weight` | 0 | 近端项权重；`prox_loss=null` 表示无配置权重 |
| `feature_align_loss` | weight=0，align_type=cosine | 特征对齐分支 |
| `rcl_loss.topk_pos / topk_neg` | 1 / 2 | 原始 top-k 参数 |
| `rcl_loss.temp` | 0.05 | 共享温度 |
| `rcl_loss.pairs` | scl / penalty，权重均为 1 | 原始配对配置；penalty 阈值为 0.7 |
| `rcl_loss.branch_level` | null | 自动选择当前模型全部隐藏层；不含分类输出 |
| `rcl_loss.feature_relu` | false | 默认使用 ReLU 前特征 |
| `local_lr_decay` | 0.998 | 每个本地 epoch 的衰减因子 |
| `grad_clip` | 10 | 梯度范数裁剪阈值 |
| `use_amp` | true | CUDA autocast 开关 |
| `metrics_only` | false | 可选复用项目的 batch_protocol 指标入口 |

`branch_level=false` 保留原语义：所有层，包括分类输出；显式索引列表用于选择层级。
共享空列表表示不计算 RCL 分支；这与 pair 空列表的语义不同。
配置的层级索引按当前模型校验。可以覆盖 pair 的采样、温度、beta、阈值、损失类型
及分支层级；未提供的项遵循原始回退规则。
pair 名称必须唯一，不能与 `cls/prox/cossim/total` 的损失或统计字段冲突。

单样本尾批次保持原掩码与哨兵处理，不强行丢弃或拼接样本。
正样本 k 不会按 batch size 自动裁剪：超过当前 batch 大小时明确报错。
默认 k=1 可覆盖非空尾批次。空训练流、非法配置或不支持的模型明确报错。

客户端始终以 `CLLoss(..., reduction=False)` 调用，再按源码顺序归约。
底层 `MultiLabelCrossEntropyLoss` 保留按非零项数归约的分支。
源码的 CLLoss 在 batch>1 时先归约为标量再 reshape 会失败；直接使用该无效组合
现在报明确错误，不替换成另一种平均公式。

与源码一致，CUDA GradScaler 独立于 autocast 开关启用；CPU 禁用 CUDA AMP 和缩放。
默认 AMP 与 FedBPC 的普通精度路径不同，已作为可见配置保留，不声称逐位数值一致。

普通运行复用全局/客户端测试指标，并记录 `fedrcl/cls_loss`、`scl_loss`、
`penalty_loss`、`cossim_loss`、`prox_loss`、`total_loss` 等本地训练均值。
统计使用训练样本数加权，不改变 batch 损失的反向传播归约。
若使用现有 `batch_protocol`，同时开启 `metrics_only`，复用公共指标报告；
本次不新增实验队列或启动任何训练。

## 验证范围

本次仅做静态验证：Python AST/语法、JSON、导入目标与入口注册、公共配置一致性、
源文件计算分支对照、模型特征位置和 scheduler 生命周期审查。
不导入执行训练模块，不运行前向、反向、训练、数据下载或数值等价测试。
静态检查不能证明运行时数值等价或实验精度。

本次静态检查结果：86 个 `src` Python 文件通过 Python 3.9 语法 AST 解析与编译，
8 个算法 JSON 文件可解析；FedRCL 与 FedBPC 的公共配置逐项一致。
新增包的本地导入目标、入口注册、依赖范围、无调试输出及改动空白检查通过。
另外通过 AST 对照检查原始 top-k 方向、归一化和 beta/log-softmax 分支、
客户端归约调用、梯度更新顺序、epoch 调度位置、当轮学习率重置、等权聚合
及 hooks 清理结构，并人工审查特征层映射和源实现的边界处理。
