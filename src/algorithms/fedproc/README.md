# FedProc integration

本实现以仓库内 `FedProc-main` 为对照，接入现有 `BaseServer` / `BaseClientTrainer` 框架。
默认采用原源码的 `ModelFedCon_noheader` 分支语义：使用项目模型分类层之前的原始特征，
不增加投影头、不归一化样本特征、不改动模型的分类前向路径。

## 使用与对比设置

在 `src` 目录执行 `python main.py --config_path ./config/fedproc.json`。
本次移植仅静态检查，未执行此训练命令。

`fedproc.json` 的 `data_setups` 以及模型、优化器、调度器、轮数、采样比例、本地 epoch、
设备和实验 seed 与当前 `fedbpc.json` 一致，FedBPC 是本移植的对比对象。
默认复用其 `legacy` 数据管线：`../../data`、CIFAR-10、100 个客户端、每客户端 2 个 shard、
batch size 50；使用相同数据划分入口、随机种子、训练变换和 DataLoader，不调用原仓库的
数据划分、数据加载或模型构造代码。改变研究设置时，各方法应同步使用相同配置。
如双方切换到项目的 longtail 管线，可共享划分 manifest；最终报告会复用现有
`save_baseline_report`，包含划分哈希。

模型来自 `train_tools.utils.create_models`：CNN 使用 `fedavg_cifar` / `fedavg_tiny` /
`fedavg_mnist`，ResNet 使用 `res10` / `res18`，同时兼容 `vgg11`。
原 FedProc 的 CNN（例如隐藏维数 84）和 ResNet 实现不等同于本项目模型，因此按用户要求，
另存 `model.py:ModelWithFeatures` 作为 FedProc 专用模型接口：内部完整使用项目构造的模型，
不改变网络层、参数数量、初始化或分类计算，只增加 `forward(data, get_features=True)`。
该接口返回 `(logits, features)`，与项目既有特征接口风格一致。通过临时的分类层 forward
pre-hook 读取输入；hook 在 `finally` 中移除，保留特征梯度且不增加一次模型前向。
封装后的 state_dict 键增加 `model.` 前缀，本方法的广播和聚合始终使用该一致格式；
未修改公共模型文件。当前默认 CNN 的特征位置与 FedBPC 相同。

FedProc 的适配层也支持项目 ResNet；现有 FedBPC 仅接受带 `get_features` 参数的公共模型，
当前公共 ResNet 没有此接口。因此默认 CNN 对比可直接复用当前配置；如需切换双方到 ResNet，
还需单独补齐 FedBPC 的特征接口，不能只修改两份配置中的模型名称。

## 源码映射与计算顺序

| 原始位置 | 移植位置 | 保留内容 |
| --- | --- | --- |
| `ContrastLoss.py:SupConLoss_new` | `criterion.py:PrototypeContrastiveLoss` | 按样本计算 exp、求和、比值、log，再对 batch 求均值；仅按列 L2 归一化类别中心 |
| `train.py:train_net_fedproc` | `ClientTrainer.py:train` | 首轮仅 CE，后续 `alpha * CE + (1 - alpha) * prototype_loss` |
| `train.py:get_global_class_center` | `ClientTrainer.py:upload_prototypes`、`utils.py:aggregate_prototypes` | 分类别累计特征总和和样本数，再跨参与客户端求和并除以该类别总样本数 |
| `main.py:fedproc` 分支 | `Server.py:_clients_training` | 首轮广播后初始化中心；所有客户端训练结束后提取新中心；随后聚合模型 |
| `main.py:server_momentum` 分支 | `Server.py:_aggregation` | `delta = old - aggregated`，`v = m * v + (1 - m) * delta`，`new = old - v` |
| `model.py:ModelFedCon` | `model.py:ModelWithProjection` | 可选的 Linear → ReLU → Linear → L2 normalize → classifier 分支 |

`round_idx` 从 0 开始。`alpha_rounds=100` 保留源码的固定 100 轮分界，独立于总轮数；
`round_idx < alpha_rounds` 时取 `round_idx / alpha_rounds`，否则为 1。
首轮和 alpha=1 时仍计算原型损失，保留源码的计算分支。没有引入 temperature、额外损失
系数、logsumexp、特征归一化补丁、原型动量或缺失类别屏蔽。

原型仅由当轮参与客户端的本地训练 DataLoader 提取，不读取验证集或全局训练集。
提取使用 `torch.no_grad()` 和训练模式，保留源码中的训练模式前向语义；如模型具有可更新
的 BatchNorm buffer，其更新后的权重也参与训练/模型聚合。初始提取完整结束后才开始本地
训练；全部本地训练结束后才进行第二次提取，避免交错顺序改变训练数据的随机流。

缺失类别保留上一轮原型；首次没有该类别则置零。源码用此前已出现类别的 `inittype`
构造零向量，在最小类别编号缺失时可能未定义；这里从任一实际特征确定相同形状/类型，
落实其原有零回退分支。若首轮所有参与训练 loader 都无样本，抛出明确错误。

## 原始聚合行为

默认 `aggregation="source"` 按用户选择保留原实现：按参与列表中的**位置** `j`
使用客户端 ID `j` 的样本量，分母为**所有客户端**的样本总量：

`w = sum_j (n_j / sum_all_clients n_i) * w_selected[j]`。

这意味着部分参与时权重和通常小于 1；参与列表顺序与客户端 ID 不一致时，权重与实际
客户端错配。项目采样器即使全参与也可能返回乱序列表，因此不能假定全参与时必然消除
索引错配。该行为原样保留，默认实验名称带 `source`，不得将此结果误记为标准 FedAvg 聚合。

显式设置 `aggregation="sampled"` 才会改为项目 `BaseServer._aggregation`：
按实际参与客户端样本量加权并归一化。这个选项改变了原实现的聚合计算，应作为单独设置
报告；不是默认设置，也不会自动切换。客户端采样仍使用项目公共采样器，以对齐比较方法。

## 可选参数及框架适配

- `use_project_head=false`：默认模型与比较方法完全相同。`out_dim` 此时不参与计算。
- `use_project_head=true`：保留原源码带头分支，`out_dim=256`，在优化器创建前安装投影头；
  此设置改变分类头结构，不属于默认的同模型对比。骨干仍使用项目模型。
- `server_momentum=0.0`：默认关闭；非零时保留原始 FedAvgM 更新公式。
- `train_setups.optimizer.name`：省略即 `sgd`，也支持源码中的 `adam` 和 `amsgrad`。
  `params` 使用所选 PyTorch 优化器的参数；切换 Adam 时移除 SGD 的 `momentum`。
  本地优化器按客户端重新初始化，再下载本轮优化器设置，不共享客户端动量。
- 学习率和 scheduler 使用项目统一设置；没有沿用源码中临时的 GPU 编号、路径或日志服务。

评估、W&B、学习率更新、长尾最终报告复用项目框架。未移植原研究的逐 epoch 调试评估、
TensorBoard 启动线程、特征 CSV 导出及独立 checkpoint 命令行；这些不是 FedProc 损失与
原型更新分支。因此本移植不宣称与原脚本的随机数消耗轨迹或原始骨干实验逐位一致。
原始 `FedProc-main` 保留作对照，不作为运行时依赖。未新增第三方库。

## 验证范围

仅检查 Python 语法/编译、AST 接口和注册、JSON 参数及与研究配置的一致性、依赖引用、
源码差异，并人工逐项核对上述损失、调度、提取顺序、缺类回退和服务器动量分支。
不执行训练、数据下载、模型前向或数值测试；静态验证不代表已经取得运行结果或复现实验指标。
