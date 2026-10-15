# FedSC 移植说明

本实现用于 FedBPC 对比实验，依据项目根目录提供的 `FedSC-main` 源码移植。
算法计算以 `main_fedsc.py`、`utils/infonce_lcl.py`、`utils/util_func.py`
的实际执行行为为准；运行时不依赖原源码目录。

**此版本保留已确认的源码行为，包括正原型列表触发异常回退、关系原型修正未写回、
按原型列表位置使用客户端权重。正常有限特征下，原始回退使 CPDR 为零。
这是所提供源码的移植，不是修复这些行为后的算法版本。**

## 运行与配置

在 `src` 目录使用项目统一入口：

```bash
python main.py --config_path ./config/fedsc.json
```

默认配置的 `data_setups`、`scenario`、`model`、`optimizer`、`scheduler`
及训练种子与当前 `config/fedbpc.json` 一致。统一 CLI 覆盖规则保持不变；
后续更改实验时，也应同时修改 FedBPC 与 FedSC 对应的公共配置。

配置参数位于 `train_setups.algo.params`：

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `cl_tau` | 0.05 | 原 RPCL 对比温度，必须为有限正数 |
| `metrics_only` | false | 批量队列启用统一指标路径，跳过额外客户端评估，不保存模型 |

不新增损失权重、原型动量、置信度筛选、投影头或稳定化规则。
首轮只计算 CE，第二轮起固定为 `CE + RPCL + CPDR`，三项系数均为 1。

## 数据、模型与实验设置

- 数据直接使用项目 `data_distributer` 或已选择的 `longtail_data_distributer`
  输出。复用相同划分种子、索引缓存、DataLoader、尾批次及数据增强，
  不执行原仓库的长尾子集选择或 Dirichlet 划分。
- 原 FedSC 在归一化后的 batch 上再次裁剪和翻转。本移植使用项目 loader 的
  裁剪、翻转、归一化和 Cutout 等既有流程，不在客户端重复增强。
  这是经确认的公共输入流程适配，不声称与原数据预处理数值一致。
- 相同数据配置及划分种子使用相同划分缓存；可核对现有日志的
  `partition/indices_sha256`。同一数据划分不意味着各算法的增强随机流逐样本相同。
- 模型完全由 `train_tools.utils.create_models` 创建，不复制原 ResNet-10，
  不改变项目模型前向或 state dict，不增加可训练参数。
  `fedavg_cifar`、`fedavg_tiny`、`vgg11` 使用已有
  `forward(data, get_features=True)`，返回 `(logits, features)`。
  原源码返回顺序为 `(features, logits)`，只在训练调用处适配返回值顺序。
  不支持该接口的模型在初始化时明确报错。
- 分类器输入的原始特征用于原型均值及 CPDR；L2 归一化只出现在原 RPCL 内部。
- 客户端抽样复用 `BaseServer._client_sampling` 的逐轮规则，与 FedBPC 一致。
  不使用原源码独立的 `np.random.RandomState(seed)` 抽样序列。

原默认实验采用 ResNet-10、10 个客户端、每轮 5 个、100 轮、10 个本地 epoch、
batch size 64 和固定 SGD 学习率 0.1，无动量、无权重衰减。
当前默认实验与 FedBPC 对齐：`fedavg_cifar`、100 个客户端、采样率 0.1、
200 轮、5 个本地 epoch、batch size 50、学习率 0.01、动量 0.9、
权重衰减 1e-5、逐轮 StepLR 衰减 0.99、训练种子 2022。
这些是公共实验设置适配，不是原论文实验设置的复现。

每次客户端下载全局参数后重新建立 SGD，加载服务器当前参数组并清空优化器状态。
客户端之间、轮次之间不保留本地动量；本地更新也不写回服务器 optimizer。
服务器 scheduler 仍由 `BaseServer` 每轮更新一次，无额外本地 scheduler。
默认零起始第 `r` 轮学习率为 `0.01 * 0.99**r`，该轮各本地 epoch 使用相同学习率。

## 计算与模块对应

| 原源码 | 移植位置 | 保留内容 |
| --- | --- | --- |
| `Local.local_train` | `ClientTrainer.py`、`criterion.py` | 首轮分支、正负原型组织、三项损失、更新及特征收集顺序 |
| `InfoNCE_LCL.infonce_lcl_loss` | `criterion.py:InfoNCELoss` | 展开共享负池、归一化、拼接 logits、温度及 CE 均值 |
| `average_kns` | `utils.py:average_prototypes` | 多样本逐项求和、单样本 detach 分支，无原型归一化 |
| `Global.get_re_p` | `utils.py:get_relational_prototypes` | 类别插入顺序、中心距离、对角置 inf、临时修正 |
| `Global.get_dis_weights` | `utils.py:get_distribution_distances` | 最后 epoch 类别计数到均匀先验的距离 |
| `Global.get_co_p` | `utils.py:get_consistent_prototypes` | sigmoid 权重、按位置原地加权及求和 |
| `Global.average_parameters` | `Server.py:_aggregation` | 先按样本数乘权重并求和，再除总样本数 |
| `FedSC_main` | `BaseServer`、`Server.py:_clients_training` | 全局下发、本地训练、原型更新、模型聚合、评估 |

`InfoNCE_LCL.sim_loss` 和 `Local.calculate_infonce` 两个未被主训练调用的
损失辅助方法也保留在 `InfoNCELoss`，不会替代实际使用的 `infonce_lcl_loss`。
演示用随机张量、调试打印、独立 main、重复 argparse 和未参与算法的实验工具不迁入。

本地每个 batch 只进行一次训练前向，顺序为：
前向 → 类别计数 → 损失 → backward → optimizer.step → zero_grad → 原型收集。
只在最后一个本地 epoch 收集该次前向的原始特征，不做训练后原型重提取，
不使用 FedBPC 的置信度、归一化或筛选逻辑。上传类别计数同样只取最后 epoch，
不累加多个 epoch；样本量聚合权重仍使用该客户端的 `datasize`。

服务器每轮在参与客户端之间按原顺序合并原型，只保留当前轮出现的类别，
随后构造一致性原型，最后聚合全体模型参数和 buffers。原型不跨轮做 EMA 或补齐。
模型聚合保留原式 `sum(local_weight * client_size) / sum(client_sizes)`，
不改为先构造 float32 比例再逐项累加。

## 源码实际行为与边界

1. **关系原型修正未写回。** `torch.stack` 生成临时张量，距离为各原型到
   重复展开的类别中心的距离，不是原型之间的距离。仍执行对角置 inf、最小索引
   及两原型平均，但只修改临时张量，不写回返回的字典。单贡献者分支也保持该流程。
2. **正原型异常回退。** 原型字典的值为列表，源码直接对其调用 `torch.mean`，
   因而进入原始宽泛异常回退。同一 try 内的两个 append 顺序保持不变。
   正原型与 CPDR 目标均退回 `features[i].detach()`；不补 `torch.stack`。
   正常有限特征下 CPDR 为零，但仍实际执行 MSE，RPCL 仍参与反向传播。
   缺少对应类别时也走原回退，不预先过滤该训练样本。
3. **共享负原型池。** 逐样本收集异类原型列表，再把整个 batch 的列表展开为
   所有 query 共用的负池，保留重复项和遍历顺序。其他样本收集的原型可能与
   当前 query 同类；不增加逐样本掩码，不去重，不裁剪负池。
4. **原型权重按位置匹配。** 令 `p[k,c]` 为客户端最后 epoch 的类别频率，
   `d[k] = sqrt(0.5 * sum_c((p[k,c] - 1/C)**2))`，
   `a[k] = sigmoid(n[k]/sum(n) - d[k]/sum(d))` 后对 `a` 作总和归一化。
   类别列表中的第 i 个原型乘 `a[i]`，即使该类别缺少部分客户端贡献。
   不按真实客户端 ID 对齐，也不按类别重新归一化。
5. **原地改写关系原型。** 上述加权会替换关系原型列表内的元素，
   随后求和得到一致性原型。下一轮下发的关系原型已经被加权，不能改成仅对副本求和。
   保留 NumPy 权重计算及原 dtype，不额外转换成 float32。
6. **首轮占位值。** 保留原源码为每类创建的未初始化占位张量；首轮损失不读取它们，
   该轮结束即用真实上传原型替换。不得用这些占位值进行预训练原型约束。

对于原源码不能有效计算的输入，保留失败语义并提供错误信息：无训练 batch、
空负原型池、客户端类别计数总和为零、聚合样本量总和为零、
所有参与客户端分布距离之和为零。最后一种情况包括各客户端类别频率完全均匀。
不增加 epsilon、均匀权重回退、补原型、跳过损失、跳过 batch 或 `nan_to_num`。
局部 epoch 数与温度进行配置校验，尾批次仍交由项目 loader 原样提供。

## 评估与日志

单次默认配置（`metrics_only=false`）沿用 `BaseServer` 的训练前评估、逐轮评估、客户端统计及 W&B 生命周期。
主结果使用已有 `server_test_acc`；另记录 `fedsc/ce_loss`、`fedsc/rpcl_loss`、
`fedsc/cpdr_loss` 和 `fedsc/total_loss`。
损失日志先取各客户端 batch 均值，再取参与客户端均值，统计不改变训练归约。

客户端标准评估在上传原型构造完成后进行，不重新计算原型。
项目指标沿用既有准确率舍入与评估口径；原源码只有每轮全局测试准确率。
额外的项目评估会遍历 loader，可能消耗随机流，因此不承诺与原脚本逐位数值相同。
批量入口及其独立指标口径见 [FedSC 批量运行说明](../../../scripts/README_fedsc.md)。
批量队列记录按实际处理样本数加权的损失，不沿用单次入口的 batch/客户端等权日志。
它不执行额外客户端评估或训练前评估，每轮完成更新后评估一次全局模型。
这里只改变日志和评估路径，损失归约与原型构造操作保持不变。

## 初次移植的验证范围（批量队列验证另见上述说明）

仅进行静态验证：Python 3.9 语法 AST 与内存编译、JSON 解析、本地导入目标、
入口注册、公共配置一致性、依赖范围、源计算分支和生命周期核对。
不导入执行训练模块，不运行前向、反向、训练、数据下载或数值测试。
静态验证不能证明运行时数值等价或实验精度。

本次静态检查结果：92 个 `src` Python 文件通过 Python 3.9 AST 解析与内存编译，
9 个算法 JSON 文件可解析，FedSC 与 FedBPC 的公共配置逐项一致。
新增包导入目标与符号、入口注册、既有依赖范围及无调试输出检查通过。
使用 AST 对照确认了原 RPCL 核心张量操作、原型均值与单样本分支、
关系原型构造以及宽泛异常回退；并检查了权重公式、原地改写、轮次映射、
最后 epoch 特征收集、优化器重置及服务器更新顺序。
文件哈希核对确认原始 FedSC 目录、已有模型、数据管线、其他算法和配置均未修改。
