# FedProto

本实现将仓库内 `FedProto-main` 的 FedProto 更新过程接入项目，使用项目的数据、模型、配置和日志。
源码依据是 `exps/federated_main.py`、`lib/update.py::update_weights_het`、
`lib/utils.py::agg_func/proto_aggregation` 和 `lib/update.py::test_inference_new_het_lt`。

## 运行与配置

在项目的 `src` 目录执行：

```bash
python main.py --config_path ./config/fedproto.json
```

`fedproto.json` 的数据、模型、通信轮数、客户端参与率、本地 epoch 数、SGD 参数、
scheduler 和 seed 与当前 `fedbpc.json` 对齐。实验中如修改 FedBPC 配置，需同步对应字段。
支持既有 legacy 和 longtail 数据管线；longtail 使用与对比方法相同的 manifest。
不调用原研究的数据划分或预训练参数下载，不增加依赖。

算法参数：

| 配置 | 含义 | 默认值 |
| --- | --- | --- |
| `lambda_proto` | 原源码 `ld`，原型 MSE 系数 | 1.0 |
| `missing_class_distance` | 原型预测中非候选类别的有限距离填充值 | 100.0 |

原源码参数解析器的 `ld` 默认值为 1；README 的 CIFAR10 示例使用 0.1。
保留原预测规则时请保持距离填充值为 100，不换成无穷大。
`optimizer.name` 支持 `sgd` 与 `adam`，参数来自 `optimizer.params`；
切换 Adam 时应移除 SGD 专有的 `momentum` 参数。
未显式指定时，SGD momentum 为原源码的 0.5，Adam weight_decay 为原源码的 1e-4。

## 总体测试性能：用于总体性能表

每轮本地训练和全局原型聚合完成后，评估**所有客户端**，包括当轮未参与训练的客户端。
每个客户端都使用自己最新的模型和当前全局原型，在**同一份完整全局测试集**上预测：

```text
overall_test_accuracy = sum(client_test_accuracy[k] for k in all_clients) / n_clients
```

主指标使用原型距离预测；候选条件仍是“存在全局原型且属于该客户端训练类别”。
类别集合只由训练 `data_map` 决定，不使用测试标签确定候选类别。
不会为了提高全类别测试结果而取消源码的候选限制，也不会筛掉无候选客户端。
所有类别先填距离 100，再对候选类别填入 MSE，最后对完整距离向量执行 `argmin`。
这保留了源码的有限填充值行为：无候选时返回类别 0；若候选距离超过 100，
非候选类别也可能因有限填充值胜出。报告额外记录无候选客户端数量。
没有任何全局原型时不伪造主指标；有效非空训练轮应生成至少一个原型，否则明确报错。

该指标衡量完整类别空间任务下的平均客户端预测效果，**不是聚合全局模型的准确率**。
与 FedBPC 对比时，FedBPC 的所有客户端共享一个全局模型，因而同一定义下的平均值
就是其全局模型准确率。表注需说明 FedProto 的客户端平均及原型候选限制。
不得将此协议与原研究基于客户端测试划分的数字视为同一评估协议。
不进行模型参数平均、预测集成或测试后微调。

所有准确率都记录为 0–1 小数。客户端标准差使用总体标准差（ddof=0），
不能用作多实验种子均值的误差条。`head/middle/tail` 根据训练总类别计数，复用
`longtail_metrics.groups_from_counts` 分组，各组指标为组内类别准确率均值，再对客户端等权平均。
macro-F1、worst20 recall 也先按客户端计算后取均值。

主要字段：

| 字段 | 含义 |
| --- | --- |
| `overall_test_accuracy` | 全部客户端完整测试集上的原型预测准确率均值，主指标 |
| `global/top1`、`server_test_acc` | 主指标的兼容别名；global 表示测试集范围 |
| `test/prototype/client_accuracy_std` | 客户端原型准确率的标准差 |
| `test/prototype/head`、`middle`、`tail` | 组内类别准确率的客户端平均 |
| `classifier_test_accuracy` | 未限制候选类别的分类头准确率客户端平均，辅助指标 |
| `test/classifier/cross_entropy` | 分类头 NLL；不冒充原型预测的分类损失 |
| `test/prototype/prototype_mse` | 原型 MSE，按测试样本数汇总所有 batch |
| `train/total_loss`、`ce_loss`、`proto_loss` | 先对 batch、再对 epoch、最后对参与客户端平均 |
| `train/last_batch_acc` | 原源码本地训练返回的最后一个 batch 准确率的客户端平均 |

实验结果只记录到 W&B，不生成自定义 CSV、JSON、JSONL 或本地结果目录：

- **Config**：完整实验配置；`fedproto_runtime` 保存评估定义、初始化、特征位置、类别分组、
  DataLoader 的 drop_last，longtail 另含 split 哈希。
- **History**：逐轮总体性能、辅助分类头性能、逐类指标、损失、学习率和参与客户端 ID。
  `clients/<客户端编号>/prototype_accuracy`、`classifier_accuracy`、`eligible_classes`
  记录每个客户端的两种准确率及候选类别数。
- **Summary**：`overall_test_accuracy`、`final_top1` 等固定最终轮汇总；
  `final/overall_test/*` 和 `final/classifier_test/*` 保存最终指标。
  有验证集时另记 `final/overall_validation/*` 和 `final/classifier_validation/*`。
  `fedproto_final` 保存最终完整测试集、验证集及原有本地测试集的每客户端明细。

需要有效的 W&B run；若 W&B 未初始化或被禁用，在训练前明确报错，避免实验结果丢失。
使用 online 模式将结果上传 W&B；W&B SDK 自身的缓存和运行日志由 SDK 管理。
现有 `batch_protocol` 的 FedProto 分支也不写本地结果文件，仍返回汇总给调用方。
longtail 的最终结果由 FedProto 记录到 W&B，不让通用报告误评未训练的服务器模板模型。
没有本地测试集的客户端不会伪造本地结果，**仍参与完整测试集的总体评估**。

## 核心计算对应关系

| 原始计算 | 项目组件 |
| --- | --- |
| `update_weights_het` | `ClientTrainer.train`、`criterion.PrototypeLoss` |
| `agg_func` | `utils.average_prototypes` |
| `proto_aggregation` | `utils.aggregate_prototypes` |
| `FedProto_taskheter/modelheter` 的共同通信轮 | `Server._clients_training`、`reporting.run_experiment` |
| `test_inference_new_het_lt` 的两种预测 | `measures.evaluate_client` |

- 分类损失保留 `NLLLoss(log_softmax(logits), labels)`。
- 原型损失保留默认 mean MSE，对 batch 和全部特征元素求平均。
- 无全局原型时使用 `0 * ce_loss`；标签缺少全局原型时目标为当前特征的 detached 副本，
  这些零误差样本仍在 MSE 分母中。
- 每个 epoch 清空样本特征列表，最终只上传最后一个 epoch 的在线特征均值。
  特征来自更新前的前向，即使收集发生在 `optimizer.step()` 之后，也不重新前向。
- 本地按样本平均，服务器按贡献客户端等权平均；保留顺序累加与单元素分支。
- 全局原型每轮仅由本轮上传重建。缺失类别不沿用旧值、不按样本量加权、不做 EMA、归一化或置信度筛选。
- 用 `detach` 代替原源码 `.data`；全局字典值直接存 tensor，省去无计算意义的单元素 list 包装。
- 客户端完整模型状态独立保留；首次参与使用相同项目初始化，之后恢复自己上次状态。
  优化器每次本地训练重建，无跨轮 momentum/Adam 状态。
- scheduler 每个通信轮推进一次，下轮所有参与客户端采用该轮学习率。
  服务器优化器只用作空状态的配置模板，无梯度的 step 不更新任何模型参数。
- 评估使用 `eval/no_grad`，不修改持久模型状态；保存和恢复随机数与评估 loader generator，
  避免新增报告消耗下一轮训练的随机流。

模型由 `create_models` 创建，模块、参数名、分类前向和初始化均沿用项目：

- `fedavg_mnist/cifar/tiny`、`vgg11`：读取分类器输入的隐藏特征。
- `res10/res18`：读取 `layer4` 池化前特征图，与原研究 ResNet 的取特征位置一致。
  不额外池化、投影或归一化。每个样本特征的空间维度保留在原型及 MSE 中。

当前 FedBPC 仍要求模型提供 `get_features=True`；本移植未更改该算法或共享模型接口，
故 FedProto 支持项目 ResNet 不代表现有 FedBPC 的 ResNet 运行条件已改变。

## 已确认的适配与评估修正

使用项目的数据划分、增强、DataLoader（包括保留尾 batch）、统一模型与初始化、
客户端抽样、优化器超参数和 scheduler。原脚本每轮全参与、独立建模、部分预训练参数、
硬编码模型异构、固定学习率和 drop_last=True 不作为当前研究协议。
源码 task/model heter 两个入口的本地更新与原型聚合逻辑相同，合并为可调用组件；
不用硬编码客户端编号制造模型异构。

修正原评估函数跨客户端和跨预测方式累计 correct/total 的问题，分别统计；
原型测试损失汇总全部 batch，不只记录最后一个 batch。
返回值统一为具名字典，消除 model_heter 的解包数量错误。
原脚本未使用的其他算法训练函数、预测集成函数及用于可视化的原型导出脚本不接入 FedProto 主流程。

本次只进行静态验证，不运行训练、模型前向/反向、数据下载或数值测试。
静态验证不能证明运行时数值一致性或实验性能。
