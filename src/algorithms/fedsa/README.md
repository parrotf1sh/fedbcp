# FedSA

将仓库内 `FedSA-main` 的 FedSA 接入项目，用作 FedBPC 的对比方法。
源码依据为 `system/flcore/clients/clientsa.py`、
`system/flcore/servers/serversa.py`、`clientbase.py` 和 `trainmodel/models.py`。
本实现不调用原仓库入口、数据生成器、模型下载或临时模型文件。

## 运行与配置

从项目的 `src` 目录运行：

```bash
python main.py --config_path ./config/fedsa.json
```

`fedsa.json` 的 `data_setups`、model、scenario、optimizer、scheduler 和实验 seed
与移植时的 `fedbpc.json` 完全一致。修改对比实验配置时需同步这些字段。
数据沿用 `_get_setups` 传入的 DataLoader：legacy 共用分区缓存与 partition_seed；
longtail 共用 manifest、数据增强和 split 哈希。原仓库的 `shuffle=False`、
`drop_last=True` 不覆盖项目设置。

| 配置 | 源码对应 | 默认值 |
| --- | --- | --- |
| `lambda_anchor` | `lamda`，锚点 MSE 权重 | 1.0 |
| `lambda_mcl` | `margin_contrastive`，间隔对比损失权重 | 1.0 |
| `lambda_cc` | `classifier_calibration`，分类器校准权重 | 1.0 |
| `anchor_momentum` | 锚点旧值系数 | 0.9999 |
| `anchor_seed` | 锚点初始化的独立 CPU seed | 0 |

原解析器真正生效的 `lamda` 默认值是 1.0；10.0 是已注释的配置。
默认锚点更新显式保留源码的两个系数 `0.9999` 和 `0.0001`。
其他 momentum 使用 `1 - anchor_momentum` 作为新原型系数。
只支持 SGD，参数从项目 JSON 读取，每次客户端参与时重建优化器，
不延续跨轮 momentum 状态。服务器的空状态优化器只承载 scheduler；
每通信轮推进一次，无梯度的模板 optimizer.step 不更新模型。

## 模型与初始化

使用 `create_models` 的原始模型及初始化，要求 `forward(data, get_features=True)`
和线性 `classifier`，与当前 FedBPC 支持的 fedavg_cifar、fedavg_tiny、vgg11 对齐。
特征直接取原分类器输入，维度读取 `classifier.in_features`。
不添加池化、投影、归一化，不改分类器参数和模型 state_dict。
每个客户端首次参与时使用同一项目模型初始化，以后恢复自己的完整模型状态。

锚点初始化在 CPU 使用以下模块，保持 training 模式和源码创建顺序：

```text
Linear(D, D) -> SyncBatchNorm(D) -> ReLU(inplace=True)
             -> Linear(D, D) -> SyncBatchNorm(D) -> ReLU(inplace=True)
```

模块创建之后生成 `randn(C, D)`，前向输出 detach 后移到实验设备。
初始化模块不参与任何优化、不放进客户端模型。直接使用 PyTorch SyncBatchNorm
替代 MMCV 的 PyTorch 包装，eps=1e-5，其余参数沿用默认值；不新增 mmcv。
参考 MMCV v1.7.2 的 `cnn/bricks/norm.py` 与 `utils/parrots_wrapper.py`。
该流程面向项目已有单进程客户端模拟，不初始化分布式进程组。
使用独立 CPU 随机数上下文保留 anchor_seed=0，退出后恢复项目随机流，
不再执行原仓库多个客户端构造器中的全局 `manual_seed(0)`。

## 核心计算与边界

本地四项损失：

```text
CE(head(features), labels)
  + lambda_cc * CE(head(anchors), all_class_labels)
  + lambda_anchor * MSE(features, anchors[labels])
  + lambda_mcl * CE(-(euclidean_distances + one_hot(labels) * gap), labels)
```

- 保留源码先累加三个锚点项、再加样本 CE 的顺序；CE/MSE 均为默认 mean。
  即使权重为零仍计算对应分支；逐类 unreduced 校准 CE 也保留并记录。
- 距离保留 `sqrt(sum(features**2) - 2 * features @ anchors.T + sum(anchors**2).T)`。
  不增加 epsilon、clamp、温度或其他数值修正。原式可能产生非有限数值，
  本移植不以改变损失的方式掩盖它；日志/评估会拒绝非有限结果。
- `gap = max(local_gap, global_gap)`；两者均先取类别间最近欧氏距离，再平均。
  本地间隔来自该客户端上一次参与产生的原型；全局间隔来自全部当前语义锚点。
  每次本地训练前计算一次，不在 batch/epoch 内重算。
- 首次没有本地原型时 local_gap=0；已有原型但类别数为 0 或 1 时 local_gap=-inf。
  -inf 保留在损失计算中；日志只记对应标志与有限值均值，避免序列化无穷大。
  数据集本身少于两类会在初始化前报错，因为初始化 SyncBN 和全局间隔不适用。
- 收集所有本地 epoch、所有 batch 的更新前特征，并 detach 后按类顺序平均。
  不能换成最后一个 epoch 或训练结束后的重新前向。单元素分支保留。
- 服务端对本轮贡献某类别的客户端等权平均，不按样本量加权，不归一化。
  全局原型字典每轮重建；缺失类别不沿用旧全局原型。
- 只更新本轮收到原型的语义锚点，其他类别锚点保留。更新不参与 autograd。
  同轮所有客户端看到相同锚点；未参与客户端的模型和历史本地原型不变。
- 参数在服务端的存储仅用于恢复客户端状态，永不参数平均或广播覆盖。
  公共模板模型只用于初始化和承载优化器配置，不参与主指标评估。
- `ClientTrainer.train_metrics(global_prototypes)` 保留源码可选的
  `CE + lambda_anchor * 全局原型MSE` 报告公式，缺失标签使用 detached 当前特征，
  仍计入 MSE 分母。默认不调用此函数，它不等于四项训练目标。
- 空训练集、无训练 batch 或本轮完全无原型明确报错，不静默跳过客户端。

共享现有 FedProto 的 `average_prototypes`、`aggregate_prototypes`、`cpu_snapshot`、
`preserve_evaluation_rng`：前两个与 FedSA 源码使用同一顺序平均算法，后两个不涉及学习规则。
不复用 FedProto 的特征位置、损失函数、候选限制或预测距离填充值。

## 已确认的实验协议适配

采用项目数据、统一模型、实验初始化、客户端抽样、SGD 超参数和 scheduler。
严格运行 n_rounds 次、轮后评估；原仓库为 global_rounds+1 次、轮前评估。
固定参与率、完整本地 epochs，关闭慢客户端模拟、随机参与率和提前停止。
原 FedSA 上传路径本来就没有调用基类的掉线/时间筛选逻辑，不补加该逻辑。
图像任务不接入原仓库 NLP 列表输入及异构模型族。
这些是已确认的实验环境差异，不宣称复现原仓库的逐位数值或论文成绩。

## 指标与报告

默认单次入口每轮用所有客户端的最新模型（包括未参与本轮者），在同一完整全局测试集上评估，
主指标为每客户端原型预测准确率的等权平均。
原型候选为本轮全局原型的所有类别，**不限制客户端训练类别**。
距离矩阵先填 +inf，有原型类别填 MSE，最后对完整类别空间 argmin；保留并列行为。
不以语义锚点替代测试原型，不做模型/预测集成。

初始化尚无全局原型时主指标为不可评估，不把源码 `(0, 1e-5, 0)` 占位计数当真实准确率。
评估函数区分 None 与空字典；空字典保持全 +inf 的 argmin 行为，
但正常训练轮若完全未生成原型会提前报错。
原源码恒为 0 的 AUC 占位不作为有效 AUC 报告。

所有准确率为 0–1 小数。head/middle/tail 分组只依据训练类别计数，复用
`longtail_metrics.groups_from_counts`。macro-F1、worst20 recall 和分组准确率
先按客户端计算，再等权平均。客户端标准差不代表多实验 seed 的标准差。
分类头准确率和 CE 独立作为辅助指标，不冒充原型指标或四项训练损失。
已有本地测试划分时，最终另报告原服务器口径的样本加权原型准确率；
无本地测试集的客户端不伪造该项结果，仍参加完整测试集评估。

| 字段 | 含义 |
| --- | --- |
| `overall_test_accuracy`、`global/top1`、`server_test_acc` | 全部客户端完整测试集上的原型准确率均值 |
| `classifier_test_accuracy` | 辅助分类头准确率均值 |
| `train/total_loss`、`ce_loss`、`anchor_loss`、`cc_loss`、`mcl_loss` | batch 均值后对参与客户端等权平均；分项为未加权值 |
| `train/local_gap` | 排除 -inf 后的本地间隔均值；无有限值则不记录 |
| `train/local_gap_finite_clients` | 本地间隔均值的有效客户端数量 |
| `train/local_gap_is_negative_infinity` | 当轮 local_gap=-inf 的客户端比例 |
| `anchors/margin/*`、`prototype/margin/*` | 源码类间距诊断；后者保留缺失类补零，仅供报告 |
| `fedsa_runtime` | 配置元数据、评估定义、seed、初始化和分区信息 |
| `fedsa_final` | 最终完整测试、可选验证及本地测试的客户端明细 |
| `final/local_test/*` | 已有本地测试划分上的样本加权结果 |

结果通过项目现有 SwanLab 接口保存；需要有效的 SwanLab run。
兼容 batch_protocol 的返回汇总，默认沿用 FedProto 的 SwanLab 结果路径，不另写自定义结果文件。
[批量队列](../../../scripts/README_fedsa.md) 显式开启 `fedsa_batch` 和 `write_metrics_files`，
启用指标/状态文件和指定评估轮次：默认 300 轮训练中，首轮、每 10 轮及最后 10 轮评估全部客户端。
未评估轮次不记录旧准确率；最后 10 轮指标使用真实的最后 10 个训练轮次。
参数、原型、锚点及优化器始终只在内存中存在，不序列化为文件。
longtail 不调用通用模板模型报告。评估使用 eval/no_grad 并恢复随机数及 loader generator，
避免新增报告改变后续训练随机流；保存的客户端模型状态不受评估覆盖。

## 初次移植的静态验证范围（批量队列验证另见运行说明）

本次只做 Python AST/编译、JSON 解析、配置对齐、导入/注册/接口和源码分支核对、
diff 格式检查。未运行训练、模型前向/反向、数据下载或数值测试。
静态验证不能证明设备运行兼容性、数值一致性、收敛或实验性能。
