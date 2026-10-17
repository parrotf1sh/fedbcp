# TinyProto

将项目内 `TinyProto-main` 的 TinyProto-FP 接入现有框架。源码依据：

- `system/flcore/clients/clienttinyfp.py`：本地训练、`agg_func`、原型测试。
- `system/flcore/servers/servertinyfp.py`：掩码生成、`proto_aggregation`。
- `system/flcore/clients/clientbase.py`、`system/main.py`：交叉熵、SGD 和配置。
- 论文：[Communication-Efficient Heterogeneous Federated Learning with Sparse Prototypes
  in Resource-Constrained Environments](https://arxiv.org/abs/2507.04327)。

这是经确认的源码缺陷修复与框架适配，不是逐字执行原仓库的残缺行为。
不增加新损失、投影层、归一化、历史原型融合或性能优化。

## 运行与配置

在项目 `src` 目录运行：

```bash
python main.py --config_path ./config/tinyproto.json
```

模型、数据、轮数、客户端采样比例、本地 epoch、batch size 和训练 seed
与当前 `fedbpc.json` 一致。数据直接使用项目 legacy/longtail 管线，沿用
划分缓存或 manifest、增强、shuffle、drop_last 和全局测试集。
修改 FedBPC 的实验设置后，需要同步 TinyProto 对应字段。

保留原始普通 SGD：默认固定学习率 0.01，每次客户端参与重新创建优化器。
不使用 momentum、weight decay、Nesterov 或 scheduler；不继承 FedBPC 的这些配置。
非零 momentum/weight_decay/dampening、其他优化器及启用 scheduler 会明确报错，
不会被静默忽略。学习率由 `train_setups.optimizer.params.lr` 设置。

| 算法配置 | 默认值 | 对应源码或含义 |
| --- | --- | --- |
| `lambda_proto` | 10.0 | `lamda`，完整特征向量 MSE 系数 |
| `add_cps` | true | 开启类别原型稀疏化 |
| `cps_ratio` | 10 | 每个类别掩码选中的类别块百分比，不是置零比例 |
| `simple_scale` | true | 开启原型计数缩放及服务器缩放 |
| `constant_scale_factor` | 0.00015 | 原仓库 README 示例值；未经当前研究设置调优 |
| `mask_max_attempts` | 10000 | 原掩码贪心搜索每次选择的尝试上限 |
| `mask_cache_path` | null | 可选掩码缓存文件；默认只保留内存掩码 |

原仓库解析器的 CPS/APS 开关默认关闭，缩放常数默认 1；其 README 示例开启 APS、
使用 0.00015，但未开启 CPS。本项目配置明确开启 CPS 和 APS，作为完整 TinyProto-FP
比较方法。两开关的四种组合均保留，可通过配置关闭相应分支。
这组参数仅为移植起始配置，不代表在 CIFAR100/TinyImageNet 或长尾数据上的最佳值。

## 算法对应

### 模型与状态

`Server`、`ClientTrainer` 继承项目基类。服务器使用独立轮次循环，禁止调用
FedAvg 参数聚合。所有客户端首次使用同一个项目初始化，之后保存和恢复各自完整
`state_dict`（含缓冲区），未参与客户端保持上次状态；服务器只共享全局原型。
CPU 模型快照是单进程模拟中的客户端状态存储，不计作模型上传或全局模型。

通过分类器输入 hook 观察项目模型的一次原有 forward，不替换层、不改变参数名，
也不额外执行特征网络。`fedavg_mnist/cifar/tiny`、`vgg11`、`res10/res18`
均取最终线性分类头的输入；ResNet 对应原有池化后的向量。
不使用现有 FedProto 的 ResNet 池化前特征图，也不将项目特征投影到上游的 500 维。
TinyProto 支持 ResNet 不改变 FedBPC 当前对 `get_features=True` 接口的要求；
两者配对比较应选择 FedBPC 已支持的相同模型，默认是 `fedavg_cifar`。

### 本地训练与原型

每批执行一次 `logits, features = forward_with_features(model, data)`。
初轮无全局原型时只计算 CE；后续为：

```text
loss = cross_entropy(logits, targets) + lambda_proto * mse(prototype_targets, features)
```

`mse` 对完整 batch 和全部特征维度取均值。CPS 目标通过 `scatter_` 补零恢复；
未选中坐标的目标为零，但不会预先把本地特征清零，梯度仍作用于完整特征。
非 CPS 分支使用完整全局原型。所有目标均 detach。

原型列表在本地 epoch 循环外创建，累计**全部 epoch、训练模式、更新前**的特征；
不改为最后一个 epoch，也不在训练结束后重新前向收集。
保留源码逐项求和、多样本求均值及单样本分支。令该均值为 `p[i,c]`：

```text
simple_scale=False: local = p[i,c]
simple_scale=True:  local = train_class_count[i,c] * p[i,c]
add_cps=True:       upload = local[mask_indices[c]]
add_cps=False:      upload = local
```

类别计数读取项目训练 `data_map`，不乘本地 epoch 数。
原仓库计数来自本地 train/test 再划分之前；这里按已确认的研究设置使用真实训练计数，
不将验证或测试标签参与 APS。

### CPS 掩码

令 `K=num_classes`、`D=feature_dim`，块宽 `L=D//K`，每类块数
`q=int(K*(cps_ratio/100))`。保留原贪心算法：首向量末尾 q 个位置为 1，
随机候选保持 q 个 1，去重，最大化与已选向量的 Hamming 距离之和，
严格大于时更新最优候选，保留尝试上限及失败分支。
不替换为单位矩阵或其他更快构造，不改变算法的搜索次数。

将 K 位向量按块宽 L 扩展为 D 维掩码，剩余 `D % K` 维保持零。
例如 D=512、K=10、比例 10% 时每类传输 51 个坐标，尾部 2 维均未选中。
`D<K`、q=0 或不能生成足够独特向量时明确拒绝，不自动修正比例。
掩码生成前后恢复训练随机状态；固定实验 seed 下可复现，也避免缓存命中与否
改变后续训练随机流。类别掩码在整个实验中固定，运行元数据记录实际索引。

可选缓存保留“已有则读取，否则生成并保存”的分支。缓存包含维度、类别数、比例、
搜索上限及索引，读取时校验完整块、长度、范围和跨类别唯一性。
使用同一缓存路径可明确固定不同实验种子的掩码；需要各 seed 独立掩码时使用 null
或不同路径。原仓库裸字典缓存不能直接作为本项目带元数据的缓存使用。

### 服务端聚合

仅收集本轮参与客户端上传的类别，不融合旧原型。同轮所有客户端使用同一份
上一轮全局原型。对类别 c 的上传张量列表 `values`，严格保留源码表达式：

```text
simple_scale=False: global[c] = mean(stack(values), dim=0).detach()
simple_scale=True:  global[c] = len(values) * constant_scale_factor
                               * mean(stack(values), dim=0).detach()
```

保留 `len(values)` 因子及乘法次序，不替换为按总类别样本数归一化的加权均值，
也不按论文公式（10）—（11）删除该因子。源码中收集但未用于返回值的
`uploaded_n_ij`、`uploaded_data_num`、`proto_weights` 不作为新的聚合权重。

## 已批准的修复与边界适配

1. 将 `clientePBFL/clientTinyFP` 命名不一致统一为项目 `ClientTrainer`，
   将 `add_csr/csr_ratio` 与 `add_cps/cps_ratio` 统一为后者。
2. 聚合 `return` 移出类别循环，所有类别执行相同原始表达式。
3. CPS 分支补回训练 MSE：原 `train()` 中仅非 CPS 分支加 MSE，
   但 `train_metrics()` 两个分支均加入，论文也要求稀疏原型约束训练。
   该修复恢复原方法的核心约束，会改变残缺源码的梯度，不能保证准确率不变。
4. 部分参与导致类别缺失时，该类训练样本以自身 detach 特征为目标，
   原型误差为零，完整 batch MSE 分母不变；测试距离设为正无穷。
   不缓存旧原型、不补造原型、不改变客户端采样。此规则扩展了源码未处理的边界，
   类别齐全时保持原损失和预测公式。
5. 初始没有原型时主指标不可用，不沿用原 `(0, 1e-5, 0)` 伪样本数；
   有效训练轮没有产生任何原型、空客户端、非法掩码及非有限数值均明确报错。
6. 使用项目 `n_rounds` 个实际更新轮并在聚合后评估；原脚本的
   `range(global_rounds+1)` 在训练前评估，会额外执行一轮未纳入最终评估的训练。
   移植采用固定最终轮，原脚本可选早停、模拟慢客户端和计时休眠不进入本研究协议。
7. 原脚本未实现的 `diverge`、`calc_proto_distance`、距离/角度统计引用，
   临时整模型文件往返、调试打印和绘图不接入；非有限计算通过明确异常停止。
   项目统一 W&B 日志负责有效的训练与测试统计。

## 评估与日志

每轮在同一份完整全局测试集上评估所有持久客户端，包括未参与本轮的客户端：

```text
overall_test_accuracy = mean(all_client_prototype_accuracies)
prediction = argmin_c mean((features - restored_global_prototype[c]) ** 2)
```

采用原源码的 batch 广播 MSE 公式，不限制为本地训练类别；距离相等时保留
按类别序号 `argmin` 的行为。缺类距离为正无穷；若没有任何原型则不生成主指标。
分类头 argmax 单独统计为辅助指标，不作为 TinyProto 原型预测的替代品。

主指标是完整测试集上的客户端平均性能，不是聚合全局模型准确率，也不同于原论文
的本地测试划分协议。保留项目已有本地测试 loader 的最终逐客户端结果，
另附按本地测试样本数加权的准确率；longtail 无本地测试集时不自行切分。
评估使用 eval/no_grad，保存并恢复随机状态与 loader generator，不写回客户端快照。

复用项目指标定义，记录 accuracy、macro accuracy/F1、worst20 recall、head/middle/tail、
逐类准确率及客户端标准差（非多 seed 标准差）。所有准确率为 0–1。
训练损失按 batch 平均，再对当轮参与客户端平均；原型误差是未乘 lambda 的 MSE。
测试原型 MSE 的缺类样本贡献为零，与训练分母规则一致；缺类测试样本仍计入准确率分母。

结果使用项目 `experiment_logging` 写入 W&B history/summary，要求有效 run。
`tinyproto_runtime` 记录公式、配置、掩码、特征位置、修复和评估定义；
longtail 记录 split 哈希，legacy 沿用入口已有划分元数据。
兼容指标 `server_test_acc`、`global/top1` 表示全局测试集范围，不表示存在全局模型。
最终结果保存在 `tinyproto_final`、`final/*` 及 `batch_summary`。
不新增 HDF5/CSV 或本地结果文件依赖；可选掩码缓存和 W&B 自身缓存除外。

## 静态验证范围

仅进行 Python 语法/AST、JSON、相对导入与导出、入口注册、配置一致性及关键控制流核对。
核对范围包括四种 CPS/APS 组合的代码分支、全部 epoch 原型累计、逐类聚合返回位置、
损失分母、缺类分支、固定 SGD、客户端状态隔离及全类别 MSE 预测。
不导入执行算法，不运行模型前向/反向、训练、数值测试、数据下载或 W&B 实验。
静态检查不证明运行时数值等价或实验性能。
