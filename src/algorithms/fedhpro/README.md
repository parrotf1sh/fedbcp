# FedHPro 移植说明

本实现依据项目根目录提供的 `FedHPro-main/main_run.py`、
`utils/gradient_matching.py` 和 `utils/param_aug.py`，用于 FedBPC 对比实验。
运行时不依赖原源码目录，不新增第三方依赖。

## 运行与公共配置

在 `src` 目录使用统一入口：

```bash
python main.py --config_path ./config/fedhpro.json
```

默认 JSON 的数据、模型、scenario、optimizer、scheduler 和 seed 与当前
`config/fedbpc.json` 一致。后续实验应同步这些公共设置。
默认是 `fedavg_cifar`、100 个客户端、每轮采样 10%、200 轮、5 个本地 epoch、
batch size 50、SGD lr=0.01、momentum=0.9、weight_decay=1e-5、
StepLR gamma=0.99、step_size=1、seed=2022。
这是共同实验协议下的移植，不是原论文实验设置的复现。

- 数据直接复用现有 `data_distributer` 或选定的 `longtail_data_distributer`。
  不执行原仓库长尾子集生成和 Dirichlet 划分；缓存/manifest、类别计数、
  训练子集、尾批次和测试集均来自共享入口。客户端抽样沿用 BaseServer。
- 普通训练直接使用项目 loader 的增强和归一化，不重复添加源码的 batch 裁剪/翻转。
  真实梯度收集仍先遍历同一客户端 dataset 构造图像集合，再重复按类别抽样并应用 DSA。
  项目 dataset 的增强也会在这些遍历中发生；不能声称与原静态预处理数值相同。
- 模型由 `create_models` 创建，不复制原 ResNet、不新增投影头、不修改前向。
  支持 `fedavg_cifar`、`fedavg_tiny`、`vgg11` 的 `(logits, features)` 接口，
  使用 `classifier.in_features`，默认维度为 512。分类头必须是带偏置的线性层。
  梯度收集在 no_grad 下通过该接口取得特征，丢弃 logits，再单独运行分类头求导；
  相比源码 `.features()`，额外的无梯度线性输出不用于任何损失。
- 客户端 SGD 加载共享优化器参数组，每次参与都清空状态，客户端之间不共享动量。
  BaseServer 每轮末更新一次 scheduler。原型和服务器分类头优化器使用独立设置。
- 保留标准本地/全局评估及 W&B 日志。额外本地评估保存/恢复 Python、NumPy、
  Torch 及 loader generator 随机状态，避免统计过程推进训练随机流。
  现有标准 loader 不启用 persistent workers。

## 算法配置

参数位于 `train_setups.algo.params`；未知参数、非法取值和不兼容模型明确报错。

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| metrics_only | false | 批量模式使用统一全局指标，跳过额外客户端评估 |
| dsa_seed_mode | wall_clock | wall_clock / deterministic；批量队列采用后者 |
| length_of_hp | 5 | 每类超原型数量 |
| rounds_of_hp | 30 | 每轮超原型梯度匹配步数 |
| hp_info_t | 0.05 | 本地对比温度 |
| lr_hp | 0.1 | 超原型、元原型 SGD 学习率 |
| meta_length | 50 | 每类元原型数量 |
| batch_real | 100 | 每次每类最多抽取的真实样本数 |
| repeat_num | 1 | 真实梯度采样重复次数 |
| use_margin | false | 是否计算客户端 margin |
| dis_metric | matching | matching / mse / cos |
| meta_rounds | 100 | 元原型梯度匹配步数 |
| meta_epochs | 100 | 每轮新分类头训练 epoch 数 |
| meta_lr | 0.01 | 新分类头 SGD 学习率 |
| meta_batch_size | 50 | 元原型分类头训练 batch size |
| margin_batch_size | 100 | margin 全客户端扫描 batch size |
| method | DSA | 仅字符串 DSA 开启梯度收集增强，其余值关闭 |
| dsa_strategy | color_crop_cutout_flip_scale_rotate | DSA 策略 |

`augmentation` 保留原默认 aug_mode=S、prob_flip=0.5、ratio_scale=1.2、
ratio_rotate=15、ratio_crop_pad=0.125、ratio_cutout=0.5、ratio_noise=0.05、
brightness=1、saturation=2、contrast=0.5。ratio_noise 在原代码中未使用，本实现同样不启用它。
保留 S 随机选一种策略、M 顺序执行全部策略，以及空字符串/none/None 返回原输入的分支。
DSA 保留源码 `int(time.time() * 1000) % 100000` 种子及全局 Torch 重置行为，
包括同一批内共享随机参数；因此固定实验 seed 不保证在默认时间戳模式下启用 DSA 后完全可重复。
批量队列显式改用 `dsa_seed_mode=deterministic`，以训练种子、零起始轮次、客户端、
重复编号和类别确定性派生 seed；保留增强算子、参数、共享随机参数和全局 Torch 重置方式。
单次入口默认仍为 `wall_clock`。详见 [批量说明](../../../scripts/README_fedhpro.md)。

## 轮次与核心计算

1. 获取本轮全局模型和旧超原型；用保存的 meta head 替换一个模型参数副本的分类头。
   第一轮 meta head 独立随机初始化，不直接使用初始全局模型分类头。
2. 客户端在本地优化前收集每类 CE 对特征以及分类头 weight/bias 的梯度。
   骨干 eval/no_grad、分类头 train；特征 detach 后作为可导叶子。
   每类随机无放回抽取至多 batch_real 个样本，少于该值时使用全部样本。
   特征梯度在重复和样本维度求均值，参数梯度按重复次数等权平均。
   不抵消 CE 自带的 batch 均值因子。
3. 重新加载本轮原始全局模型，使用旧超原型训练。首轮也计算全部三项损失。
   每 batch 顺序：前向、CE、可选 margin 扫描、HPCL/HPAL、zero_grad、backward、step。
4. 独立上传每个客户端模型快照，按 `sum(weights * size) / sum(size)` 聚合全部
   state_dict 参数和 buffers，保留先乘、求和、再除的源码顺序。
5. 每类真实梯度只在拥有该类的参与客户端间等权平均，不按样本数加权。
   固定轮次开始时的 meta head，以参数梯度匹配更新元原型。
6. 在所有类别元原型的 detached 副本上训练一个全新随机线性头，再替换聚合分类头。
   不 warm-start，不把此 SGD 更新写入原型或客户端 optimizer。
7. 加载替换后的全局模型，使用新分类头的特征梯度匹配旧头下收集的真实梯度，更新超原型。
   此前后分类头不相同的时序按源码保留。更新后的超原型下一轮才下发。
8. BaseServer 完成全局评估、scheduler 和日志。

本地损失固定为 `CE + HPCL + HPAL`，各系数为 1，没有新增权重：

- HPCL：超原型逐个 L2 归一化后按类求均值，与归一化特征点积并除温度。
  正类 logits 由标签索引，分母通过 logsumexp 计算；非零 margin 用
  logaddexp(log_den, log(margin)) 加入。最后保留源码 `mean() / 当前 batch_size`。
- HPAL：先对同类原始超原型求均值，再归一化，与归一化特征计算
  smooth_l1_loss(reduction="mean")。不能改成先归一化后平均。
- margin：每个训练 batch 内重新扫描客户端全部样本，保持当前 train 模式。
  原始类均值的两两内积去对角线后，除 `(全局类别数 - 1) ** 2` 并求和。
  不归一化、不缓存、不改用本地有效类别数。零 margin 跳过 log 分支。
- matching：特征梯度先对合成样本求均值，再逐坐标调用 distance_wb；
  参数梯度则分别匹配 weight/bias，保留 1/2/3/4 维 reshape 分支及 1e-6。
  mse、cos、create_graph=True 和原始归约方式全部保留，不向量化改写。
- 单样本 calculate_infonce 辅助方法保留，但不替代主流程的 batch 版本。

缺失类别不参与当轮梯度匹配，对应原型保留旧值；分类头重训仍使用所有类别元原型。
不引入补齐、类别重加权、原型 EMA、warmup、数值 clamp 或负 margin 修正。
空客户端/无梯度等不能构成有效训练轮次的输入明确报错，不静默回退。

## 已确认的源码修正 B

上游复用同一个 Local，local_train 直接返回 model.state_dict() 并加入列表。
其中张量共享模型存储，后续客户端的加载/训练会覆盖先前记录，导致聚合缺少各客户端独立更新。
本移植按用户确认的 B 方案，直接继承 BaseClientTrainer.upload_local 的 deepcopy，
每次训练完成后立即保存独立快照。这是已确认的源码缺陷修正，不是原共享覆盖行为的复现。
该问题在上游复用模型的流程中也存在，并非仅由目标框架不同导致。

## 批量实验

批量入口 `python scripts/run_fedhpro_experiments.py` 使用两训练种子、44 次训练、每次 300 轮。
已确认的服务器预算为 meta_rounds=20、meta_epochs=10、rounds_of_hp=10；
这属于计算预算适配，不是原 100/100/30 默认配置复现。单次 JSON 默认参数保持不变。
批量运行逐轮评估分类头修正后的全局模型，并按处理样本数加权汇总在线训练损失；
不改变反向传播的原损失归约。服务器匹配/分类头损失明确记录为内层最后一步值。
阶段计时和通信定义、恢复原预算方式及验证范围见 [批量说明](../../../scripts/README_fedhpro.md)。

## 初次移植的模块对应与验证范围

Global 的聚合、set_meta_params、meta_revised、obtain_hyper_prototypes 对应 Server；
Local 对应 ClientTrainer；两类 InfoNCE、HPAL 和 gradient_matching 对应 criterion；
param_aug 对应 augmentation；模型接口、分类头替换和评估随机状态管理位于 utils。
不迁入原 main、argparse、数据分割、调试输出、未使用的 FedProx mu 或 encoder。

本次仅进行静态验证：AST/语法、JSON/公共配置一致性、导入和注册、源码计算与分支对照、
独立上传快照路径、git diff 空白检查。未运行训练、数据下载、模型前向、梯度计算或运行时测试。
静态验证不等价于已验证运行时数值一致性或训练收敛。
