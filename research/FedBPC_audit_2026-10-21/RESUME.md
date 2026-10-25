# FedBPC 研究续接记录

> **最新入口（2026-10-22）：[独立原型方案续接记录](../FedBPC_local_prototypes_2026-10-22/RESUME.md)。** 已完成新候选实现、128 项测试与合成对照。下方“未修改代码/未训练”等描述属于 10 月 21 日历史状态，已被后续工作覆盖。当前仅研究 sharding/LDA 本地类别不平衡。

## 2026-10-22 用户纠正（优先于下文旧计划）

- 目标仅为 sharding / LDA 的本地类别不平衡和缺类，不以全局长尾为研究目标。旧报告中的长尾实验不是当前方向的立项条件。
- 用户明确要求继续构建原型方案；无需论文级新颖性，以合理机制、可运行和验证为准。
- 已开始独立实现 FedBPC-Complete：同一轮全局编码器提取原型；用原型补足本地不足类别的分类损失质量，同时从过多类别等量扣除；真实特征方向约束保留。原 FedBPC 保持不变作为对照。
- 找到已有可用环境 /Users/ygding/opt/anaconda3/envs/clsimb_py39/bin/python（torch2.2.2、torchvision0.17.2、numpy1.26.4、wandb；CPU，无CUDA/MPS）。此前“系统Python缺torch”不等于本机没有训练环境。
- 当前先做性质测试、CPU合成数据小实验与入口验证，不启动大规模CIFAR训练。

更新时间：2026-10-21（Asia/Shanghai）。状态：本次请求的第一轮研究报告与优先级实验计划已完成；未进入算法改写和训练阶段。

## 恢复时先读这里

1. REPORT.md：完整结论、代码证据、8个可证伪假设、A/B/C数学设计与反方审查。
2. EXPERIMENT_PLAN.md：D1—D8诊断、分阶段预算、公平协议、文件级实施计划。
3. LITERATURE.md：一手链接与核验深度，包含最终补查的Fed-CBS、FedPAC及UAI2026可靠性聚合近邻。
4. LOG_TABLES.md 与 audit_results.json：全部历史复算；reproduce_audit.py可重建。

不要重启一轮泛泛文献搜索，也不要自动启动全矩阵。下一步应先确认实际训练环境和可用检查点，执行D1—D4；本机系统Python无torch。若用户只问进度，说明报告已交付、尚无新训练结果。

当前推荐：A先修复/对齐；B是有条件的论文候选；C为更高成本备选。B若精确全量原型无帮助则停止。C的类别感知采样已有Fed-CBS，不能宣称概念创新。B的CV未归一化矩可无偏，但方向归一化会引入偏差，缓存错误可增大方差，带符号残差可出现负坐标/超单位球，均已写入反例和诊断。

## 最后完成的工作与后续待办

- 已补读FedYoYo方法部分、FedProto出版页，核验BTR transfer调用和RNG/loader重放边界。
- 已补充无放回batch精确反例、反预测缓存方差3.0625（原方差1）、Bernoulli参与两类最差方差1→.5的枚举；这些是玩具性质验证，不是精度提升。
- 已完成报告、文献表、分阶段计划、代码实施表及论文停止条件；已核对本地链接与关键数值。
- 已复跑既有splitter 9项测试，通过；未运行需要torch的测试。未改src、scripts、tests、原始日志；未提交Git；仅新增本研究目录。
- 尚缺：真实GPU运行时验证、原始FedBPC各CSV有效配置/commit/checkpoint、同目标强基线多seed对照、D1—D8真实模型结果。
- 部分论文只核验元数据/摘要，尤其RUCR、MMC-FL、FedUDRP首次上线日期；方法定稿前补齐全文。已检索不等于穷尽所有近邻。
- 用户回来如要求继续实施，优先按I0/I1和固定模型诊断推进，维持原始BPC可复现版本。完整训练需先用训练机测资源预算，再由用户决定规模。

可直接用于续接的一句话：“读取 research/FedBPC_audit_2026-10-21/RESUME.md 和 EXPERIMENT_PLAN.md，在保留已审计结论的基础上继续，先做固定模型诊断，不重新开始审计。”

## 用户授权与范围

阅读原始任务：`/Users/ygding/.codex/attachments/1110d840-fd23-4707-ad48-c7ceca257d70/已粘贴的文本.txt`。
目标：深度审计 FedBPC，核验相关文献，提出至少五个可证伪假设及 A/B/C 三条路线，交付科研报告、排序实验计划、代码实施计划。
本阶段不大规模修改原算法、不启动昂贵训练。允许轻量诊断。用户特别要求保存关键判断、证据和待办，以便约五小时后恢复。
保存可核验的研究结论和决策摘要，不依赖聊天上下文。未安排自动唤醒。

## 起点

- 工作目录：`/Users/ygding/Project/FedClsimb`；初始 Git 无改动。
- 当前提交：`3472e12`（Extend FedHPro and FedSA experiments）。
- 仓库规则：根目录 `AGENT.md`；未发现 AGENTS.md。没有要求并行代理，故独立执行。
- `docs/`、`result/`、`fedbtr_run_metrics/` 被 .gitignore 忽略，但实际存在且需审计。
- 新研究产物放在本目录（未被忽略），保留既有实现不变。

## 已建立的事实（初步）

1. FedBPC 默认 CIFAR10、100 客户端、sharding 每端 2 片、每轮 10 端、200 轮、5 本地 epoch、CNN、seed=2022；不是默认全局长尾。
2. criterion.py 中 CE 为 minibatch 出现类别的均值平均；alignment 和 prototype CE 同样按出现类别平均。`class_balanced_loss=false` 只切换 CE。
3. 客户端训练后额外遍历训练 loader，提取单位方向均值原型；可靠性为 p(y|x) × 正误权重 × exp(-gamma*logit_norm)。
4. 原型以 confidence*n/(n+5) 聚合，单位化，再做 EMA；默认 singleton 更新变慢、age>5 失效。模型主体仍样本量 FedAvg。
5. 头部可靠性聚合实现存在，但当前 JSON 明确关闭；源码缺省却开启。历史日志必须区分版本配置，不能把该分支归咎于当前默认。
6. 已有 FedBTR 深度研究、20 个 experiment/final JSON 及 CSV。需充分利用，避免把已经探索过的方向当作新贡献。

## 已完成的验证与重要判断

- `reproduce_audit.py` 已执行，标准库即可复现；结果在 `audit_results.json`、`LOG_TABLES.md`。`source_inventory.json` 保存 253 个相关文件的哈希，不能解释为每个文件逐行审读。
- 既有 longtail splitter 的 9 个 unittest 全通过。本机 Python 3.9.6 未装 torch/numpy/torchvision/pandas/pytest，未安装依赖、未启动训练；数值反例不等于 PyTorch 端到端验证。
- 可靠性使用绝对 logit 范数，不满足公共平移不变性：z=[2,0,-1] 与 z+100 的 softmax 完全相同，可靠性由 .7545 变为 .000142，跨越 .05 门槛。这是公式性质；尚无证据说明历史运行发生了如此大的平移。
- 在仓库 IF100、IID、100 客户端长尾划分下，两类总量为 80 和 48，均摊后每端至多一个样本；FedBPC 默认 min_proto_samples=2 使它们永远没有原型。使用合成 CIFAR 标签数量复现；不能归因于旧的全局均衡 sharding 运行。
- 单贡献者时 confidence*n/(n+5) 在归一化中抵消，仅硬门槛和固定 singleton EMA 起作用；局部单位化也丢失集中度。
- 当前 minibatch 类均值 CE 一般不等于全局 macro 风险。双类稀有率 .01、batch50 的独立有放回模型中，稀有类期望权重约 .1975 而非 .5；真实无放回 loader 不应冒称该公式精确。
- 本地训练后的不同编码器产生的原型被聚合，再给参数平均后的模型使用；这是坐标和版本一致性假设，不是已证实主因。原型提取重用随机增强训练 loader，额外消耗 RNG。
- 旧 FedBPC CSV 缺完整配置/commit/checkpoint。head_false 等仅能按文件名分组，不能证明因果。8 份详细日志的按类采样数组相同，“more_client” 文件名不能证明参与率不同。
- FedAvg 最后20轮均值49.2595%，head_false50.7475%，re49.9395%；尾段波动约4–5pp，缺独立多种子证明。局部全局测试约20.6%，聚合全局约50.7%，不能笼统断言聚合毁掉性能。
- 记录到的少量辅助/CE 非零梯度余弦均为正；后期辅助损失数值较大但梯度范数比仅约.24。不能直接以“梯度冲突”或“辅助loss大”定案。
- 当前 FedNH JSON 是 LDA .1，与多数默认 sharding2 不同。FedProto/FedSA/TinyProto 的全客户端私有模型平均评估不等于单全局模型指标。FedSC 保留上游使部分正则失效的实现语义，比较必须说明。
- FedBTR 有20套配置/final/CSV，已核对指标。LDA IF100 seed2022：natural27.27、balanced40.38、rawteacher41.46、full41.36；seed2027：balanced40.71、raw/full42.29。主要收益来自目标重加权；完整teacher-retention尚无相对raw稳定增益。已有 transfer diagnostics 代码但历史20CSV均无 transfer/列，应复用，勿重复开发。
- 长尾 class counts=[1034,2877,371,4800,80,1725,619,133,48,222]；head=[3,1,5,0]，middle=[6,2,9]，tail=[7,4,8]。最差类不一定是最稀有类。
- BaseServer 每轮按 round_idx 重设 NumPy seed，因此不同 train seed 的参与序列相同；需单独引入/记录 sampling seed。reported round time 不含后续 FedBPC 诊断评估，不能当公平总耗时。

## 文献核验进度与创新边界

已用一手论文/出版页核验 FedAvg、SCAFFOLD、AFL、cRT、Balanced Softmax、FedRS、FedLC、FedRoD、CReFF、FedNTD、FedProc、FedNH、FedIIC、FedGraB、FedLoGe、FedTGP、RUCR、FedLF、FedYoYo、FedVARP、Optimal Client Sampling、AISTATS2026 SSDI 等。已直接读官方 FedGraB PID loss、FedTGP server 关键实现。
RUCR 已明确用全局模型提取原型，不能宣称共同编码器本身创新；FedVARP 已有服务器历史残差修正；SSDI 已区分 skew 与 scarcity。FedUDRP 2026 的一手页面明确讨论不确定性筛选及少样本过滤风险，精确在线日期尚未核实，仅作需排除的近邻。2026 年12月期刊期号的搜索结果不应自动当作截至10月已发表证据。

## 前一检查点的阶段性待办（现已完成，保留为工作轨迹）

1. 先看是否已有 REPORT.md / EXPERIMENT_PLAN.md / LITERATURE.md；若有，按其中完成状态继续，不重跑审计。
2. 补齐少数论文原文（FedYoYo方法、FedProto原文），核验 BTR transfer diagnostics 的调用边界。
3. 写最终中文报告：事实/推断/假设分层，>=5可证伪假设，A修复与对齐、B共同编码器下类别矩估计、C主动支持覆盖采样三条互相区分路线；明确这些目前是研究候选，不是已验证创新。
4. 写低成本到完整训练的分阶段计划、公平协议、消融/敏感性/失败分析、精确代码实施路径、论文定位与停止条件。
5. 更新本文件为最终续接索引；检查所有统计与链接、Git仅有研究产物；向用户交付链接和关键结论。
