# 文献核验与创新边界

检索截止：2026-10-21。优先采用论文、出版社页面、作者预印本及官方仓库。以下是面向本项目的机制核验，不是穷尽式系统综述。只核验到摘要的条目不作为详细算法或优先权判断的充分依据。局限与项目启示属于本次分析，不应误读为原论文自述。

## 1. 基础目标、漂移与长尾决策

| 文献与一手链接 | 核验与核心机制 | 对本项目的意义及边界 |
|---|---|---|
| McMahan et al., **Communication-Efficient Learning of Deep Networks from Decentralized Data**, AISTATS 2017，[论文页](https://proceedings.mlr.press/v54/mcmahan17a.html) | 论文页；本地多步训练后模型平均 | 本地 epoch、客户端样本量和参与采样共同决定更新，不能只看损失公式推断全局目标。 |
| Karimireddy et al., **SCAFFOLD: Stochastic Controlled Averaging for Federated Learning**, ICML 2020，[论文](https://proceedings.mlr.press/v119/karimireddy20a.html) | 一手论文页；控制变量缓解客户端漂移 | 目标一致性与优化漂移是两个问题；SCAFFOLD 不自动把自然频率目标变成类别均衡目标。 |
| Mohri et al., **Agnostic Federated Learning**, ICML 2019，[论文](https://proceedings.mlr.press/v97/mohri19a.html) | 一手论文页；针对未知客户端混合的鲁棒风险 | 客户端公平、类别公平、域公平不等价。最小化最差客户端风险不保证最差类别召回。 |
| Kang et al., **Decoupling Representation and Classifier for Long-Tailed Recognition**, ICLR 2020，[预印本](https://arxiv.org/abs/1910.09217) | 一手论文；解耦表征与分类器 | 冻结骨干的均衡线性探针是必要诊断。集中式结果不能证明联邦瓶颈总在分类器。 |
| Ren et al., **Balanced Meta-Softmax for Long-Tailed Visual Recognition**, NeurIPS 2020，[原文](https://proceedings.neurips.cc/paper/2020/file/2ba61cc3a8f44143e1f2f13b2b729ab3-Paper.pdf) | 原文；训练/目标类别先验下的 softmax 校正及采样设计 | Balanced Softmax、样本逆先验加权和 minibatch 出现类别均值不是同一种目标；必须各自写清公式。 |
| Li & Zhan, **FedRS: Federated Learning with Restricted Softmax for Label Distribution Non-IID Data**, KDD 2021，[作者原文](https://www.lamda.nju.edu.cn/lixc/papers/FedRS-KDD2021-Lixc.pdf)，[DOI](https://doi.org/10.1145/3447548.3467254) | 原文公式；对本地缺失类限制 softmax 中的尺度 | 不应粗略等同“删除所有缺失类梯度”。没有本地正样本不意味着该类对应的负样本信息没有价值。 |
| Zhang et al., **Federated Learning with Label Distribution Skew via Logits Calibration**, ICML 2022，[原文](https://proceedings.mlr.press/v162/zhang22p/zhang22p.pdf) | 原文；类别相关的 logits 校准 | 已覆盖局部频率诱导的分类偏置；局部计数为零、全局先验改变及参与采样仍需单独处理。 |
| Chen & Chao, **On Bridging Generic and Personalized Federated Learning for Image Classification**（FedRoD）, ICLR 2022，[预印本](https://arxiv.org/abs/2107.00778) | 一手摘要/机制；均衡通用预测与个性化残差 | 通用与个性化性能应分表；不能拿不同预测对象的数值做单一排名。 |
| Shang et al., **Federated Learning on Heterogeneous and Long-Tailed Data via Classifier Re-Training with Federated Features**（CReFF）, IJCAI 2022，[原文](https://www.ijcai.org/proceedings/2022/0308.pdf)，[官方实现](https://github.com/shangxinyi/CReFF-FL) | 原文；梯度匹配生成联邦特征后重训分类器 | 已覆盖服务端分类器修正；特征合成与服务端训练成本应计入。类均值不足以代表边界所需的全部信息。 |
| Lee et al., **Preservation of the Global Knowledge by Not-True Distillation in Federated Learning**（FedNTD）, NeurIPS 2022，[作者预印本](https://arxiv.org/abs/2106.03097) | 一手论文；非真实类别输出上的蒸馏 | 在本地输入上保留输出约束，不自动识别缺失类别输入区域的函数行为；这是需实验检验的支持域限制。 |

## 2. 原型、梯度与近期长尾方法

| 文献与一手链接 | 核验与核心机制 | 项目启示／潜在失效条件 |
|---|---|---|
| Mu et al., **FedProc: Prototypical Contrastive Federated Learning on Non-IID Data**, 2021 预印本，[原文入口](https://arxiv.org/abs/2109.12273) | 预印本；全局原型引导对比表示 | “原型 + 对比损失”本身不是本项目的新贡献；本地 port 的投影头、全体/参与者聚合模式会改变成本和含义。 |
| Tan et al., **FedProto: Federated Prototype Learning across Heterogeneous Clients**, AAAI 2022，[出版页](https://ojs.aaai.org/index.php/AAAI/article/view/20819)，[官方实现](https://github.com/ms116/fedproto) | 出版页/摘要；交换类别原型以支持异构客户端模型 | 本项目 port 保留私有模型；与 FedBPC 的共享全局模型任务不相同。不能用其低分支持“原型方法都无效”。 |
| Dai et al., **Tackling Data Heterogeneity in Federated Learning with Class Prototypes**（FedNH）, AAAI 2023，[作者预印本](https://arxiv.org/abs/2212.02758) | 一手论文；原型均匀性与语义逐步融合 | 原型几何和分类器结构联动；不能在不控制归一化、聚合权重的情况下归因于某个损失。 |
| Wu et al., **Towards Robust Federated Learning for Class-Imbalanced Medical Image Classification**（FedIIC）, MICCAI 2023，[作者预印本](https://arxiv.org/abs/2206.13803)，[官方实现](https://github.com/wnn2000/FedIIC) | 论文机制；难度/先验相关校正与客户端内外对比 | 类别频率不等于学习难度已被研究；医学域、患者划分与自然图像标签偏斜的假设不同。 |
| Xiao et al., **Fed-GraB: Federated Long-tailed Learning with Self-Adjusting Gradient Balancer**, NeurIPS 2023，[原文](https://proceedings.neurips.cc/paper_files/paper/2023/file/f4b8ddb9b1aa3cb11462d64a70b84db2-Paper-Conference.pdf)，[官方实现](https://github.com/ZackZikaiXiao/FedGraB) | 原文及官方 util/losses.py；PID 风格调节正/负梯度，官方实现使用逐类 BCE 路径 | “修复正负梯度竞争”已有直接近邻。不能用 softmax CE 的逐行 mask 近似后声称复现。其轮数、骨干和 batch 与当前实验不同。 |
| Xiao et al., **FedLoGe: Joint Local and Generic Federated Learning under Long-tailed Data**, ICLR 2024，[论文](https://openreview.net/forum?id=V3j5d0GQgH)，[官方实现](https://github.com/ZackZikaiXiao/FedLoGe) | 论文机制；稀疏 ETF 引导表示及全局/本地范数校正 | 已研究局部与通用预测、表示几何和范数偏置。当前 ReLU 非负特征不能无条件实现含负内积的理想 ETF 几何。 |
| Zhang et al., **FedTGP: Trainable Global Prototypes with Adaptive-Margin-Enhanced Contrastive Learning for Data and Model Heterogeneity in Federated Learning**, AAAI 2024，[原文入口](https://arxiv.org/abs/2401.03230)，[官方实现](https://github.com/TsingZ0/FedTGP) | 论文及官方 servertgp.py；服务端可训练原型与自适应间隔 | 可训练原型已有。官方服务器使用嵌入/MLP及原型分类训练，不是直接参数 FedAvg；额外服务端优化应单列成本。 |
| **Federated Learning With Long-Tailed Data via Representation Unification and Classifier Rectification**（RUCR）, IEEE TIFS 19, 5738–5750, 2024，[DOI](https://doi.org/10.1109/TIFS.2024.3402361) | 核验出版元数据及摘要，未取得完整原文；明确以全局模型抽取原型，并用原型混合进行分类器校正 | “统一编码器提取原型”不能宣称首次。本报告不能依据摘要断言其全部统计处理细节。 |
| Lu et al., **Adaptive Logit Adjustment and Feature Optimization in Federated Long-Tailed Learning**（FedLF）, PMLR 260, 2025，[论文页](https://proceedings.mlr.press/v260/lu25a.html)，[官方实现](https://github.com/18sym/FedLF) | 一手论文页；自适应 logits、类内聚集和特征去相关 | 三类因素联动已有近邻；更多损失不等于解释了瓶颈。这里只用出版卷年份，避免混淆会议与上线年份。 |
| Yan et al., **You Are Your Own Best Teacher: Achieving Centralized-level Performance in Federated Learning under Heterogeneous and Long-tailed Data**（FedYoYo）, ICCV 2025，[作者原文](https://arxiv.org/html/2503.06916v2)，[官方实现](https://github.com/shanss132/FedYoYo) | 已读方法部分；弱/强增强自蒸馏与局部/全局有效先验融合 | 正确弱视图才作为教师可能影响困难类覆盖；相关性先验的少样本稳定性需检查。双视图增加计算，论文题名不意味着本仓库也可达到集中式水平。 |

## 3. 参与采样、缺失与统计估计的近邻

| 文献与一手链接 | 核验与核心机制 | 对候选路线的约束 |
|---|---|---|
| Jhunjhunwala et al., **FedVARP: Tackling the Variance Due to Partial Client Participation in Federated Learning**, UAI 2022，[作者预印本](https://arxiv.org/abs/2207.14130) | 一手论文；利用历史客户端更新消减部分参与方差 | 历史残差控制变量不新。把对象换为类别矩只是候选变体；须证明类别支持稀疏性带来的独特问题与收益。 |
| Chen, Horváth & Richtárik, **Optimal Client Sampling for Federated Learning**, TMLR 2022，[作者论文页](https://wenlin-chen.github.io/publication/chen2022optimal/)，[预印本](https://arxiv.org/abs/2010.13723) | 一手论文；基于更新重要性的采样 | 优化参与概率及重要性校正不新；应区分所有候选先计算再选择上传与真正减少本地计算。 |
| Zeng et al., **Disentangling Federated Learning Heterogeneity: A Dual-Perspective Analysis of Quantifying Skew Versus Scarcity**, AISTATS 2026，PMLR 300, 5068–5076，[论文页](https://proceedings.mlr.press/v300/zeng26b.html)，[官方实现](https://github.com/wkzeng/SSDI) | 核验标题、作者、出版信息和摘要；区分标签分布偏斜与标签覆盖缺失 | 不能把“频率与覆盖不同”作为独立首创。候选贡献应落在固定目标下的估计偏差、方差和干预结果。 |
| **Rethinking sparse supervision on federated long-tailed learning**, Knowledge-Based Systems 342, 115819, 2026，[DOI](https://doi.org/10.1016/j.knosys.2026.115819)，[出版社](https://www.sciencedirect.com/science/article/pii/S0950705126005459) | 出版元数据与摘要检索内容；全文访问受限。MMC-FL 涉及缺失类别的梯度/服务端校正 | 把“缺失类”当全新问题不成立；详细比较需取得全文后完成。 |
| **Uncertainty-driven reliable prototypes for heterogeneous federated learning**（FedUDRP）, 2026，[出版社](https://www.sciencedirect.com/science/article/pii/S2666603026000242) | 出版社可检索正文片段；不确定性筛选、渐进融合、可训练原型。精确首次上线日期未核实 | 原文已讨论极少样本过滤的风险。作为高优先级近邻，不据此作未经确认的先后权判断，也不声称过滤风险首次发现。 |

检索中出现以 2026 年 12 月为期刊期号的条目；未验证其首次在线日期，未把这些条目作为截至本次审计已发表的主要依据。

## 4. 最终近邻补查

| 文献与一手链接 | 核验与核心机制 | 创新约束 |
|---|---|---|
| Zhang et al., **Fed-CBS: A Heterogeneity-Aware Client Sampling Mechanism for Federated Learning via Class-Imbalance Reduction**, ICML 2023，[论文页](https://proceedings.mlr.press/v202/zhang23y.html)，[原文](https://proceedings.mlr.press/v202/zhang23y/zhang23y.pdf) | 一手论文页和原文引言；通过选择客户端改善参与者合并数据的类别均衡 | C不能把“类别感知参与”称为新贡献；必须比较其目标与固定macro风险下的类别梯度方差，并匹配实际参与成本。 |
| Xu, Tong & Huang, **Personalized Federated Learning with Feature Alignment and Classifier Collaboration**（FedPAC）, ICLR 2023，[预印本](https://arxiv.org/abs/2306.11867)，[官方实现](https://github.com/JianXu95/FedPAC) | 一手摘要及原文搜索片段；局部/全局特征对齐与分类器协作权重。OpenReview原文访问遇验证页 | B的表示对齐叙事需对照该工作；任务为个性化，不能直接混入单全局模型排名。另有2026无监督同名FedPAC，不能混淆。 |
| Kovalchuk et al., **Who to Trust? Aggregating Client Predictions in Federated Distillation**, UAI 2026，PMLR337, 3162–3181，[论文页](https://proceedings.mlr.press/v337/kovalchuk26a.html) | 已核验出版页/摘要；利用密度相关不确定性处理共享公共输入上的预测聚合 | 可靠性聚合已有近期理论近邻；其公共未标注数据假设不同于本仓库，不能直接声称其解决了无公共数据的缺类输入问题。 |

## 5. 真正值得检验的交叉问题

**问题一：原型的估计对象是否在过滤过程中发生了变化？** 现有代码估计的是“通过信任门槛的客户端单位方向的加权方向”，并非声明过的全局类均值。若保留完整类别样本质量而降低估计噪声，在严重分布偏斜下能否改善决策？这比笼统“提高原型质量”更具体。需要对照 RUCR、FedNH、FedTGP、FedUDRP；若统一编码器已经解释全部收益，则只得到工程修复结论。

**问题二：类别集中在少数客户端时，减少原型估计误差是否比增加正样本出现机会更关键？** 固定全局计数、固定局部优化，分别改变支持覆盖与参与概率，才可区分二者。SSDI 已给出相关概念，Optimal Client Sampling 已给出采样理论；本项目必须提供更具体的机制或可证伪边界，不能重新命名它们。

**问题三：修正在哪一步失效？** 同一轮、同一验证样本、同一本地增强批次，比较校准模型、局部模型、局部输出集成与参数平均模型，追踪修正是否存活。仓库 FedBTR 已有此诊断框架。只有确认失效位置，才能判断应调整表示、分类器还是聚合。

这些都是研究候选。当前证据不足以认定一个尚无人研究且已经有效的新算法。

