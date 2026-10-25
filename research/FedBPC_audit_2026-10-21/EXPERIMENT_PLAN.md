# 2026-10-22 范围更新
本文是历史实验计划。当前优先级已调整为全局类别均衡下的 sharding/LDA 局部不平衡，见 [新设计与验证计划](../FedBPC_local_prototypes_2026-10-22/DESIGN.md)；不自动执行原全局长尾矩阵。

---

# 实验优先级与代码实施计划

日期：2026-10-21。所有训练预算均为提案，**本次未启动训练、未实现A/B/C**。先执行能否定假设的检查；只有通过关卡才扩大预算。背景与公式见[报告](/Users/ygding/Project/FedClsimb/research/FedBPC_audit_2026-10-21/REPORT.md)。

## 1. 第一优先级：锁定可比较的研究对象

先建立三个场景，不混用结果：

| ID | 目的 | 数据协议 |
|---|---|---|
| S0 | 解释旧默认实验 | CIFAR-10全局均衡、legacy sharding2、100端；固定原划分指纹 |
| S1 | 只引入全局长尾 | longtail CIFAR-10 IF100、IID、100端、4%独立验证；共享现有5fb90b93ce0d…划分 |
| S2 | 全局长尾与标签异质性耦合 | longtail CIFAR-10 IF100、LDA .1、100端；共享现有177ea16e5d0f…划分 |

历史指纹前缀只用于识别，真正启动前比较完整64位哈希及数据内容；重新生成时不能把不同图像顺序的合成标签指纹当原始CIFAR指纹。S0不能与S1直接相减并归因于“长尾”，因为数据量和验证留出也变了。

完整比较另建共同longtail流水线的2×2控制网格：IF=1/100 × IID/LDA .1，均使用相同验证比例、类别顺序、预处理。IF变化会同时改变总训练量，主表描述此事实；机制实验增加匹配总量的均衡子集，对“样本量下降”和“类别比例下降”作区分。

**类别覆盖独立干预：** 在固定全局类计数、固定各端总样本量的计数矩阵中，通过交换分配构造同一稀有类的集中/分散支持；再随机分配该类内部样本，保持图像集合不变。不是仅调整Dirichlet α就声称只改变coverage。优先做计数可行性检查；若无法同时满足约束，记录改变的量并缩小因果主张。

## 2. 阶段0：已完成的零训练诊断

| 检查 | 结果与用途 |
|---|---|
| 12份旧CSV、20套FedBTR元数据、20份CSV复算 | 已生成LOG_TABLES与audit_results；未补造缺失配置 |
| 公共logit平移 | 同预测，可靠性跨阈值；证伪当前不变性 |
| min2原型资格 | S1计数条件下类4/8全体不合格；需优先消除硬失效 |
| 类均衡反例 | 有/无放回模型都显示batch类均值不等于全局macro |
| singleton权重 | 连续置信度和count shrinkage归一化后抵消 |
| 缺类classifier行屏蔽 | 反例证明改变合法CE梯度方向 |
| 历史残差与采样 | 穷举验证无偏性、预测历史降方差、反预测历史升方差 |
| 既有splitter测试 | 9项通过；未覆盖torch训练运行时 |

可复现入口为[reproduce_audit.py](/Users/ygding/Project/FedClsimb/research/FedBPC_audit_2026-10-21/reproduce_audit.py)。从仓库根目录运行 Python 3.9+ 的该脚本即可重建JSON与表格；不训练、不修改输入。

## 3. 阶段1：固定检查点诊断，先决定该不该保留原型路线

需要训练机上的torch环境、实际CIFAR数据、完整模型/优化器/RNG/DataLoader状态。历史未保存的checkpoint不能从CSV恢复。若没有可用检查点，先限定为S1/S2各一条30轮、单seed的目标对齐FedAvg轨迹，在第1/10/30轮保存；这是**2条短轨迹、60轮、600次客户端调用、E=5时3000客户端epoch**，不是整个实验矩阵的授权。

| 实验 | 操作及控制 | 记录 | 两类结果如何解释 |
|---|---|---|---|
| D1 目标梯度审计（H1） | 冻结同一θ；流式计算训练macro参考梯度；重放同批次比较present-class与global-weighted | 按类/整体梯度cos、范数、1步验证变化；固定优化步长 | 一致则H1不能解释该场景；不一致且global目标的一步变化更好，才进入目标修复 |
| D2 筛选偏差（H2） | 同一模型、样本、特征，重放min1/2、gamma0/.05、gate0/.05；不重新训练 | 每类总量/上传量/拒绝原因、均值误差、有效端数、置信度分布 | 纳入低支持降低误差支持修复；加噪时恶化则需明确噪声权衡 |
| D3 编码器/版本（H3） | 同轮相同训练输入；global-pre、local-post、global-post各提取一次 | 对当前全局完整矩的相对L2/角误差；原型分类竞争间隔 | 统一编码器不能改善则不继续该叙事；改善统计但无决策变化只算统计现象 |
| D4 原型可用性上限（H5） | 用全训练数据固定θ，比较精确类均值NCM、均衡线性probe、当前head；训练和验证严格分离 | perclass recall、macro-F1、最差类、两两混淆；probe同预算 | 精确NCM和原型引导一步都无用，停止B；probe有效则偏向classifier问题 |
| D5 参与与历史（H4） | 固定θ/局部完整矩，重复均匀抽样；BPC/B0/B1共用子集 | perclass MSE、bias²、variance、零向量率、方向误差、age，以及CV负坐标/模长越界率 | B1未优于B0，放弃缓存复杂度；小q才改善则限定适用范围 |
| D6 梯度关系（H6） | 选前/中/后检查点；按类和支持层分组；同批数据 | CE/align/proto各项在head/backbone的范数、cos；未定义cos记null | 无冲突则不做梯度手术；局部冲突存在也需一步验证风险支持 |
| D7 修正存活（H7） | 复用FedBTR TransferDiagnostics；固定“原全局错、teacher对”的验证cohort；批次哈希一致 | teacher→local→输出集成→参数平均→后续轮的修正留存、margin、失误迁移 | teacher先失效、local失效、aggregation失效分别对应不同机制；全为零cohort不能算100%保留 |
| D8 缺类负梯度 | 固定模型，比较完整CE与缺类行mask的完整macro梯度 | 梯度偏差、head类间margin变化 | 若mask只在少数偏置条件有效，必须明确改变目标；不把它作为普适修复 |

全训练精确矩和集中式线性probe是**仿真诊断参照**，不作为可部署联邦算法，不参与测试集训练。NCM弱不单独充分否定辅助正则，因此D4同时测原型引导的受控一步作用。

最低checkpoint先用第10/30轮；第1轮用于检查初始化边界。已有检查点足够就不重训。每个检查点先处理全体训练特征一次并缓存离线诊断数据，避免每个分支重复前向；该缓存仅用于仿真诊断，单列成本与访问假设。

### 阶段1通过门槛

- 完整配置、manifest、参数、RNG和batch trace可复现；
- A的代码修复能通过性质测试；
- B必须同时显示统计误差问题和决策/梯度层面的有用空间；
- C只有参与供给表现为主要瓶颈才进入，不因玩具方差下降直接立项。

不设未经测量的“至少提升X个百分点”门槛。正式短训练前，用基线重复性和研究所需效果定义实际有意义差异δ，并冻结判断规则。

## 4. 阶段2：短训练，单因素筛选

第一批仅4臂：FedAvg natural、FedAvg global_macro、原FedBPC、最小A；选择S2单seed、30轮。预算为120轮、1200次客户端调用。这里30轮只能发现崩溃和明显趋势，不能发布最终排名。

通过后，展开以下7臂，S1/S2 × 2个训练seed × 50轮，**上限28条短运行、1400轮、14000次客户端调用**。上一批可用记录不能因轮数不同直接合并为终点对照。

| 臂 | 相对父臂的唯一改动／用途 |
|---|---|
| F0 | 普通FedAvg CE，常规强制参考 |
| F1 | F0只换global_macro，分离目标校正收益 |
| P0 | 原FedBPC，完整显式配置 |
| P1 | P0仅确定性统计loader与随机状态隔离，观察统计视图效应 |
| P2 | P1仅换global_macro CE |
| P3 | P1仅做筛选修复；先在D2确定gamma与min_count各自效果后使用最小组合 |
| A | P2加P3被D2支持的修复，检验交互 |

P3若同时变两项，不把它当二者各自的因果证明；D2/必要的额外短分支负责拆开。P1也不是“毫无算法影响”的基础设施修复，确定性提取改变了统计视图，必须保留该对照。

B0/B1是下一轮候选，不在本批同时扩大参数空间。只有A与原方法的差异被解释清楚，才用A作为共同目标/预算底座比较共同编码器、HT和历史修正。

调参只读验证集。所有方案同数量候选、同停止规则；发现NaN/所有尾类缺原型等结构失败可提前停止并记录，不能只保留成功种子。

## 5. 阶段3：公平完整比较，分批预算

核心批：F0、F1、P0、A、B0、B1，共6方法 × S1/S2 × 3种子 × 200轮 = **36次完整运行、7200全局轮、72000次客户端调用**。E=5时为360000客户端epoch，样本不等大，最终计算预算以真实训练样本前反向次数为准。

只有B有独立增量才补论文相关方法：FedNH、FedGraB、RUCR、FedYoYo，4方法 × 2场景 × 3种子 × 200轮 = **24次、48000次客户端调用**。原文需要更长训练/不同架构时另列忠实协议；统一200轮表只代表共同预算，不能据此说较慢收敛方法本身劣。

若研究转向表示几何，加入FedLoGe；转向可训练原型，加入FedTGP；转向分类器重训，加入CReFF/FedHPro。不要为了凑数量把所有端口全跑一遍。FedProto/FedSA等个性化方法独立表；必须评估通用输出时单独明确推理规则。

C先用共同点FedSGD对共同点FedSGD，统一实际梯度样本数、期望参与数、累计上传字节，并补Fed-CBS与范数重要性采样对照。不要拿一遍梯度方案和E=5 FedAvg只按轮数对比。E>1扩展若做，标记理论不适用并补漂移实测。

训练时间不能从旧CSV可靠推算。每种方法先测5轮稳态的客户端训练、统计前向、服务器优化、评估及同步时间，再估算剩余GPU小时；记录GPU型号、精度、并行度、CPU/数据加载配置。**到此再审阅实测资源预算后决定完整批次，当前不启动。**

## 6. 消融、敏感性与失败矩阵

消融应沿一条因果链展开：

- A：present-class/global-macro；统计视图；min_count；logit norm；连续权重是否真的影响singleton；align/proto各去一项及都去除。
- B：local-post/global-pre；局部均值是否再次单位化；参与者归一化/固定Nc的HT；β=0/1；同样额外同步但不用原型的控制。
- C：均匀Bernoulli/计数代理/历史范数；有无HT；去掉概率下限。最后一项主要展示风险，不能用偏置版本取得高分后称目标无偏。
- BTR：优先复用raw/full/teacher-init/retention-off等已有分支，机制诊断重放相同数据，避免再次无目的扫全部参数。

敏感性先单因素小网格，禁止一次全笛卡尔积：

| 因素 | 候选 | 控制与读法 |
|---|---|---|
| 全局IF | 1、10、100 | 共用类别顺序；补等总量控制 |
| 异质性 | IID、LDA .5/.1 | 记录支持数、客户端样本量、实际分布距离，不只报告α |
| 参与率 | .05/.1/.2；1.0作诊断上限 | 同累计参与预算也给一张图，区分更多通信的收益 |
| 本地epoch | 1、5 | 同轮数及同累计训练样本预算两种解释 |
| 原型温度 | .1/.5/1 | 联动记录类数与有效类数、梯度范数；不按loss绝对值选 |
| 辅助权重 | 0、默认一半、默认、默认两倍 | 固定CE目标，分别检验align与proto |
| 计数阈值/收缩 | min1/2；s=0/5/20 | 验证尾类资格与noise权衡 |
| B历史 | β=0/1；可预测的历史诊断 | 观察实际残差方差，不假定缓存越长越好 |
| C概率下限 | 平均q的 .2/.5；均匀基线 | 控制期望总参与数，报告实际累计量 |

明确失败场景：单端单样本、同类跨域多模态、少量错误标签、连续多轮无该类贡献者、快速表示漂移、尾类本身不可分、全局均衡且支持广泛、C=100/200的小τ几何限制。噪声/域实验是扩展，不与初始主表混在一起。

## 7. 评价与统计协议

主终点预注册为**固定200轮的balanced accuracy/macro recall**，次终点为validation-best checkpoint的测试表现；两种选模规则各自成表。验证时选中模型必须保存并复评，不能回看测试峰值。末20轮均值作为稳定性描述，不把20轮当20个独立重复。

记录：

- confusion matrix、overall accuracy、macro recall、macro-F1、逐类precision/recall/F1；
- 训练计数定义的head/middle/tail，CIFAR10为4/3/3分组；最差类及其身份；
- 自然分布目标下的加权准确率另列，不以macro改进掩盖头类下降；
- perclass margin、NCM与linear-head差异、ECE/NLL（校准诊断，不等同原型置信度有效性）；
- 客户端测试macro、10%分位数与样本加权均值；需要独立客户端测试划分。现有全局test按类重加权只能称代理；
- 每类上传质量、可用原型数、当前更新覆盖、贡献端数、零贡献连续时长、age、局部原始均值模长；
- 真实训练/校准/teacher前向样本数、反向样本数、客户端/服务端字节数、峰值内存、同步次数及墙钟时间。

种子层次独立记录：data split、class order、initialization/training、client sampling、augmentation/DataLoader。先固定split做paired training-seed比较，再在至少两种split/class-order上验证外推。3个seed只作最低限度筛查，正式关键结果扩到5个；不要用每个测试图像当算法重复样本。

报告每个种子差值、均值与标准差，配对置信区间及完整点图。若跨split，按split分层/聚类，避免把同一划分的重复误当独立分布样本。种子少时承认区间不稳定；不凭单次p值宣称“显著优于”。

最差类是易波动的极值；同时报告各类及其计数。不允许自动把最差类称tail类。

## 8. 文件级实施顺序

以下为**拟改动**，不表示当前已经存在这些功能。每次修改独立版本与配置，原配置保留。实现分支使用codex/前缀。

| 次序 | 文件/函数 | 具体改动与验收 |
|---|---|---|
| I0 | [main.py](/Users/ygding/Project/FedClsimb/src/main.py:170)、[experiment_logging.py](/Users/ygding/Project/FedClsimb/src/experiment_logging.py)、[longtail_report.py](/Users/ygding/Project/FedClsimb/src/longtail_report.py) | 所有全局算法统一保存resolved config、commit/dirty diff、数据哈希、seed层次、环境、最终和验证选模规则；记录原型与模型版本；不依赖W&B在线可用性 |
| I0 | [BaseServer._client_sampling](/Users/ygding/Project/FedClsimb/src/algorithms/BaseServer.py:110) | 显式sampling_seed与局部RandomState；legacy模式复现原round_idx计划；保存客户端列表。默认语义变化需配置隔离 |
| I0 | [measures.py](/Users/ygding/Project/FedClsimb/src/algorithms/measures.py)、[longtail_metrics.py](/Users/ygding/Project/FedClsimb/src/longtail_metrics.py) | 统一confusion matrix与macro-F1，未定义项记null而不是0；全局/客户端推理对象写入schema |
| I0 | [longtail_datasetter.py](/Users/ygding/Project/FedClsimb/src/train_tools/preprocessing/longtail_datasetter.py:84) | 复用现有calibration/train_eval；给legacy数据增加同样的确定性统计视图；独立generator，不在训练loader上做额外诊断 |
| I1 | [ClientTrainer._calibrated_confidence](/Users/ygding/Project/FedClsimb/src/algorithms/fedbpc/ClientTrainer.py:200) | 新配置显式none/centered/legacy范数模式；保留原行为分支用于复现，增加平移不变性检验 |
| I1 | [ClientTrainer._build_prototype_payload](/Users/ygding/Project/FedClsimb/src/algorithms/fedbpc/ClientTrainer.py:137) | 上传count、原始单位特征和/均值模长、reject_reason、encoder_version；允许min1；不把n1当高可靠 |
| I1 | [criterion.py](/Users/ygding/Project/FedClsimb/src/algorithms/fedbpc/criterion.py:85) | 显式present_class/global_macro/natural三模式；global权重先乘后平均，测试batch=1避免权重被内部归一化抵消 |
| I1 | [aggregate_prototypes](/Users/ygding/Project/FedClsimb/src/algorithms/fedbpc/utils.py:8) | 显式singleton更新步长、保留旧策略；零模长不伪造成有效方向；拆分“无贡献”“被拒绝”“过期” |
| I1 | [Server._evaluate_server_prototype_alignment](/Users/ygding/Project/FedClsimb/src/algorithms/fedbpc/Server.py:591) | 诊断改为独立验证集或标记为test-only观察；按类分层采样，替换仅前500个测试样本；独立计时 |
| I2 | 拟新增src/algorithms/fedbpc_moment/{Server,ClientTrainer,criterion,utils}.py | 按AGENT.md注册新算法；使用自定义两阶段round loop；实现B0后再B1。先只支持当前无BN的CNN，不声称任意backbone通用 |
| I2 | [fedbtr/diagnostics.py](/Users/ygding/Project/FedClsimb/src/algorithms/fedbtr/diagnostics.py:22) | 复用RandomSnapshot、BatchTrace与cohort思想；开启已有transfer开关。抽公共模块前先确认不会改变BTR结果 |
| I3 | 拟新增src/algorithms/fed_class_sampling/ | C仅在D5通过后实现；概率求解、实际包含概率、空轮、HT聚合和预算计数是独立模块；不复用随机分母FedAvg |
| I3 | [models/resnet.py](/Users/ygding/Project/FedClsimb/src/train_tools/models/resnet.py)、[train_tools/utils.py](/Users/ygding/Project/FedClsimb/src/train_tools/utils.py) | 扩模型前加统一feature接口并核对BN行为；新骨干只在独立泛化实验中使用，不在主消融中悄悄替换 |

必要的研究性质测试：公共logit平移、batch=1目标权重、n=1不消失、singleton步长可观察、全参与B0等于集中式矩、穷举HT/CV均值、零向量/无贡献/过期边界、checkpoint与loader RNG重放一致、诊断开关不改变训练批次、macro-F1由手工混淆矩阵验证。测试这些科学不变量，不写只重复实现的测试。

## 9. 继续工作时的第一个动作

先读[RESUME.md](/Users/ygding/Project/FedClsimb/research/FedBPC_audit_2026-10-21/RESUME.md)确认状态，再查看是否已提供训练机/环境和检查点。没有额外信息时，可以在本地完成I0/I1的设计与小型实现审查；**当前用户请求的交付阶段到研究报告和计划为止，不自动进入大规模算法改写。**

下一次最有信息价值的工作是D1—D4的固定模型诊断，而不是再次搜索几十篇泛相关论文或启动全基线网格。
