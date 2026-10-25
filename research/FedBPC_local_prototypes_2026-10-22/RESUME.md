# 研究续接记录：2026-10-22

此文件是当前唯一优先续接入口。旧 2026-10-21 审计报告保留为历史材料，其长尾/论文创新优先级已经被用户后续要求覆盖。

## 用户当前要求

1. FedBPC 不以解决全局长尾为目标，只关注 sharding / LDA 的本地类别不平衡与缺类。
2. 沿原型路线构建实用方案；不发表论文，无需强创新性。
3. 可以完全放下旧 FedBPC 结构，独立从优化问题出发设计。
4. 机制需要有必要性和性能价值，不能为了复杂而堆叠。
5. 保存关键结论、证据、代码状态和下一步，供约五小时后用户恢复。没有安排自动唤醒。
6. 原附件限制仍适用：先做轻量验证；昂贵完整训练应先给预算，不自动展开。

## 当前交付状态

已完成独立候选 FedBPC-Complete 的实现与初步验证；原 FedBPC 未改动。核心文件：

- src/algorithms/fedbpc_complete/{Server.py,ClientTrainer.py,criterion.py,utils.py,config.py,__init__.py,README.md}
- src/config/fedbpc_complete_sharding.json 与 fedbpc_complete_lda.json
- scripts/run_fedbpc_complete.py：单次训练/消融入口，独立结果目录、离线日志、不下载。
- scripts/diagnose_fedbpc_complete.py：配对合成诊断入口。
- tests/test_fedbpc_complete.py：21 项新测试。

公共路径只增加算法导出、注册，以及 fedavg_batch.py 中仅针对新算法的非空输出目录保护；.gitignore 增加实验输出目录。未提交 Git，工作区修改需要保留。

完整设计见 DESIGN.md；结果见 RESULTS.md；逐轮原始数据和协议在 synthetic/。不要重新启动一轮泛泛文献检索。

## 当前保留的算法决策

同一轮全局编码器，在参与客户端训练图像的确定性视图上提取 raw feature sum / squared sum / count。float64 合并，得到 raw mean 和对角样本方差；N=1 保留均值但方差不可估计，记录为零。不按置信度筛掉困难样本。不混合不同当前编码器版本，不做跨版本 EMA。

π 为本地训练类别先验，p 为全局训练类别先验。d=(p-π)+，e=(π-p)+，a=αrd，A=Σa，D=Σe。真实分类头权重 b=1-(A/D)e/π；D=0 时 b=1,a=0。保证 Σπb+Σa=1。r=valid/(1+age)，过期 5 轮失效。这是类别质量校正，不是完整真实风险等价性。

原型分类风险采用 Gaussian Jensen CE 上界：竞争 logit 加 .5γ Σd vcd(wjd-wcd)^2。由 ISDA 原理得到，公式不是原创；只在假设的高斯分布下是风险上界。分布原型 detach，只训练头。

真实 CE 的头梯度按 b 加权，骨干保留自然 CE 梯度。代码用 naturalCE+(b-1)*CE(head(stopgrad(features))) 实现；前向值等于加权 CE，但梯度路由不同。λ=.1 的类均衡方向约束是可消融假设。

默认 α=.5、γ=1、λ=.1、warmup=1、max_age=5、preserve_backbone_gradients=True、统计 worker=0。默认未基于真实数据调参。保留 mean_only、no_alignment、alignment_only、coupled_backbone 消融。普通 sample-weighted FedAvg 聚合、普通全局线性头推理。

## 已验证什么

- 最终全仓库 unittest：128 项通过（其中 21 项新测试）；日志 tests_final.log。
- 质量守恒、缺原型回退、batch=1 权重不被归一化消除。
- 缺类正类梯度流向头，原型/骨干不接收回放梯度。
- 梯度路由：骨干自然 CE、头加权 CE 的导数逐项相等。
- 同编码器矩聚合与集中统计一致，保留客户端间均值差异及单例/零均值。
- 方差风险的解析结果、梯度与高斯确定性积分检验。
- 缓存年龄、版本拒绝、训练 RNG 隔离、两轮训练/CSV/检查点、动量不串客户端。
- 辅助项关闭时客户端更新与 FedAvg 对齐。
- CIFAR 默认模型 feature 接口与统计提取通过；不是 CIFAR 性能训练。
- launcher dry-run 可用，已有结果目录不会被新算法覆盖。

原公共 scheduler 在服务器模板 optimizer 没有 step 时会发 PyTorch 警告，客户端实际进行了 step；新集成测试确认两轮 LR=.03×.9²。未修改公共调度规则。

## 实验事实与不能夸大的部分

全部合成数据全局均衡；四类、八客户端、每轮两客户端、E=3、20 轮、小型 MLP。每组方法严格配对，三个训练 seed 共享同一个数据/划分。共 156 次短程运行（含重复基线），记录训练与评估总计约 174.6 秒，非大规模训练。

1. synthetic/20261022_161208_e1d2c9e0：初版耦合真实骨干梯度；6 方法×2 划分×3 seeds。
2. synthetic/20261022_161502_200f7749：当前路由版，同设置；7 方法×2×3。
3. synthetic/20261022_161635_9d4bd12b：换数据、划分、初始化；7 方法×2×3。
4. synthetic/20261022_162021_dc8d2fe2：固定原型头梯度修正探索；3 方法×2×3。
5. synthetic/20261022_162122_dc64df6b：该探索在第三组新数据/划分复查；3 方法×2×3。

当前完整方法在三套不同 LDA 设置的最终 Top-1 均值为 94.82、92.81、93.07%；FedAvg 为 92.45、62.57、84.28%。但部分 seed 有退化、分片设置无一致优势，第二套原 FedBPC 的最终均值 94.73% 高于新方法 92.81%。对齐和路由的必要性证据混合；不得宣称已全面超过原 FedBPC。

不能只报告 LDA 最终一轮：第一套完整方法末5轮均值 88.17%，原 FedBPC 为 88.52%；第二套 86.80% 对原 FedBPC 85.12%。所有正负结果已写入 RESULTS.md。

固定梯度修正（control_variate）在 LDA 严重失败，已从正式模块、入口和测试删除。快照 rejected_control_variate_snapshot.tar.gz 保留历史源码与实验入口，供独立副本审计；**不要恢复到正式候选，更不要无记录反复调参把失败抹掉**。其负系数代理风险/重复固定修正可能不稳定，是解释假设而非已隔离证明的原因。

## 环境、边界

- repo：/Users/ygding/Project/FedClsimb；起始提交 3472e12。
- Python：/Users/ygding/opt/anaconda3/envs/clsimb_py39/bin/python。
- torch2.2.2、torchvision0.17.2、numpy1.26.4，CPU；无 CUDA/MPS。
- 系统 Python 没有 torch；不能据此误判整台机器没有 PyTorch。
- 仍未定位到可用 CIFAR 数据；没有下载数据或启动 GPU/真实图像完整训练。
- 未使用子 agent，未设置自动化，未发布/提交远端。
- final_state.pt 只供检查，不支持精确训练断点续跑。研究续接依赖本文件和保存工件，不依赖隐含聊天状态。

## 下一步（不重新做已通过的工作）

1. 用户提供训练机/数据位置后，复用共享划分缓存，先 profile 三轮，估计资源。
2. 最小真实筛查：FedAvg、原 FedBPC、完整候选、均值消融 × sharding2/LDA.1 × seed2022，30 轮；先说明预算再执行。每 run 期望约75万次 SGD 样本访问，候选另15万统计前向。
3. 原 FedBPC 仍用原日志入口；比较时统一全局评估，不把不同本地诊断次数当公平训练时间。不要直接用 run_tracked 包旧服务器，它未提供新式 batch_summary。
4. 同时看末段均值、最差类、各类 Recall、运行成本与头/骨干的局部遗忘，不能凭一轮 Top-1 判断。
5. 若真实筛查失败，先测原型代理风险/梯度和真实类条件风险/梯度的差异；不默认继续堆新模块。若某机制无稳定收益，关闭或删除。
6. 全训练为下一阶段：200 轮、多 seed、多 partition seed；当前未执行，不能假装完成真实性能验证。

可用恢复指令：“读取 research/FedBPC_local_prototypes_2026-10-22/RESUME.md、DESIGN.md、RESULTS.md，继续分片/LDA局部类别不平衡的原型研究；保留全部正负实验，不恢复已拒绝的固定梯度分支。”
