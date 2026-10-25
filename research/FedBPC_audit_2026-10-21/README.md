# FedBPC 研究交付索引

本轮完成了核心代码与历史证据审计、相关文献核验、轻量数值验证和后续研究设计。原算法及原始结果未修改，未运行新训练。

建议按以下顺序阅读：

1. [科研报告](/Users/ygding/Project/FedClsimb/research/FedBPC_audit_2026-10-21/REPORT.md)：主要判断、8个假设、3套方案与推荐。
2. [实验与代码实施计划](/Users/ygding/Project/FedClsimb/research/FedBPC_audit_2026-10-21/EXPERIMENT_PLAN.md)：先证伪再扩大实验，包含预算和停止条件。
3. [文献核验表](/Users/ygding/Project/FedClsimb/research/FedBPC_audit_2026-10-21/LITERATURE.md)：原文/官方实现入口、研究近邻和创新边界。
4. [历史复算表](/Users/ygding/Project/FedClsimb/research/FedBPC_audit_2026-10-21/LOG_TABLES.md)：可读结果；完整机器数据见[audit_results.json](/Users/ygding/Project/FedClsimb/research/FedBPC_audit_2026-10-21/audit_results.json)。
5. [恢复记录](/Users/ygding/Project/FedClsimb/research/FedBPC_audit_2026-10-21/RESUME.md)：保留证据、判断、限制与下一步，便于稍后继续。

[复算脚本](/Users/ygding/Project/FedClsimb/research/FedBPC_audit_2026-10-21/reproduce_audit.py)只依赖Python标准库。[源文件清单](/Users/ygding/Project/FedClsimb/research/FedBPC_audit_2026-10-21/source_inventory.json)记录253个相关文件哈希；这是输入快照，不代表所有文件均逐行验证。

当前建议：先验证目标与筛选修复A；只有精确原型确有用，再研究类别矩估计B。主动采样C成本更高且有直接近邻，暂列备选。

