# FedBPC-Complete：分布原型补全

独立候选算法，针对 sharding / LDA 的本地类别不平衡与缺类。原 fedbpc 代码不变。完整设计、数学边界和已运行实验见 [DESIGN.md](../../../research/FedBPC_local_prototypes_2026-10-22/DESIGN.md)。

当前状态：性质测试与 CPU 集成通过；合成 LDA 有收益潜力，分片无稳定收益；尚无真实 CIFAR 性能结论。不能将本 README 视为推荐全面替换原 FedBPC。

## 运行一个明确的实验

在仓库根目录，先检查配置；dry-run 不训练、不写结果、不下载数据：

    python scripts/run_fedbpc_complete.py --partition sharding --dry-run
    python scripts/run_fedbpc_complete.py --partition lda --variant mean_only --dry-run

在有 PyTorch、torchvision、W&B 和本地 CIFAR 数据的训练环境：

    python scripts/run_fedbpc_complete.py --partition sharding --data-root /absolute/path/to/data --device cuda:0 --seed 2022 --rounds 200

数据根目录应包含 cifar10 子目录，符合本仓库原始数据布局。入口不会下载缺失数据；W&B 默认离线。一次命令只跑一个实验，不会自动展开全矩阵。

每次生成独立目录，保留配置、分区信息、每轮客户端、逐轮指标、算法说明、summary 和最终模型/原型检查点。直接使用固定配置重复运行时也会拒绝覆盖非空目录。FedAvg 对照沿用仓库 metrics-only 逻辑，不保存最终模型。

final_state.pt 是检查用的最终状态，**没有实现精确中断续训**；恢复本研究任务使用研究目录的 RESUME.md。不要把研究续接与训练 RNG/loader 精确恢复混为一谈。

## 消融

| --variant | 改动 |
|---|---|
| full | 默认：质量补全、方差风险、骨干梯度保留、方向对齐 |
| mean_only | variance_scale=0 |
| no_alignment | lambda_align=0 |
| alignment_only | completion_strength=0 |
| coupled_backbone | preserve_backbone_gradients=False |
| fedavg | 同模型/划分/优化预算下的普通 FedAvg |

原 FedBPC 对照使用原配置和原实现；合成诊断入口已经包含它。真实训练使用其原日志协议，需要统一提取指标；不能假定它已经支持本新模块的 metrics-only 输出。

## 默认参数

completion_strength=.5、variance_scale=1、lambda_align=.1、warmup_rounds=1、max_prototype_age=5、preserve_backbone_gradients=True、statistics_num_workers=0。它们是初始假设，不是已调优超参数。第一轮保留自然 CE，此后按有效原型覆盖生效。

若用模块 API，自定义数据集可提供 local[k]["calibration"] 顺序 loader，必须遍历相同训练样本一次。默认从训练集复制数据视图并使用测试集的确定性 transform，不读取测试样本。模型要求普通线性 classifier 和 get_features=True；首版拒绝 BatchNorm。

## 轻量验证与合成对照

本机已确认可用解释器：

    /Users/ygding/opt/anaconda3/envs/clsimb_py39/bin/python

运行测试：

    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -B -m unittest discover -s tests -v

运行短程合成对照（全局均衡，不下载、不连接日志服务）：

    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -B scripts/diagnose_fedbpc_complete.py

可通过 --methods、--seeds、--data-seeds、--partition-seed 选择明确对照。程序核对初始化哈希、划分哈希、参与客户端和类别计数一致性，保存每轮指标。各方法本地 SGD 样本顺序配对。原 FedBPC 多做的统计与本地评估会增加运行时间，因此此诊断的耗时不能作为严格算法速度排名。

已拒绝的固定梯度修正实验不在正式入口中；历史代码和失败数据保存在研究目录。
