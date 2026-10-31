# SwanLab：迁移已有结果与继续批量训练

在已登录 SwanLab 的服务器、原训练环境中执行。以下使用此前日志中的项目路径
`/root/fedbcp`、数据路径 `/root/data` 和输出目录 `/root/fedbcp/fedhpro_experiments`。
若实际目录不同，请替换路径。先把本次修改后的代码同步到服务器，保留原实验目录。

## 1. 安装 SDK

```bash
cd /root/fedbcp
python -m pip install -r requirements-tracking.txt
python -c 'import swanlab; print(swanlab.__version__)'
```

锁定已验证的 `swanlab==0.10.3`；在原 PyTorch 环境内安装即可。
`src/requirements.txt` 是历史 Linux conda 导出格式，不是 pip requirements；
此次增量安装请使用根目录的 `requirements-tracking.txt`。
使用服务器已有 SwanLab 登录凭据，不需要 W&B key，不需要在脚本中粘贴任何 key。

默认目标是当前 SwanLab 账号的个人空间，项目 `Point1`。
组织空间可在迁移命令中加 `--workspace 组织用户名`；训练使用
`SWANLAB_WORKSPACE=组织用户名`。W&B entity 不会自动当作 SwanLab workspace。

## 2. 先预览历史数据

```bash
python scripts/migrate_results_to_swanlab.py \
  --source /root/fedbcp/fedhpro_experiments \
  --project Point1 \
  --dry-run
```

`--source` 是包含配置和指标的实验输出目录，不是只有终端日志的 `result/`。
也可以指定某个任务目录或 `attempt-*` 目录。预览只读文件，不联网、不创建实验、
不修改原始结果，也不需要安装 SwanLab/PyTorch。

支持的格式：

| 保存格式 | 迁移行为 |
| --- | --- |
| `config.json + metrics.csv + summary.json` | 原始 round 为横轴，逐列上传数值，保留指标名和数值单位 |
| `config.json + metrics.jsonl + summary.json` | FedSA 优先使用逐轮 JSONL；没有评估的轮次保留为空，不补 0 |
| `experiment.json + rounds.jsonl + final.json` | FedBTR 按项目原有规则展开嵌套指标和逐类数组 |
| `final.json` 含 config，或 config + 成功 summary | 仅上传现有最终结果，不编造历史曲线 |

默认只选成功结果；同一任务有多个成功 attempt 时只选最新的一个。
`--include-incomplete` 可额外导入已有轮次的失败/中断 attempt，标签和
`config.migration.source_status` 会保留原训练状态。
W&B 初始化就失败、没有任何轮次的 attempt 会跳过，不创建空实验。
JSONL 比派生 CSV 优先，CSV 空格单元格表示缺测。

预览会拒绝损坏文件、截断行、重复/倒退轮次、NaN/Inf，以及成功 summary 与历史
轮数不符的记录。在这些问题解决前不会开始云端上传。
如果只有 `.wandb` 二进制缓存或 `train.log`，本脚本不会尝试猜测完整曲线；
没有自定义本地记录的旧 FedProto/TinyProto 等运行不能凭空恢复历史。

## 3. 先迁移一个实验，再迁移其余实验

```bash
python scripts/migrate_results_to_swanlab.py \
  --source /root/fedbcp/fedhpro_experiments \
  --project Point1 \
  --state-dir /root/fedbcp/.swanlab-migration \
  --limit 1
```

打开打印的实验链接，检查名称、300 轮曲线、最终结果。然后去掉 `--limit 1`：

```bash
python scripts/migrate_results_to_swanlab.py \
  --source /root/fedbcp/fedhpro_experiments \
  --project Point1 \
  --state-dir /root/fedbcp/.swanlab-migration
```

同一来源可重复运行：成功导入且经过云端校验的实验会跳过；中断重试使用确定的
SwanLab 实验 ID 和 `resume="allow"`。状态文件只存在 `--state-dir`，不改写原
`config/metrics/summary/status`、manifest 或 aggregate.csv。
来源文件改变时会作为新版本生成新 ID；请导入已经停止写入的结果目录。
脚本会检查上传前后文件哈希，防止把正在训练中的半份结果当成完整结果。

上传结束会通过 SwanLab API 回读**所有迁移的数值指标完整序列**，比较轮次和数值，
不是抽样曲线。暂未完成入库时会短暂重试；仍失败则保留状态并返回非零退出码，
下次执行同一命令重试。`verified` 代表数值序列核对通过，原始附件由 SDK 上传。
登录用户、目标项目、工作空间或 API host 改变时不会错误沿用旧目标的成功状态。

同一批迁移完成后，SwanLab 显示的运行结束表示“导入完成”；原始训练成功/失败状态
保存在 `source-success/source-failed/source-interrupted` 标签及 migration 配置中。
原始绝对时间不伪造；图表 step/round 和 `time/*` 数值保留，云端创建时间是迁移时间。

多个算法可重复指定 `--source`：

```bash
python scripts/migrate_results_to_swanlab.py \
  --source /root/fedbcp/fedhpro_experiments \
  --source /root/fedbcp/fedavg_experiments \
  --source /root/fedbcp/moon_experiments \
  --project Point1
```

只列出实际存在的目录。对整项目目录递归扫描可能包含你不希望导入的历史实验，
建议明确指定各算法输出目录。

`--mode offline` 可先生成 SwanLab 离线缓存，状态为 `offline_saved`，不会标为云端已验证。
正常迁移请直接用默认 online；离线与在线是两个独立的迁移目标。

## 4. 继续批量训练

先保留原数据路径、输出目录和训练超参数，预览队列：

```bash
python scripts/run_fedhpro_experiments.py \
  --data-root /root/data \
  --output-root /root/fedbcp/fedhpro_experiments \
  --dry-run
```

然后启动训练，另写新日志以保留之前的报错日志：

```bash
mkdir -p /root/fedbcp/result
SWANLAB_MODE=online nohup python -u scripts/run_fedhpro_experiments.py \
  --data-root /root/data \
  --output-root /root/fedbcp/fedhpro_experiments \
  > /root/fedbcp/result/fedhpro_swanlab_batch.log 2>&1 &
```

八个批量入口都使用 SwanLab。平台配置不参与训练任务 ID；在同样数据根路径和训练
设置下，原成功任务继续跳过。此前 35 个成功任务应 SKIP，9 个失败任务重跑。
迁移是上传历史记录，不会改变训练队列状态。不要加 `--rerun-successful`。
失败任务仍从第 1 轮开始；此次没有新增模型 checkpoint 或断点续训。

环境变量可控制所有训练入口：

```bash
export SWANLAB_PROJ_NAME=Point1
# 仅组织空间需要设置：export SWANLAB_WORKSPACE=组织用户名
export SWANLAB_MODE=online
```

遇到临时断网，可以将 `SWANLAB_MODE` 设置成 `offline`，不会再被调度器写死覆盖。
每个新 attempt 的缓存保存在 `attempt-*/swanlog/`。
`run_fedbpc_complete.py` 单实验入口保持原来的默认 offline；同样可用环境变量切换。

## 汇总与兼容细节

- 新配置使用 `swanlab_setups`、`experiment_name`、`workspace`、`logdir`；
  新结果使用 `swanlab_run_id` 和 `swanlab_url`。
- SwanLab 没有原来可直接写入的 summary 字典。适配层将完整汇总保存到
  `config.result_summary`，数值字段同时写为 `summary/*` 指标，使用最后一个真实 step，
  不添加一轮训练。只有最终汇总的历史数据使用 summary 的 step 0，没有 round 曲线。
- 迁移保留名称、group、job_type、tags、训练配置、原指标列名和精度；
  `clients.jsonl`、划分统计、配置、summary、原 CSV/JSONL 等作为原始文件附件上传。
  不上传数据集、模型权重或 checkpoint。
- 原 W&B 字段只保留在旧结果解析和历史来源信息中。训练不导入、不初始化、不调用 W&B。
- 验证使用 SwanLab 0.10.3 的真实离线 SDK、现有 CPU/模拟算法测试、迁移格式和重试测试；
  本机未使用服务器凭据做真实云端上传，也未重跑你的 GPU 实验。

SDK 参考：[初始化](https://docs.swanlab.cn/api/py-init.html)、
[指标横轴](https://docs.swanlab.cn/api/py-define_metric.html)、
[断点恢复实验](https://docs.swanlab.cn/guide_cloud/experiment_track/resume-experiment.html)。
