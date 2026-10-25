"""Rebuild transparent tables from every recorded short synthetic study."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STUDIES = [
    ("20261022_161208_e1d2c9e0", "初版：真实样本梯度与分类头一起降权"),
    ("20261022_161502_200f7749", "修订版：保留骨干真实 CE 梯度，同数据配对复查"),
    ("20261022_161635_9d4bd12b", "修订版：更换数据、划分与训练种子"),
    ("20261022_162021_dc8d2fe2", "固定梯度修正分支：初始设置，已拒绝"),
    ("20261022_162122_dc64df6b", "固定梯度修正分支：第三组独立设置，已拒绝"),
]


def main():
    lines = [
        "# 合成诊断完整结果",
        "",
        "所有数值来自实际 CPU PyTorch 运行；四个全局均衡类别、八客户端、每轮两客户端、20 轮、E=3。"
        "每类训练 128、测试 256 个 8 维高斯输入，模型为小型 MLP。它们不是 CIFAR 结果。",
        "",
        "每组三个训练 seed，标准差为样本标准差。每组方法共享训练/测试数据、划分、初始化、参与序列和训练 batch 顺序。"
        "同一组的三个 seed 只改变初始化与训练顺序，不能视为三个独立数据划分。测试集从未用于梯度或原型生成。",
        "",
        "按实验发生顺序保留全部结果；第二组设计根据第一组观察修订，因此属于探索性对照。"
        "第三组在修订版确定后使用新数据/新划分；第五组再使用第三套新数据/划分，只有部分方法。"
        "重复基线不是新增独立证据。所有指标均按百分比显示，末 5 轮为每 seed 的五轮平均后再跨 seed 平均。",
        "",
        "特别注意：LDA 个别 seed 出现严重退化，平均值差距很大；不应将其当作现实任务的提升幅度。"
        "需要同时查看每 seed、逐轮曲线、末段均值，以及原 FedBPC 对照，不能只看最后一轮。",
    ]
    total_runs, seconds = 0, 0.
    for name, title in STUDIES:
        directory = ROOT / "synthetic" / name
        protocol = json.loads((directory / "protocol.json").read_text())
        rows = json.loads((directory / "aggregate.json").read_text())
        lines.extend(["", "## " + title, "",
            "原始目录：[{}](synthetic/{}/aggregate.json)。训练 seeds={}；数据 seeds={}；partition_seed={}。".format(
                name, name, protocol["seeds"], protocol["data_seeds"], protocol["partition_seed"]),
            "", "| 划分 | 方法 | 最终 Top-1 ± std | 末5轮 Top-1 | 最终 Macro-F1 | 最差类 Recall |",
            "|---|---|---:|---:|---:|---:|"])
        for row in rows:
            lines.append("| {} | {} | {:.2f} ± {:.2f} | {:.2f} | {:.2f} | {:.2f} |".format(
                row["partition"], row["method"], 100*row["global/top1_mean"],
                100*row["global/top1_std"], 100*row["last5_top1_mean"],
                100*row["global/macro_f1_mean"], 100*row["global/worst20_recall_mean"]))
        lines.extend(["", "此处只有四类，worst20_recall 取 ceil(4×.2)=1 类，等于最差类 Recall。"])
        for path in directory.glob("*.json"):
            if path.name in ("aggregate.json", "protocol.json"):
                continue
            result = json.loads(path.read_text())
            total_runs += 1
            seconds += result["elapsed_seconds"]
        if name == "20261022_161635_9d4bd12b":
            lines.extend(["", "逐 seed 的 LDA 主对照：", "",
                          "| seed | FedAvg | 原 FedBPC | 完整候选 |",
                          "|---|---:|---:|---:|"])
            for seed in protocol["seeds"]:
                values = [json.loads((directory / ("lda_{}_{}.json".format(seed, method))).read_text())
                          ["final"]["global/top1"]*100
                          for method in ("fedavg", "original_fedbpc", "complete")]
                lines.append("| {} | {:.2f} | {:.2f} | {:.2f} |".format(seed, *values))
    lines.extend(["", "## 决策", "",
        "保留分布原型补全作为可运行候选。方差版本在已记录 LDA 中较均值版更稳，但 sharding 没有一致优势；"
        "方向对齐与骨干路由的消融结果混合。因此目前不推荐全面替换原 FedBPC，也不把全部模块称为必要。",
        "",
        "固定梯度修正分支在 LDA 明显失败，已从正式代码删除。其快照为 "
        "[rejected_control_variate_snapshot.tar.gz](rejected_control_variate_snapshot.tar.gz)，"
        "仅供审计；如需复现，在独立仓库副本中解压，不要覆盖当前实现。快照依赖本仓库公共模块与测试环境。",
        "",
        "共 {} 次短程运行，其中包含重复配对基线；记录的训练及逐轮评估时间之和约 {:.1f} 秒，"
        "不含各进程导入与部分构造开销。没有进行真实图像数据训练。".format(total_runs, seconds),
        "",
        "由 summarize_diagnostics.py 从原始 JSON 重建。"])
    (ROOT / "RESULTS.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
