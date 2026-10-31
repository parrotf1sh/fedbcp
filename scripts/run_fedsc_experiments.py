#!/usr/bin/env python3
"""44 serial FedSC experiments preserving the provided source behavior."""
import argparse
import runpy
from pathlib import Path
import sys

# Supports both direct execution and import by the dependency-free tests.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_fedavg_experiments as queue


# ======================== Experiment settings ========================
REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
DATA_ROOT = REPO_ROOT / "data"
OUTPUT_ROOT = REPO_ROOT / "fedsc_experiments"
DATASETS = ["cifar10", "cifar100", "tiny-imagenet"]
LDA_ALPHAS = [0.03, 0.1, 0.3]
SHARDS_PER_CLIENT = [2, 4, 6]
TRAIN_SEEDS = [2022, 2023]
N_CLIENTS = 100
LOCAL_EPOCHS = 5
N_ROUNDS = 300
SAMPLE_RATIO = 0.1
BATCH_SIZE = 50
DEVICE = "cuda:0"
OPTIMIZER = {"lr": 0.01, "momentum": 0.9, "weight_decay": 1e-5}
SCHEDULER = {"enabled": True, "name": "step",
             "params": {"gamma": 0.99, "step_size": 1}}
MODELS = {"cifar10": "fedavg_cifar", "cifar100": "fedavg_cifar",
          "tinyimagenet": "fedavg_tiny"}
CL_TAU = 0.05
DOWNLOAD_IF_MISSING = True
INCLUDE_SUPPLEMENTARY = True
SUPPLEMENTARY_DATASET = "cifar100"
SUPPLEMENTARY_ALPHA = 0.1
PARTICIPATION_RATES = [0.05, 0.1, 0.2]
LOCAL_EPOCH_VALUES = [1, 5, 10]
SWANLAB_PROJECT = "Point1"
SWANLAB_WORKSPACE = None
BATCH_NAME = "fedsc-source-300rounds"
SKIP_SUCCESSFUL = True
# Keep the same fixed partition/sampling protocol as FedAvg.
PARTITION_SEED = 19940817
CLIENT_SAMPLING_RULE = "numpy.seed(round_index), zero_based"
PARTITION_CACHE_ENABLED = True
PARTITION_CACHE_DIR = None
LDA_MIN_SAMPLES = 10
LDA_MAX_ATTEMPTS = 128
LDA_INSUFFICIENT_POLICY = "repair"
# =====================================================================


def build_tasks(data_root=None):
    # This config module is stdlib-only; dry-run never imports torch or CUDA.
    resolve = runpy.run_path(str(REPO_ROOT / "src/algorithms/fedsc/config.py"))["resolve_config"]
    params = resolve({"metrics_only": True, "cl_tau": CL_TAU})
    settings = dict(globals())
    settings.update(
        ALGO_NAME="fedsc", ALGO_PARAMS=params,
        RUN_SUFFIX="_source_tau{:g}".format(CL_TAU),
        PROTOCOL_EXTRAS={
            "fedsc_implementation": "provided_source_behavior_v1",
            "representation": "raw_backbone_features_no_projection_head",
            "model_aggregation": "sum_weight_times_samples_then_divide_total",
            "prototype_storage": "memory_only",
            "prototype_extraction": "last_local_epoch_training_forward_no_extra_pass",
            "prototype_communication": "tensor_payload_and_int64_labels_counts_list_lengths_v1",
            "fedsc_metrics_version": 1,
        },
    )
    tasks = queue.build_tasks(data_root, settings=settings)
    for task in tasks:
        task["config"]["swanlab_setups"]["tags"].extend([
            "source-behavior", "no-projection-head", "tau-{:g}".format(CL_TAU)])
    return tasks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Print/validate tasks only; no training, downloads or SwanLab")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--python", default=PYTHON)
    parser.add_argument("--rerun-successful", action="store_true")
    args = parser.parse_args()
    tasks = build_tasks(args.data_root)
    print("{} FedSC tasks; {} main, {} additional; {} rounds; train seeds {}; source behavior; tau={}".format(
        len(tasks), sum("main" in t["categories"] for t in tasks),
        sum("main" not in t["categories"] for t in tasks), N_ROUNDS, TRAIN_SEEDS, CL_TAU))
    output = args.output_root.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    with queue.queue_lock(output):
        queue.write_json(output / "manifest.json", tasks)
        if args.dry_run:
            for index, task in enumerate(tasks, 1):
                print("{:02d}. {} [{}]".format(index, task["name"], ", ".join(task["categories"])))
            return 0
        return queue.execute(tasks, output, args.python, SKIP_SUCCESSFUL and not args.rerun_successful)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
