#!/usr/bin/env python3
"""44 serial FedRCL experiments preserving the provided source behavior."""
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
OUTPUT_ROOT = REPO_ROOT / "fedrcl_experiments"
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
RCL_TEMP = 0.05
TOPK_POS = 1
TOPK_NEG = 2
PENALTY_THRESHOLD = 0.7
CE_WEIGHT = 1.0
SCL_WEIGHT = 1.0
PENALTY_WEIGHT = 1.0
PROX_WEIGHT = 0.0
FEATURE_ALIGN_WEIGHT = 0.0
BRANCH_LEVEL = [0, 1, 2]
FEATURE_RELU = False
LOCAL_LR_DECAY = 0.998
GRAD_CLIP = 10.0
USE_AMP = True
DOWNLOAD_IF_MISSING = True
INCLUDE_SUPPLEMENTARY = True
SUPPLEMENTARY_DATASET = "cifar100"
SUPPLEMENTARY_ALPHA = 0.1
PARTICIPATION_RATES = [0.05, 0.1, 0.2]
LOCAL_EPOCH_VALUES = [1, 5, 10]
WANDB_PROJECT = "Point1"
WANDB_ENTITY = None
BATCH_NAME = "fedrcl-source-300rounds"
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
    resolve = runpy.run_path(str(REPO_ROOT / "src/algorithms/fedrcl/config.py"))["resolve_config"]
    if any(name not in ("fedavg_cifar", "fedavg_tiny") for name in MODELS.values()):
        raise ValueError("This batch protocol uses the three-hidden-layer project CNNs")
    params = resolve({
        "metrics_only": True,
        "ce_loss": {"weight": CE_WEIGHT},
        "prox_loss": {"weight": PROX_WEIGHT},
        "feature_align_loss": {"weight": FEATURE_ALIGN_WEIGHT, "align_type": "cosine"},
        "rcl_loss": {
            "temp": RCL_TEMP, "topk_pos": TOPK_POS, "topk_neg": TOPK_NEG,
            "branch_level": BRANCH_LEVEL, "feature_relu": FEATURE_RELU,
            "pairs": [
                {"name": "scl", "pos": "nn", "neg": "nn", "weight": SCL_WEIGHT,
                 "loss_type": "supervised"},
                {"name": "penalty", "pos": "nn", "neg": "nn", "weight": PENALTY_WEIGHT,
                 "loss_type": "unsupervised", "neg_sample_type": "intra_class_thresholding",
                 "threshold": PENALTY_THRESHOLD},
            ],
        },
        "local_lr_decay": LOCAL_LR_DECAY, "grad_clip": GRAD_CLIP, "use_amp": USE_AMP,
    }, num_layers=4)
    settings = dict(globals())
    settings.update(
        ALGO_NAME="fedrcl", ALGO_PARAMS=params,
        RUN_SUFFIX="_equal_tau{:g}_ld{:g}_amp{}".format(RCL_TEMP, LOCAL_LR_DECAY, int(USE_AMP)),
        PROTOCOL_EXTRAS={
            "fedrcl_implementation": "project_source_behavior_v1",
            "representation": "multilayer_backbone_features_no_projection_head",
            "model_aggregation": "equal_participant_full_state_mean",
            "feature_layer_mapping": ["conv2d_1", "conv2d_2", "linear_1", "classifier"],
            "reference_model": "frozen_parameters_train_mode_no_grad_client_memory_only",
            "local_lr_rule": "round_initial_lr * local_lr_decay ** zero_based_epoch",
            "grad_scaler_rule": "enabled_on_cuda_independent_of_autocast",
            "fedrcl_metrics_version": 1,
        },
    )
    tasks = queue.build_tasks(data_root, settings=settings)
    for task in tasks:
        task["config"]["wandb_setups"]["tags"].extend([
            "source-behavior", "equal-client-aggregation", "no-projection-head",
            "tau-{:g}".format(RCL_TEMP), "local-decay-{:g}".format(LOCAL_LR_DECAY),
            "amp-{}".format(int(USE_AMP))])
    return tasks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Print/validate tasks only; no training, downloads or W&B")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--python", default=PYTHON)
    parser.add_argument("--rerun-successful", action="store_true")
    args = parser.parse_args()
    tasks = build_tasks(args.data_root)
    print("{} FedRCL tasks; {} main, {} additional; {} rounds; train seeds {}; equal aggregation; tau={}; local_decay={}; AMP={}".format(
        len(tasks), sum("main" in t["categories"] for t in tasks),
        sum("main" not in t["categories"] for t in tasks), N_ROUNDS, TRAIN_SEEDS, RCL_TEMP, LOCAL_LR_DECAY, USE_AMP))
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
