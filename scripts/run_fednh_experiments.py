#!/usr/bin/env python3
"""44 serial FedNH experiments with equal-client updates and a normalized prototype head."""
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
OUTPUT_ROOT = REPO_ROOT / "fednh_experiments"
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
HEAD_INIT = "orthogonal"
SMOOTHING = 0.9
CLIENT_ADV_PROTOTYPE_AGG = False
SERVER_ADV_PROTOTYPE_AGG = False
FIX_SCALING = False
SCALING_INIT = None
FIXED_SCALING = 30.0
MAX_GRAD_NORM = 10.0
SERVER_LR = 1.0
SERVER_LR_DECAY = 1.0
CLIENT_LR_SCHEDULER = "project"
CLIENT_LR_DECAY = 0.99
EXCLUDE = []
DOWNLOAD_IF_MISSING = True
INCLUDE_SUPPLEMENTARY = True
SUPPLEMENTARY_DATASET = "cifar100"
SUPPLEMENTARY_ALPHA = 0.1
PARTICIPATION_RATES = [0.05, 0.1, 0.2]
LOCAL_EPOCH_VALUES = [1, 5, 10]
WANDB_PROJECT = "Point1"
WANDB_ENTITY = None
BATCH_NAME = "fednh-300rounds"
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
    resolve = runpy.run_path(str(REPO_ROOT / "src/algorithms/fednh/config.py"))["resolve_config"]
    params = resolve({
        "metrics_only": True, "head_init": HEAD_INIT, "smoothing": SMOOTHING,
        "client_adv_prototype_agg": CLIENT_ADV_PROTOTYPE_AGG,
        "server_adv_prototype_agg": SERVER_ADV_PROTOTYPE_AGG,
        "fix_scaling": FIX_SCALING, "scaling_init": SCALING_INIT,
        "fixed_scaling": FIXED_SCALING, "max_grad_norm": MAX_GRAD_NORM,
        "server_lr": SERVER_LR, "server_lr_decay": SERVER_LR_DECAY,
        "client_lr_scheduler": CLIENT_LR_SCHEDULER, "client_lr_decay": CLIENT_LR_DECAY,
        "exclude": EXCLUDE,
    })
    if HEAD_INIT != "orthogonal":
        raise ValueError("The batch image backbones require orthogonal initialization; uniform needs 2D features")
    settings = dict(globals())
    scale = "fixed{:g}".format(FIXED_SCALING) if FIX_SCALING else "learned"
    settings.update(
        ALGO_NAME="fednh", ALGO_PARAMS=params,
        RUN_SUFFIX="_{}_s{:g}_adv{}_scale{}".format(HEAD_INIT, SMOOTHING, int(SERVER_ADV_PROTOTYPE_AGG), scale),
        PROTOCOL_EXTRAS={
            "representation": "normalized_backbone_features_no_projection_head",
            "classifier": "fednh_normalized_prototype_head_with_scaling",
            "model_aggregation": "equal_participant_model_deltas",
            "prototype_storage": "memory_only",
            "prototype_extraction": "eval_no_grad_on_augmented_training_loader_after_local_training",
            "prototype_communication": "full_model_includes_head_plus_separate_dense_payload_upload_v1",
            "fednh_metrics_version": 1,
        },
    )
    tasks = queue.build_tasks(data_root, settings=settings)
    for task in tasks:
        task["config"]["wandb_setups"]["tags"].extend([
            "equal-client-aggregation", "nh-" + HEAD_INIT,
            "smoothing-{:g}".format(SMOOTHING), "scale-" + scale,
            "advanced-{}".format(int(SERVER_ADV_PROTOTYPE_AGG))])
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
    print("{} FedNH tasks; {} main, {} additional; {} rounds; train seeds {}; smoothing={}, fixed_scaling={}, advanced={}".format(
        len(tasks), sum("main" in t["categories"] for t in tasks),
        sum("main" not in t["categories"] for t in tasks), N_ROUNDS, TRAIN_SEEDS,
        SMOOTHING, FIX_SCALING, SERVER_ADV_PROTOTYPE_AGG))
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
