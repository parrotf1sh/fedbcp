#!/usr/bin/env python3
"""44 serial FedHPro experiments with an explicit reduced server budget and snapshot fix B."""
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
OUTPUT_ROOT = REPO_ROOT / "fedhpro_experiments"
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
# Reduced server budget: explicit protocol adaptation, not source defaults.
META_ROUNDS = 20
META_EPOCHS = 10
ROUNDS_OF_HP = 10
LENGTH_OF_HP = 5
META_LENGTH = 50
HP_INFO_T = 0.05
LR_HP = 0.1
META_LR = 0.01
META_BATCH_SIZE = 50
BATCH_REAL = 100
REPEAT_NUM = 1
USE_MARGIN = False
MARGIN_BATCH_SIZE = 100
DIS_METRIC = "matching"
METHOD = "DSA"
DSA_STRATEGY = "color_crop_cutout_flip_scale_rotate"
DSA_SEED_MODE = "deterministic"
AUGMENTATION = {
    "aug_mode": "S", "prob_flip": 0.5, "ratio_scale": 1.2,
    "ratio_rotate": 15.0, "ratio_crop_pad": 0.125, "ratio_cutout": 0.5,
    "ratio_noise": 0.05, "brightness": 1.0, "saturation": 2.0, "contrast": 0.5,
}
DOWNLOAD_IF_MISSING = True
INCLUDE_SUPPLEMENTARY = True
SUPPLEMENTARY_DATASET = "cifar100"
SUPPLEMENTARY_ALPHA = 0.1
PARTICIPATION_RATES = [0.05, 0.1, 0.2]
LOCAL_EPOCH_VALUES = [1, 5, 10]
WANDB_PROJECT = "Point1"
WANDB_ENTITY = None
BATCH_NAME = "fedhpro-budget-300rounds"
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
    resolve = runpy.run_path(str(REPO_ROOT / "src/algorithms/fedhpro/config.py"))["resolve_config"]
    params = resolve({
        "metrics_only": True, "dsa_seed_mode": DSA_SEED_MODE,
        "length_of_hp": LENGTH_OF_HP, "rounds_of_hp": ROUNDS_OF_HP,
        "hp_info_t": HP_INFO_T, "lr_hp": LR_HP, "meta_length": META_LENGTH,
        "batch_real": BATCH_REAL, "repeat_num": REPEAT_NUM, "use_margin": USE_MARGIN,
        "dis_metric": DIS_METRIC, "meta_rounds": META_ROUNDS, "meta_epochs": META_EPOCHS,
        "meta_lr": META_LR, "meta_batch_size": META_BATCH_SIZE,
        "margin_batch_size": MARGIN_BATCH_SIZE, "method": METHOD,
        "dsa_strategy": DSA_STRATEGY, "augmentation": AUGMENTATION,
    })
    settings = dict(globals())
    budget = "mr{}_me{}_hp{}".format(META_ROUNDS, META_EPOCHS, ROUNDS_OF_HP)
    settings.update(
        ALGO_NAME="fedhpro", ALGO_PARAMS=params,
        RUN_SUFFIX="_B_{}_{}".format(budget, DSA_SEED_MODE),
        PROTOCOL_EXTRAS={
            "fedhpro_implementation": "independent_client_snapshot_B",
            "server_budget": {"meta_rounds": META_ROUNDS, "meta_epochs": META_EPOCHS,
                              "rounds_of_hp": ROUNDS_OF_HP},
            "source_server_budget": {"meta_rounds": 100, "meta_epochs": 100, "rounds_of_hp": 30},
            "dsa_seed_derivation": ("sha256_fedhpro_dsa_v1_trainseed_round_client_repeat_class"
                                    if DSA_SEED_MODE == "deterministic" else "wall_clock_ms_mod100000"),
            "dsa_seed_mode": DSA_SEED_MODE,
            "model_aggregation": "sample_weighted_then_replace_classifier_with_fresh_meta_head",
            "real_gradient_aggregation": "equal_contributors_per_class",
            "representation": "classifier_input_no_projection_head",
            "gradient_communication": "shared_backbone_once_extra_head_hp_and_sparse_gradients_v1",
            "state_storage": "memory_only",
            "fedhpro_metrics_version": 1,
        },
    )
    tasks = queue.build_tasks(data_root, settings=settings)
    for task in tasks:
        task["config"]["wandb_setups"]["tags"].extend([
            "snapshot-B", budget, "dsa-" + DSA_SEED_MODE, "no-projection-head",
            "source-budget" if (META_ROUNDS, META_EPOCHS, ROUNDS_OF_HP) == (100, 100, 30) else "adapted-budget"])
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
    print("{} FedHPro tasks; {} main, {} additional; {} rounds; train seeds {}; server budget={}/{}/{}; DSA seed mode={}".format(
        len(tasks), sum("main" in t["categories"] for t in tasks),
        sum("main" not in t["categories"] for t in tasks), N_ROUNDS, TRAIN_SEEDS, META_ROUNDS, META_EPOCHS, ROUNDS_OF_HP, DSA_SEED_MODE))
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
