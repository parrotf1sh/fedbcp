#!/usr/bin/env python3
"""44 serial FedSA experiments preserving the provided source behavior."""
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
OUTPUT_ROOT = REPO_ROOT / "fedsa_experiments"
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
LAMBDA_ANCHOR = 1.0
LAMBDA_MCL = 1.0
LAMBDA_CC = 1.0
ANCHOR_MOMENTUM = 0.9999
ANCHOR_SEED = 0
EVAL_INTERVAL = 10
FINAL_EVAL_ROUNDS = 10
DOWNLOAD_IF_MISSING = True
INCLUDE_SUPPLEMENTARY = True
SUPPLEMENTARY_DATASET = "cifar100"
SUPPLEMENTARY_ALPHA = 0.1
PARTICIPATION_RATES = [0.05, 0.1, 0.2]
LOCAL_EPOCH_VALUES = [1, 5, 10]
SWANLAB_PROJECT = "Point1"
SWANLAB_WORKSPACE = None
BATCH_NAME = "fedsa-source-300rounds"
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
    resolve = runpy.run_path(str(REPO_ROOT / "src/algorithms/fedsa/config.py"))["resolve_config"]
    params = resolve({
        "lambda_anchor": LAMBDA_ANCHOR, "lambda_mcl": LAMBDA_MCL,
        "lambda_cc": LAMBDA_CC, "anchor_momentum": ANCHOR_MOMENTUM,
        "anchor_seed": ANCHOR_SEED,
    })
    schedule = runpy.run_path(str(REPO_ROOT / "src/algorithms/fedsa/batch.py"))["evaluation_rounds"]
    evaluation = schedule(N_ROUNDS, EVAL_INTERVAL, FINAL_EVAL_ROUNDS)
    settings = dict(globals())
    settings.update(
        ALGO_NAME="fedsa", ALGO_PARAMS=params,
        RUN_SUFFIX="_anchor{:g}_aseed{}_eval{}".format(ANCHOR_MOMENTUM, ANCHOR_SEED, EVAL_INTERVAL),
        PROTOCOL_EXTRAS={
            "fedsa_batch": True, "write_metrics_files": True,
            "fedsa_implementation": "persistent_client_semantic_anchors_v1",
            "model_aggregation": "none_persistent_client_models",
            "prototype_aggregation": "equal_contributor_mean_per_class",
            "prototype_collection": "all_local_epochs_pre_step_features",
            "prediction": "all_client_mean_global_prototype_mse_argmin",
            "evaluation": "all_clients_full_test_on_scheduled_rounds",
            "eval_interval": EVAL_INTERVAL, "final_eval_rounds": FINAL_EVAL_ROUNDS,
            "evaluation_rounds": evaluation,
            "representation": "classifier_input_no_projection_head",
            "state_storage": "cpu_client_model_and_prototype_memory_only",
            "communication": "anchors_down_prototypes_up_training_only_v1",
            "initial_model_distribution": "one_time_estimate_reported_separately",
            "fedsa_metrics_version": 1,
        },
    )
    tasks = queue.build_tasks(data_root, settings=settings)
    for task in tasks:
        task["config"]["swanlab_setups"]["tags"].extend([
            "persistent-client-models", "prototype-prediction", "no-projection-head",
            "anchor-seed-{}".format(ANCHOR_SEED), "eval-every-{}".format(EVAL_INTERVAL)])
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
    print("{} FedSA tasks; {} main, {} additional; {} rounds; train seeds {}; persistent client models; anchor_seed={}; eval_interval={}".format(
        len(tasks), sum("main" in t["categories"] for t in tasks),
        sum("main" not in t["categories"] for t in tasks), N_ROUNDS, TRAIN_SEEDS, ANCHOR_SEED, EVAL_INTERVAL))
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
