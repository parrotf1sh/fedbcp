#!/usr/bin/env python3
"""Sequential FedAvg experiments. The coordinator needs only Python's stdlib."""
import argparse
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time
import traceback
from contextlib import contextmanager
from types import SimpleNamespace


# ======================== Experiment settings ========================
REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
DATA_ROOT = REPO_ROOT / "data"
OUTPUT_ROOT = REPO_ROOT / "fedavg_experiments"
DATASETS = ["cifar10", "cifar100", "tiny-imagenet"]
LDA_ALPHAS = [0.03, 0.1, 0.3]
SHARDS_PER_CLIENT = [2, 4, 6]
TRAIN_SEEDS = [2022, 2023, 2024]
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
DOWNLOAD_IF_MISSING = True
INCLUDE_SUPPLEMENTARY = True
SUPPLEMENTARY_DATASET = "cifar100"
SUPPLEMENTARY_ALPHA = 0.1
PARTICIPATION_RATES = [0.05, 0.1, 0.2]
LOCAL_EPOCH_VALUES = [1, 5, 10]
WANDB_PROJECT = "Point1"
WANDB_ENTITY = None  # Uses the logged-in account's default entity.
BATCH_NAME = "fedavg-300rounds"
SKIP_SUCCESSFUL = True
# These describe the existing protocol; they are deliberately NOT swept.
PARTITION_SEED = 19940817
CLIENT_SAMPLING_RULE = "numpy.seed(round_index), zero_based"
ALGO_NAME = "fedavg"
ALGO_PARAMS = {"metrics_only": True}
RUN_SUFFIX = ""
PROTOCOL_EXTRAS = {}
PARTITION_CACHE_ENABLED = True
PARTITION_CACHE_DIR = None  # Default: <data-root>/<dataset>/.partition_cache
LDA_MIN_SAMPLES = 10
LDA_MAX_ATTEMPTS = 128
LDA_INSUFFICIENT_POLICY = "repair"  # Shared by all algorithms; transfers are recorded.
# =====================================================================


def canonical_dataset(name):
    return "tinyimagenet" if name in ("tiny-imagenet", "tiny_imagenet") else name


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:16]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def build_tasks(data_root=None, settings=None):
    """Deduplicate on all effective training settings, independently of labels."""
    s = SimpleNamespace(**(globals() if settings is None else settings))
    if len(s.TRAIN_SEEDS) != 3 or len(set(s.TRAIN_SEEDS)) != 3:
        raise ValueError("TRAIN_SEEDS must contain three distinct training seeds")
    if min(s.N_CLIENTS, s.LOCAL_EPOCHS, s.N_ROUNDS, s.BATCH_SIZE) < 1:
        raise ValueError("Clients, epochs, rounds and batch size must be positive")
    tasks = {}
    classes = {"cifar10": 10, "cifar100": 100, "tinyimagenet": 200}

    def add(dataset, partition, category, ratio=None, epochs=None):
        ratio = s.SAMPLE_RATIO if ratio is None else ratio
        epochs = s.LOCAL_EPOCHS if epochs is None else epochs
        dataset = canonical_dataset(dataset)
        if dataset not in classes or dataset not in s.MODELS:
            raise ValueError("Unsupported dataset: " + dataset)
        if not 0 < ratio <= 1 or epochs < 1:
            raise ValueError("Participation must be in (0, 1]; epochs must be positive")
        if partition["method"] == "sharding":
            shards = partition["shard_per_user"]
            per_class, remainder = divmod(s.N_CLIENTS * shards, classes[dataset])
            test_per_class = 1000 if dataset == "cifar10" else (100 if dataset == "cifar100" else 50)
            if shards < 1 or remainder or not 1 <= per_class <= test_per_class:
                raise ValueError("Invalid sharding for {}: N={} and shards={}; "
                                 "N * shards must be divisible by the class count "
                                 "and not create empty test shards".format(dataset, s.N_CLIENTS, shards))
            label = "shards-{}".format(shards)
        else:
            if partition["alpha"] <= 0:
                raise ValueError("Dirichlet alpha must be positive")
            label = "lda-a{:g}".format(partition["alpha"])
            partition = dict(partition, min_samples=s.LDA_MIN_SAMPLES,
                             max_attempts=s.LDA_MAX_ATTEMPTS,
                             insufficient_policy=s.LDA_INSUFFICIENT_POLICY)
        for seed in s.TRAIN_SEEDS:
            config = {
                "data_setups": {"root": str(Path(data_root or s.DATA_ROOT).expanduser().resolve()),
                                "dataset_name": dataset, "batch_size": s.BATCH_SIZE,
                                "n_clients": s.N_CLIENTS, "partition": copy.deepcopy(partition),
                                "partition_seed": s.PARTITION_SEED,
                                "partition_cache": {"enabled": s.PARTITION_CACHE_ENABLED,
                                                    "directory": str(s.PARTITION_CACHE_DIR) if s.PARTITION_CACHE_DIR else None}},
                "train_setups": {
                    "algo": {"name": s.ALGO_NAME, "params": copy.deepcopy(s.ALGO_PARAMS)},
                    "scenario": {"n_rounds": s.N_ROUNDS, "sample_ratio": ratio,
                                 "local_epochs": epochs, "device": s.DEVICE},
                    "model": {"name": s.MODELS[dataset], "params": {}},
                    "optimizer": {"params": copy.deepcopy(s.OPTIMIZER)},
                    "scheduler": copy.deepcopy(s.SCHEDULER), "seed": seed},
                "batch_protocol": {"version": 1, "partition_seed": s.PARTITION_SEED,
                                   "client_sampling_rule": s.CLIENT_SAMPLING_RULE,
                                   "download_if_missing": s.DOWNLOAD_IF_MISSING,
                                   "save_checkpoints": False,
                                   "evaluation": "global_each_round",
                                   "accuracy_unit": "fraction"},
            }
            config["batch_protocol"]["partition_implementation"] = "shared_partition_cache_v1"
            config["batch_protocol"].update(copy.deepcopy(s.PROTOCOL_EXTRAS))
            task_id = digest(config)
            if task_id in tasks:
                if category not in tasks[task_id]["categories"]:
                    tasks[task_id]["categories"].append(category)
                continue
            group_config = copy.deepcopy(config)
            del group_config["train_setups"]["seed"]
            condition = digest(group_config)
            name = "{}_{}_{}_n{}_q{:g}_e{}_r{}{}_seed{}".format(
                s.ALGO_NAME, dataset, label, s.N_CLIENTS, ratio, epochs,
                s.N_ROUNDS, s.RUN_SUFFIX, seed)
            tasks[task_id] = {"id": task_id, "condition_id": condition, "name": name,
                              "categories": [category], "config": config}

    for alpha in s.LDA_ALPHAS:
        for dataset in s.DATASETS:
            add(dataset, {"method": "lda", "alpha": alpha}, "main")
    for shards in s.SHARDS_PER_CLIENT:
        for dataset in s.DATASETS:
            add(dataset, {"method": "sharding", "shard_per_user": shards}, "main")
    if s.INCLUDE_SUPPLEMENTARY:
        partition = {"method": "lda", "alpha": s.SUPPLEMENTARY_ALPHA}
        for ratio in s.PARTICIPATION_RATES:
            add(s.SUPPLEMENTARY_DATASET, partition, "participation", ratio=ratio)
        for epochs in s.LOCAL_EPOCH_VALUES:
            add(s.SUPPLEMENTARY_DATASET, partition, "local_epochs", epochs=epochs)
    for task in tasks.values():
        cfg = task["config"]
        cfg["wandb_setups"] = {
            "project": s.WANDB_PROJECT, "entity": s.WANDB_ENTITY, "mode": "online", "force": True,
            "name": task["name"], "group": s.BATCH_NAME + "-" + task["condition_id"],
            "job_type": s.ALGO_NAME, "save_code": False,
            "tags": [s.ALGO_NAME, cfg["data_setups"]["dataset_name"],
                     cfg["data_setups"]["partition"]["method"],
                     "seed-{}".format(cfg["train_setups"]["seed"]), s.BATCH_NAME] + task["categories"],
        }
    return list(tasks.values())


@contextmanager
def queue_lock(output_root):
    # flock is released automatically even if the coordinator crashes.
    import fcntl
    with (output_root / ".queue.lock").open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another scheduler is using " + str(output_root)) from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def run_child(command, log_path, env):
    """Stream output, reap each process, and stop its group on user interrupt."""
    with Path(log_path).open("w", buffering=1) as log:
        process = subprocess.Popen(command, cwd=str(REPO_ROOT / "src"), env=env,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, errors="replace", bufsize=1, start_new_session=True)
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
            return process.wait()
        finally:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            process.stdout.close()


def successful(directory):
    try:
        status = json.loads((directory / "status.json").read_text())
        summary = json.loads((directory / "summary.json").read_text())
        return status.get("status") == "success" and summary.get("status") == "success"
    except (OSError, ValueError):
        return False


def aggregate_results(tasks, output_root):
    groups = {}
    for task in tasks:
        directory = output_root / task["id"]
        group = groups.setdefault(task["condition_id"], {"tasks": [], "summaries": []})
        group["tasks"].append(task)
        if successful(directory):
            group["summaries"].append(json.loads((directory / "summary.json").read_text()))
    rows = []
    metrics = ["final_top1", "last10_top1_mean", "final_loss", "final_macro_f1",
               "final_worst20_recall", "total_train_seconds", "communication_bytes"]
    for condition, group in groups.items():
        example = group["tasks"][0]["config"]
        row = {"condition_id": condition, "dataset": example["data_setups"]["dataset_name"],
               "partition": json.dumps(example["data_setups"]["partition"], sort_keys=True),
               "sample_ratio": example["train_setups"]["scenario"]["sample_ratio"],
               "local_epochs": example["train_setups"]["scenario"]["local_epochs"],
               "n_success": len(group["summaries"]), "n_expected": len(group["tasks"])}
        for metric in metrics:
            values = [s[metric] for s in group["summaries"] if metric in s]
            row[metric + "_mean"] = statistics.mean(values) if values else ""
            row[metric + "_std"] = statistics.stdev(values) if len(values) > 1 else ""
        rows.append(row)
    if rows:
        with (output_root / "aggregate.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def execute(tasks, output_root, python=PYTHON, skip_successful=True):
    counts = {"success": 0, "failed": 0, "skipped": 0}
    for index, task in enumerate(tasks, 1):
        directory = output_root / task["id"]
        directory.mkdir(parents=True, exist_ok=True)
        if skip_successful and successful(directory):
            print("[{}/{}] SKIP {}".format(index, len(tasks), task["name"]), flush=True)
            counts["skipped"] += 1
            continue
        # Keep failed-attempt logs, and always allocate a new W&B run (no checkpoint resume).
        attempt = directory / ("attempt-" + str(time.time_ns()))
        attempt.mkdir()
        cfg = copy.deepcopy(task["config"])
        cfg["batch_protocol"].update(output_dir=str(attempt), task_id=task["id"],
                                     condition_id=task["condition_id"], categories=task["categories"])
        cfg["wandb_setups"]["dir"] = str(attempt)
        write_json(attempt / "config.json", cfg)
        status = {"status": "running", "name": task["name"], "attempt_dir": str(attempt),
                  "started_at": time.time()}
        write_json(directory / "status.json", status)
        print("[{}/{}] START {}".format(index, len(tasks), task["name"]), flush=True)
        env = dict(os.environ, PYTHONUNBUFFERED="1", WANDB_MODE="online",
                   PYTHONHASHSEED=str(cfg["train_setups"]["seed"]))
        # Never attach to a run ID inherited from a parent shell or sweep.
        for key in ("WANDB_RUN_ID", "WANDB_RESUME", "WANDB_SWEEP_ID"):
            env.pop(key, None)
        try:
            code = run_child([python, "-u", str(REPO_ROOT / "src" / "main.py"),
                              "--config_path", str(attempt / "config.json")],
                             attempt / "train.log", env)
            status["exit_code"] = code
            if code:
                raise RuntimeError("Training exited with code {}; see {}".format(code, attempt / "train.log"))
            summary = json.loads((attempt / "summary.json").read_text())
            if summary.get("status") != "success":
                raise RuntimeError("Training did not produce a successful summary")
            write_json(directory / "summary.json", summary)
            status["status"] = "success"
        except KeyboardInterrupt:
            status.update(status="interrupted", finished_at=time.time())
            write_json(directory / "status.json", status)
            aggregate_results(tasks, output_root)
            print("Interrupted by user; queue stopped.", flush=True)
            raise
        except Exception as exc:
            status.update(status="failed", error=str(exc))
            traceback.print_exc()
        status["finished_at"] = time.time()
        write_json(directory / "status.json", status)
        counts[status["status"]] += 1
        aggregate_results(tasks, output_root)
        print("[{}] {}".format(status["status"].upper(), task["name"]), flush=True)
    aggregate_results(tasks, output_root)
    write_json(output_root / "queue_summary.json", counts)
    print("Queue finished: " + json.dumps(counts), flush=True)
    return 1 if counts["failed"] else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Validate and print tasks without training/downloading/W&B")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--python", default=PYTHON)
    parser.add_argument("--rerun-successful", action="store_true")
    args = parser.parse_args()
    tasks = build_tasks(args.data_root)
    print("{} tasks; {} main, {} additional; {} rounds; train seeds {}".format(
        len(tasks), sum("main" in t["categories"] for t in tasks),
        sum("main" not in t["categories"] for t in tasks), N_ROUNDS, TRAIN_SEEDS))
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    with queue_lock(output_root):
        write_json(output_root / "manifest.json", tasks)
        if args.dry_run:
            for i, task in enumerate(tasks, 1):
                print("{:02d}. {} [{}]".format(i, task["name"], ", ".join(task["categories"])))
            return 0
        return execute(tasks, output_root, args.python, SKIP_SUCCESSFUL and not args.rerun_successful)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
