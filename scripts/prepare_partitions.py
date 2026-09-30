#!/usr/bin/env python3
"""Prepare the 18 shared baseline splits in parallel, without training."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src" / "train_tools" / "preprocessing"))


def unique_configs(tasks):
    """Training seed, epochs, participation and algorithm do not define a split."""
    configurations = {}
    for task in tasks:
        cfg = task["config"]
        data = cfg["data_setups"]
        identity = {key: value for key, value in data.items() if key != "batch_size"}
        key = json.dumps(identity, sort_keys=True)
        configurations.setdefault(key, copy.deepcopy(cfg))
    return list(configurations.values())


def prepare_one(job):
    # NumPy only: spawned workers must not import torch or initialize CUDA.
    from partition_cache import load_or_create_partition
    data, labels, test_labels = job
    root = Path(data["root"]) / data["dataset_name"]
    cache = data.get("partition_cache", {})
    _, _, metadata = load_or_create_partition(
        labels, data["n_clients"], data["partition"], data["dataset_name"],
        cache.get("directory") or root / ".partition_cache",
        seed=data.get("partition_seed", 19940817), test_targets=test_labels)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("fedavg", "moon", "fedproc"), default="fedavg",
                        help="Read the experiment settings from this scheduler")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--dry-run", action="store_true", help="List configurations only, without downloading/generating")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.source == "moon":
        import run_moon_experiments as scheduler
    elif args.source == "fedproc":
        import run_fedproc_experiments as scheduler
    else:
        import run_fedavg_experiments as scheduler
    configs = unique_configs(scheduler.build_tasks(args.data_root))
    print("{} distinct splits; {} CPU workers; source={}".format(len(configs), args.workers, args.source), flush=True)
    if args.dry_run:
        for cfg in configs:
            data = cfg["data_setups"]
            print(data["dataset_name"], data["partition"], "seed=", data["partition_seed"])
        return 0
    for cfg in configs:
        if not cfg["data_setups"].get("partition_cache", {}).get("enabled", True):
            raise ValueError("Enable PARTITION_CACHE_ENABLED before preparing shared splits")
    # Limit native-library thread pools; the independent jobs use the CPU cores.
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[variable] = "1"
    sys.path.insert(0, str(ROOT / "src"))
    from fedavg_data import prepare_data
    from train_tools.preprocessing.datasetter import DATA_INSTANCES
    labels_by_dataset = {}
    # Download each dataset once before starting workers to avoid extraction races.
    for cfg in configs:
        data = cfg["data_setups"]
        dataset = data["dataset_name"]
        if dataset in labels_by_dataset:
            continue
        print("[prepare] Checking/downloading " + dataset, flush=True)
        prepare_data(cfg)
        root = str(Path(data["root"]) / dataset)
        labels_by_dataset[dataset] = (DATA_INSTANCES[dataset](root), DATA_INSTANCES[dataset](root, train=False))
    jobs = []
    for cfg in configs:
        data = cfg["data_setups"]
        train_labels, test_labels = labels_by_dataset[data["dataset_name"]]
        jobs.append((data, train_labels, test_labels if data["partition"]["method"] == "sharding" else None))
    started = time.perf_counter()
    failures, results = [], []
    with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs)),
                             mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = {pool.submit(prepare_one, job): job[0] for job in jobs}
        for future in as_completed(futures):
            data = futures[future]
            description = "{} {}".format(data["dataset_name"], data["partition"])
            try:
                results.append(future.result())
                print("[prepare] READY " + description, flush=True)
            except Exception as exc:
                failures.append({"data": data, "error": str(exc)})
                print("[prepare] FAILED {}: {}".format(description, exc), flush=True)
    manifest = {"ready": results, "failed": failures, "seconds": time.perf_counter() - started}
    destination = args.data_root.expanduser().resolve() / "partition_cache_manifest.json"
    writer = scheduler if args.source == "fedavg" else scheduler.queue
    writer.write_json(destination, manifest)
    print("[prepare] {} ready, {} failed ({:.2f}s); {}".format(
        len(results), len(failures), manifest["seconds"], destination), flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
