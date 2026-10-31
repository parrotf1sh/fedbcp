#!/usr/bin/env python3
"""Paired, globally balanced synthetic study; no datasets/downloads/SwanLab service.

This is a mechanism diagnostic, not a substitute for CIFAR performance evidence.
Hyperparameters and seeds are fixed before inspecting outcomes. Re-runs use a new
directory so an unfavorable result is never silently overwritten.
"""
import argparse
import copy
from datetime import datetime
import hashlib
import json
from pathlib import Path
import random
import statistics
import sys
import tempfile
import time
import uuid

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from algorithms.fedavg.Server import Server as FedAvg
from algorithms.fedbpc.Server import Server as Original
from algorithms.fedbpc_complete.Server import Server as Complete
from algorithms.fedbpc_complete.config import resolve_config
from algorithms.fedavg.reporting import evaluate, write_json
from train_tools.preprocessing.partition_cache import load_or_create_partition
from utils import ConfLoader

METHODS = ("fedavg", "original_fedbpc", "complete", "mean_only", "no_alignment", "alignment_only", "coupled_backbone")


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 8), nn.ReLU())
        self.classifier = nn.Linear(8, 4)

    def forward(self, images, get_features=False):
        features = self.encoder(images)
        logits = self.classifier(features)
        return (logits, features) if get_features else logits


def samples(seed, count):
    generator = torch.Generator().manual_seed(seed)
    labels = torch.arange(4).repeat_interleave(count)
    centers = torch.zeros(4, 8)
    centers[:, :4] = 3 * torch.eye(4)
    return centers[labels] + .9 * torch.randn(len(labels), 8, generator=generator), labels


def data_for(partition, cache, data_seeds, partition_seed):
    x, y = samples(data_seeds[0], 128)
    test_x, test_y = samples(data_seeds[1], 256)
    mapping, _, metadata = load_or_create_partition(
        y.numpy(), 8, partition, "balanced_synthetic", cache, seed=partition_seed,
        test_targets=test_y.numpy(), progress=False)
    local, counts = {}, []
    for k, ids in mapping.items():
        dataset = TensorDataset(x[ids], y[ids])
        counts.append(torch.bincount(y[ids], minlength=4).tolist())
        local[k] = {
            "datasize": len(ids),
            "train": DataLoader(dataset, batch_size=16, shuffle=True,
                                generator=torch.Generator().manual_seed(k)),
            "calibration": DataLoader(dataset, batch_size=64, shuffle=False,
                                      generator=torch.Generator().manual_seed(800+k)),
        }
    test = DataLoader(TensorDataset(test_x, test_y), batch_size=256,
                      generator=torch.Generator().manual_seed(900))
    return {"local": local, "global": {"test": test}, "num_classes": 4,
            "data_map": np.array(counts), "partition_metadata": metadata}


def train(method, partition, seed, rounds, cache, data_seeds, partition_seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    data = data_for(partition, cache, data_seeds, partition_seed)
    model = Model()
    initial_hash = hashlib.sha256(b"".join(t.detach().numpy().tobytes()
        for t in model.state_dict().values())).hexdigest()
    optimizer = torch.optim.SGD(model.parameters(), lr=.04, momentum=.9, weight_decay=1e-5)
    params = resolve_config({})
    cls = Complete
    if method == "fedavg":
        cls, params = FedAvg, {"metrics_only": True}
    elif method == "original_fedbpc":
        cls = Original
        params = ConfLoader(str(ROOT / "src/config/fedbpc.json")).opt.train_setups.algo.params
        params.diagnostics.enabled = False
    elif method == "mean_only":
        params["variance_scale"] = 0.
    elif method == "no_alignment":
        params["lambda_align"] = 0.
    elif method == "alignment_only":
        params["completion_strength"] = 0.
    elif method == "coupled_backbone":
        params["preserve_backbone_gradients"] = False
    server = cls(params, model, data, optimizer, None, n_rounds=rounds,
                 sample_ratio=.25, local_epochs=3, device="cpu")
    history, participation = [], []
    started = time.perf_counter()
    for r in range(rounds):
        selected = server._client_sampling(r)
        server.server_results["client_history"].append(selected)
        participation.append([int(k) for k in selected])
        # Same minibatch order for all methods; extra payload/evaluation passes
        # after local training cannot change the following round's stream.
        for k, local in data["local"].items():
            local["train"].generator.manual_seed(seed * 10000 + r * 8 + k)
        weights, sizes, local = server._clients_training(selected)
        server.model.load_state_dict(server._aggregation(weights, sizes))
        metrics = evaluate(server.model, server.testloader, 4, "cpu")
        if cls is Complete:
            metrics.update(server.batch_round_metrics(local))
        history.append(dict(round=r+1, **metrics))
    return dict(method=method, seed=seed, partition=partition, rounds=rounds,
                settings=dict(lr=.04, momentum=.9, local_epochs=3, batch_size=16,
                              n_clients=8, participation=.25, scheduler=None, params=params),
                initial_model_sha256=initial_hash, participants=participation,
                partition_sha256=data["partition_metadata"]["indices_sha256"],
                class_counts=data["data_map"].tolist(), elapsed_seconds=time.perf_counter()-started,
                history=history, final=history[-1],
                last5_top1=float(np.mean([row["global/top1"] for row in history[-5:]])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=ROOT / "research/FedBPC_local_prototypes_2026-10-22/synthetic")
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 11, 19])
    parser.add_argument("--data-seeds", type=int, nargs=2, default=[71001, 71002])
    parser.add_argument("--partition-seed", type=int, default=19940817)
    parser.add_argument("--methods", choices=METHODS, nargs="+", default=list(METHODS))
    args = parser.parse_args()
    if args.rounds < 1 or not args.seeds or min(args.seeds) < 0:
        parser.error("Positive rounds and nonnegative seeds required")
    torch.set_num_threads(1)
    output = args.output_root / (datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8])
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "protocol.json", {
        "purpose": "Synthetic mechanism diagnostic ONLY, not CIFAR performance evidence",
        "torch": str(torch.__version__), "numpy": np.__version__, "python": sys.version,
        "rounds": args.rounds, "seeds": args.seeds, "methods": args.methods,
        "data": "4 balanced Gaussian input classes; 128 train and 256 test per class; 8-D",
        "data_seeds": args.data_seeds, "partition_seed": args.partition_seed,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "algorithm_sources_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((ROOT / "src/algorithms/fedbpc_complete").glob("*.py"))},
        "interpretation": "fixed split and data; seeds vary initialization/training order; no tuning on results",
    })
    results = []
    with tempfile.TemporaryDirectory() as cache:
        for partition in ({"method": "sharding", "shard_per_user": 2},
                          {"method": "lda", "alpha": .1, "min_samples": 10}):
            for seed in args.seeds:
                paired = []
                for method in args.methods:
                    result = train(method, partition, seed, args.rounds, cache,
                                   args.data_seeds, args.partition_seed)
                    results.append(result)
                    paired.append(result)
                    name = "{}_{}_{}.json".format(partition["method"], seed, method)
                    write_json(output / name, result)
                    print("{} seed={} {} final={:.4f} last5={:.4f}".format(
                        partition["method"], seed, method, result["final"]["global/top1"],
                        result["last5_top1"]), flush=True)
                for result in paired[1:]:
                    for key in ("initial_model_sha256", "partition_sha256", "participants", "class_counts"):
                        if result[key] != paired[0][key]:
                            raise AssertionError("Unpaired comparison: " + key)
    aggregate = []
    for partition in ("sharding", "lda"):
        for method in args.methods:
            selected = [r for r in results if r["method"] == method and r["partition"]["method"] == partition]
            row = dict(partition=partition, method=method, n=len(selected))
            for metric in ("global/top1", "global/macro_f1", "global/worst20_recall", "last5_top1"):
                values = [r[metric] if metric == "last5_top1" else r["final"][metric] for r in selected]
                row[metric + "_mean"] = statistics.mean(values)
                row[metric + "_std"] = statistics.stdev(values) if len(values) > 1 else 0.
            aggregate.append(row)
    write_json(output / "aggregate.json", aggregate)
    print("Saved: " + str(output), flush=True)


if __name__ == "__main__":
    main()
