"""Metrics-only evaluation for the optional FedAvg experiment queue.

No model/optimizer serialization is performed here. All metrics are fractions.
"""
import csv
import json
import math
from pathlib import Path
import time

import torch
import torch.nn.functional as F
import wandb


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def tensor_bytes(value):
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(tensor_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(tensor_bytes(item) for item in value)
    return 0


def synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def evaluate(model, loader, num_classes, device):
    model.eval()
    model.to(device)
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    loss_sum, count = 0.0, 0
    for data, targets in loader:
        data, targets = data.to(device), targets.to(device)
        logits = model(data)
        loss_sum += F.cross_entropy(logits, targets, reduction="sum").item()
        predictions = logits.argmax(1)
        indices = (targets * num_classes + predictions).cpu()
        confusion += torch.bincount(indices, minlength=num_classes ** 2).reshape(num_classes, num_classes)
        count += targets.numel()
    if not count:
        raise ValueError("Global evaluation dataset is empty")
    support = confusion.sum(1)
    predicted = confusion.sum(0)
    true_positive = confusion.diag().double()
    recall = true_positive / support.clamp_min(1)
    f1 = 2 * true_positive / (support + predicted).clamp_min(1)
    present = support > 0
    worst_count = max(1, math.ceil(int(present.sum()) * 0.2))
    metrics = {"global/top1": true_positive.sum().item() / count,
               "global/loss": loss_sum / count,
               "global/macro_f1": f1[present].mean().item(),
               "global/worst20_recall": recall[present].sort().values[:worst_count].mean().item()}
    metrics.update({"class_recall/{:03d}".format(i): recall[i].item()
                    for i in range(num_classes) if present[i]})
    return metrics


def log_partition(server, output):
    counts = torch.as_tensor(server.data_distributed["data_map"], dtype=torch.float64)
    sizes = counts.sum(1)
    probabilities = counts / sizes.clamp_min(1).unsqueeze(1)
    entropies = -(probabilities * probabilities.clamp_min(1e-30).log()).sum(1)
    stats = {"samples_per_client": sizes.tolist(),
             "classes_per_client": (counts > 0).sum(1).tolist(),
             "label_entropy_per_client": entropies.tolist(),
             "clients_per_class": (counts > 0).sum(0).tolist(),
             "class_counts_per_client": counts.long().tolist()}
    write_json(output / "partition_stats.json", stats)
    wandb.run.summary.update({"data/client_samples_min": sizes.min().item(),
                             "data/client_samples_max": sizes.max().item(),
                             "data/client_classes_mean": (counts > 0).sum(1).double().mean().item(),
                             "data/label_entropy_mean": entropies.mean().item()})


def run_metrics(server):
    config = server.experiment_config
    output = Path(config["batch_protocol"]["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    server._print_start()
    log_partition(server, output)
    wandb.define_metric("round")
    wandb.define_metric("*", step_metric="round")
    history = []
    communication = 0
    train_seconds = 0.0
    processed = 0
    start = time.perf_counter()
    with (output / "metrics.csv").open("w", newline="") as handle, \
            (output / "clients.jsonl").open("w") as clients_handle:
        writer = None
        for round_idx in range(server.n_rounds):
            sampled = server._client_sampling(round_idx)  # Keep legacy round-only seeding.
            server.server_results["client_history"].append(sampled)
            learning_rate = server.optimizer.param_groups[0]["lr"]
            model_size = tensor_bytes(server.model.state_dict())
            optimizer_size = tensor_bytes(server.optimizer.state_dict())
            synchronize(server.device)
            tick = time.perf_counter()
            weights, sizes, local = server._clients_training(sampled)
            aggregated = server._aggregation(weights, sizes)
            server.model.load_state_dict(aggregated)
            synchronize(server.device)
            round_training = time.perf_counter() - tick
            del weights, aggregated
            tick = time.perf_counter()
            metrics = evaluate(server.model, server.testloader, server.num_classes, server.device)
            synchronize(server.device)
            evaluation_seconds = time.perf_counter() - tick
            train_seconds += round_training
            round_bytes = len(sampled) * (2 * model_size + optimizer_size)
            communication += round_bytes
            seen = sum(local["seen"])
            processed += seen
            metrics.update({"round": round_idx + 1,
                            "server_test_acc": metrics["global/top1"],
                            "train/loss": sum(local["loss_sum"]) / seen,
                            "train/online_top1": sum(local["correct"]) / seen,
                            "train/learning_rate": learning_rate,
                            "train/processed_samples": processed,
                            "time/train_round_seconds": round_training,
                            "time/eval_round_seconds": evaluation_seconds,
                            "time/round_seconds": round_training + evaluation_seconds,
                            "time/cumulative_train_seconds": train_seconds,
                            "time/elapsed_seconds": time.perf_counter() - start,
                            "communication/round_bytes": round_bytes,
                            "communication/cumulative_bytes": communication})
            if server.scheduler is not None:
                server.scheduler.step()
            if writer is None:
                writer = csv.DictWriter(handle, fieldnames=list(metrics))
                writer.writeheader()
            writer.writerow(metrics)
            handle.flush()
            clients_handle.write(json.dumps({"round": round_idx + 1,
                                             "clients": [int(i) for i in sampled]}) + "\n")
            clients_handle.flush()
            wandb.log(metrics, step=round_idx + 1)
            server.server_results["test_accuracy"].append(metrics["global/top1"])
            history.append(metrics["global/top1"])
            print("[Round {}/{}] global Top-1={:.4f} loss={:.4f} "
                  "train={:.1f}s eval={:.1f}s".format(round_idx + 1, server.n_rounds,
                    metrics["global/top1"], metrics["global/loss"], round_training,
                    evaluation_seconds), flush=True)
    server.batch_summary = {
        "completed_rounds": len(history), "final_top1": history[-1],
        "last10_top1_mean": sum(history[-10:]) / len(history[-10:]),
        "final_loss": metrics["global/loss"], "final_macro_f1": metrics["global/macro_f1"],
        "final_worst20_recall": metrics["global/worst20_recall"],
        "total_train_seconds": train_seconds, "communication_bytes": communication,
        "communication_definition": "sum over clients of model upload + model download + optimizer tensor download; excludes transport overhead",
        "seed": config["train_setups"]["seed"],
    }
    wandb.run.summary.update(server.batch_summary)
