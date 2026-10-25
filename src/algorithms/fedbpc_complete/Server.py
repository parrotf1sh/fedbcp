"""Two-phase round: shared-encoder statistics, then local prototype completion."""
import copy
import inspect
from pathlib import Path
import time

import torch
import torch.nn as nn

from ..BaseServer import BaseServer
from ..fedavg.reporting import run_metrics, synchronize, tensor_bytes, write_json
from .ClientTrainer import ClientTrainer
from .config import resolve_config
from .utils import PrototypeBank, collect_moments, make_statistics_loader


class Server(BaseServer):
    def __init__(self, algo_params, model, data_distributed, optimizer, scheduler, **kwargs):
        super().__init__(algo_params, model, data_distributed, optimizer, scheduler, **kwargs)
        self.cfg = resolve_config(algo_params)
        if "get_features" not in inspect.signature(model.forward).parameters:
            raise ValueError("FedBPC-Complete needs forward(..., get_features=True)")
        if not isinstance(getattr(model, "classifier", None), nn.Linear):
            raise ValueError("FedBPC-Complete needs a linear model.classifier")
        if model.classifier.out_features != self.num_classes:
            raise ValueError("Classifier and dataset class counts disagree")
        if any(isinstance(module, nn.modules.batchnorm._BatchNorm) for module in model.modules()):
            raise ValueError("First implementation supports backbones without BatchNorm")
        if not 0 < self.sample_ratio <= 1 or self.local_epochs < 1 or self.n_rounds < 1:
            raise ValueError("Invalid training rounds, epochs or participation ratio")
        raw_counts = torch.as_tensor(data_distributed["data_map"])
        if not torch.isfinite(raw_counts).all() or not torch.equal(raw_counts, raw_counts.long()):
            raise ValueError("Partition counts must be finite integers")
        self.counts = raw_counts.long()
        if self.counts.shape != (self.n_clients, self.num_classes) or (self.counts < 0).any():
            raise ValueError("Invalid per-client class counts")
        if (self.counts.sum(1) == 0).any():
            raise ValueError("Empty clients are unsupported")
        self.global_prior = self.counts.sum(0).float() / self.counts.sum()
        self.bank = PrototypeBank(self.num_classes, model.classifier.in_features,
                                  self.cfg["max_prototype_age"])
        self.statistics_loaders = {}
        for index in range(self.n_clients):
            local = data_distributed["local"][index]
            if local["datasize"] != self.counts[index].sum().item():
                raise ValueError("Partition counts and client sizes disagree")
            loader = local.get("calibration")
            if loader is None:
                loader = make_statistics_loader(local["train"], self.testloader,
                    seed=910000 + index, num_workers=self.cfg["statistics_num_workers"])
            if len(loader.dataset) != local["datasize"]:
                raise ValueError("Statistics must use the local training sample set")
            self.statistics_loaders[index] = loader
        self.client = ClientTrainer(algo_params=self.cfg, model=copy.deepcopy(model),
            local_epochs=self.local_epochs, device=self.device, num_classes=self.num_classes)
        self.round_statistics = {}
        self.statistics_samples = self.prototype_bytes = 0

    def run(self):
        if not getattr(self, "experiment_config", {}).get("batch_protocol"):
            raise ValueError("Use the provided fedbpc_complete configuration with batch_protocol")
        output = Path(self.experiment_config["batch_protocol"]["output_dir"])
        write_json(output / "algorithm.json", {
            "algorithm": "fedbpc_complete", "version": 3, "parameters": self.cfg,
            "global_prior": self.global_prior.tolist(),
            "prototype_space": "raw feature means and diagonal sample variances from the pre-local global model",
            "replay": "Gaussian Jensen CE upper bound; detached moments; affine head only",
            "gradient_routing": "natural real CE for encoder, completed class mass for head",
            "aggregation": "sample-weighted FedAvg", "synchronizations_per_round": 2,
            "prediction": "single global linear head; no test-time prototype ensemble",
        })
        return run_metrics(self)

    def _set_client_data(self, client_idx):
        super()._set_client_data(client_idx)
        self.client.class_counts = self.counts[client_idx].tolist()

    def _clients_training(self, sampled_clients):
        round_idx = len(self.server_results["client_history"]) - 1
        server_weights = copy.deepcopy(self.model.state_dict())
        server_optimizer = self.optimizer.state_dict()
        payloads = []
        synchronize(self.device)
        start = time.perf_counter()
        # All statistics use this SAME frozen model before ANY client training.
        self.client.model.load_state_dict(server_weights)
        for index in sampled_clients:
            payload = collect_moments(self.client.model, self.statistics_loaders[int(index)],
                self.num_classes, self.device, encoder_version=round_idx)
            if not torch.equal(payload["class_counts"], self.counts[index]):
                raise ValueError("Statistics labels disagree with the training partition")
            payloads.append(payload)
        update = self.bank.update(payloads, encoder_version=round_idx)
        synchronize(self.device)
        seconds = time.perf_counter() - start
        uplink = tensor_bytes(payloads)
        downlink = len(sampled_clients) * (tensor_bytes(self.bank.state_dict())
                                         + tensor_bytes(self.global_prior))
        setup_bytes = tensor_bytes(self.counts) if round_idx == 0 else 0
        self.round_bytes = uplink + downlink + setup_bytes
        self.prototype_bytes += self.round_bytes
        self.round_statistics = {
            "prototype/statistics_seconds": seconds,
            "prototype/processed_samples": int(update["counts"].sum()),
            "prototype/current_coverage": update["updated"].float().mean().item(),
            "prototype/memory_coverage": self.bank.valid.float().mean().item(),
            "prototype/uplink_bytes": uplink, "prototype/downlink_bytes": downlink,
            "prototype/setup_count_bytes": setup_bytes,
            "prototype/expired_classes": int(update["expired"].sum()),
        }
        self.statistics_samples += int(update["counts"].sum())
        for c in range(self.num_classes):
            prefix = "prototype/class_{:03d}/".format(c)
            self.round_statistics.update({
                prefix + "samples": int(update["counts"][c]),
                prefix + "contributors": int(update["contributors"][c]),
                prefix + "valid": int(self.bank.valid[c]),
                prefix + "age": int(self.bank.age[c]),
                prefix + "norm": self.bank.means[c].norm().item(),
                prefix + "variance_trace": self.bank.variances[c].sum().item(),
                prefix + "support": int(self.bank.support[c]),
            })
        weights, sizes, results = [], [], {}
        for index in sampled_clients:
            self._set_client_data(index)
            self.client.download_global(server_weights, server_optimizer,
                self.bank, self.global_prior, round_idx)
            local, size = self.client.train()
            weights.append({key: value.detach().cpu().clone()
                            for key, value in self.client.model.state_dict().items()})
            sizes.append(size)
            results = self._results_updater(results, local)
            self.client.reset()
        return weights, sizes, results

    def batch_communication_bytes(self):
        return self.round_bytes

    def batch_round_metrics(self, local):
        seen = sum(local["seen"])
        metrics = dict(self.round_statistics)
        for key in ("real_ce", "replay_ce", "align", "natural_ce"):
            metrics["train/" + key] = sum(local[key + "_sum"]) / seen
        for key in ("completion_mass", "missing_completion_mass",
                    "minority_completion_mass", "classification_mass"):
            metrics["completion/" + key] = sum(local[key]) / len(local[key])
        metrics["prototype/cumulative_samples"] = self.statistics_samples
        metrics["prototype/cumulative_bytes"] = self.prototype_bytes
        return metrics

    def batch_summary_metrics(self, metrics):
        protocol = self.experiment_config["batch_protocol"]
        if protocol.get("save_checkpoints", False):
            # Inspection checkpoint, not a claim of exact resumable-loader support.
            torch.save({
                "model": {k: v.detach().cpu() for k, v in self.model.state_dict().items()},
                "prototype_bank": self.bank.state_dict(), "global_prior": self.global_prior,
                "parameters": self.cfg, "completed_rounds": self.n_rounds,
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
            }, Path(protocol["output_dir"]) / "final_state.pt")
        return {
            "algorithm": "fedbpc_complete",
            "prototype_processed_samples": self.statistics_samples,
            "prototype_communication_bytes": self.prototype_bytes,
            "final_completion_mass": metrics["completion/completion_mass"],
            "final_prototype_coverage": metrics["prototype/memory_coverage"],
            "communication_definition": (
                "model upload/download + optimizer tensor download per participant; "
                "plus float64 raw feature sums/squared sums and counts upload, "
                "prototype bank/prior download, "
                "and initial all-client class counts; excludes transport overhead"),
        }
