import copy
import math

import torch

from ..BaseServer import BaseServer
from longtail_metrics import groups_from_counts
from .ClientTrainer import ClientTrainer
from .config import resolve_config
from .measures import average_client_metrics, evaluate_client
from .utils import (
    aggregate_prototypes,
    classifier_module,
    cpu_snapshot,
    initialize_anchors,
    preserve_evaluation_rng,
    update_anchors,
)

__all__ = ["Server"]


class Server(BaseServer):
    def __init__(self, algo_params, model, data_distributed, optimizer, scheduler, **kwargs):
        super().__init__(algo_params, model, data_distributed, optimizer, scheduler, **kwargs)
        self.cfg = resolve_config(algo_params)
        classifier = classifier_module(model)
        for name in ("n_rounds", "local_epochs", "n_clients", "num_classes"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError("{} must be a positive integer".format(name))
        if self.num_classes < 2:
            raise ValueError("FedSA anchor SyncBN and class margins require at least two classes")
        if classifier.out_features != self.num_classes:
            raise ValueError("FedSA model output size must match dataset classes")
        if not math.isfinite(self.sample_ratio) or not 0 < self.sample_ratio <= 1:
            raise ValueError("sample_ratio must be in (0, 1]")
        if set(data_distributed["local"]) != set(range(self.n_clients)):
            raise ValueError("FedSA requires contiguous project client IDs")
        if type(optimizer) is not torch.optim.SGD:
            raise ValueError("FedSA uses SGD with the project's experiment parameters")
        if len(optimizer.param_groups) != 1 or optimizer.state:
            raise ValueError("FedSA requires one untrained optimizer parameter group")
        counts = torch.as_tensor(data_distributed["data_map"])
        if (counts.shape != (self.n_clients, self.num_classes)
                or not torch.isfinite(counts).all() or (counts < 0).any()):
            raise ValueError("Invalid FedSA client class counts")
        for client_idx, loaders in data_distributed["local"].items():
            if loaders["datasize"] <= 0 or loaders["train"] is None or len(loaders["train"]) == 0:
                raise ValueError("FedSA client {} has no training batches".format(client_idx))
        self.groups = groups_from_counts(counts.sum(dim=0).tolist())
        self.initial_weights = cpu_snapshot(model.state_dict())
        self.local_weights = {}
        self.local_prototypes = {}
        self.global_prototypes = None
        self.semantic_anchors = initialize_anchors(
            self.num_classes, classifier.in_features, self.cfg["anchor_seed"], self.device
        )
        self.client = ClientTrainer(
            cfg=self.cfg, algo_params=algo_params, model=copy.deepcopy(model),
            local_epochs=self.local_epochs, device=self.device, num_classes=self.num_classes,
        )
        self.server_results["classifier_test_accuracy"] = []

    def run(self):
        # BaseServer.run broadcasts/averages weights, which would change FedSA.
        from .reporting import run_experiment
        return run_experiment(self)

    def _clients_training(self, sampled_clients):
        if len(sampled_clients) == 0:
            raise ValueError("FedSA requires at least one selected client")
        uploads, round_results = [], {}
        server_optimizer = self.optimizer.state_dict()
        for client_idx in sampled_clients:
            client_idx = int(client_idx)
            self._set_client_data(client_idx)
            self.client.download_global(
                self.local_weights.get(client_idx, self.initial_weights),
                server_optimizer, self.semantic_anchors,
                self.local_prototypes.get(client_idx),
            )
            local_results, _ = self.client.train()
            self.local_weights[client_idx] = self.client.upload_local()
            prototypes = self.client.upload_prototypes()
            uploads.append(prototypes)
            self.local_prototypes[client_idx] = cpu_snapshot(prototypes)
            round_results = self._results_updater(round_results, local_results)
            self.client.reset()

        # Every selected client used the same anchors, and its own older prototypes.
        self.global_prototypes = aggregate_prototypes(uploads)
        if not self.global_prototypes:
            raise ValueError("FedSA round produced no global prototypes")
        update_anchors(self.semantic_anchors, self.global_prototypes, self.cfg["anchor_momentum"])
        return round_results

    def _aggregation(self, w, ns):
        raise NotImplementedError("FedSA aggregates prototypes, never model parameters")

    def evaluate_clients(self, loader=None, local=False):
        if loader is None and not local:
            loader = self.testloader
        loaders = ([self.data_distributed["local"][index].get("test")
                    for index in range(self.n_clients)] if local else [loader])
        records = []
        with preserve_evaluation_rng(loaders):
            for client_idx in range(self.n_clients):
                client_loader = loaders[client_idx] if local else loader
                if client_loader is None:
                    continue
                self.client.model.load_state_dict(
                    self.local_weights.get(client_idx, self.initial_weights), strict=True
                )
                result = evaluate_client(
                    self.client.model, client_loader, self.global_prototypes,
                    self.num_classes, self.groups, self.device,
                )
                records.append(dict(client_idx=client_idx, **result))
        if local:
            return records
        return dict(prototype=average_client_metrics(records, "prototype"),
                    classifier=average_client_metrics(records, "classifier"), clients=records)

    def step_scheduler(self):
        if self.scheduler is not None:
            # This untrained optimizer only carries the shared learning-rate schedule.
            self.optimizer.zero_grad(set_to_none=True)
            self.optimizer.step()
            self.scheduler.step()
