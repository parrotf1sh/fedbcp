import copy
import math

import torch

from ..BaseServer import BaseServer
from longtail_metrics import groups_from_counts
from .ClientTrainer import ClientTrainer
from .config import resolve_config
from .measures import average_client_metrics, evaluate_client
from .utils import (
    aggregate_prototypes, classifier_module, cpu_snapshot, create_mask_indices,
    preserve_evaluation_rng,
)

__all__ = ["Server"]


class Server(BaseServer):
    def __init__(self, algo_params, model, data_distributed, optimizer, scheduler, **kwargs):
        super().__init__(algo_params, model, data_distributed, optimizer, scheduler, **kwargs)
        self.cfg = resolve_config(algo_params)
        for name in ("n_rounds", "local_epochs", "n_clients", "num_classes"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError("TinyProto {} must be a positive integer".format(name))
        if (isinstance(self.sample_ratio, bool) or not math.isfinite(self.sample_ratio)
                or not 0 < self.sample_ratio <= 1):
            raise ValueError("TinyProto sample_ratio must be in (0, 1]")
        if set(data_distributed["local"]) != set(range(self.n_clients)):
            raise ValueError("TinyProto requires contiguous project client IDs")
        if type(optimizer) is not torch.optim.SGD or len(optimizer.param_groups) != 1 or optimizer.state:
            raise ValueError("TinyProto requires one untrained plain SGD parameter group")
        options = optimizer.param_groups[0]
        if (any(options.get(key, 0) != 0 for key in ("momentum", "weight_decay", "dampening"))
                or options.get("nesterov", False) or options.get("maximize", False)):
            raise ValueError("TinyProto requires plain SGD without momentum or weight decay")
        self.learning_rate = options["lr"]
        if (type(self.learning_rate) not in (int, float)
                or not math.isfinite(self.learning_rate) or self.learning_rate <= 0):
            raise ValueError("TinyProto learning rate must be finite and positive")
        if scheduler is not None:
            raise ValueError("TinyProto preserves the source fixed learning rate; disable scheduler")
        classifier = classifier_module(model)
        self.feature_dim = classifier.in_features
        if classifier.out_features != self.num_classes:
            raise ValueError("TinyProto model output size does not match the dataset")
        counts = torch.as_tensor(data_distributed["data_map"], dtype=torch.float64)
        if (counts.shape != (self.n_clients, self.num_classes)
                or not torch.isfinite(counts).all() or (counts < 0).any()
                or (counts != counts.floor()).any()):
            raise ValueError("Invalid TinyProto client class counts")
        for client_idx, loaders in data_distributed["local"].items():
            if loaders["datasize"] <= 0 or loaders["train"] is None or len(loaders["train"]) == 0:
                raise ValueError("TinyProto client {} has no training batches".format(client_idx))
            if counts[client_idx].sum().item() != loaders["datasize"]:
                raise ValueError("TinyProto data_map must count only each client's training samples")
        self.class_counts = counts.tolist()
        self.groups = groups_from_counts(counts.sum(dim=0).tolist())
        self.mask_indices = create_mask_indices(self.feature_dim, self.num_classes, self.cfg)
        self.initial_weights = cpu_snapshot(model.state_dict())
        self.local_weights = {}
        self.global_prototypes = {}
        self.client = ClientTrainer(
            cfg=self.cfg, mask_indices=self.mask_indices, learning_rate=self.learning_rate,
            algo_params=algo_params, model=copy.deepcopy(model),
            local_epochs=self.local_epochs, device=self.device, num_classes=self.num_classes,
        )
        self.server_results["classifier_test_accuracy"] = []

    def run(self):
        # BaseServer.run broadcasts/averages weights; TinyProto must not use it.
        from .reporting import run_experiment
        return run_experiment(self)

    def _clients_training(self, sampled_clients):
        if len(sampled_clients) == 0:
            raise ValueError("TinyProto requires at least one participating client")
        local_prototypes, round_results = [], {}
        for client_idx in sampled_clients:
            client_idx = int(client_idx)
            self._set_client_data(client_idx)
            self.client.download_global(
                self.local_weights.get(client_idx, self.initial_weights),
                self.global_prototypes, self.class_counts[client_idx],
            )
            local_results, _ = self.client.train()
            self.local_weights[client_idx] = self.client.upload_local()
            local_prototypes.append(self.client.upload_prototypes())
            round_results = self._results_updater(round_results, local_results)
            self.client.reset()
        # Every participant sees the same previous-round prototypes. No history merge.
        self.global_prototypes = aggregate_prototypes(local_prototypes, self.cfg)
        if not self.global_prototypes:
            raise ValueError("TinyProto round produced no global prototypes")
        if any(not torch.isfinite(value).all() for value in self.global_prototypes.values()):
            raise FloatingPointError("Non-finite TinyProto aggregated prototype")
        return round_results

    def _aggregation(self, w, ns):
        raise NotImplementedError("TinyProto aggregates prototypes, not model parameters")

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
                    if local:
                        continue
                    raise ValueError("TinyProto requires a complete global test loader")
                self.client.model.load_state_dict(
                    self.local_weights.get(client_idx, self.initial_weights), strict=True
                )
                result = evaluate_client(
                    self.client.model, client_loader, self.global_prototypes,
                    self.mask_indices, self.num_classes, self.groups, self.device,
                )
                records.append(dict(client_idx=client_idx, **result))
        if local:
            return records
        return dict(prototype=average_client_metrics(records, "prototype"),
                    classifier=average_client_metrics(records, "classifier"), clients=records)
