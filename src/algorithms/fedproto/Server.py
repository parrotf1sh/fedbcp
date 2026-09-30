import copy
import math

import torch

from ..BaseServer import BaseServer
from longtail_metrics import groups_from_counts
from .ClientTrainer import ClientTrainer
from .config import resolve_config
from .measures import average_client_metrics, evaluate_client
from .utils import aggregate_prototypes, cpu_snapshot, feature_module, preserve_evaluation_rng

__all__ = ["Server"]


class Server(BaseServer):
    def __init__(self, algo_params, model, data_distributed, optimizer, scheduler, **kwargs):
        super().__init__(algo_params, model, data_distributed, optimizer, scheduler, **kwargs)
        self.cfg = resolve_config(algo_params)
        feature_module(model)
        for name in ("n_rounds", "local_epochs", "n_clients", "num_classes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError("{} must be a positive integer".format(name))
        if not math.isfinite(self.sample_ratio) or not 0 < self.sample_ratio <= 1:
            raise ValueError("sample_ratio must be in (0, 1]")
        if set(data_distributed["local"]) != set(range(self.n_clients)):
            raise ValueError("FedProto requires contiguous project client IDs")
        if type(optimizer) not in (torch.optim.SGD, torch.optim.Adam):
            raise ValueError("FedProto supports SGD and Adam")
        if len(optimizer.param_groups) != 1 or optimizer.state:
            raise ValueError("FedProto requires one untrained optimizer parameter group")
        counts = torch.as_tensor(data_distributed["data_map"])
        if counts.shape != (self.n_clients, self.num_classes) or (counts < 0).any():
            raise ValueError("Invalid FedProto client class counts")
        self.local_classes = {}
        for client_idx, loaders in data_distributed["local"].items():
            if loaders["datasize"] <= 0 or loaders["train"] is None or len(loaders["train"]) == 0:
                raise ValueError("FedProto client {} has no training batches".format(client_idx))
            self.local_classes[client_idx] = set(torch.nonzero(
                counts[client_idx] > 0, as_tuple=False
            ).view(-1).tolist())
            if not self.local_classes[client_idx]:
                raise ValueError("FedProto client {} has no training classes".format(client_idx))
        self.groups = groups_from_counts(counts.sum(dim=0).tolist())
        self.initial_weights = cpu_snapshot(model.state_dict())
        self.local_weights = {}
        self.global_prototypes = {}
        self.client = ClientTrainer(
            lambda_proto=self.cfg["lambda_proto"], optimizer_class=type(optimizer),
            algo_params=algo_params, model=copy.deepcopy(model),
            local_epochs=self.local_epochs, device=self.device, num_classes=self.num_classes,
        )
        self.server_results["classifier_test_accuracy"] = []

    def run(self):
        # BaseServer.run broadcasts and averages weights; FedProto must not use it.
        from .reporting import run_experiment
        return run_experiment(self)

    def _clients_training(self, sampled_clients):
        local_prototypes, round_results = [], {}
        server_optimizer = self.optimizer.state_dict()
        for client_idx in sampled_clients:
            client_idx = int(client_idx)
            self._set_client_data(client_idx)
            self.client.download_global(
                self.local_weights.get(client_idx, self.initial_weights),
                server_optimizer, self.global_prototypes,
            )
            local_results, _ = self.client.train()
            self.local_weights[client_idx] = self.client.upload_local()
            local_prototypes.append(self.client.upload_prototypes())
            round_results = self._results_updater(round_results, local_results)
            self.client.reset()
        # All clients in this round saw the SAME previous-round global prototypes.
        self.global_prototypes = aggregate_prototypes(local_prototypes)
        if not self.global_prototypes:
            raise ValueError("FedProto round produced no global prototypes")
        return round_results

    def _aggregation(self, w, ns):
        raise NotImplementedError("FedProto aggregates prototypes, not model parameters")

    def evaluate_clients(self, loader=None, local=False):
        """Evaluate all client states, including clients not sampled this round."""
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
                    self.local_classes[client_idx], self.num_classes, self.groups,
                    self.device, self.cfg["missing_class_distance"],
                )
                records.append(dict(client_idx=client_idx, **result))
        if local:
            return records
        return dict(
            prototype=average_client_metrics(records, "prototype"),
            classifier=average_client_metrics(records, "classifier"),
            clients=records,
        )

    def step_scheduler(self):
        if self.scheduler is not None:
            # The template has no gradients and is never trained or evaluated.
            # Its no-op step keeps PyTorch's optimizer/scheduler order valid.
            self.optimizer.zero_grad(set_to_none=True)
            self.optimizer.step()
            self.scheduler.step()
