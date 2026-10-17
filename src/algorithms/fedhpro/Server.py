import copy
import math

import torch
import wandb
from torch.utils.data import DataLoader, TensorDataset

from ..BaseServer import BaseServer
from .ClientTrainer import ClientTrainer
from .config import resolve_config
from .criterion import match_loss, match_meta
from .utils import classifier_module, replace_classifier

__all__ = ["Server"]


class Server(BaseServer):
    def __init__(
        self, algo_params, model, data_distributed, optimizer, scheduler, **kwargs
    ):
        super().__init__(
            algo_params, model, data_distributed, optimizer, scheduler, **kwargs
        )
        self.cfg = resolve_config(algo_params)
        classifier = classifier_module(model)
        self.feature_dim = classifier.in_features
        for name in ("num_classes", "n_clients", "n_rounds", "local_epochs"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError("FedHPro {} must be a positive integer".format(name))
        if self.num_classes < 2 or classifier.out_features != self.num_classes:
            raise ValueError("FedHPro requires at least two classes and a matching classifier")
        if not math.isfinite(self.sample_ratio) or not 0 < self.sample_ratio <= 1:
            raise ValueError("FedHPro sample_ratio must be in (0, 1]")
        if type(optimizer) is not torch.optim.SGD:
            raise ValueError("FedHPro uses SGD with the shared experiment settings")
        if len(optimizer.param_groups) != 1 or optimizer.state:
            raise ValueError("FedHPro requires one untrained optimizer parameter group")
        if set(data_distributed["local"]) != set(range(self.n_clients)):
            raise ValueError("FedHPro requires contiguous project client IDs")
        counts = torch.as_tensor(data_distributed["data_map"])
        if (counts.shape != (self.n_clients, self.num_classes)
                or not torch.isfinite(counts).all() or (counts < 0).any()
                or not torch.equal(counts, counts.long().to(counts.dtype))):
            raise ValueError("Invalid FedHPro client class counts")
        for client_idx, loaders in data_distributed["local"].items():
            loader = loaders["train"]
            if (loaders["datasize"] <= 0 or loader is None or len(loader) == 0
                    or len(loader.dataset) != loaders["datasize"]
                    or counts[client_idx].sum().item() != loaders["datasize"]):
                raise ValueError("Invalid FedHPro training data for client {}".format(client_idx))

        self.model.to(self.device)
        self.hyper_prototypes = torch.randn(
            self.num_classes * self.cfg["length_of_hp"], self.feature_dim,
            dtype=torch.float, requires_grad=True, device=self.device,
        )
        self.meta_prototypes = torch.randn(
            self.num_classes * self.cfg["meta_length"], self.feature_dim,
            dtype=torch.float, requires_grad=True, device=self.device,
        )
        self.hyper_labels = torch.arange(
            self.num_classes, dtype=torch.long, device=self.device
        ).repeat_interleave(self.cfg["length_of_hp"])
        self.meta_labels = torch.arange(
            self.num_classes, dtype=torch.long, device=self.device
        ).repeat_interleave(self.cfg["meta_length"])
        self.hyper_optimizer = torch.optim.SGD([self.hyper_prototypes], lr=self.cfg["lr_hp"])
        self.meta_optimizer = torch.optim.SGD([self.meta_prototypes], lr=self.cfg["lr_hp"])
        self.meta_model = torch.nn.Linear(self.feature_dim, self.num_classes).to(self.device)
        # Upstream creates a separate random classifier for first-round gradients.
        initial_meta = torch.nn.Linear(self.feature_dim, self.num_classes).to(self.device)
        self.meta_weights = copy.deepcopy(initial_meta.state_dict())
        self.client = ClientTrainer(
            cfg=self.cfg, algo_params=self.algo_params, model=copy.deepcopy(model),
            local_epochs=self.local_epochs, device=self.device, num_classes=self.num_classes,
        )
        self.global_meta_weights = None
        self.client_gradients = []
        self.client_meta_gradients = []
        self.prototype_metrics = {}

    def _clients_training(self, sampled_clients):
        if len(sampled_clients) == 0:
            raise ValueError("FedHPro requires at least one selected client")
        server_weights = copy.deepcopy(self.model.state_dict())
        server_optimizer = self.optimizer.state_dict()
        hyper_prototypes = self.hyper_prototypes.detach().clone()
        self.global_meta_weights = replace_classifier(
            copy.deepcopy(server_weights), self.meta_weights
        )
        self.client_gradients, self.client_meta_gradients = [], []
        updated_local_weights, client_sizes, round_results = [], [], {}
        for client_idx in sampled_clients:
            client_idx = int(client_idx)
            self._set_client_data(client_idx)
            gradients, meta_gradients = self.client.compute_gradient(
                self.global_meta_weights, self.data_distributed["data_map"][client_idx]
            )
            self.client_gradients.append(gradients)
            self.client_meta_gradients.append(meta_gradients)
            # Restore the original global head after collecting meta gradients.
            self.client.download_global(server_weights, server_optimizer, hyper_prototypes)
            local_results, local_size = self.client.train()
            # B: BaseClientTrainer.upload_local deep-copies each state_dict.
            # Upstream's reused Local returned aliased tensors instead.
            updated_local_weights.append(self.client.upload_local())
            client_sizes.append(local_size)
            round_results = self._results_updater(round_results, local_results)
            self.client.reset()
        return updated_local_weights, client_sizes, round_results

    def _aggregation(self, w, ns):
        if not w or len(w) != len(ns) or sum(ns) <= 0:
            raise ValueError("FedHPro cannot aggregate empty or inconsistent updates")
        if self.global_meta_weights is None or not self.client_meta_gradients:
            raise RuntimeError("Collect FedHPro client gradients before aggregation")
        aggregated = copy.deepcopy(w[0])
        # Preserve source multiply -> sum -> divide order for parameters/buffers.
        for name in w[0]:
            values = [weights[name] * size for weights, size in zip(w, ns)]
            aggregated[name] = sum(values) / sum(ns)
        self._update_meta_prototypes()
        self.meta_weights, aggregated = self._revise_classifier(aggregated)
        self.model.load_state_dict(aggregated)
        # Real gradients came from the PREVIOUS meta head, but synthetic HP
        # gradients use this NEWLY fitted head, exactly as in the source.
        self._update_hyper_prototypes()
        self.global_meta_weights = None
        self.client_gradients, self.client_meta_gradients = [], []
        return aggregated

    def _update_meta_prototypes(self):
        self.meta_model.load_state_dict({
            "weight": self.global_meta_weights["classifier.weight"],
            "bias": self.global_meta_weights["classifier.bias"],
        })
        self.meta_model.train()
        parameters = list(self.meta_model.parameters())
        gradients_by_class = {label: [] for label in range(self.num_classes)}
        for payload in self.client_meta_gradients:
            for label, gradients in payload.items():
                gradients_by_class[label].append(gradients)
        average = {label: [] for label in range(self.num_classes)}
        for label in range(self.num_classes):
            contributions = gradients_by_class[label]
            if len(contributions) != 0:
                weight = 1.0 / len(contributions)
                for param_idx in range(2):
                    values = [gradients[param_idx] * weight for gradients in contributions]
                    average[label].append(sum(values))
        if not any(average.values()):
            raise ValueError("FedHPro received no real classifier gradients")
        for _ in range(self.cfg["meta_rounds"]):
            loss = torch.tensor(0.0).to(self.device)
            for label in range(self.num_classes):
                if len(average[label]) != 0:
                    start = label * self.cfg["meta_length"]
                    end = (label + 1) * self.cfg["meta_length"]
                    features = self.meta_prototypes[start:end].reshape(
                        self.cfg["meta_length"], self.feature_dim
                    )
                    logits = self.meta_model(features)
                    loss_meta = self.criterion(logits, self.meta_labels[start:end])
                    gradients = torch.autograd.grad(loss_meta, parameters, create_graph=True)
                    loss += match_meta(gradients, average[label], self.cfg["dis_metric"])
            self.meta_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.meta_optimizer.step()
        self.prototype_metrics["fedhpro/meta_matching_loss"] = loss.detach().item()
        self.prototype_metrics["fedhpro/gradient_classes"] = sum(bool(values) for values in average.values())

    def _revise_classifier(self, aggregated):
        dataset = TensorDataset(
            self.meta_prototypes.detach().clone().float(), self.meta_labels.detach().clone()
        )
        # Fresh random initialization EVERY round; never warm-start this head.
        classifier = torch.nn.Linear(self.feature_dim, self.num_classes).to(self.device)
        optimizer = torch.optim.SGD(classifier.parameters(), lr=self.cfg["meta_lr"])
        loader = DataLoader(
            dataset, batch_size=self.cfg["meta_batch_size"], shuffle=True, num_workers=0
        )
        classifier.train()
        for _ in range(self.cfg["meta_epochs"]):
            for features, targets in loader:
                features, targets = features.to(self.device), targets.to(self.device)
                loss = self.criterion(classifier(features), targets)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        classifier.eval()
        weights = classifier.state_dict()
        aggregated = replace_classifier(aggregated, weights)
        self.prototype_metrics["fedhpro/meta_classifier_loss"] = loss.detach().item()
        return copy.deepcopy(weights), copy.deepcopy(aggregated)

    def _update_hyper_prototypes(self):
        self.model.eval()
        classifier = classifier_module(self.model)
        gradients_by_class = {label: [] for label in range(self.num_classes)}
        for payload in self.client_gradients:
            for label, gradient in payload.items():
                gradients_by_class[label].append(gradient)
        average = {}
        for label in range(self.num_classes):
            contributions = gradients_by_class[label]
            average[label] = (torch.stack(contributions, dim=0).mean(dim=0)
                              if len(contributions) > 0 else None)
        if not any(value is not None for value in average.values()):
            raise ValueError("FedHPro received no real feature gradients")
        for _ in range(self.cfg["rounds_of_hp"]):
            loss = torch.tensor(0.0).to(self.device)
            for label in range(self.num_classes):
                if average[label] is not None:
                    start = label * self.cfg["length_of_hp"]
                    end = (label + 1) * self.cfg["length_of_hp"]
                    features = self.hyper_prototypes[start:end].reshape(
                        self.cfg["length_of_hp"], self.feature_dim
                    )
                    logits = classifier(features)
                    loss_syn = self.criterion(logits, self.hyper_labels[start:end])
                    gradient = torch.autograd.grad(loss_syn, features, create_graph=True)[0]
                    loss += match_loss(gradient, average[label], self.cfg["dis_metric"])
            self.hyper_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.hyper_optimizer.step()
        self.prototype_metrics["fedhpro/hyper_matching_loss"] = loss.detach().item()

    def _wandb_logging(self, round_results, round_idx):
        super()._wandb_logging(round_results, round_idx)
        metrics = dict(self.prototype_metrics)
        metrics.update({
            "fedhpro/" + name: sum(values) / len(values)
            for name, values in round_results.items() if name.endswith("_loss")
        })
        wandb.log(metrics, step=round_idx)
