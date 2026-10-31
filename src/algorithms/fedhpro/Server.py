import copy
import math

import torch
import experiment_tracker as tracking
from torch.utils.data import DataLoader, TensorDataset

from ..BaseServer import BaseServer
from .ClientTrainer import ClientTrainer
from .config import resolve_config
from .batch import stage_timer, sparse_payload_bytes
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
        # Capture once, before any DSA resets Torch's global RNG.
        self.training_seed = torch.initial_seed()
        self.stage_seconds = {}
        self.total_stage_seconds = {}
        self.total_extra_communication = 0
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

    def run(self):
        if self.cfg["metrics_only"]:
            if not getattr(self, "experiment_config", {}).get("batch_protocol"):
                raise ValueError("FedHPro metrics_only requires batch_protocol")
            from ..fedavg.reporting import run_metrics
            return run_metrics(self)
        return super().run()

    def _clients_training(self, sampled_clients):
        if len(sampled_clients) == 0:
            raise ValueError("FedHPro requires at least one selected client")
        round_idx = len(self.server_results["client_history"]) - 1
        if round_idx < 0:
            raise RuntimeError("Record sampled clients before FedHPro local training")
        self.stage_seconds = {}
        server_weights = copy.deepcopy(self.model.state_dict())
        server_optimizer = self.optimizer.state_dict()
        hyper_prototypes = self.hyper_prototypes.detach().clone()
        self.global_meta_weights = replace_classifier(
            copy.deepcopy(server_weights), self.meta_weights
        )
        self.client_gradients, self.client_meta_gradients = [], []
        updated_local_weights, client_sizes, round_results = [], [], {}
        if self.cfg["metrics_only"]:
            from ..fedavg.reporting import tensor_bytes
            self.communication_metrics = {
                "fedhpro/hyper_download_bytes": len(sampled_clients) * tensor_bytes(hyper_prototypes),
                "fedhpro/meta_head_download_bytes": len(sampled_clients) * tensor_bytes(self.meta_weights),
                "fedhpro/feature_gradient_upload_bytes": 0,
                "fedhpro/head_gradient_upload_bytes": 0,
            }
        for client_idx in sampled_clients:
            client_idx = int(client_idx)
            self._set_client_data(client_idx)
            self.client.dsa_context = (self.training_seed, round_idx, client_idx)
            with stage_timer(self.stage_seconds, "real_gradient_seconds", self.device, self.cfg["metrics_only"]):
                gradients, meta_gradients = self.client.compute_gradient(
                    self.global_meta_weights, self.data_distributed["data_map"][client_idx]
                )
            if self.cfg["metrics_only"]:
                self.communication_metrics["fedhpro/feature_gradient_upload_bytes"] += sparse_payload_bytes(gradients)
                self.communication_metrics["fedhpro/head_gradient_upload_bytes"] += sparse_payload_bytes(meta_gradients)
            self.client_gradients.append(gradients)
            self.client_meta_gradients.append(meta_gradients)
            # Restore the original global head after collecting meta gradients.
            self.client.download_global(server_weights, server_optimizer, hyper_prototypes)
            with stage_timer(self.stage_seconds, "local_train_seconds", self.device, self.cfg["metrics_only"]):
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
        with stage_timer(self.stage_seconds, "meta_matching_seconds", self.device, self.cfg["metrics_only"]):
            self._update_meta_prototypes()
        with stage_timer(self.stage_seconds, "classifier_revision_seconds", self.device, self.cfg["metrics_only"]):
            self.meta_weights, aggregated = self._revise_classifier(aggregated)
        self.model.load_state_dict(aggregated)
        # Real gradients came from the PREVIOUS meta head, but synthetic HP
        # gradients use this NEWLY fitted head, exactly as in the source.
        with stage_timer(self.stage_seconds, "hyper_matching_seconds", self.device, self.cfg["metrics_only"]):
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

    def _tracking_logging(self, round_results, round_idx):
        super()._tracking_logging(round_results, round_idx)
        metrics = dict(self.prototype_metrics)
        metrics.update({
            "fedhpro/" + name: sum(values) / len(values)
            for name, values in round_results.items() if name.endswith("_loss")
        })
        tracking.log(metrics, step=round_idx)

    def batch_communication_bytes(self):
        # Common reporter already counts full model up/down and optimizer tensors.
        # The shared backbone in global_meta_weights is reconstructed locally.
        return sum(self.communication_metrics.values())

    def batch_round_metrics(self, local):
        from ..fedavg.reporting import tensor_bytes
        seen = sum(local["seen"])
        metrics = {"train/" + name: sum(local[name + "_sum"]) / seen
                   for name in ("ce_loss", "hpcl_loss", "hpal_loss", "total_loss")}
        for name, value in self.prototype_metrics.items():
            # These are last inner-step losses, not averages over an entire stage.
            metrics[name + "_last_step" if name.endswith("_loss") else name] = value
        metrics.update(self.communication_metrics)
        for name, value in self.stage_seconds.items():
            metrics["time/fedhpro_" + name] = value
            self.total_stage_seconds[name] = self.total_stage_seconds.get(name, 0.0) + value
        self.total_extra_communication += self.batch_communication_bytes()
        metrics.update({
            "fedhpro/extra_communication_cumulative_bytes": self.total_extra_communication,
            "fedhpro/hyper_prototype_bytes": tensor_bytes(self.hyper_prototypes),
            "fedhpro/meta_prototype_bytes": tensor_bytes(self.meta_prototypes),
            "fedhpro/classifier_updates_per_round": self.cfg["meta_epochs"] * math.ceil(
                self.num_classes * self.cfg["meta_length"] / self.cfg["meta_batch_size"]),
        })
        return metrics

    def batch_summary_metrics(self, metrics):
        summary = {"final_train_" + name: metrics["train/" + name]
                   for name in ("ce_loss", "hpcl_loss", "hpal_loss", "total_loss")}
        summary.update({
            "fedhpro_implementation": "independent_client_snapshot_B",
            "dsa_seed_mode": self.cfg["dsa_seed_mode"],
            "training_seed_for_dsa": self.training_seed,
            "meta_rounds": self.cfg["meta_rounds"],
            "meta_epochs": self.cfg["meta_epochs"],
            "rounds_of_hp": self.cfg["rounds_of_hp"],
            "extra_communication_bytes": self.total_extra_communication,
            "stage_seconds": dict(self.total_stage_seconds),
            "model_parameter_count": sum(p.numel() for p in self.model.parameters()),
            "communication_definition": (
                "full global model down + client model up + optimizer tensor down per participant; "
                "add old meta head and hyper-prototype down, per-class feature/head gradient up; "
                "each sparse gradient dictionary includes int64 class IDs; shared backbone sent once; "
                "meta-prototypes remain server-local; excludes control metadata, transport overhead "
                "and simulator CPU/GPU transfers"),
        })
        for name, value in metrics.items():
            if name.startswith("fedhpro/") and name.endswith("_loss_last_step"):
                summary["final_" + name.replace("/", "_")] = value
        return summary
