import copy
import time

from ..BaseServer import BaseServer
from .ClientTrainer import ClientTrainer
from .config import resolve_config, loss_weights
from .model import ModelWithFeatures, ModelWithProjection
from .utils import aggregate_prototypes

__all__ = ["Server"]


class Server(BaseServer):
    def __init__(
        self, algo_params, model, data_distributed, optimizer, scheduler, **kwargs
    ):
        super().__init__(
            algo_params, model, data_distributed, optimizer, scheduler, **kwargs
        )
        self.cfg = resolve_config(algo_params)
        if not isinstance(model, (ModelWithFeatures, ModelWithProjection)):
            raise ValueError("Use the FedProc model adapter before creating the optimizer")
        if self.cfg["use_project_head"] != isinstance(model, ModelWithProjection):
            raise ValueError("Configure the FedProc projection head before creating the optimizer")
        self.global_prototypes = None
        self.server_velocity = None
        self.prototype_seen_classes = set()
        self.prototype_total_bytes = 0
        self.prototype_total_samples = 0
        self.prototype_total_seconds = 0.0
        self.client = ClientTrainer(
            alpha_rounds=self.cfg["alpha_rounds"],
            optimizer_class=type(optimizer),
            algo_params=self.algo_params,
            model=copy.deepcopy(model),
            local_epochs=self.local_epochs,
            device=self.device,
            num_classes=self.num_classes,
        )

    def run(self):
        if self.cfg["metrics_only"]:
            from ..fedavg.reporting import run_metrics
            return run_metrics(self)
        return super().run()

    def _clients_training(self, sampled_clients):
        """Preserve initialization, local training, then prototype refresh order."""
        round_idx = len(self.server_results["client_history"]) - 1
        server_weights = self.model.state_dict()
        server_optimizer = self.optimizer.state_dict()
        initial_weights = [server_weights for _ in sampled_clients]
        if self.cfg["metrics_only"]:
            self.prototype_metrics = {
                "fedproc/prototype_phase_seconds": 0.0,
                "fedproc/prototype_processed_samples": 0,
                "fedproc/prototype_upload_bytes": 0,
                "fedproc/prototype_download_bytes": 0,
                "fedproc/prototype_initialization_upload_bytes": 0,
                "fedproc/prototype_refresh_upload_bytes": 0,
                "fedproc/prototype_initialization_classes": 0,
                "fedproc/prototype_refresh_classes": 0,
                "fedproc/prototype_preserved_classes": 0,
                "fedproc/prototype_uninitialized_classes": 0,
            }

        # First-round prototypes use the broadcast model on selected clients only.
        if self.global_prototypes is None:
            initial_weights = self._collect_prototypes(sampled_clients, initial_weights)

        if self.cfg["metrics_only"]:
            from ..fedavg.reporting import tensor_bytes
            self.prototype_metrics["fedproc/prototype_training_known_classes"] = len(self.prototype_seen_classes)
            self.prototype_metrics["fedproc/prototype_download_bytes"] = (
                len(sampled_clients) * tensor_bytes(self.global_prototypes))

        updated_local_weights, client_sizes = [], []
        round_results = {}
        for client_idx, weights in zip(sampled_clients, initial_weights):
            self._set_client_data(client_idx)
            self.client.download_global(
                weights, server_optimizer, self.global_prototypes, round_idx
            )
            local_results, local_size = self.client.train()
            updated_local_weights.append(self.client.upload_local())
            client_sizes.append(local_size)
            round_results = self._results_updater(round_results, local_results)
            self.client.reset()

        # Revisit each trained local model before aggregating model parameters.
        updated_local_weights = self._collect_prototypes(
            sampled_clients, updated_local_weights
        )
        if self.cfg["metrics_only"]:
            self.prototype_total_bytes += self.batch_communication_bytes()
            self.prototype_total_samples += self.prototype_metrics["fedproc/prototype_processed_samples"]
            self.prototype_total_seconds += self.prototype_metrics["fedproc/prototype_phase_seconds"]
        return updated_local_weights, client_sizes, round_results

    def _collect_prototypes(self, sampled_clients, local_weights):
        metrics_only = self.cfg["metrics_only"]
        initializing = self.global_prototypes is None
        if metrics_only:
            from ..fedavg.reporting import synchronize, tensor_bytes
            synchronize(self.device)
            started = time.perf_counter()
        payloads, updated_weights = [], []
        for client_idx, weights in zip(sampled_clients, local_weights):
            self._set_client_data(client_idx)
            self.client.model.load_state_dict(weights)
            payloads.append(self.client.upload_prototypes())
            # Prototype extraction also updates buffers in models with tracked BN.
            updated_weights.append(self.client.upload_local())
            self.client.reset()

        self.global_prototypes = aggregate_prototypes(
            payloads, self.num_classes, self.global_prototypes
        )
        if metrics_only:
            synchronize(self.device)
            elapsed = time.perf_counter() - started
            observed = set().union(*(payload["class_counts"] for payload in payloads))
            # Logical wire format: one int64 class ID and int64 sample count
            # per uploaded feature-sum vector; Python container overhead excluded.
            uploaded = sum(tensor_bytes(payload["feature_sums"]) + 16 * len(payload["class_counts"])
                           for payload in payloads)
            samples = sum(sum(payload["class_counts"].values()) for payload in payloads)
            phase = "initialization" if initializing else "refresh"
            self.prototype_metrics["fedproc/prototype_{}_classes".format(phase)] = len(observed)
            self.prototype_metrics["fedproc/prototype_{}_upload_bytes".format(phase)] += uploaded
            self.prototype_metrics["fedproc/prototype_upload_bytes"] += uploaded
            self.prototype_metrics["fedproc/prototype_processed_samples"] += samples
            self.prototype_metrics["fedproc/prototype_phase_seconds"] += elapsed
            self.prototype_metrics["fedproc/prototype_preserved_classes"] = len(self.prototype_seen_classes - observed)
            self.prototype_seen_classes.update(observed)
            self.prototype_metrics["fedproc/prototype_uninitialized_classes"] = self.num_classes - len(self.prototype_seen_classes)
            self.prototype_metrics["fedproc/prototype_cache_bytes"] = tensor_bytes(self.global_prototypes)
        return updated_weights

    def batch_communication_bytes(self):
        return (self.prototype_metrics["fedproc/prototype_upload_bytes"]
                + self.prototype_metrics["fedproc/prototype_download_bytes"])

    def batch_round_metrics(self, local):
        seen = sum(local["seen"])
        ce_weight, proto_weight = loss_weights(
            len(self.server_results["client_history"]) - 1, self.cfg["alpha_rounds"])
        metrics = dict(self.prototype_metrics)
        for name in ("ce_loss", "prototype_loss", "weighted_ce_loss", "weighted_prototype_loss"):
            metrics["train/" + name] = sum(local[name + "_sum"]) / seen
        metrics.update({
            "train/total_loss": sum(local["loss_sum"]) / seen,
            "fedproc/ce_weight": ce_weight,
            "fedproc/prototype_weight": proto_weight,
            "fedproc/prototype_cumulative_bytes": self.prototype_total_bytes,
            "fedproc/prototype_cumulative_samples": self.prototype_total_samples,
            "fedproc/prototype_cumulative_seconds": self.prototype_total_seconds,
        })
        return metrics

    def batch_summary_metrics(self, metrics):
        return {
            "final_train_ce_loss": metrics["train/ce_loss"],
            "final_train_prototype_loss": metrics["train/prototype_loss"],
            "final_train_total_loss": metrics["train/total_loss"],
            "prototype_communication_bytes": self.prototype_total_bytes,
            "prototype_processed_samples": self.prototype_total_samples,
            "prototype_phase_seconds": self.prototype_total_seconds,
            "aggregation": self.cfg["aggregation"],
            "use_project_head": self.cfg["use_project_head"],
            "out_dim": self.cfg["out_dim"],
            "model_parameter_count": sum(p.numel() for p in self.model.parameters()),
            "communication_definition": (
                "model upload + model download + optimizer tensor download per participant; "
                "plus prototype feature-sum uploads with int64 class IDs/counts and dense prototype downloads; "
                "includes first-round initialization and refresh uploads; clients retain their model between "
                "prototype extraction and training; excludes transport and simulator CPU/GPU transfers"),
        }

    def _aggregation(self, w, ns):
        if self.cfg["aggregation"] == "sampled":
            weights = super()._aggregation(w, ns)
        else:
            # Literal upstream behavior: positional client IDs and all-client total.
            # This intentionally retains its partial-participation weight mismatch.
            sizes = [
                self.data_distributed["local"][client_idx]["datasize"]
                for client_idx in range(self.n_clients)
            ]
            total_size = sum(sizes)
            weights = copy.deepcopy(w[0])
            for key in weights:
                weights[key] = w[0][key] * (sizes[0] / total_size)
                for client_idx in range(1, len(w)):
                    weights[key] += w[client_idx][key] * (sizes[client_idx] / total_size)

        momentum = self.cfg["server_momentum"]
        if momentum:
            old_weights = self.model.state_dict()
            if self.server_velocity is None:
                self.server_velocity = {key: 0 for key in weights}
            for key in weights:
                delta = old_weights[key] - weights[key]
                self.server_velocity[key] = (
                    momentum * self.server_velocity[key] + (1 - momentum) * delta
                )
                weights[key] = old_weights[key] - self.server_velocity[key]

        return weights
