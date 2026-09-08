import copy
import inspect
import os
import sys

import torch
import wandb

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))

from algorithms.BaseServer import BaseServer
from algorithms.fedbpc.ClientTrainer import ClientTrainer
from algorithms.fedbpc.criterion import BalancedPrototypeCalibrationLoss
from algorithms.fedbpc.utils import aggregate_prototypes

__all__ = ["Server"]


class Server(BaseServer):
    def __init__(
        self, algo_params, model, data_distributed, optimizer, scheduler=None, **kwargs
    ):
        super(Server, self).__init__(
            algo_params, model, data_distributed, optimizer, scheduler, **kwargs
        )
        self._validate_feature_interface(model)

        self.global_prototypes = None
        self.prototype_valid = None
        self.prototype_age = None
        self.prototype_metrics = {
            "prototype_coverage": 0.0,
            "prototype_confidence_mean": 0.0,
        }
        self._validate_algo_params(algo_params)
        self.head_agg_params = self._get_head_agg_params(algo_params)
        self.head_agg_metrics = {}
        self._last_logit_norms = []

        local_criterion = BalancedPrototypeCalibrationLoss(
            lambda_align=algo_params.lambda_align,
            lambda_proto=algo_params.lambda_proto,
            proto_tau=algo_params.proto_tau,
            class_balanced_loss=algo_params.get("class_balanced_loss", True),
            min_proto_classes=algo_params.get("min_proto_classes", 2),
        )
        self.client = ClientTrainer(
            local_criterion,
            min_confidence=algo_params.min_confidence,
            min_proto_samples=algo_params.get("min_proto_samples", 1),
            confidence_logit_norm_gamma=algo_params.get(
                "confidence_logit_norm_gamma", 0.0
            ),
            wrong_prediction_weight=algo_params.get("wrong_prediction_weight", 0.5),
            algo_params=algo_params,
            model=copy.deepcopy(model),
            local_epochs=self.local_epochs,
            device=self.device,
            num_classes=self.num_classes,
        )

        print("\n>>> FedBPC Server initialized...\n")

    def _clients_training(self, sampled_clients):
        """Conduct local training and get weights plus prototype payloads."""
        updated_local_weights, client_sizes = [], []
        prototype_payloads = []
        round_results = {}
        self._last_logit_norms = []

        server_weights = self.model.state_dict()
        server_optimizer = self.optimizer.state_dict()

        for client_idx in sampled_clients:
            self._set_client_data(client_idx)
            self.client.download_global(
                server_weights,
                server_optimizer,
                self.global_prototypes,
                self.prototype_valid,
            )

            local_results, local_size = self.client.train()
            prototype_payload = self.client.upload_prototypes()

            updated_local_weights.append(self.client.upload_local())
            prototype_payloads.append(prototype_payload)
            self._last_logit_norms.append(prototype_payload["mean_logit_norm"].item())
            round_results = self._results_updater(round_results, local_results)
            client_sizes.append(local_size)

            self.client.reset()

        (
            self.global_prototypes,
            self.prototype_valid,
            self.prototype_age,
            self.prototype_metrics,
        ) = aggregate_prototypes(
            prototype_payloads,
            self.num_classes,
            self.global_prototypes,
            self.prototype_valid,
            self.prototype_age,
            proto_momentum=self.algo_params.proto_momentum,
            count_shrinkage=self.algo_params.get("count_shrinkage", 5.0),
            min_proto_contributors=self.algo_params.get("min_proto_contributors", 1),
            max_prototype_age=self.algo_params.get("max_prototype_age", None),
        )

        return updated_local_weights, client_sizes, round_results

    def _aggregation(self, w, ns):
        """Average backbone normally and aggregate classifier head by reliability."""
        if not self.head_agg_params["enabled"]:
            return super()._aggregation(w, ns)

        head_keys = self._get_head_keys(w[0])
        if len(head_keys) == 0:
            return super()._aggregation(w, ns)

        backbone_prop = torch.tensor(ns, dtype=torch.float)
        backbone_prop /= torch.sum(backbone_prop)
        head_prop = self._get_head_aggregation_weights(w, ns, head_keys)

        w_avg = copy.deepcopy(w[0])
        for key in w_avg.keys():
            prop = head_prop if key in head_keys else backbone_prop
            w_avg[key] = self._weighted_average_parameter(w, key, prop)

        return copy.deepcopy(w_avg)

    def _wandb_logging(self, round_results, round_idx):
        super()._wandb_logging(round_results, round_idx)
        wandb.log(self.prototype_metrics, step=round_idx)
        if self.head_agg_metrics:
            wandb.log(self.head_agg_metrics, step=round_idx)

    def _validate_feature_interface(self, model):
        forward_signature = inspect.signature(model.forward)
        if "get_features" not in forward_signature.parameters:
            raise NotImplementedError(
                "FedBPC currently supports models with forward(..., get_features=True), "
                "such as fedavg_cifar, fedavg_tiny, and vgg11."
            )

    def _validate_algo_params(self, algo_params):
        if algo_params.proto_tau <= 0:
            raise ValueError("FedBPC proto_tau must be > 0.")
        if not 0 <= algo_params.proto_momentum <= 1:
            raise ValueError("FedBPC proto_momentum must be in [0, 1].")
        if not 0 <= algo_params.min_confidence <= 1:
            raise ValueError("FedBPC min_confidence must be in [0, 1].")
        if algo_params.get("min_proto_samples", 1) < 1:
            raise ValueError("FedBPC min_proto_samples must be >= 1.")
        if algo_params.get("count_shrinkage", 5.0) < 0:
            raise ValueError("FedBPC count_shrinkage must be >= 0.")
        if algo_params.get("min_proto_contributors", 1) < 1:
            raise ValueError("FedBPC min_proto_contributors must be >= 1.")
        if algo_params.get("max_prototype_age", None) is not None:
            if algo_params.max_prototype_age < 0:
                raise ValueError("FedBPC max_prototype_age must be >= 0.")
        if algo_params.get("confidence_logit_norm_gamma", 0.0) < 0:
            raise ValueError("FedBPC confidence_logit_norm_gamma must be >= 0.")
        if not 0 <= algo_params.get("wrong_prediction_weight", 0.5) <= 1:
            raise ValueError("FedBPC wrong_prediction_weight must be in [0, 1].")
        if algo_params.get("min_proto_classes", 2) < 1:
            raise ValueError("FedBPC min_proto_classes must be >= 1.")

    def _get_head_agg_params(self, algo_params):
        defaults = {
            "enabled": True,
            "gamma_logit": 0.5,
            "gamma_update": 0.5,
            "eps": 1e-8,
            "head_keyword": "classifier",
        }
        if "head_agg" not in algo_params:
            return defaults

        head_agg = algo_params.head_agg
        for key in defaults.keys():
            if key in head_agg:
                defaults[key] = head_agg[key]

        return defaults

    def _get_head_keys(self, weights):
        keyword = self.head_agg_params["head_keyword"]
        return [key for key in weights.keys() if keyword in key]

    def _get_head_aggregation_weights(self, local_weights, client_sizes, head_keys):
        eps = self.head_agg_params["eps"]
        logit_norms = torch.tensor(self._last_logit_norms, dtype=torch.float)
        update_norms = self._get_head_update_norms(local_weights, head_keys)

        logit_penalty = self._robust_positive_penalty(logit_norms, eps)
        update_penalty = self._robust_positive_penalty(update_norms, eps)
        reliability = torch.exp(
            -self.head_agg_params["gamma_logit"] * logit_penalty
            - self.head_agg_params["gamma_update"] * update_penalty
        )

        size_prop = torch.tensor(client_sizes, dtype=torch.float)
        raw_prop = size_prop * reliability
        if raw_prop.sum().item() <= eps:
            raw_prop = size_prop
        head_prop = raw_prop / raw_prop.sum()

        self.head_agg_metrics = {
            "head_reliability_mean": reliability.mean().item(),
            "head_reliability_min": reliability.min().item(),
            "head_logit_norm_mean": logit_norms.mean().item(),
            "head_update_norm_mean": update_norms.mean().item(),
        }

        return head_prop

    def _get_head_update_norms(self, local_weights, head_keys):
        eps = self.head_agg_params["eps"]
        server_weights = self.model.state_dict()
        update_norms = []

        for weights in local_weights:
            update_sq = torch.tensor(0.0)
            server_sq = torch.tensor(0.0)

            for key in head_keys:
                local_tensor = weights[key].detach().float().cpu()
                server_tensor = server_weights[key].detach().float().cpu()
                update_sq += torch.sum((local_tensor - server_tensor) ** 2)
                server_sq += torch.sum(server_tensor ** 2)

            update_norm = torch.sqrt(update_sq) / (torch.sqrt(server_sq) + eps)
            update_norms.append(update_norm)

        return torch.stack(update_norms)

    def _robust_positive_penalty(self, values, eps):
        if values.numel() <= 1:
            return torch.zeros_like(values)

        median = values.median()
        mad = torch.abs(values - median).median()
        z_scores = (values - median) / (mad + eps)

        return torch.clamp(z_scores, min=0.0)

    def _weighted_average_parameter(self, local_weights, key, prop):
        first_value = local_weights[0][key]
        if not torch.is_floating_point(first_value):
            return copy.deepcopy(first_value)

        weighted_value = torch.zeros_like(first_value)
        for idx, weights in enumerate(local_weights):
            scalar = prop[idx].to(device=weights[key].device, dtype=weights[key].dtype)
            weighted_value += weights[key] * scalar

        return weighted_value
