import copy
import inspect
import math
import os
import random
import sys

import numpy as np
import torch
import wandb

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))

from algorithms.BaseServer import BaseServer
from algorithms.measures import evaluate_model
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
        self.singleton_proto_update_params = (
            self._get_singleton_proto_update_params(algo_params)
        )
        self.diagnostic_params = self._get_diagnostic_params(algo_params)
        self.head_agg_metrics = {}
        self._last_logit_norms = []
        self._current_round_idx = None
        self._last_sampled_clients = []
        self._sampling_metrics = {}
        self._client_diagnostic_records = []
        self._last_head_diagnostics = {}
        self._shadow_weights = None
        self._shadow_model = (
            copy.deepcopy(model) if self.diagnostic_params["enabled"] else None
        )

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

    def _client_sampling(self, round_idx):
        sampled_clients = super()._client_sampling(round_idx)
        self._current_round_idx = round_idx
        self._last_sampled_clients = [int(item) for item in sampled_clients]
        self._sampling_metrics = self._get_sampling_metrics(sampled_clients)
        return sampled_clients

    def _clients_training(self, sampled_clients):
        """Conduct local training and get weights plus prototype payloads."""
        updated_local_weights, client_sizes = [], []
        prototype_payloads = []
        round_results = {}
        self._last_logit_norms = []
        self._client_diagnostic_records = []
        self._last_head_diagnostics = {}

        server_weights = self.model.state_dict()
        server_optimizer = self.optimizer.state_dict()

        for client_slot, client_idx in enumerate(sampled_clients):
            self._set_client_data(client_idx)
            self.client.download_global(
                server_weights,
                server_optimizer,
                self.global_prototypes,
                self.prototype_valid,
            )

            collect_gradients = self._should_collect_gradient_diagnostics(
                client_slot
            )
            local_results, local_size = self.client.train(
                collect_gradient_diagnostics=collect_gradients
            )
            prototype_payload = self.client.upload_prototypes()

            updated_local_weights.append(self.client.upload_local())
            prototype_payloads.append(prototype_payload)
            self._last_logit_norms.append(prototype_payload["mean_logit_norm"].item())
            round_results = self._results_updater(round_results, local_results)
            client_sizes.append(local_size)
            self._client_diagnostic_records.append(
                self._build_client_diagnostic_record(
                    client_slot,
                    client_idx,
                    local_size,
                    local_results,
                    prototype_payload,
                )
            )

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
            singleton_update_enabled=self.singleton_proto_update_params["enabled"],
            singleton_update_scale=self.singleton_proto_update_params[
                "update_scale"
            ],
        )

        return updated_local_weights, client_sizes, round_results

    def _aggregation(self, w, ns):
        """Average backbone normally and aggregate classifier head by reliability."""
        if self.diagnostic_params["enabled"]:
            shadow_weights = super()._aggregation(w, ns)
            self._shadow_weights = self._state_dict_to_cpu(shadow_weights)

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

        if self.diagnostic_params["enabled"] and self._shadow_weights is not None:
            self.head_agg_metrics.update(
                self._get_head_counterfactual_parameter_metrics(
                    w_avg, self._shadow_weights, head_keys
                )
            )

        return copy.deepcopy(w_avg)

    def _wandb_logging(self, round_results, round_idx):
        super()._wandb_logging(round_results, round_idx)
        wandb.log(self.prototype_metrics, step=round_idx)
        if self.head_agg_metrics:
            wandb.log(self.head_agg_metrics, step=round_idx)
        if self.diagnostic_params["enabled"]:
            diagnostic_metrics = self._collect_diagnostic_metrics(round_results)
            prototype_alignment_metrics = (
                self._evaluate_server_prototype_alignment(round_idx)
            )
            shadow_metrics = self._evaluate_shadow_counterfactual(round_idx)
            diagnostic_metrics.update(prototype_alignment_metrics)
            diagnostic_metrics.update(shadow_metrics)
            wandb.log(diagnostic_metrics, step=round_idx)

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
        singleton_update = algo_params.get("singleton_proto_update", {})
        singleton_update_enabled = singleton_update.get("enabled", False)
        singleton_update_scale = singleton_update.get("update_scale", 2.0 / 3.0)
        if not 0 < singleton_update_scale <= 1:
            raise ValueError(
                "FedBPC singleton_proto_update.update_scale must be in (0, 1]."
            )
        if (
            singleton_update_enabled
            and algo_params.get("min_proto_contributors", 1) != 1
        ):
            raise ValueError(
                "FedBPC singleton prototype updates require "
                "min_proto_contributors == 1."
            )
        if algo_params.get("max_prototype_age", None) is not None:
            if algo_params.max_prototype_age < 0:
                raise ValueError("FedBPC max_prototype_age must be >= 0.")
        if algo_params.get("confidence_logit_norm_gamma", 0.0) < 0:
            raise ValueError("FedBPC confidence_logit_norm_gamma must be >= 0.")
        if not 0 <= algo_params.get("wrong_prediction_weight", 0.5) <= 1:
            raise ValueError("FedBPC wrong_prediction_weight must be in [0, 1].")
        if algo_params.get("min_proto_classes", 2) < 1:
            raise ValueError("FedBPC min_proto_classes must be >= 1.")
        diagnostics = algo_params.get("diagnostics", {})
        if diagnostics.get("shadow_eval_interval", 1) < 1:
            raise ValueError("FedBPC diagnostics.shadow_eval_interval must be >= 1.")
        if diagnostics.get("prototype_eval_interval", 1) < 1:
            raise ValueError("FedBPC diagnostics.prototype_eval_interval must be >= 1.")
        if diagnostics.get("prototype_eval_batches", 10) < 1:
            raise ValueError("FedBPC diagnostics.prototype_eval_batches must be >= 1.")

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

    def _get_singleton_proto_update_params(self, algo_params):
        defaults = {
            "enabled": False,
            "update_scale": 2.0 / 3.0,
        }
        singleton_update = algo_params.get("singleton_proto_update", {})
        for key in defaults.keys():
            if key in singleton_update:
                defaults[key] = singleton_update[key]

        return defaults

    def _get_diagnostic_params(self, algo_params):
        defaults = {
            "enabled": False,
            "shadow_eval_interval": 1,
            "prototype_eval_interval": 1,
            "prototype_eval_batches": 10,
            "gradient_rounds": [
                1,
                49,
                99,
                149,
                max(self.n_rounds - 1, 0),
            ],
            "gradient_client_slots": [0],
            "log_client_slots": True,
        }
        diagnostics = algo_params.get("diagnostics", {})
        for key in defaults.keys():
            if key in diagnostics:
                defaults[key] = diagnostics[key]

        defaults["gradient_rounds"] = {
            int(item) for item in defaults["gradient_rounds"]
        }
        defaults["gradient_client_slots"] = {
            int(item) for item in defaults["gradient_client_slots"]
        }
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
            "head_weight_max": head_prop.max().item(),
            "head_weight_min": head_prop.min().item(),
            "head_weight_entropy": self._normalized_entropy(head_prop),
            "head_effective_clients": (1.0 / torch.sum(head_prop ** 2)).item(),
        }
        self._last_head_diagnostics = {
            "head_prop": head_prop.detach().cpu(),
            "reliability": reliability.detach().cpu(),
            "logit_penalty": logit_penalty.detach().cpu(),
            "update_penalty": update_penalty.detach().cpu(),
            "logit_norm": logit_norms.detach().cpu(),
            "update_norm": update_norms.detach().cpu(),
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

    def _should_collect_gradient_diagnostics(self, client_slot):
        return (
            self.diagnostic_params["enabled"]
            and self._current_round_idx
            in self.diagnostic_params["gradient_rounds"]
            and client_slot in self.diagnostic_params["gradient_client_slots"]
        )

    def _get_sampling_metrics(self, sampled_clients):
        client_maps = torch.as_tensor(
            self.data_distributed["data_map"][sampled_clients], dtype=torch.float
        )
        sample_counts = client_maps.sum(dim=0)
        client_counts = (client_maps > 0).float().sum(dim=0)
        present = sample_counts > 0
        probabilities = sample_counts / sample_counts.sum().clamp_min(1.0)
        nonzero_probabilities = probabilities[present]
        entropy = -torch.sum(
            nonzero_probabilities * torch.log(nonzero_probabilities)
        )
        if self.num_classes > 1:
            entropy /= math.log(self.num_classes)

        metrics = {
            "sampled_class_coverage": present.float().mean().item(),
            "sampled_class_count": present.float().sum().item(),
            "sampled_missing_class_count": (~present).float().sum().item(),
            "sampled_class_entropy": entropy.item(),
            "sampled_class_sample_count_std": sample_counts.std(
                unbiased=False
            ).item(),
            "sampled_class_client_count_std": client_counts.std(
                unbiased=False
            ).item(),
            "sampled_class_client_count_max": client_counts.max().item(),
            "sampled_class_client_count_min": client_counts.min().item(),
        }
        for class_idx in range(self.num_classes):
            prefix = "diag_sampled_class_{:03d}".format(class_idx)
            metrics[prefix + "_samples"] = sample_counts[class_idx].item()
            metrics[prefix + "_clients"] = client_counts[class_idx].item()

        return metrics

    def _build_client_diagnostic_record(
        self,
        client_slot,
        client_idx,
        local_size,
        local_results,
        prototype_payload,
    ):
        class_counts = torch.as_tensor(
            self.data_distributed["data_map"][int(client_idx)], dtype=torch.float
        )
        present_classes = torch.nonzero(
            class_counts > 0, as_tuple=False
        ).view(-1)
        sorted_classes = torch.argsort(class_counts, descending=True)

        record = {
            "slot": int(client_slot),
            "client_id": int(client_idx),
            "local_size": int(local_size),
            "num_classes": int(present_classes.numel()),
            "primary_class": int(sorted_classes[0].item())
            if sorted_classes.numel() > 0
            else -1,
            "secondary_class": int(sorted_classes[1].item())
            if present_classes.numel() > 1
            else -1,
            "valid_prototypes": int(
                prototype_payload["prototype_valid"].sum().item()
            ),
        }
        for key in (
            "train_acc",
            "test_acc",
            "ce_loss",
            "align_loss",
            "proto_loss",
            "feature_norm",
            "mean_logit_norm",
            "mean_feature_norm_payload",
            "diag_grad_ce_norm",
            "diag_grad_align_norm",
            "diag_grad_proto_norm",
            "diag_grad_aux_norm",
            "diag_grad_aux_to_ce_ratio",
            "diag_grad_ce_aux_cosine",
        ):
            if key in local_results:
                record[key] = float(local_results[key])

        return record

    def _collect_diagnostic_metrics(self, round_results):
        metrics = dict(self._sampling_metrics)
        for key in (
            "ce_loss",
            "align_loss",
            "proto_loss",
            "feature_norm",
            "mean_logit_norm",
            "mean_feature_norm_payload",
        ):
            values = round_results.get(key, [])
            if len(values) > 0:
                tensor = torch.tensor(values, dtype=torch.float)
                metrics["diag_{}_mean".format(key)] = tensor.mean().item()
                metrics["diag_{}_std".format(key)] = tensor.std(
                    unbiased=False
                ).item()

        ce_mean = metrics.get("diag_ce_loss_mean", 0.0)
        weighted_align = (
            self.algo_params.lambda_align
            * metrics.get("diag_align_loss_mean", 0.0)
        )
        weighted_proto = (
            self.algo_params.lambda_proto
            * metrics.get("diag_proto_loss_mean", 0.0)
        )
        metrics["diag_weighted_align_loss_mean"] = weighted_align
        metrics["diag_weighted_proto_loss_mean"] = weighted_proto
        metrics["diag_aux_to_ce_loss_ratio"] = (
            (weighted_align + weighted_proto) / max(ce_mean, 1e-12)
        )

        for key in (
            "diag_grad_ce_norm",
            "diag_grad_align_norm",
            "diag_grad_proto_norm",
            "diag_grad_aux_norm",
            "diag_grad_aux_to_ce_ratio",
            "diag_grad_ce_aux_cosine",
        ):
            values = round_results.get(key, [])
            if len(values) > 0:
                metrics[key] = float(sum(values) / len(values))

        if self.diagnostic_params["log_client_slots"]:
            metrics.update(self._get_client_slot_metrics())

        return metrics

    def _get_client_slot_metrics(self):
        metrics = {}
        head_diagnostics = self._last_head_diagnostics
        for record in self._client_diagnostic_records:
            slot = record["slot"]
            prefix = "diag_client_{:02d}".format(slot)
            for key, value in record.items():
                if key == "slot":
                    continue
                metrics[prefix + "_" + key] = value

            for key, values in head_diagnostics.items():
                if slot < values.numel():
                    metrics[prefix + "_" + key] = values[slot].item()

        return metrics

    def _evaluate_shadow_counterfactual(self, round_idx):
        if self._shadow_weights is None or self._shadow_model is None:
            return {}
        if round_idx % self.diagnostic_params["shadow_eval_interval"] != 0:
            self._shadow_weights = None
            return {}

        rng_state = self._capture_rng_state()

        try:
            self._shadow_model.load_state_dict(self._shadow_weights)
            shadow_accuracy = evaluate_model(
                self._shadow_model, self.testloader, device=self.device
            )
        finally:
            self._shadow_model.to("cpu")
            self._restore_rng_state(rng_state)
            self._shadow_weights = None

        actual_accuracy = self.server_results["test_accuracy"][-1]
        return {
            "shadow_fedavg_head_eval_ran": 1.0,
            "shadow_fedavg_head_test_acc": shadow_accuracy,
            "shadow_fedavg_head_minus_actual_acc": shadow_accuracy
            - actual_accuracy,
        }

    @torch.no_grad()
    def _evaluate_server_prototype_alignment(self, round_idx):
        if self.global_prototypes is None or self.prototype_valid is None:
            return {}
        if round_idx % self.diagnostic_params["prototype_eval_interval"] != 0:
            return {}

        rng_state = self._capture_rng_state()
        feature_sums = None
        class_counts = torch.zeros(self.num_classes, device=self.device)
        prototype_correct = 0
        sample_count = 0

        self.model.eval()
        self.model.to(self.device)
        valid_classes = torch.nonzero(
            self.prototype_valid.to(self.device), as_tuple=False
        ).view(-1)
        valid_prototypes = torch.nn.functional.normalize(
            self.global_prototypes.to(self.device)[valid_classes], dim=1
        )

        try:
            for batch_idx, (data, targets) in enumerate(self.testloader):
                if batch_idx >= self.diagnostic_params["prototype_eval_batches"]:
                    break
                data, targets = data.to(self.device), targets.to(self.device)
                _, features = self.model(data, get_features=True)
                features = torch.nn.functional.normalize(features, dim=1)

                if feature_sums is None:
                    feature_sums = torch.zeros(
                        self.num_classes, features.size(1), device=self.device
                    )
                feature_sums.index_add_(0, targets, features)
                class_counts.index_add_(
                    0, targets, torch.ones(targets.size(0), device=self.device)
                )

                if valid_classes.numel() > 0:
                    similarities = torch.matmul(features, valid_prototypes.t())
                    predictions = valid_classes[similarities.argmax(dim=1)]
                    prototype_correct += predictions.eq(targets).sum().item()
                sample_count += targets.size(0)
        finally:
            self._restore_rng_state(rng_state)

        if feature_sums is None:
            return {}

        centroids = torch.zeros_like(feature_sums)
        has_samples = class_counts > 0
        centroids[has_samples] = feature_sums[has_samples] / class_counts[
            has_samples
        ].view(-1, 1)
        centroids[has_samples] = torch.nn.functional.normalize(
            centroids[has_samples], dim=1
        )
        comparable = has_samples & self.prototype_valid.to(self.device)
        comparable_classes = torch.nonzero(comparable, as_tuple=False).view(-1)
        centroid_cosines = torch.zeros(self.num_classes, device=self.device)
        centroid_margins = torch.zeros(self.num_classes, device=self.device)
        margin_valid = torch.zeros(
            self.num_classes, dtype=torch.bool, device=self.device
        )

        if comparable_classes.numel() > 0:
            normalized_global = torch.nn.functional.normalize(
                self.global_prototypes.to(self.device), dim=1
            )
            centroid_cosines[comparable_classes] = torch.sum(
                centroids[comparable_classes]
                * normalized_global[comparable_classes],
                dim=1,
            )

            if valid_classes.numel() > 1:
                similarities = torch.matmul(
                    centroids[comparable_classes],
                    normalized_global[valid_classes].t(),
                )
                class_to_position = torch.full(
                    (self.num_classes,),
                    -1,
                    dtype=torch.long,
                    device=self.device,
                )
                class_to_position[valid_classes] = torch.arange(
                    valid_classes.numel(), device=self.device
                )
                own_positions = class_to_position[comparable_classes]
                own_similarities = similarities[
                    torch.arange(comparable_classes.numel(), device=self.device),
                    own_positions,
                ]
                similarities[
                    torch.arange(comparable_classes.numel(), device=self.device),
                    own_positions,
                ] = -float("inf")
                centroid_margins[comparable_classes] = (
                    own_similarities - similarities.max(dim=1)[0]
                )
                margin_valid[comparable_classes] = True

        metrics = {
            "prototype_server_eval_sample_count": float(sample_count),
            "prototype_nearest_class_acc": prototype_correct
            / max(sample_count, 1),
            "prototype_server_centroid_cosine_mean": centroid_cosines[
                comparable
            ].mean().item()
            if comparable.any()
            else 0.0,
            "prototype_server_centroid_cosine_min": centroid_cosines[
                comparable
            ].min().item()
            if comparable.any()
            else 0.0,
            "prototype_server_centroid_margin_mean": centroid_margins[
                margin_valid
            ].mean().item()
            if margin_valid.any()
            else 0.0,
            "prototype_server_centroid_margin_min": centroid_margins[
                margin_valid
            ].min().item()
            if margin_valid.any()
            else 0.0,
        }
        for class_idx in range(self.num_classes):
            prefix = "diag_proto_class_{:03d}".format(class_idx)
            metrics[prefix + "_server_sample_count"] = class_counts[
                class_idx
            ].item()
            metrics[prefix + "_server_centroid_valid"] = comparable[
                class_idx
            ].float().item()
            metrics[prefix + "_server_centroid_cosine"] = centroid_cosines[
                class_idx
            ].item()
            metrics[prefix + "_server_centroid_margin"] = centroid_margins[
                class_idx
            ].item()

        return metrics

    def _get_head_counterfactual_parameter_metrics(
        self, actual_weights, shadow_weights, head_keys
    ):
        difference_sq = torch.tensor(0.0)
        shadow_sq = torch.tensor(0.0)
        for key in head_keys:
            actual = actual_weights[key].detach().float().cpu()
            shadow = shadow_weights[key].detach().float().cpu()
            difference_sq += torch.sum((actual - shadow) ** 2)
            shadow_sq += torch.sum(shadow ** 2)

        return {
            "head_shadow_delta_norm": torch.sqrt(difference_sq).item(),
            "head_shadow_relative_delta_norm": (
                torch.sqrt(difference_sq) / torch.sqrt(shadow_sq).clamp_min(1e-12)
            ).item(),
        }

    def _state_dict_to_cpu(self, state_dict):
        return {
            key: value.detach().cpu().clone()
            if torch.is_tensor(value)
            else copy.deepcopy(value)
            for key, value in state_dict.items()
        }

    def _normalized_entropy(self, probabilities):
        if probabilities.numel() <= 1:
            return 0.0
        positive = probabilities[probabilities > 0]
        entropy = -torch.sum(positive * torch.log(positive))
        return (entropy / math.log(probabilities.numel())).item()

    def _capture_rng_state(self):
        return {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None,
        }

    def _restore_rng_state(self, state):
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"])
        if state["cuda"] is not None:
            torch.cuda.set_rng_state_all(state["cuda"])
