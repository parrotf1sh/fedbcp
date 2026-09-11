import copy
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))

from algorithms.BaseClientTrainer import BaseClientTrainer

__all__ = ["ClientTrainer"]


class ClientTrainer(BaseClientTrainer):
    def __init__(
        self,
        criterion,
        min_confidence=0.0,
        min_proto_samples=1,
        confidence_logit_norm_gamma=0.0,
        wrong_prediction_weight=0.5,
        **kwargs
    ):
        super(ClientTrainer, self).__init__(**kwargs)
        self.criterion = criterion
        self.min_confidence = min_confidence
        self.min_proto_samples = min_proto_samples
        self.confidence_logit_norm_gamma = confidence_logit_norm_gamma
        self.wrong_prediction_weight = wrong_prediction_weight
        self.global_prototypes = None
        self.prototype_valid = None
        self.prototype_payload = None

    def train(self, collect_gradient_diagnostics=False):
        """Local training with balanced prototype calibration."""
        self.model.train()
        self.model.to(self.device)

        local_size = self.datasize
        loss_results = {
            "ce_loss": [],
            "align_loss": [],
            "proto_loss": [],
            "feature_norm": [],
        }
        gradient_diagnostics = {}
        gradient_diagnostics_collected = False

        for _ in range(self.local_epochs):
            for data, targets in self.trainloader:
                self.optimizer.zero_grad()

                data, targets = data.to(self.device), targets.to(self.device)
                logits, features = self._forward_with_features(data)
                should_collect_gradients = (
                    collect_gradient_diagnostics
                    and not gradient_diagnostics_collected
                )
                criterion_output = self.criterion(
                    logits,
                    targets,
                    features,
                    self.global_prototypes,
                    self.prototype_valid,
                    return_raw=should_collect_gradients,
                )
                if should_collect_gradients:
                    loss, loss_dict, raw_losses = criterion_output
                    gradient_diagnostics = self._measure_gradient_diagnostics(
                        raw_losses
                    )
                    gradient_diagnostics_collected = True
                else:
                    loss, loss_dict = criterion_output

                loss.backward()
                self.optimizer.step()

                for key, item in loss_dict.items():
                    loss_results[key].append(item.item())
                loss_results["feature_norm"].append(
                    torch.norm(features.detach(), p=2, dim=1).mean().item()
                )

        self.prototype_payload = self._build_prototype_payload()
        local_results = self._get_local_stats()
        local_results.update(self._summarize_losses(loss_results))
        local_results["mean_logit_norm"] = self.prototype_payload[
            "mean_logit_norm"
        ].item()
        local_results["mean_feature_norm_payload"] = self.prototype_payload[
            "mean_feature_norm"
        ].item()
        local_results.update(gradient_diagnostics)

        return local_results, local_size

    def download_global(
        self,
        server_weights,
        server_optimizer,
        global_prototypes=None,
        prototype_valid=None,
    ):
        """Load global model, optimizer, and current global prototypes."""
        self.model.load_state_dict(server_weights)
        self.optimizer.load_state_dict(server_optimizer)

        if global_prototypes is None:
            self.global_prototypes = None
            self.prototype_valid = None
        else:
            self.global_prototypes = copy.deepcopy(global_prototypes).to(self.device)
            self.prototype_valid = copy.deepcopy(prototype_valid).to(self.device)

    def upload_prototypes(self):
        """Upload prototype payload with reliability statistics."""
        return copy.deepcopy(self.prototype_payload)

    def reset(self):
        super().reset()
        self.global_prototypes = None
        self.prototype_valid = None
        self.prototype_payload = None

    def _forward_with_features(self, data):
        try:
            return self.model(data, get_features=True)
        except TypeError:
            raise NotImplementedError(
                "FedBPC requires a model forward(data, get_features=True) interface. "
                "Use fedavg_cifar, fedavg_tiny, or vgg11 for the first version."
            )

    @torch.no_grad()
    def _build_prototype_payload(self):
        self.model.eval()
        self.model.to(self.device)

        feature_sums = None
        confidence_sums = torch.zeros(self.num_classes, device=self.device)
        class_counts = torch.zeros(self.num_classes, device=self.device)
        logit_norm_sum = torch.tensor(0.0, device=self.device)
        feature_norm_sum = torch.tensor(0.0, device=self.device)
        logit_count = 0

        for data, targets in self.trainloader:
            data, targets = data.to(self.device), targets.to(self.device)
            logits, features = self._forward_with_features(data)
            feature_norm_sum += torch.norm(features, p=2, dim=1).sum()
            features = F.normalize(features, dim=1)
            confidence = self._calibrated_confidence(logits, targets)

            if feature_sums is None:
                feature_sums = torch.zeros(
                    self.num_classes, features.size(1), device=self.device
                )

            feature_sums.index_add_(0, targets, features)
            confidence_sums.index_add_(0, targets, confidence)
            class_counts.index_add_(0, targets, torch.ones_like(confidence))
            logit_norm_sum += torch.norm(logits, p=2, dim=1).sum()
            logit_count += logits.size(0)

        if feature_sums is None:
            feature_sums = torch.zeros(self.num_classes, 0, device=self.device)

        prototype_valid = class_counts >= self.min_proto_samples
        confidence = torch.zeros(self.num_classes, device=self.device)
        prototypes = torch.zeros_like(feature_sums)

        class_has_samples = class_counts > 0
        if class_has_samples.any():
            prototypes[class_has_samples] = feature_sums[class_has_samples] / class_counts[
                class_has_samples
            ].view(-1, 1)
            prototypes[class_has_samples] = F.normalize(
                prototypes[class_has_samples], dim=1
            )
            confidence[class_has_samples] = confidence_sums[
                class_has_samples
            ] / class_counts[class_has_samples]

        prototype_valid = prototype_valid & (confidence >= self.min_confidence)
        prototypes[~prototype_valid] = 0
        confidence[~prototype_valid] = 0
        mean_logit_norm = logit_norm_sum / max(logit_count, 1)
        mean_feature_norm = feature_norm_sum / max(logit_count, 1)

        return {
            "prototypes": prototypes.detach().cpu(),
            "prototype_valid": prototype_valid.detach().cpu(),
            "confidence": confidence.detach().cpu(),
            "class_counts": class_counts.detach().cpu(),
            "mean_logit_norm": mean_logit_norm.detach().cpu(),
            "mean_feature_norm": mean_feature_norm.detach().cpu(),
        }

    def _calibrated_confidence(self, logits, targets):
        probs = torch.softmax(logits, dim=1)
        target_confidence = probs.gather(1, targets.view(-1, 1)).view(-1)
        predicted = probs.argmax(dim=1)
        correct = predicted.eq(targets)

        correctness_weight = torch.ones_like(target_confidence)
        correctness_weight[~correct] = self.wrong_prediction_weight

        logit_norm = torch.norm(logits.detach(), p=2, dim=1)
        logit_norm_weight = torch.exp(
            -self.confidence_logit_norm_gamma * logit_norm
        )

        return target_confidence * correctness_weight * logit_norm_weight

    def _summarize_losses(self, loss_results):
        summarized = {}
        for key, values in loss_results.items():
            if len(values) == 0:
                summarized[key] = 0.0
            else:
                summarized[key] = sum(values) / len(values)

        return summarized

    def _measure_gradient_diagnostics(self, raw_losses):
        """Measure CE/auxiliary gradient interaction without changing .grad."""
        head_keyword = self.algo_params.get("head_agg", {}).get(
            "head_keyword", "classifier"
        )
        shared_parameters = [
            parameter
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad and head_keyword not in name
        ]
        if len(shared_parameters) == 0:
            return {}

        ce_grads = torch.autograd.grad(
            raw_losses["ce_loss"],
            shared_parameters,
            retain_graph=True,
            allow_unused=True,
        )
        ce_norm = self._gradient_norm(ce_grads)
        diagnostics = {"diag_grad_ce_norm": ce_norm.item()}

        weighted_align = self.criterion.lambda_align * raw_losses["align_loss"]
        weighted_proto = self.criterion.lambda_proto * raw_losses["proto_loss"]
        weighted_aux = weighted_align + weighted_proto

        for name, component in (
            ("align", weighted_align),
            ("proto", weighted_proto),
            ("aux", weighted_aux),
        ):
            if not component.requires_grad:
                diagnostics["diag_grad_{}_norm".format(name)] = 0.0
                continue

            component_grads = torch.autograd.grad(
                component,
                shared_parameters,
                retain_graph=True,
                allow_unused=True,
            )
            component_norm = self._gradient_norm(component_grads)
            diagnostics["diag_grad_{}_norm".format(name)] = component_norm.item()

            if name == "aux":
                dot_product = self._gradient_dot(ce_grads, component_grads)
                denominator = ce_norm * component_norm
                diagnostics["diag_grad_aux_to_ce_ratio"] = (
                    component_norm / ce_norm.clamp_min(1e-12)
                ).item()
                diagnostics["diag_grad_ce_aux_cosine"] = (
                    dot_product / denominator.clamp_min(1e-12)
                ).item()

        return diagnostics

    def _gradient_norm(self, gradients):
        squared_norm = None
        for gradient in gradients:
            if gradient is None:
                continue
            item = torch.sum(gradient.detach().float() ** 2)
            squared_norm = item if squared_norm is None else squared_norm + item

        if squared_norm is None:
            return torch.tensor(0.0, device=self.device)
        return torch.sqrt(squared_norm)

    def _gradient_dot(self, left_gradients, right_gradients):
        dot_product = None
        for left, right in zip(left_gradients, right_gradients):
            if left is None or right is None:
                continue
            item = torch.sum(left.detach().float() * right.detach().float())
            dot_product = item if dot_product is None else dot_product + item

        if dot_product is None:
            return torch.tensor(0.0, device=self.device)
        return dot_product
