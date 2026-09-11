import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["BalancedPrototypeCalibrationLoss"]


class BalancedPrototypeCalibrationLoss(nn.Module):
    """Cross entropy with class-balanced prototype calibration."""

    def __init__(
        self,
        lambda_align=0.1,
        lambda_proto=0.5,
        proto_tau=0.5,
        class_balanced_loss=True,
        min_proto_classes=2,
    ):
        super(BalancedPrototypeCalibrationLoss, self).__init__()
        self.lambda_align = lambda_align
        self.lambda_proto = lambda_proto
        self.proto_tau = proto_tau
        self.class_balanced_loss = class_balanced_loss
        self.min_proto_classes = min_proto_classes

    def forward(
        self,
        logits,
        targets,
        features,
        global_prototypes,
        prototype_valid,
        return_raw=False,
    ):
        ce_loss = self._classification_loss(logits, targets)

        if global_prototypes is None or prototype_valid is None:
            zero = features.new_tensor(0.0)
            return self._format_output(ce_loss, ce_loss, zero, zero, return_raw)

        prototype_valid = prototype_valid.to(device=features.device, dtype=torch.bool)
        if not prototype_valid.any():
            zero = features.new_tensor(0.0)
            return self._format_output(ce_loss, ce_loss, zero, zero, return_raw)

        global_prototypes = global_prototypes.to(features.device)
        norm_features = F.normalize(features, dim=1)
        norm_prototypes = F.normalize(global_prototypes, dim=1)

        target_has_proto = prototype_valid[targets]
        align_loss = self._alignment_loss(
            norm_features, targets, norm_prototypes, target_has_proto
        )
        proto_loss = self._prototype_classification_loss(
            norm_features, targets, norm_prototypes, prototype_valid, target_has_proto
        )

        total_loss = (
            ce_loss
            + self.lambda_align * align_loss
            + self.lambda_proto * proto_loss
        )
        return self._format_output(
            total_loss, ce_loss, align_loss, proto_loss, return_raw
        )

    def _format_output(
        self, total_loss, ce_loss, align_loss, proto_loss, return_raw
    ):
        detached = {
            "ce_loss": ce_loss.detach(),
            "align_loss": align_loss.detach(),
            "proto_loss": proto_loss.detach(),
        }
        if not return_raw:
            return total_loss, detached

        raw = {
            "ce_loss": ce_loss,
            "align_loss": align_loss,
            "proto_loss": proto_loss,
        }
        return total_loss, detached, raw

    def _classification_loss(self, logits, targets):
        losses = F.cross_entropy(logits, targets, reduction="none")
        if not self.class_balanced_loss:
            return losses.mean()

        return self._balanced_mean_by_class(losses, targets)

    def _alignment_loss(self, norm_features, targets, norm_prototypes, target_has_proto):
        if not target_has_proto.any():
            return norm_features.new_tensor(0.0)

        valid_targets = targets[target_has_proto]
        target_prototypes = norm_prototypes[valid_targets]
        cosine_sim = (norm_features[target_has_proto] * target_prototypes).sum(dim=1)
        losses = 1.0 - cosine_sim

        return self._balanced_mean_by_class(losses, valid_targets)

    def _prototype_classification_loss(
        self, norm_features, targets, norm_prototypes, prototype_valid, target_has_proto
    ):
        if not target_has_proto.any():
            return norm_features.new_tensor(0.0)

        valid_classes = torch.nonzero(prototype_valid, as_tuple=False).view(-1)
        if valid_classes.numel() < self.min_proto_classes:
            return norm_features.new_tensor(0.0)

        proto_logits = (
            torch.matmul(norm_features[target_has_proto], norm_prototypes[valid_classes].t())
            / self.proto_tau
        )

        class_to_position = torch.full(
            (prototype_valid.numel(),),
            -1,
            dtype=torch.long,
            device=targets.device,
        )
        class_to_position[valid_classes] = torch.arange(
            valid_classes.numel(), dtype=torch.long, device=targets.device
        )
        proto_targets = class_to_position[targets[target_has_proto]]

        losses = F.cross_entropy(proto_logits, proto_targets, reduction="none")
        return self._balanced_mean_by_class(losses, targets[target_has_proto])

    def _balanced_mean_by_class(self, losses, targets):
        classes = torch.unique(targets)
        class_losses = []
        for class_idx in classes:
            class_losses.append(losses[targets == class_idx].mean())

        if len(class_losses) == 0:
            return losses.new_tensor(0.0)

        return torch.stack(class_losses).mean()
