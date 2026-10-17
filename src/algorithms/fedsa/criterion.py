import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["SemanticAnchorLoss"]


def local_gap(local_prototypes):
    """Use this client's previous participation, including None/singleton cases."""
    if local_prototypes is None:
        return 0.0
    if len(local_prototypes) <= 1:
        return float("-inf")
    prototypes = torch.stack(list(local_prototypes.values()))
    return global_gap(prototypes)


def global_gap(semantic_anchors):
    distances = torch.cdist(semantic_anchors, semantic_anchors, p=2)
    distances.fill_diagonal_(float("inf"))
    return torch.mean(torch.min(distances, dim=1).values)


class SemanticAnchorLoss(nn.Module):
    def __init__(self, lambda_anchor, lambda_mcl, lambda_cc):
        super().__init__()
        self.lambda_anchor = lambda_anchor
        self.lambda_mcl = lambda_mcl
        self.lambda_cc = lambda_cc
        self.classification_loss = nn.CrossEntropyLoss()
        self.anchor_loss = nn.MSELoss()

    def forward(self, logits, features, targets, classifier, semantic_anchors,
                local_margin, global_margin):
        ce_loss = self.classification_loss(logits, targets)

        anchor_labels = torch.arange(semantic_anchors.shape[0], device=features.device)
        anchor_logits = classifier(semantic_anchors)
        cc_loss = self.classification_loss(anchor_logits, anchor_labels)
        anchor_class_losses = F.cross_entropy(anchor_logits, anchor_labels, reduction="none")
        anchors_loss = cc_loss * self.lambda_cc

        # Same label indexing as the upstream concatenation, without copying
        # each scalar label through a new tensor/device.
        anchor_loss = self.anchor_loss(features, semantic_anchors[targets])
        anchors_loss = anchors_loss + anchor_loss * self.lambda_anchor

        features_square = torch.sum(torch.pow(features, 2), 1, keepdim=True)
        anchors_square = torch.sum(torch.pow(semantic_anchors, 2), 1, keepdim=True)
        features_into_anchors = torch.matmul(features, semantic_anchors.T)
        distances = features_square - 2 * features_into_anchors + anchors_square.T
        distances = torch.sqrt(distances)
        one_hot = F.one_hot(targets, semantic_anchors.shape[0]).to(features.device)
        gap = max(local_margin, global_margin)
        distances = distances + one_hot * gap
        mcl_loss = self.classification_loss(-distances, targets)
        anchors_loss = anchors_loss + mcl_loss * self.lambda_mcl
        loss = ce_loss + anchors_loss

        # Even zero-weight terms are computed, just as in clientSA.
        return loss, dict(total_loss=loss.detach(), ce_loss=ce_loss.detach(),
                          anchor_loss=anchor_loss.detach(), cc_loss=cc_loss.detach(),
                          mcl_loss=mcl_loss.detach()), anchor_class_losses.detach()


def prototype_report_loss(logits, features, targets, global_prototypes, lambda_anchor):
    """Upstream train_metrics only; this is NOT the four-term training objective."""
    loss = F.cross_entropy(logits, targets)
    if global_prototypes is not None:
        prototype_targets = features.detach().clone()
        for index, label in enumerate(targets):
            prototype = global_prototypes.get(label.item(), [])
            if not isinstance(prototype, list):
                prototype_targets[index] = prototype.detach()
        loss = loss + F.mse_loss(prototype_targets, features) * lambda_anchor
    return loss
