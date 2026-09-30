import torch.nn as nn
import torch.nn.functional as F

__all__ = ["PrototypeLoss"]


def prototype_loss(features, targets, global_prototypes):
    """Missing labels keep detached local targets, still in the MSE divisor."""
    prototype_targets = features.detach().clone()
    for index, label in enumerate(targets):
        class_idx = label.item()
        if class_idx in global_prototypes:
            prototype = global_prototypes[class_idx].detach()
            if prototype.shape != features[index].shape:
                raise ValueError("FedProto prototype and local feature shapes must match")
            prototype_targets[index] = prototype
    return F.mse_loss(prototype_targets, features)


class PrototypeLoss(nn.Module):
    def __init__(self, lambda_proto=1.0):
        super().__init__()
        self.lambda_proto = lambda_proto
        self.classification_loss = nn.NLLLoss()

    def forward(self, logits, targets, features, global_prototypes):
        log_probs = F.log_softmax(logits, dim=1)
        ce_loss = self.classification_loss(log_probs, targets)
        if len(global_prototypes) == 0:
            proto_loss = 0 * ce_loss
        else:
            proto_loss = prototype_loss(features, targets, global_prototypes)
        loss = ce_loss + self.lambda_proto * proto_loss
        return loss, {"total_loss": loss.detach(), "ce_loss": ce_loss.detach(),
                      "proto_loss": proto_loss.detach()}
