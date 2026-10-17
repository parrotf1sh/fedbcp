import torch
import torch.nn as nn
import torch.nn.functional as F


def prototype_loss(features, targets, global_prototypes, mask_indices):
    """Keep the full batch/dimension MSE denominator, including missing labels."""
    prototype_targets = features.detach().clone()
    if mask_indices:
        rows = [index for index, label in enumerate(targets)
                if label.item() in global_prototypes]
        if rows:
            indices = torch.stack([mask_indices[targets[index].item()] for index in rows])
            values = torch.stack([global_prototypes[targets[index].item()].detach()
                                  for index in rows])
            sparse_targets = torch.zeros_like(features[rows].detach())
            sparse_targets.scatter_(1, indices.to(features.device), values.to(features.device))
            prototype_targets[rows] = sparse_targets
    else:
        for index, label in enumerate(targets):
            if label.item() in global_prototypes:
                prototype = global_prototypes[label.item()].detach()
                if prototype.shape != features[index].shape:
                    raise ValueError("TinyProto prototype and feature dimensions differ")
                prototype_targets[index] = prototype
    return F.mse_loss(prototype_targets, features)


class PrototypeLoss(nn.Module):
    def __init__(self, lambda_proto):
        super().__init__()
        self.lambda_proto = lambda_proto
        self.classification_loss = nn.CrossEntropyLoss()

    def forward(self, logits, targets, features, global_prototypes, mask_indices):
        ce_loss = self.classification_loss(logits, targets)
        proto_loss = (prototype_loss(features, targets, global_prototypes, mask_indices)
                      if global_prototypes else 0 * ce_loss)
        # Approved repair: both dense and CPS targets contribute to training.
        loss = ce_loss + self.lambda_proto * proto_loss
        return loss, dict(total_loss=loss.detach(), ce_loss=ce_loss.detach(),
                          proto_loss=proto_loss.detach())
