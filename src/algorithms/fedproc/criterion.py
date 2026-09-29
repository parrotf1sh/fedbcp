import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["PrototypeContrastiveLoss"]


class PrototypeContrastiveLoss(nn.Module):
    """FedProc's SupConLoss_new, without temperature or feature normalization."""

    def __init__(self):
        super().__init__()

    def forward(self, features, targets, prototypes):
        centers = torch.stack(tuple(prototypes), dim=-1)
        centers = F.normalize(centers, p=2, dim=0)

        batch_size = features.shape[0]
        loss = None
        for sample_idx in range(batch_size):
            label = targets[sample_idx].item()
            feature = features[sample_idx]
            center = centers[:, label]
            numerator = torch.exp(feature.matmul(center))

            denominator = torch.mm(feature.view(1, -1), centers)
            denominator = torch.sum(torch.exp(denominator))
            if loss is None:
                loss = -1 * torch.log(numerator / denominator)
            else:
                loss += -1 * torch.log(numerator / denominator)

        return loss / batch_size
