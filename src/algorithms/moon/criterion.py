import torch
import torch.nn as nn
import math

__all__ = ["ModelContrastiveLoss"]


class ModelContrastiveLoss(nn.Module):
    def __init__(self, mu=0.001, tau=3):
        super(ModelContrastiveLoss, self).__init__()
        if not math.isfinite(mu) or mu < 0 or not math.isfinite(tau) or tau <= 0:
            raise ValueError("MOON requires finite mu >= 0 and finite tau > 0")
        self.mu = mu
        self.tau = tau
        self.ce = nn.CrossEntropyLoss()
        self.sim = nn.CosineSimilarity(dim=-1)

    def forward(self, logits, targets, z, z_prev, z_g, return_components=False):
        device = logits.device
        loss1 = self.ce(logits, targets)

        positive = self.sim(z, z_g).reshape(-1, 1)
        negative = self.sim(z, z_prev).reshape(-1, 1)
        moon_logits = torch.cat([positive, negative], dim=1)
        moon_logits /= self.tau
        moon_labels = torch.zeros(z.size(0)).to(device).long()

        loss2 = self.ce(moon_logits, moon_labels)

        total_loss = loss1 + self.mu * loss2

        if return_components:
            return total_loss, {"ce_loss": loss1.detach(),
                                "contrastive_loss": loss2.detach(),
                                "weighted_contrastive_loss": (self.mu * loss2).detach()}

        return total_loss
