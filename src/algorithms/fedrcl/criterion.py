import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["MultiLabelCrossEntropyLoss", "CLLoss"]


class MultiLabelCrossEntropyLoss(nn.Module):
    """FedRCL's masked log-softmax, including its nonzero-count reduction."""

    def __init__(self, eps=0.0, alpha=0.2, topk_pos=-1, temp=1.0, **kwargs):
        super().__init__()
        self.eps = eps
        self.alpha = alpha
        self.topk_pos = topk_pos
        self.temp = temp

    def forward(self, input, targets, reduction=True, beta=None):
        input[input == float("inf")] = -float("inf")
        if beta is not None:
            weights = torch.ones_like(targets)
            weights[targets == 0] = beta
            input += torch.log(weights) / self.temp

        log_probs = F.log_softmax(input, dim=1)
        loss = -targets * log_probs
        loss[loss == float("inf")] = 0.0
        loss[loss == -float("inf")] = 0.0
        loss[loss.isnan()] = 0.0
        loss = loss.sum(dim=1)
        with torch.no_grad():
            non_zero_count = max(loss.nonzero(as_tuple=False).size(0), 1)
        if reduction:
            loss = loss.sum() / non_zero_count
        return loss


class CLLoss(nn.Module):
    """Port of the source CLLoss; this is not an all-pairs SupCon objective."""

    def __init__(
        self, topk_pos=-1, topk_neg=-1, temp=1.0, eps=0.0, pair=None,
        loss_type=None, beta=None, pos_sample_type=None, neg_sample_type=None,
        pos_loss_type=None, neg_loss_type=None, threshold=1.0, **kwargs
    ):
        super().__init__()
        self.pair = pair
        # Preserve the source's truthy override semantics, including zero fallback.
        self.topk_pos = pair.get("topk_pos") or topk_pos
        self.topk_neg = pair.get("topk_neg") or topk_neg
        self.pos_sample_type = pair.get("pos_sample_type") or pos_sample_type
        self.neg_sample_type = pair.get("neg_sample_type") or neg_sample_type
        self.temp = pair.get("temp") or temp
        self.loss_type = pair.get("loss_type") or loss_type
        self.pos_loss_type = pair.get("pos_loss_type") or pos_loss_type
        self.neg_loss_type = pair.get("neg_loss_type") or neg_loss_type
        self.beta = pair.get("beta") or beta
        self.threshold = pair.get("threshold") or threshold
        # The source uses the shared temperature here, even with a pair override.
        self.criterion = MultiLabelCrossEntropyLoss(
            topk_pos=topk_pos, temp=temp, eps=eps
        )

    def get_classwise_mask(self, targets):
        batch_size = targets.size(0)
        return targets.expand(batch_size, batch_size).eq(
            targets.expand(batch_size, batch_size).T
        )

    def get_topk_neg(self, sim, topk_neg=None, topk_pos=None, labels=None):
        sim_neg = sim.clone()
        batch_size = sim_neg.size(0)
        loss_type = self.neg_loss_type if self.neg_loss_type else self.loss_type
        if "unsupervised" in loss_type:
            pos_mask = self.get_classwise_mask(
                torch.arange(batch_size, device=labels.device)
            )
        else:
            pos_mask = self.get_classwise_mask(labels)

        if self.neg_sample_type:
            if self.neg_sample_type == "intra_class_thresholding":
                classwise_mask = self.get_classwise_mask(labels)
                sim_neg[~classwise_mask] = float("inf")
                sim_neg[torch.eye(batch_size, device=sim.device) == 1] = float("inf")
                sim_neg = torch.topk(
                    sim_neg, min(topk_pos, batch_size), dim=1, largest=False
                )[0]
                sim_neg[sim_neg < self.threshold] = -1
                sim_neg[sim_neg == float("inf")] = -1
            else:
                raise ValueError("Unknown FedRCL negative sampling: " + self.neg_sample_type)
        else:
            sim_neg[pos_mask == 1] = -float("inf")
            sim_neg = torch.topk(
                sim_neg, min(topk_neg, batch_size), dim=1, largest=True
            )[0]
        return sim_neg

    def get_topk_pos(self, sim, topk_pos=None, labels=None):
        sim_pos = sim.clone()
        batch_size = sim.size(0)
        loss_type = self.pos_loss_type if self.pos_loss_type else self.loss_type
        if loss_type == "unsupervised":
            pos_mask = self.get_classwise_mask(
                torch.arange(batch_size, device=labels.device)
            )
        else:
            pos_mask = self.get_classwise_mask(labels)
        sim_pos[pos_mask == 0] = float("inf")
        # Do not clamp positive k: unlike negative k, the source does not do so.
        return torch.topk(sim_pos, topk_pos, dim=1, largest=False)[0]

    def forward(
        self, old_feat, new_feat, target, reduction=True, pair=None,
        topk_pos=None, topk_neg=None
    ):
        if old_feat.dim() > 2:
            old_feat = old_feat.squeeze(-1).squeeze(-1)
        if new_feat.dim() > 2:
            new_feat = new_feat.squeeze(-1).squeeze(-1)
        batch_size, _ = new_feat.size()
        if batch_size == 0:
            raise ValueError("FedRCL requires a nonempty batch")
        if topk_pos is None:
            topk_pos = self.topk_pos
        if topk_neg is None:
            topk_neg = self.topk_neg
        if not 1 <= topk_pos <= batch_size:
            raise ValueError("FedRCL topk_pos must be in [1, batch_size], including the last batch")
        if reduction and batch_size > 1:
            raise ValueError("Source CLLoss requires reduction=False for batches larger than one")

        old_feat = F.normalize(old_feat, p=2, dim=1)
        new_feat = F.normalize(new_feat, p=2, dim=1)
        all_pair_types = set(self.pair["pos"].split(" ") + self.pair["neg"].split(" "))
        sims = {"nn": torch.mm(new_feat, new_feat.t()) if "nn" in all_pair_types else None}
        sim_pos, sim_neg = {}, {}
        for pair_type in all_pair_types:
            sim_pos[pair_type] = self.get_topk_pos(
                sims[pair_type], topk_pos=topk_pos, labels=target
            )
            sim_neg[pair_type] = self.get_topk_neg(
                sims[pair_type], topk_neg=topk_neg, topk_pos=topk_pos, labels=target
            )

        if pair is None:
            pair = self.pair
        pair_pos = torch.cat([sim_pos[name] for name in pair["pos"].split(" ")], 1)
        pair_neg = torch.cat([sim_neg[name] for name in pair["neg"].split(" ")], 1)
        pair_pos = pair_pos.unsqueeze(2).repeat(1, 1, 1)
        pair_neg = pair_neg.unsqueeze(1).repeat(1, pair_pos.shape[1], 1)
        pair_all = torch.cat((pair_pos, pair_neg), 2)
        binary_labels = torch.zeros_like(pair_all)
        binary_labels[:, :, 0] = 1
        loss = self.criterion(
            input=pair_all.reshape(-1, pair_all.size(2)) / self.temp,
            targets=binary_labels.reshape(-1, pair_all.size(2)),
            reduction=reduction, beta=self.beta,
        )
        return loss.reshape(batch_size, -1).mean(1)
