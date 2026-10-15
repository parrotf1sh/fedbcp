import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    """Source InfoNCE_LCL, including its shared, repeated negative prototype pool."""

    def __init__(self, tau=0.05):
        super().__init__()
        self.tau = tau

    def normalize(self, *xs):
        return [None if x is None else F.normalize(x, dim=-1) for x in xs]

    def sim_loss(self, query, pos_key, neg_keys):
        """Retain the source alternative loss; the training path does not call it."""
        query, pos_key, neg_keys = self.normalize(query, pos_key, neg_keys)
        pos_sim = torch.exp(torch.sum(query * pos_key, dim=1) / self.tau)
        query = query.unsqueeze(1)
        neg_keys = neg_keys.transpose(-2, -1)
        neg_sim = torch.exp((query @ neg_keys) / self.tau)
        neg_sim = neg_sim.squeeze(1)
        loss_all = -torch.log(pos_sim / torch.sum(neg_sim, dim=1))
        return torch.sum(loss_all) / len(loss_all)

    def calculate_infonce(self, query, pos_key, neg_keys):
        """Retain Local.calculate_infonce without activating it in training."""
        prototypes = torch.cat((pos_key, neg_keys), dim=0)
        logits = torch.cosine_similarity(query, prototypes, dim=1)
        logits = logits / self.tau
        exp_logits = torch.exp(logits).view(1, -1)
        pos_mask = [1 for _ in range(pos_key.shape[0])]
        pos_mask += [0 for _ in range(neg_keys.shape[0])]
        pos_mask = torch.tensor(pos_mask, dtype=torch.float, device=query.device)
        pos_mask = pos_mask.view(1, -1)
        pos_logits = exp_logits * pos_mask
        sum_pos_logits = pos_logits.sum(1)
        sum_exp_logits = exp_logits.sum(1)
        return -torch.log(sum_pos_logits / sum_exp_logits)

    def infonce_lcl_loss(self, query, pos_key, neg_keys):
        neg_keys_list = []
        for class_prototypes in neg_keys:
            for prototype in class_prototypes:
                neg_keys_list.append(prototype)
        if not neg_keys_list:
            raise ValueError("FedSC source RPCL requires a nonempty negative prototype pool")
        pos_key = torch.stack(pos_key)
        neg_keys = torch.stack(neg_keys_list)
        query, pos_key, neg_keys = self.normalize(query, pos_key, neg_keys)
        pos_logit = torch.sum(query * pos_key, dim=1, keepdim=True)
        query = query.unsqueeze(1)
        neg_keys = neg_keys.transpose(-2, -1)
        neg_logits = query @ neg_keys
        neg_logits = neg_logits.squeeze(1)
        logits = torch.cat([pos_logit, neg_logits], dim=1)
        labels = torch.zeros(len(logits), dtype=torch.long, device=query.device)
        return F.cross_entropy(logits / self.tau, labels, reduction="mean")


class SemanticCollaborationLoss(nn.Module):
    """Preserve CE + RPCL + CPDR and the source prototype lookup fallback."""

    def __init__(self, cl_tau=0.05, collect_metrics=False):
        super().__init__()
        self.collect_metrics = collect_metrics
        self.last_metrics = {}
        self.ce_loss = nn.CrossEntropyLoss()
        self.rpcl_loss = InfoNCELoss(cl_tau)

    def forward(
        self, logits, features, targets, relational_prototypes,
        consistent_prototypes, round_idx,
    ):
        if self.collect_metrics:
            self.last_metrics = {"fallback_count": 0, "lookup_count": 0, "negative_pool_size": 0}
        ce_loss = self.ce_loss(logits, targets)
        rpcl_loss = features.new_zeros(())
        cpdr_loss = features.new_zeros(())
        # Project rounds are zero-based; source round_id > 1 starts at round two.
        if round_idx > 0:
            pos_key, neg_keys, reg_vector = [], [], []
            for sample_idx in range(len(targets)):
                label = targets[sample_idx].item()
                if self.collect_metrics:
                    self.last_metrics["lookup_count"] += 1
                try:
                    # Intentionally do not stack this list: the source falls back.
                    pos_key.append(torch.mean(relational_prototypes[label], dim=0))
                    reg_vector.append(consistent_prototypes[label])
                except:
                    # Preserve the original broad fallback and append order.
                    pos_key.append(features[sample_idx].detach())
                    reg_vector.append(features[sample_idx].detach())
                    if self.collect_metrics:
                        self.last_metrics["fallback_count"] += 1
                negatives = [
                    value for key, value in relational_prototypes.items()
                    if key != label
                ]
                for class_prototypes in negatives:
                    neg_keys.append(class_prototypes)
            if self.collect_metrics:
                self.last_metrics["negative_pool_size"] = sum(len(values) for values in neg_keys)
            rpcl_loss = self.rpcl_loss.infonce_lcl_loss(features, pos_key, neg_keys)
            cpdr_loss = F.mse_loss(features, torch.stack(reg_vector))
            loss = ce_loss + rpcl_loss + cpdr_loss
        else:
            loss = ce_loss
        return loss, {
            "ce_loss": ce_loss,
            "rpcl_loss": rpcl_loss,
            "cpdr_loss": cpdr_loss,
            "total_loss": loss,
        }
