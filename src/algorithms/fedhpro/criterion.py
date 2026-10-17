import torch
import torch.nn.functional as F


class HyperPrototypeLoss(torch.nn.Module):
    def __init__(self, hp_info_t):
        super().__init__()
        self.hp_info_t = hp_info_t

    def forward(self, loss_ce, features, targets, hp_class_norm_mean, hp_avg, margin):
        loss_hpcl = self.calculate_infonce_batch(
            features, targets, hp_class_norm_mean, margin
        )
        features_norm = F.normalize(features, dim=1)
        hp_avg_norm = F.normalize(hp_avg[targets], dim=1)
        loss_hpal = F.smooth_l1_loss(features_norm, hp_avg_norm, reduction="mean")
        loss = loss_ce + loss_hpcl + loss_hpal
        return loss, {
            "ce_loss": loss_ce, "hpcl_loss": loss_hpcl,
            "hpal_loss": loss_hpal, "total_loss": loss,
        }

    def calculate_infonce_batch(self, features, labels, hp_class_norm_mean, margin):
        features_norm = F.normalize(features, dim=1)
        batch_size = features.size(0)
        logits = features_norm.matmul(hp_class_norm_mean.T) / self.hp_info_t
        pos_logits = logits.gather(1, labels.view(-1, 1)).squeeze(1)
        log_den = torch.logsumexp(logits, dim=1)
        if margin != 0.0:
            log_margin = torch.as_tensor(
                margin, device=features.device, dtype=logits.dtype
            ).log()
            log_den = torch.logaddexp(log_den, log_margin)
        # The additional division is present in the provided executable source.
        return (log_den - pos_logits).mean() / batch_size

    def calculate_infonce(self, feature, pos_hp, neg_hp, margin):
        """Preserve the source's unused single-sample helper separately."""
        pos_hp, neg_hp = pos_hp.to(feature.device), neg_hp.to(feature.device)
        combined = torch.cat((pos_hp.unsqueeze(0), neg_hp), dim=0)
        expanded = feature.unsqueeze(0).expand(combined.size(0), combined.size(1), -1)
        similarity = F.cosine_similarity(expanded, combined, dim=2)
        logits = similarity.mean(dim=1) / self.hp_info_t
        exp_logits = torch.exp(logits).view(1, -1)
        pos_mask = torch.tensor(
            [1] + [0 for _ in range(combined.size(0) - 1)],
            dtype=torch.float, device=feature.device,
        ).view(1, -1)
        sum_pos = (exp_logits * pos_mask).sum(1)
        sum_exp = exp_logits.sum(1) + margin
        return -torch.log(sum_pos / sum_exp)


def match_loss(gw_syn, gw_real, dis_metric):
    dis = torch.tensor(0.0).to(gw_syn.device)
    avg_gw_syn = gw_syn.mean(dim=0)

    if dis_metric == "matching":
        for ig in range(len(gw_real)):
            gwr = gw_real[ig]
            gws = avg_gw_syn[ig]
            dis += distance_wb(gwr, gws)

    elif dis_metric == "mse":
        gw_real_vec = []
        gw_syn_vec = []
        for ig in range(len(gw_real)):
            gw_real_vec.append(gw_real[ig].reshape((-1)))
            gw_syn_vec.append(avg_gw_syn[ig].reshape((-1)))
        gw_real_vec = torch.cat(gw_real_vec, dim=0)
        gw_syn_vec = torch.cat(gw_syn_vec, dim=0)
        dis = torch.sum((gw_syn_vec - gw_real_vec) ** 2)

    elif dis_metric == "cos":
        gw_real_vec = []
        gw_syn_vec = []
        for ig in range(len(gw_real)):
            gw_real_vec.append(gw_real[ig].reshape((-1)))
            gw_syn_vec.append(avg_gw_syn[ig].reshape((-1)))
        gw_real_vec = torch.cat(gw_real_vec, dim=0)
        gw_syn_vec = torch.cat(gw_syn_vec, dim=0)
        dis = 1 - torch.sum(gw_real_vec * gw_syn_vec, dim=-1) / (
            torch.norm(gw_real_vec, dim=-1) * torch.norm(gw_syn_vec, dim=-1) + 0.000001
        )

    else:
        raise ValueError("Unknown gradient distance: {}".format(dis_metric))

    return dis


def match_meta(gw_syn, gw_real, dis_metric):
    dis = torch.tensor(0.0).to(gw_syn[0].device)

    if dis_metric == "matching":
        for ig in range(len(gw_real)):
            gwr = gw_real[ig]
            gws = gw_syn[ig]
            dis += distance_wb(gwr, gws)

    elif dis_metric == "mse":
        gw_real_vec = []
        gw_syn_vec = []
        for ig in range(len(gw_real)):
            gw_real_vec.append(gw_real[ig].reshape((-1)))
            gw_syn_vec.append(gw_syn[ig].reshape((-1)))
        gw_real_vec = torch.cat(gw_real_vec, dim=0)
        gw_syn_vec = torch.cat(gw_syn_vec, dim=0)
        dis = torch.sum((gw_syn_vec - gw_real_vec) ** 2)

    elif dis_metric == "cos":
        gw_real_vec = []
        gw_syn_vec = []
        for ig in range(len(gw_real)):
            gw_real_vec.append(gw_real[ig].reshape((-1)))
            gw_syn_vec.append(gw_syn[ig].reshape((-1)))
        gw_real_vec = torch.cat(gw_real_vec, dim=0)
        gw_syn_vec = torch.cat(gw_syn_vec, dim=0)
        dis = 1 - torch.sum(gw_real_vec * gw_syn_vec, dim=-1) / (
            torch.norm(gw_real_vec, dim=-1) * torch.norm(gw_syn_vec, dim=-1) + 0.000001
        )

    else:
        raise ValueError("Unknown gradient distance: {}".format(dis_metric))

    return dis


def distance_wb(gwr, gws):
    shape = gwr.shape
    if len(shape) == 4:  # conv, out*in*h*w
        gwr = gwr.reshape(shape[0], shape[1] * shape[2] * shape[3])
        gws = gws.reshape(shape[0], shape[1] * shape[2] * shape[3])
    elif len(shape) == 3:  # layernorm, C*h*w
        gwr = gwr.reshape(shape[0], shape[1] * shape[2])
        gws = gws.reshape(shape[0], shape[1] * shape[2])
    elif len(shape) == 2:  # linear, out*in
        pass
    elif len(shape) == 1:  # batchnorm/instancenorm, C; groupnorm x, bias
        gwr = gwr.reshape(1, shape[0])
        gws = gws.reshape(1, shape[0])

    dis_weight = torch.sum(
        1 - torch.sum(gwr * gws, dim=-1) / (
            torch.norm(gwr, dim=-1) * torch.norm(gws, dim=-1) + 0.000001
        )
    )
    dis = dis_weight

    return dis
