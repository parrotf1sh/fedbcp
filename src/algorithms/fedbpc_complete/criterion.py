"""Conserve classification mass while filling underrepresented classes."""
import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.no_grad()
def completion_weights(class_counts, global_prior, reliability, strength):
    """Return real-sample weights and prototype mass; no batch renormalization.

    pi @ real_weights + replay_weights.sum() == 1.
    With fresh full coverage and strength=1, class mass equals global_prior.
    Unavailable/stale prototypes reduce BOTH replay and real-data subtraction.
    """
    counts = class_counts.detach().float()
    prior = global_prior.detach().to(counts)
    reliability = reliability.detach().to(counts)
    if counts.ndim != 1 or prior.shape != counts.shape or reliability.shape != counts.shape:
        raise ValueError("Counts, prior and reliability must be same-length vectors")
    if not all(torch.isfinite(x).all() for x in (counts, prior, reliability)):
        raise ValueError("Completion weights must be finite")
    if (counts < 0).any() or counts.sum() <= 0 or (prior < 0).any():
        raise ValueError("Nonnegative counts/prior and a nonempty client are required")
    if not torch.isclose(prior.sum(), prior.new_tensor(1.0), atol=1e-6):
        raise ValueError("Global prior must sum to one")
    if (reliability < 0).any() or (reliability > 1).any() or not 0 <= strength <= 1:
        raise ValueError("Reliability and completion strength must be in [0, 1]")
    pi = counts / counts.sum()
    deficit = (prior - pi).clamp_min(0)
    excess = (pi - prior).clamp_min(0)
    replay = strength * reliability * deficit
    real = torch.ones_like(pi)
    excess_mass = excess.sum()
    if excess_mass > 1e-8:
        subtraction = replay.sum() / excess_mass
        present = pi > 0
        real[present] -= subtraction * excess[present] / pi[present]
    else:
        # Equal priors: suppress roundoff-scale pseudo observations.
        replay.zero_()
    return real.clamp(0, 1), replay


@torch.no_grad()
def alignment_weights(class_counts, reliability):
    """Unbiased batch estimator of a full-local present-class alignment mean."""
    counts = class_counts.detach().float()
    present = counts > 0
    weights = torch.zeros_like(counts)
    if present.any():
        weights[present] = (
            counts.sum() / (present.sum() * counts[present])
            * reliability.to(counts)[present]
        )
    return weights


def distributional_replay(classifier, means, variances, targets, variance_scale):
    """Gaussian Jensen upper bound on affine-head CE, not an exact risk.

    For target c, add .5 * scale * (w_j-w_c)^T diag(v_c) (w_j-w_c)
    to each logit j. Uses O(CD + C^2) memory, no C x C x D tensor.
    Moments are detached; both sides of the weight difference train the head.
    The correction is the ISDA Gaussian bound (Wang et al., NeurIPS 2019).
    """
    means, variances = means.detach(), variances.detach()
    logits = classifier(means)
    if variance_scale > 0:
        weights = classifier.weight
        target_weights = weights[targets]
        distance = (variances @ weights.square().T
                    - 2 * (variances * target_weights) @ weights.T
                    + (variances * target_weights.square()).sum(1, keepdim=True))
        # Only cancel roundoff: the true-class distance is mathematically zero.
        distance = distance.clamp_min(0).scatter(1, targets[:, None], 0.)
        logits = logits + .5 * variance_scale * distance
    return F.cross_entropy(logits, targets, reduction="none")


class PrototypeCompletionLoss(nn.Module):
    def forward(self, logits, targets, features, classifier, prototypes,
                real_weights, replay_weights, align_weights, lambda_align,
                variances=None, variance_scale=0., preserve_backbone_gradients=True):
        per_example = F.cross_entropy(logits, targets, reduction="none")
        if preserve_backbone_gradients:
            # Correct the HEAD's real-data class mass, but retain natural CE
            # gradients for the encoder: detached replay cannot replace lost
            # image-based encoder learning. Both paths share head parameters.
            head_ce = F.cross_entropy(classifier(features.detach()), targets, reduction="none")
            real_ce = per_example.mean() + ((real_weights[targets] - 1) * head_ce).mean()
        else:
            real_ce = (per_example * real_weights[targets]).mean()
        zero = logits.sum() * 0.0
        replay_ce = zero
        replay_classes = torch.nonzero(replay_weights > 0, as_tuple=False).flatten()
        if replay_classes.numel():
            # The feature memory is detached: synthetic features train ONLY the head.
            variance = (torch.zeros_like(prototypes) if variances is None else variances)
            replay_ce = (
                distributional_replay(classifier, prototypes[replay_classes],
                    variance[replay_classes], replay_classes, variance_scale)
                * replay_weights[replay_classes]
            ).sum()

        align = zero
        if lambda_align > 0:
            usable = ((align_weights[targets] > 0) & (features.detach().norm(dim=1) > 1e-8)
                      & (prototypes[targets].norm(dim=1) > 1e-8))
            if usable.any():
                cosine = F.cosine_similarity(
                    features[usable], prototypes[targets[usable]].detach(), dim=1, eps=1e-8)
                # Divide by the ORIGINAL batch size, not the surviving sample count.
                align = ((1 - cosine) * align_weights[targets[usable]]).sum() / targets.numel()
        total = real_ce + replay_ce + lambda_align * align
        return total, {"real_ce": real_ce, "replay_ce": replay_ce,
                       "align": align, "natural_ce": per_example.mean()}
