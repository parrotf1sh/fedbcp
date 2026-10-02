import copy

import torch
import torch.nn.functional as F


def create_optimizer(model, name="sgd", **params):
    """Preserve all optimizer branches used by the upstream FedNH client."""
    parameters = filter(lambda parameter: parameter.requires_grad, model.parameters())
    name = name.lower()
    if name == "sgd":
        return torch.optim.SGD(parameters, **params)
    if name == "adam":
        params.setdefault("weight_decay", 1e-5)
        return torch.optim.Adam(parameters, **params)
    if name == "rmsprop":
        params.setdefault("eps", 1e-8)
        return torch.optim.RMSprop(parameters, **params)
    raise ValueError("Unknown FedNH optimizer: {}".format(name))


def client_learning_rate(cfg, initial_lr, round_idx, n_rounds, project_lr):
    if cfg["client_lr_scheduler"] == "project":
        return project_lr
    if cfg["client_lr_scheduler"] == "diminishing":
        return initial_lr * cfg["client_lr_decay"] ** round_idx
    # Upstream rounds start at 1 and use a strict < comparison.
    if round_idx + 1 < n_rounds // 2:
        return initial_lr
    return initial_lr * 0.1


def linear_combination_state_dict(left, right, left_weight, right_weight, exclude):
    weights = copy.deepcopy(left)
    for key in left:
        if key not in exclude:
            weights[key] = left[key] * left_weight + right[key] * right_weight
    return weights


@torch.no_grad()
def aggregate_weights(server_weights, local_weights, payloads, cfg, round_idx, exclude):
    """Literal FedNH update order: model deltas, prototype mean, then EMA."""
    if not local_weights or len(local_weights) != len(payloads):
        raise ValueError("FedNH requires one prototype payload per participating client")
    server_lr = cfg["server_lr"] * cfg["server_lr_decay"] ** round_idx
    prototype = server_weights["prototype"]
    class_counts = torch.zeros(prototype.shape[0], device=prototype.device)
    aggregation_weights = {}
    update_direction = None
    for client_idx, (weights, payload) in enumerate(zip(local_weights, payloads)):
        if not cfg["server_adv_prototype_agg"]:
            class_counts += payload["count_by_class_full"]
        else:
            aggregation_weights[client_idx] = torch.exp(torch.sum(
                prototype * payload["adv_agg_prototype"], dim=1, keepdim=True
            ))
        update = linear_combination_state_dict(weights, server_weights, 1.0, -1.0, exclude)
        if client_idx == 0:
            update_direction = update
        else:
            update_direction = linear_combination_state_dict(
                update_direction, update, 1.0, 1.0, exclude
            )
    weights = linear_combination_state_dict(
        server_weights, update_direction, 1.0, server_lr / len(local_weights), exclude
    )

    average_prototype = torch.zeros_like(weights["prototype"])
    if not cfg["server_adv_prototype_agg"]:
        for payload in payloads:
            average_prototype += payload["scaled_prototype"] / class_counts.view(-1, 1)
    else:
        weight_sum = torch.zeros((prototype.shape[0], 1), device=prototype.device)
        for client_idx, payload in enumerate(payloads):
            weight_sum += aggregation_weights[client_idx]
            average_prototype += aggregation_weights[client_idx] * payload["adv_agg_prototype"]
        average_prototype /= weight_sum

    average_prototype = F.normalize(average_prototype, dim=1)
    smoothed = (cfg["smoothing"] * weights["prototype"]
                + (1 - cfg["smoothing"]) * average_prototype)
    weights["prototype"].copy_(F.normalize(smoothed, dim=1))
    return weights
