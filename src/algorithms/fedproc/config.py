import math


DEFAULTS = dict(
    alpha_rounds=100,
    server_momentum=0.0,
    aggregation="source",
    use_project_head=False,
    out_dim=256,
    metrics_only=False,
)


def resolve_config(params):
    unknown = set(params) - set(DEFAULTS)
    if unknown:
        raise ValueError("Unknown FedProc settings: {}".format(sorted(unknown)))
    cfg = dict(DEFAULTS, **params)
    for key in ("alpha_rounds", "out_dim"):
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError("{} must be a positive integer".format(key))
    if not math.isfinite(cfg["server_momentum"]) or not 0 <= cfg["server_momentum"] <= 1:
        raise ValueError("server_momentum must be in [0, 1]")
    if cfg["aggregation"] not in ("sampled", "source"):
        raise ValueError("aggregation must be sampled or source")
    for key in ("use_project_head", "metrics_only"):
        if type(cfg[key]) is not bool:
            raise ValueError("{} must be a JSON boolean".format(key))
    return cfg


def loss_weights(round_idx, alpha_rounds):
    """Effective weights; the first round is CE-only despite scheduled alpha=0."""
    ce_weight = 1.0 if round_idx == 0 else min(round_idx / alpha_rounds, 1.0)
    return ce_weight, 1.0 - ce_weight
