import math


DEFAULTS = dict(
    alpha_rounds=100,
    server_momentum=0.0,
    aggregation="source",
    use_project_head=False,
    out_dim=256,
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
    if type(cfg["use_project_head"]) is not bool:
        raise ValueError("use_project_head must be a JSON boolean")
    return cfg
