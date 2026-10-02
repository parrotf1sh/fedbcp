import math


DEFAULTS = dict(
    head_init="orthogonal",
    smoothing=0.9,
    client_adv_prototype_agg=False,
    server_adv_prototype_agg=False,
    fix_scaling=False,
    scaling_init=None,
    fixed_scaling=30.0,
    max_grad_norm=10.0,
    server_lr=1.0,
    server_lr_decay=1.0,
    client_lr_scheduler="project",
    client_lr_decay=0.99,
    exclude=[],
)


def resolve_config(params):
    unknown = set(params) - set(DEFAULTS)
    if unknown:
        raise ValueError("Unknown FedNH settings: {}".format(sorted(unknown)))
    cfg = dict(DEFAULTS, **params)
    for key in ("client_adv_prototype_agg", "server_adv_prototype_agg", "fix_scaling"):
        if type(cfg[key]) is not bool:
            raise ValueError("{} must be a JSON boolean".format(key))
    if cfg["client_adv_prototype_agg"] != cfg["server_adv_prototype_agg"]:
        raise ValueError("FedNH client and server advanced aggregation must match")
    if cfg["head_init"] not in ("orthogonal", "uniform"):
        raise ValueError("head_init must be orthogonal or uniform")
    if cfg["client_lr_scheduler"] not in ("project", "diminishing", "stepwise"):
        raise ValueError("client_lr_scheduler must be project, diminishing or stepwise")
    for key in ("smoothing", "fixed_scaling", "max_grad_norm", "server_lr",
                "server_lr_decay", "client_lr_decay", "scaling_init"):
        value = cfg[key]
        if key == "scaling_init" and value is None:
            continue
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value)):
            raise ValueError("{} must be a finite number".format(key))
        if key == "smoothing":
            if not 0 <= value <= 1:
                raise ValueError("smoothing must be in [0, 1]")
        elif key in ("server_lr", "server_lr_decay", "client_lr_decay"):
            if value < 0:
                raise ValueError("{} must be nonnegative".format(key))
        elif key == "max_grad_norm" and value <= 0:
            raise ValueError("max_grad_norm must be positive")
    if (not isinstance(cfg["exclude"], (list, tuple))
            or any(not isinstance(key, str) or not key for key in cfg["exclude"])):
        raise ValueError("exclude must be a list of nonempty parameter key substrings")
    cfg["exclude"] = list(cfg["exclude"])
    return cfg
