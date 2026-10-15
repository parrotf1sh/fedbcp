import math


DEFAULTS = {
    "cl_tau": 0.05,
    "metrics_only": False,
}


def resolve_config(params):
    """Resolve the source contrastive temperature without adding loss weights."""
    unknown = set(params) - set(DEFAULTS)
    if unknown:
        raise ValueError("Unknown FedSC settings: {}".format(sorted(unknown)))
    cfg = dict(DEFAULTS)
    cfg.update(params)
    tau = cfg["cl_tau"]
    if type(tau) not in (int, float) or not math.isfinite(tau) or tau <= 0:
        raise ValueError("FedSC cl_tau must be a finite positive number")
    if type(cfg["metrics_only"]) is not bool:
        raise ValueError("FedSC metrics_only must be a boolean")
    return cfg
