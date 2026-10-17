import math


DEFAULTS = dict(
    lambda_anchor=1.0,
    lambda_mcl=1.0,
    lambda_cc=1.0,
    anchor_momentum=0.9999,
    anchor_seed=0,
)


def resolve_config(params):
    """Resolve upstream loss weights and the experiment's anchor initialization."""
    unknown = set(params) - set(DEFAULTS)
    if unknown:
        raise ValueError("Unknown FedSA settings: {}".format(sorted(unknown)))
    cfg = dict(DEFAULTS, **params)
    for key in ("lambda_anchor", "lambda_mcl", "lambda_cc", "anchor_momentum"):
        value = cfg[key]
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError("{} must be finite and nonnegative".format(key))
    if cfg["anchor_momentum"] > 1:
        raise ValueError("anchor_momentum must be in [0, 1]")
    if type(cfg["anchor_seed"]) is not int or not 0 <= cfg["anchor_seed"] < 2 ** 63:
        raise ValueError("anchor_seed must be an integer in [0, 2**63)")
    return cfg
