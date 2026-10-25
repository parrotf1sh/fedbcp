"""Small, explicit configuration for prototype prior completion."""
import math

DEFAULTS = {
    "completion_strength": 0.5,
    "lambda_align": 0.1,
    "variance_scale": 1.0,
    "preserve_backbone_gradients": True,
    "warmup_rounds": 1,
    "max_prototype_age": 5,
    "statistics_num_workers": 0,
}


def resolve_config(params):
    unknown = set(params) - set(DEFAULTS)
    if unknown:
        raise ValueError("Unknown FedBPC-Complete settings: {}".format(sorted(unknown)))
    cfg = dict(DEFAULTS, **params)
    for key in ("completion_strength", "lambda_align", "variance_scale"):
        value = cfg[key]
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError("{} must be finite and nonnegative".format(key))
    if cfg["completion_strength"] > 1:
        raise ValueError("completion_strength must be in [0, 1]")
    if type(cfg["preserve_backbone_gradients"]) is not bool:
        raise ValueError("preserve_backbone_gradients must be a boolean")
    for key in ("warmup_rounds", "max_prototype_age", "statistics_num_workers"):
        if type(cfg[key]) is not int or cfg[key] < 0:
            raise ValueError("{} must be a nonnegative integer".format(key))
    return cfg
