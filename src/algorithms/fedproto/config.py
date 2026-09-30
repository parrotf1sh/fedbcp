import math


DEFAULTS = dict(
    lambda_proto=1.0,
    missing_class_distance=100.0,
)


def resolve_config(params):
    unknown = set(params) - set(DEFAULTS)
    if unknown:
        raise ValueError("Unknown FedProto settings: {}".format(sorted(unknown)))
    cfg = dict(DEFAULTS, **params)
    for key in ("lambda_proto", "missing_class_distance"):
        value = cfg[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("{} must be a finite number".format(key))
        if not math.isfinite(value) or value < 0:
            raise ValueError("{} must be finite and nonnegative".format(key))
    if cfg["missing_class_distance"] == 0:
        raise ValueError("missing_class_distance must be positive")
    return cfg
