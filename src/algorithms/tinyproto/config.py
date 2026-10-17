import math


DEFAULTS = dict(
    lambda_proto=10.0,
    add_cps=True,
    cps_ratio=10,
    simple_scale=True,
    constant_scale_factor=0.00015,
    mask_max_attempts=10000,
    mask_cache_path=None,
)


def resolve_config(params):
    unknown = set(params) - set(DEFAULTS)
    if unknown:
        raise ValueError("Unknown TinyProto settings: {}".format(sorted(unknown)))
    cfg = dict(DEFAULTS, **params)
    for name in ("add_cps", "simple_scale"):
        if type(cfg[name]) is not bool:
            raise ValueError("TinyProto {} must be a boolean".format(name))
    for name in ("lambda_proto", "constant_scale_factor"):
        value = cfg[name]
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError("TinyProto {} must be finite and nonnegative".format(name))
    if type(cfg["cps_ratio"]) is not int or not 0 < cfg["cps_ratio"] < 100:
        raise ValueError("TinyProto cps_ratio must be an integer in (0, 100)")
    if type(cfg["mask_max_attempts"]) is not int or cfg["mask_max_attempts"] < 1:
        raise ValueError("TinyProto mask_max_attempts must be a positive integer")
    path = cfg["mask_cache_path"]
    if path is not None and (not isinstance(path, str) or not path.strip()):
        raise ValueError("TinyProto mask_cache_path must be a nonempty string or null")
    return cfg
