import copy
import math


DEFAULTS = {
    "ce_loss": {"weight": 1.0},
    "prox_loss": {"weight": 0.0},
    "feature_align_loss": {"weight": 0.0, "align_type": "cosine"},
    "rcl_loss": {
        "loss_type": "supervised",
        "topk_pos": 1,
        "topk_neg": 2,
        "temp": 0.05,
        "eps": 0.0,
        "beta": None,
        "pos_sample_type": None,
        "neg_sample_type": None,
        "pos_loss_type": None,
        "neg_loss_type": None,
        "threshold": 1.0,
        "weight": 1.0,
        "branch_level": None,
        "feature_relu": False,
        "pairs": [
            {"name": "scl", "pos": "nn", "neg": "nn", "weight": 1.0,
             "loss_type": "supervised"},
            {"name": "penalty", "pos": "nn", "neg": "nn", "weight": 1.0,
             "loss_type": "unsupervised", "neg_sample_type": "intra_class_thresholding",
             "threshold": 0.7},
        ],
    },
    "local_lr_decay": 0.998,
    "grad_clip": 10.0,
    "use_amp": True,
    "metrics_only": False,
}

PAIR_OVERRIDES = (
    "topk_pos", "topk_neg", "temp", "loss_type", "beta", "pos_sample_type",
    "neg_sample_type", "pos_loss_type", "neg_loss_type", "threshold",
)


def _copy_config(value):
    """Copy nested JSON without probing ConfLoader dictionaries for copy hooks."""
    if isinstance(value, dict):
        return {key: _copy_config(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_copy_config(item) for item in value]
    return copy.deepcopy(value)


def _finite(value, name):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("{} must be a finite number".format(name))


def _branches(value, num_layers, name):
    if value is False or value is None:
        return
    if not isinstance(value, list) or any(
        type(index) is not int or not 0 <= index < num_layers for index in value
    ):
        raise ValueError("{} must be null, false, or valid layer indices".format(name))


def _loss_config(cfg):
    for key, minimum in (("topk_pos", 1), ("topk_neg", 0)):
        if type(cfg[key]) is not int or cfg[key] < minimum:
            raise ValueError("{} must be an integer >= {}".format(key, minimum))
    _finite(cfg["temp"], "temp")
    if cfg["temp"] <= 0:
        raise ValueError("FedRCL temp must be positive")
    _finite(cfg["threshold"], "threshold")
    if cfg["beta"] is not None:
        _finite(cfg["beta"], "beta")
        if cfg["beta"] < 0:
            raise ValueError("FedRCL beta must be nonnegative")
    for key in ("loss_type", "pos_loss_type", "neg_loss_type"):
        value = cfg[key]
        if value is not None and not isinstance(value, str):
            raise ValueError("{} must be a string or null".format(key))
    if not (cfg["neg_loss_type"] or cfg["loss_type"]):
        raise ValueError("FedRCL requires loss_type or neg_loss_type")
    if cfg["neg_sample_type"] not in (None, "", "intra_class_thresholding"):
        raise ValueError("Unsupported FedRCL neg_sample_type")


def resolve_config(params, num_layers):
    """Resolve source loss options and the existing model's branch mapping."""
    unknown = set(params) - set(DEFAULTS)
    if unknown:
        raise ValueError("Unknown FedRCL settings: {}".format(sorted(unknown)))
    cfg = copy.deepcopy(DEFAULTS)
    for key, value in params.items():
        if key == "prox_loss" and value is None:
            cfg[key] = None
        elif isinstance(cfg[key], dict):
            if not isinstance(value, dict):
                raise ValueError("{} must be a JSON object".format(key))
            unknown = set(value) - set(cfg[key])
            if unknown:
                raise ValueError("Unknown FedRCL {} settings: {}".format(key, sorted(unknown)))
            cfg[key].update(_copy_config(value))
        else:
            cfg[key] = _copy_config(value)

    for key in ("use_amp", "metrics_only"):
        if type(cfg[key]) is not bool:
            raise ValueError("{} must be a JSON boolean".format(key))
    for key in ("local_lr_decay", "grad_clip"):
        _finite(cfg[key], key)
        if cfg[key] <= 0:
            raise ValueError("{} must be positive".format(key))
    for key in ("ce_loss", "prox_loss", "feature_align_loss"):
        if cfg[key] is not None:
            _finite(cfg[key]["weight"], key + ".weight")
    if cfg["feature_align_loss"]["align_type"] not in ("l2", "cosine"):
        raise ValueError("FedRCL align_type must be l2 or cosine")

    rcl = cfg["rcl_loss"]
    if type(rcl["feature_relu"]) is not bool:
        raise ValueError("feature_relu must be a JSON boolean")
    # null adapts the source's hidden-layer selection to the chosen backbone.
    # false retains the source meaning: all layers, including classifier logits.
    if rcl["branch_level"] is None:
        rcl["branch_level"] = list(range(num_layers - 1))
    _branches(rcl["branch_level"], num_layers, "branch_level")
    _finite(rcl["eps"], "eps")
    _finite(rcl["weight"], "rcl_loss.weight")
    _loss_config(rcl)
    if not isinstance(rcl["pairs"], list):
        raise ValueError("FedRCL pairs must be a JSON list")

    names = set()
    allowed = set(PAIR_OVERRIDES) | {"name", "pos", "neg", "weight", "branch_level"}
    for pair in rcl["pairs"]:
        if not isinstance(pair, dict) or set(pair) - allowed:
            raise ValueError("Invalid FedRCL pair settings")
        name = pair.get("name")
        if (not isinstance(name, str) or not name or name in names
                or name in ("cls", "prox", "cossim", "total")):
            raise ValueError("FedRCL pair names must be unique and not cls, prox, cossim, or total")
        names.add(name)
        for key in ("pos", "neg"):
            if not isinstance(pair.get(key), str) or set(pair[key].split(" ")) != {"nn"}:
                raise ValueError("The FedRCL source implements only nn feature pairs")
        _finite(pair.get("weight"), name + ".weight")
        _branches(pair.get("branch_level"), num_layers, name + ".branch_level")
        resolved = dict(rcl)
        for key in PAIR_OVERRIDES:
            resolved[key] = pair.get(key) or rcl[key]
        _loss_config(resolved)
    return cfg
