import math


DEFAULTS = {
    "metrics_only": False,
    "dsa_seed_mode": "wall_clock",
    "length_of_hp": 5,
    "rounds_of_hp": 30,
    "hp_info_t": 0.05,
    "lr_hp": 0.1,
    "meta_length": 50,
    "batch_real": 100,
    "repeat_num": 1,
    "use_margin": False,
    "dis_metric": "matching",
    "meta_rounds": 100,
    "meta_epochs": 100,
    "meta_lr": 0.01,
    "meta_batch_size": 50,
    "margin_batch_size": 100,
    "method": "DSA",
    "dsa_strategy": "color_crop_cutout_flip_scale_rotate",
}

AUGMENT_DEFAULTS = {
    "aug_mode": "S",
    "prob_flip": 0.5,
    "ratio_scale": 1.2,
    "ratio_rotate": 15.0,
    "ratio_crop_pad": 0.125,
    "ratio_cutout": 0.5,
    "ratio_noise": 0.05,
    "brightness": 1.0,
    "saturation": 2.0,
    "contrast": 0.5,
}


def resolve_config(params):
    """Keep source defaults; expose its fixed meta-training loop settings."""
    unknown = set(params) - set(DEFAULTS) - {"augmentation"}
    if unknown:
        raise ValueError("Unknown FedHPro settings: {}".format(sorted(unknown)))
    cfg = dict(DEFAULTS)
    cfg.update(params)
    for name in (
        "length_of_hp", "rounds_of_hp", "meta_length", "batch_real",
        "repeat_num", "meta_rounds", "meta_epochs", "meta_batch_size",
        "margin_batch_size",
    ):
        if type(cfg[name]) is not int or cfg[name] < 1:
            raise ValueError("FedHPro {} must be a positive integer".format(name))
    for name in ("hp_info_t", "lr_hp", "meta_lr"):
        value = cfg[name]
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError("FedHPro {} must be finite and positive".format(name))
    if type(cfg["metrics_only"]) is not bool:
        raise ValueError("FedHPro metrics_only must be a boolean")
    if cfg["dsa_seed_mode"] not in ("wall_clock", "deterministic"):
        raise ValueError("FedHPro dsa_seed_mode must be wall_clock or deterministic")
    if type(cfg["use_margin"]) is not bool:
        raise ValueError("FedHPro use_margin must be a boolean")
    if cfg["dis_metric"] not in ("matching", "mse", "cos"):
        raise ValueError("Unknown FedHPro gradient distance")
    if not isinstance(cfg["method"], str):
        raise ValueError("FedHPro method must be a string; DSA enables augmentation")
    strategy = cfg["dsa_strategy"]
    if not isinstance(strategy, str):
        raise ValueError("FedHPro dsa_strategy must be a string")
    if strategy not in ("", "None", "none"):
        if set(strategy.split("_")) - {"color", "crop", "cutout", "flip", "scale", "rotate"}:
            raise ValueError("Unknown FedHPro DSA strategy")

    options = params.get("augmentation", {})
    if not isinstance(options, dict) or set(options) - set(AUGMENT_DEFAULTS):
        raise ValueError("Invalid FedHPro augmentation settings")
    # Plain dictionaries also support ConfLoader's attribute-access mappings.
    augmentation = dict(AUGMENT_DEFAULTS)
    augmentation.update(options)
    if augmentation["aug_mode"] not in ("S", "M"):
        raise ValueError("FedHPro aug_mode must be S or M")
    for name, value in augmentation.items():
        if name == "aug_mode":
            continue
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError("FedHPro augmentation {} must be finite and nonnegative".format(name))
    if augmentation["ratio_scale"] <= 0 or augmentation["prob_flip"] > 1:
        raise ValueError("FedHPro requires ratio_scale > 0 and prob_flip <= 1")
    cfg["augmentation"] = augmentation
    return cfg
