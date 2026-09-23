"""Explicit settings; typoed options fail instead of silently changing an experiment."""

import math


DEFAULTS = dict(
    teacher_mode="balanced", teacher_lr=0.05, teacher_balance_power=1.0,
    teacher_grad_clip=5.0, student_balance_power=1.0,
    retention_mode="band", retention_weight=0.1, support_scale=5.0,
    uniform_support=False, band_min=0.25, band_max=1.0,
    teacher_error_weight=0.25, kl_temperature=2.0, student_init="global",
    warmup_rounds=0, diagnostic_interval=10, diagnostic_clients=1,
    eval_interval=1, test_interval=0, output_dir="./results/fedbtr",
    save_checkpoints=False, save_local_history=False, experiment_seed=2022,
    transfer_diagnostics=False, transfer_shadow_branches=True,
    transfer_followup_rounds=(1, 5, 10),
)


def resolve_config(params):
    unknown = set(params) - set(DEFAULTS)
    if unknown:
        raise ValueError("Unknown FedBTR settings: {}".format(sorted(unknown)))
    cfg = dict(DEFAULTS, **params)
    for key, choices in (
        ("teacher_mode", ("balanced", "global", "none")),
        ("retention_mode", ("band", "kl", "none")),
        ("student_init", ("global", "teacher")),
    ):
        if cfg[key] not in choices:
            raise ValueError("{} must be one of {}".format(key, choices))
    for key in ("teacher_lr", "teacher_grad_clip", "retention_weight", "band_min", "band_max"):
        if not math.isfinite(cfg[key]) or cfg[key] < 0:
            raise ValueError("{} must be finite and nonnegative".format(key))
    for key in ("support_scale", "kl_temperature"):
        if not math.isfinite(cfg[key]) or cfg[key] <= 0:
            raise ValueError("{} must be finite and positive".format(key))
    for key in ("teacher_balance_power", "student_balance_power", "teacher_error_weight"):
        if not math.isfinite(cfg[key]) or not 0 <= cfg[key] <= 1:
            raise ValueError("{} must be in [0, 1]".format(key))
    for key in ("warmup_rounds", "diagnostic_interval", "diagnostic_clients", "test_interval"):
        if int(cfg[key]) != cfg[key] or cfg[key] < 0:
            raise ValueError("{} must be a nonnegative integer".format(key))
        cfg[key] = int(cfg[key])
    if int(cfg["eval_interval"]) != cfg["eval_interval"] or cfg["eval_interval"] < 1:
        raise ValueError("eval_interval must be a positive integer")
    cfg["eval_interval"] = int(cfg["eval_interval"])
    for key in ("uniform_support", "save_checkpoints", "save_local_history",
                "transfer_diagnostics", "transfer_shadow_branches"):
        if type(cfg[key]) is not bool:
            raise ValueError("{} must be a JSON boolean".format(key))
    if cfg["band_min"] > cfg["band_max"]:
        raise ValueError("band_min must not exceed band_max")
    if cfg["teacher_mode"] == "none":
        if cfg["retention_mode"] != "none" or cfg["student_init"] != "global":
            raise ValueError("teacher_mode=none requires retention_mode=none and student_init=global")
    if cfg["teacher_mode"] == "balanced" and cfg["teacher_lr"] == 0:
        raise ValueError("Use teacher_mode=global for an uncalibrated teacher")
    lags = cfg["transfer_followup_rounds"]
    if not isinstance(lags, (list, tuple)) or any(
            type(value) is not int or value < 1 for value in lags):
        raise ValueError("transfer_followup_rounds must be a list of positive integer round offsets")
    if len(set(lags)) != len(lags):
        raise ValueError("transfer_followup_rounds must not contain duplicates")
    cfg["transfer_followup_rounds"] = sorted(lags)
    if cfg["transfer_diagnostics"]:
        if (cfg["teacher_mode"] != "balanced" or cfg["retention_mode"] != "band"
                or cfg["student_init"] != "global" or cfg["retention_weight"] <= 0):
            raise ValueError("Transfer diagnostics require balanced teacher, band retention, "
                             "global student initialization and positive retention_weight")
        if cfg["diagnostic_interval"] < 1 or cfg["diagnostic_clients"] < 1:
            raise ValueError("Transfer diagnostics require positive diagnostic_interval/clients")
    return cfg
