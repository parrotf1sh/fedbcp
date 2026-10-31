"""Dependency-free tracking configuration, including read-only legacy conversion."""
import os


def tracking_settings(config):
    """Old saved configs can be replayed without importing or contacting W&B.

    A W&B entity is never treated as a SwanLab workspace: these are unrelated
    accounts. Use SWANLAB_WORKSPACE or the new workspace setting explicitly.
    """
    current = config.get("swanlab_setups")
    legacy = current is None
    source = dict(config.get("wandb_setups", {}) if legacy else current)
    for key in ("entity", "force", "save_code"):
        source.pop(key, None)
    for old, new in (("name", "experiment_name"), ("dir", "logdir")):
        if old in source:
            source.setdefault(new, source.pop(old))
    if legacy:
        # Old remote IDs/resume flags must never attach to a new service's run.
        source.pop("id", None)
        source.pop("resume", None)
    source["mode"] = os.environ.get("SWANLAB_MODE", source.get("mode", "online"))
    if source["mode"] not in ("online", "offline", "local", "disabled"):
        raise ValueError("Unsupported SWANLAB_MODE: " + source["mode"])
    for env, key in (("SWANLAB_PROJ_NAME", "project"), ("SWANLAB_WORKSPACE", "workspace")):
        if os.environ.get(env):
            source[key] = os.environ[env]
    return source
