"""SwanLab adapter for this project's round history and final reports.

SwanLab has no writable training-run summary dictionary. Preserve the complete
summary as config.result_summary and expose its numeric values as summary/*
metrics at the last real step, rather than inventing an additional round.
The SDK is loaded lazily so migration previews need only the standard library.
"""
from collections import UserDict
import importlib

from experiment_logging import flatten_metrics

run = None


def _sdk():
    try:
        sdk = importlib.import_module("swanlab")
    except ImportError as exc:
        raise RuntimeError("Install tracking dependencies: python -m pip install -r requirements-tracking.txt") from exc
    version = tuple(int(part) for part in sdk.__version__.split(".")[:3])
    if not (0, 10, 3) <= version < (0, 11):
        raise RuntimeError("This project requires swanlab>=0.10.3,<0.11")
    return sdk


class Summary(UserDict):
    def __init__(self, owner):
        self.owner = owner
        super().__init__()

    def __setitem__(self, key, value):
        self.data[key] = value
        self.owner.config.update({"result_summary": dict(self.data)})

    def update(self, *args, **kwargs):
        self.data.update(*args, **kwargs)
        if self.data:
            self.owner.config.update({"result_summary": dict(self.data)})


class Run:
    def __init__(self, native):
        self.native = native
        self.config = native.config
        self.disabled = native.mode == "disabled"
        self.id = native.id
        self.url = native.url if native.mode == "online" else None
        self.path = native.path.lstrip("/") if native.mode == "online" else None
        self.summary = Summary(self)
        self.last_step = None
        self.finished = False

    def define_metric(self, key, step_metric=None, **kwargs):
        if step_metric is not None:
            kwargs["x_axis"] = step_metric
        self.native.define_metric(key, **kwargs)

    def log(self, metrics, step=None):
        step = (0 if self.last_step is None else self.last_step + 1) if step is None else step
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ValueError("Metric step must be a nonnegative integer")
        self.native.log(metrics, step=step)
        self.last_step = max(step, self.last_step if self.last_step is not None else step)

    def save(self, path, base_path=None):
        return self.native.save(str(path), base_path=str(base_path) if base_path is not None else None, policy="now")

    def finish(self, exit_code=0):
        global run
        if self.finished:
            return
        state = "success" if exit_code == 0 else ("aborted" if exit_code == 130 else "crashed")
        try:
            if self.native.alive:
                try:
                    scalars = flatten_metrics(dict(self.summary), "summary")
                    if scalars:
                        # Summary-only imports have a summary point at 0, no round history.
                        self.native.define_metric("summary/*", x_axis="_step")
                        self.native.log(scalars, step=self.last_step if self.last_step is not None else 0)
                except Exception as exc:
                    self.native.finish(state="crashed", error=str(exc))
                    raise
                else:
                    self.native.finish(state=state, error=self.summary.get("error") if exit_code else None)
        finally:
            self.finished = True
            if run is self:
                run = None


def init(config=None, **settings):
    global run
    if run is not None and not run.finished:
        raise RuntimeError("Finish the previous tracking run before starting another")
    native = _sdk().init(config=config, **settings)
    run = Run(native)
    return run


def log(metrics, step=None):
    if run is None:
        raise RuntimeError("Initialize SwanLab before logging metrics")
    run.log(metrics, step=step)


def define_metric(key, **kwargs):
    if run is None:
        raise RuntimeError("Initialize SwanLab before defining metrics")
    run.define_metric(key, **kwargs)


def __getattr__(name):
    if name == "__version__":
        return _sdk().__version__
    if name == "config" and run is not None:
        return run.config
    raise AttributeError(name)
