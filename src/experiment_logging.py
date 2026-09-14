"""CSV-friendly W&B metrics; pure-Python helpers accept an explicit run object."""

import math


def tracking_enabled(run):
    return run is not None and not getattr(run, "disabled", False)


def flatten_metrics(record, prefix=""):
    """Expand lists into stable columns; absent diagnostics remain absent.

    Diagnostic slots intentionally do not use client IDs in column names, to
    avoid creating K*C time series. Each slot logs its actual client ID.
    """
    result = {}
    if isinstance(record, dict):
        for key, value in record.items():
            path = prefix + "/" + str(key) if prefix else str(key)
            result.update(flatten_metrics(value, path))
    elif isinstance(record, (list, tuple)):
        kind = "class" if prefix.rsplit("/", 1)[-1] in (
            "class_accuracy", "class_count", "class_delta") else "slot"
        for index, value in enumerate(record):
            result.update(flatten_metrics(value, "{}/{}_{:03d}".format(prefix, kind, index)))
    elif isinstance(record, (int, float)) and not isinstance(record, bool):
        if not math.isfinite(record):
            raise ValueError("Non-finite metric: {}".format(prefix))
        result[prefix] = record
    return result


def configure_tracking(run, metadata):
    if tracking_enabled(run):
        run.config.update({"fedbtr_runtime": metadata})
        run.define_metric("round")
        run.define_metric("*", step_metric="round")


def log_round(run, record):
    if tracking_enabled(run):
        run.log(flatten_metrics(record), step=record["round"])


def log_final(run, results):
    if tracking_enabled(run):
        # Summary is separate from history: do not create a fake final round.
        run.summary.update(flatten_metrics(results, "final"))
