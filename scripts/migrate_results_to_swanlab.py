#!/usr/bin/env python3
"""Replay saved FedClsimb results into SwanLab, without training or W&B access."""
import argparse
from contextlib import contextmanager
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from experiment_logging import flatten_metrics

FILES = ("config.json", "experiment.json", "summary.json", "final.json",
         "metrics.csv", "metrics.jsonl", "rounds.jsonl", "clients.jsonl",
         "partition_metadata.json", "partition_stats.json", "evaluation_rounds.json", "algorithm.json")
PRUNE = {".git", "wandb", "swanlog", ".swanlab-migration", "__pycache__", "data"}


def read_json(path):
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object: " + str(path))
    return value


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def csv_number(value):
    try:
        return int(value)
    except ValueError:
        return float(value)


def read_history(directory):
    # FedSA JSONL is the flushed source of truth; CSV is a derived sparse table.
    history = next((directory / name for name in ("metrics.jsonl", "metrics.csv", "rounds.jsonl")
                    if (directory / name).is_file()), None)
    if history is None:
        return None, []
    rows = []
    with history.open(encoding="utf-8", newline="") as handle:
        if history.suffix == ".csv":
            reader = csv.DictReader(handle)
            fields = reader.fieldnames or []
            if not fields or len(set(fields)) != len(fields) or "round" not in fields:
                raise ValueError("CSV requires unique headers including round: " + str(history))
            for line, row in enumerate(reader, 2):
                if None in row or any(value is None for value in row.values()):
                    raise ValueError("Truncated/malformed CSV at {}:{}".format(history, line))
                # Blank cells mean 'not evaluated', never zero or forward-filled.
                rows.append({key: csv_number(value) for key, value in row.items() if value.strip()})
        else:
            for line, text in enumerate(handle, 1):
                if not text.strip():
                    continue
                value = json.loads(text)
                if not isinstance(value, dict):
                    raise ValueError("Expected metric object at {}:{}".format(history, line))
                rows.append(flatten_metrics(value))
    previous = -1
    for row in rows:
        step = row.get("round")
        if (isinstance(step, bool) or not isinstance(step, (int, float))
                or not math.isfinite(step) or step < 0 or step != int(step) or step <= previous):
            raise ValueError("Rounds must be nonnegative, integral and strictly increasing: " + str(history))
        row["round"] = previous = int(step)
        if not all(math.isfinite(value) for value in row.values()):
            raise ValueError("Non-finite metric in " + str(history))
    return history, rows


def load_result(directory):
    files = [directory / name for name in FILES if (directory / name).is_file()]
    hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    if (directory / "config.json").exists():
        config = read_json(directory / "config.json")
    elif (directory / "experiment.json").exists():
        config = read_json(directory / "experiment.json").get("experiment") or {}
    elif (directory / "final.json").exists():
        config = read_json(directory / "final.json").get("config", {})
    else:
        raise ValueError("No saved experiment configuration in " + str(directory))
    summary = read_json(directory / "summary.json") if (directory / "summary.json").exists() else {}
    final = read_json(directory / "final.json") if (directory / "final.json").exists() else {}
    if final and not summary:
        if "last_global_validation" in final and "config" in final:
            selected = {key: final[key] for key in ("last_global_validation", "last_global_test")}
        else:
            selected = final
        summary = flatten_metrics(selected, "final")
        # final.json is only written after completing the fixed-round report.
        summary["status"] = "success"
    history, rows = read_history(directory)
    if not rows and not summary:
        raise ValueError("No saved metrics or final summary in " + str(directory))
    status = summary.get("status", "incomplete")
    protocol = config.get("batch_protocol", {})
    expected = config.get("train_setups", {}).get("scenario", {}).get("n_rounds")
    if status == "success" and rows and expected is not None and rows[-1]["round"] != expected:
        raise ValueError("Successful summary disagrees with last saved round in " + str(directory))
    if status == "success" and rows and "completed_rounds" in summary and summary["completed_rounds"] != len(rows):
        raise ValueError("Successful summary disagrees with saved row count in " + str(directory))
    settings = config.get("swanlab_setups", config.get("wandb_setups", {}))
    name = settings.get("experiment_name", settings.get("name", directory.name))
    identity = {"task_id": protocol.get("task_id"), "attempt": directory.name, "files": hashes}
    return dict(directory=directory, config=config, summary=summary, status=status, name=name,
                settings=settings, history=history, rows=rows, files=files, hashes=hashes,
                source_id=digest(identity))


def discover(sources, include_incomplete=False):
    directories = set()
    for source in sources:
        source = source.expanduser().resolve()
        if not source.is_dir():
            raise ValueError("Source directory does not exist: " + str(source))
        for root, children, files in os.walk(source):
            children[:] = sorted(child for child in children if child not in PRUNE and not child.startswith("."))
            if (set(files) & {"config.json", "experiment.json", "final.json"}
                    and set(files) & {"metrics.csv", "metrics.jsonl", "rounds.jsonl", "summary.json", "final.json"}):
                directories.add(Path(root))
    selected, skipped, errors = {}, [], []
    for directory in sorted(directories):
        try:
            item = load_result(directory)
            if item["status"] != "success" and not include_incomplete:
                skipped.append((directory, "source status=" + item["status"]))
                continue
            if not item["rows"] and item["status"] != "success":
                skipped.append((directory, "no saved rounds (e.g. tracking initialization failed)"))
                continue
            # A queue may have multiple attempts. Prefer its latest successful
            # attempt; --include-incomplete also keeps partial attempts separately.
            key = directory if include_incomplete or not directory.name.startswith("attempt-") else directory.parent
            if key in selected:
                skipped.append((selected[key]["directory"], "superseded successful attempt"))
            selected[key] = item
        except (ValueError, OSError, TypeError, KeyError) as exc:
            errors.append((directory, str(exc)))
    return list(selected.values()), skipped, errors


def assert_unchanged(item):
    for path in item["files"]:
        if hashlib.sha256(path.read_bytes()).hexdigest() != item["hashes"][path.name]:
            raise RuntimeError("Source changed during migration; stop training before retrying: " + str(path))


def expected_series(item):
    result = {}
    for row in item["rows"]:
        for key, value in row.items():
            result.setdefault(key, {})[row["round"]] = value
    step = item["rows"][-1]["round"] if item["rows"] else 0
    for key, value in flatten_metrics(item["summary"], "summary").items():
        result.setdefault(key, {})[step] = value
    return result


def compare_series(expected, response):
    """Validate SwanLab 0.10 full-resolution API output, not sampled curves."""
    actual = {}
    for series in response.get("list", []):
        points = {}
        for point in series.get("metrics", []):
            value = point.get("value", point.get("data"))
            # Exported sparse tables can contain empty cells.
            if value is None or (isinstance(value, float) and not math.isfinite(value)):
                continue
            step = point.get("step", point.get("index"))
            if step in points:
                raise ValueError("Duplicate remote step for " + series["key"])
            points[step] = value
        actual[series["key"]] = points
    for key, points in expected.items():
        remote = actual.get(key, {})
        if set(remote) != set(points):
            raise ValueError("Cloud steps not yet complete or differ for " + key)
        for step, value in points.items():
            if not math.isclose(float(remote[step]), value, rel_tol=1e-6, abs_tol=1e-9):
                raise ValueError("Cloud value differs for {} at round {}".format(key, step))


def verify_cloud(api, path, item):
    expected = expected_series(item)
    for attempt in range(3):
        try:
            remote = api.run(path)
            keys = list(expected)
            for start in range(0, len(keys), 20):
                batch = keys[start:start + 20]
                response = remote.metrics(keys=batch, all=True, x_axis="step")
                compare_series({key: expected[key] for key in batch}, response)
            if not keys:
                # A metadata-only import still has to exist on the service.
                _ = remote.state
            return
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2)


@contextmanager
def migration_lock(state_dir):
    import fcntl
    state_dir.mkdir(parents=True, exist_ok=True)
    with (state_dir / ".lock").open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another migration is using " + str(state_dir)) from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def migrate(items, state_dir, project, workspace=None, mode="online", tracker=None, api=None):
    if tracker is None:
        import experiment_tracker as tracker
    if mode == "online" and api is None:
        import swanlab
        api = swanlab.Api()
    if mode == "online" and not workspace:
        # Resolve the actual account so changing login cannot incorrectly skip runs.
        workspace = api.username
    counts = dict(uploaded=0, offline=0, skipped=0, failed=0)
    with migration_lock(state_dir):
        for item in items:
            target = dict(project=project, workspace=workspace, mode=mode,
                          host=os.environ.get("SWANLAB_API_HOST", "https://api.swanlab.cn"))
            key = digest(dict(source=item["source_id"], target=target))
            state_path = state_dir / (key + ".json")
            state = read_json(state_path) if state_path.exists() else {}
            if state.get("status") in ("verified", "offline_saved"):
                print("[SKIP migrated] " + item["name"], flush=True)
                counts["skipped"] += 1
                continue
            run = None
            try:
                assert_unchanged(item)
                state = dict(status="uploading", source=str(item["directory"]),
                             source_id=item["source_id"], target=target,
                             run_id="migrate_" + key[:32], rows=len(item["rows"]))
                # Persist identity before touching the network. Retry reuses this ID.
                write_json(state_path, state)
                settings = item["settings"]
                config = dict(item["config"])
                config["migration"] = dict(source_id=item["source_id"],
                    source_status=item["status"], source_directory=str(item["directory"]),
                    history_file=item["history"].name if item["history"] else None,
                    history_rows=len(item["rows"]), source_files_sha256=item["hashes"],
                    note="Replayed local results; original rounds/values retained, upload time is not original wall time")
                tags = list(settings.get("tags") or []) + ["migrated-local", "source-" + item["status"]]
                run = tracker.init(project=project, workspace=workspace, mode=mode,
                    experiment_name=item["name"], group=settings.get("group"), job_type=settings.get("job_type"),
                    tags=list(dict.fromkeys(tags)), config=config,
                    logdir=str(state_dir / "swanlog" / key), id=state["run_id"], resume="allow")
                state.update(url=run.url, run_path=run.path)
                write_json(state_path, state)
                if item["rows"]:
                    run.define_metric("round")
                    run.define_metric("*", step_metric="round")
                    for row in item["rows"]:
                        run.log(row, step=row["round"])
                run.summary.update(item["summary"])
                for path in item["files"]:
                    run.save(path, base_path=item["directory"])
                assert_unchanged(item)
                # Remote completion means replay completed. source_status and tags
                # retain whether the original training failed or was interrupted.
                run.finish()
                if mode == "online":
                    verify_cloud(api, state["run_path"], item)
                state["status"] = "verified" if mode == "online" else "offline_saved"
                write_json(state_path, state)
                counts["uploaded" if mode == "online" else "offline"] += 1
                print("[{}] {} ({} rows) {}".format(state["status"], item["name"], len(item["rows"]), run.url or ""), flush=True)
            except BaseException as exc:
                state.update(status="failed", error="{}: {}".format(type(exc).__name__, exc))
                write_json(state_path, state)
                if run is not None:
                    try:
                        run.finish(exit_code=130 if isinstance(exc, KeyboardInterrupt) else 1)
                    except Exception:
                        pass
                if not isinstance(exc, Exception):
                    raise
                counts["failed"] += 1
                print("[FAILED] {}: {}".format(item["name"], exc), file=sys.stderr, flush=True)
    print("Migration finished: " + json.dumps(counts), flush=True)
    return 1 if counts["failed"] else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, action="append", required=True,
                        help="Result root, task or attempt directory; repeat for multiple algorithms")
    parser.add_argument("--project", default=os.environ.get("SWANLAB_PROJ_NAME", "Point1"))
    parser.add_argument("--workspace", default=os.environ.get("SWANLAB_WORKSPACE"))
    parser.add_argument("--state-dir", type=Path, default=Path(".swanlab-migration"))
    parser.add_argument("--dry-run", action="store_true", help="Validate/preview only; no SDK, network or writes")
    parser.add_argument("--include-incomplete", action="store_true", help="Also import nonempty partial attempts with explicit source-status tags")
    parser.add_argument("--mode", choices=("online", "offline"), default="online")
    parser.add_argument("--limit", type=int, help="Migrate only the first N selected experiments")
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    items, skipped, errors = discover(args.source, args.include_incomplete)
    for directory, reason in skipped:
        print("[SKIP source] {}: {}".format(directory, reason))
    for directory, reason in errors:
        print("[INVALID] {}: {}".format(directory, reason), file=sys.stderr)
    for item in items:
        steps = "{}..{}".format(item["rows"][0]["round"], item["rows"][-1]["round"]) if item["rows"] else "summary only; no saved curve"
        print("[PLAN] {} | {} | {} rows | {} | {}".format(item["name"], item["status"], len(item["rows"]), steps, item["directory"]))
    print("Selected={}, skipped={}, invalid={}".format(len(items), len(skipped), len(errors)))
    if errors:
        print("Validation failed; no experiments uploaded. Fix the listed input or narrow --source.", file=sys.stderr)
        return 1
    if not items:
        print("No matching local results. W&B binary caches and console logs are not CSV/JSON histories.", file=sys.stderr)
        return 1
    if args.dry_run:
        return 0
    return migrate(items[:args.limit] if args.limit else items, args.state_dir.expanduser().resolve(),
                   args.project, args.workspace, args.mode)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
    except (ValueError, RuntimeError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
