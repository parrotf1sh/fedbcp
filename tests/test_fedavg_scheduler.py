"""Non-training checks: no datasets, GPU, torch or W&B account required."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
import zipfile


ROOT = Path(__file__).resolve().parents[1]


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


scheduler = load_file("fedavg_scheduler", ROOT / "scripts/run_fedavg_experiments.py")
data_helper = load_file("fedavg_data", ROOT / "src/fedavg_data.py")


class SchedulerTests(unittest.TestCase):
    def test_matrix_and_seed_protocol(self):
        tasks = scheduler.build_tasks()
        self.assertEqual(len(tasks), 66)
        self.assertEqual(len({t["id"] for t in tasks}), 66)
        main = [t for t in tasks if "main" in t["categories"]]
        self.assertEqual(len(main), 54)
        self.assertEqual(sum(t["config"]["data_setups"]["partition"]["method"] == "lda"
                             for t in main), 27)
        groups = {}
        for task in tasks:
            cfg = task["config"]
            self.assertEqual(cfg["train_setups"]["scenario"]["n_rounds"], 300)
            self.assertEqual(cfg["batch_protocol"]["partition_seed"], 19940817)
            self.assertEqual(cfg["batch_protocol"]["client_sampling_rule"],
                             "numpy.seed(round_index), zero_based")
            self.assertFalse(cfg["batch_protocol"]["save_checkpoints"])
            partition = cfg["data_setups"]["partition"]
            self.assertEqual(set(partition), {"method", "alpha"} if partition["method"] == "lda"
                             else {"method", "shard_per_user"})
            groups.setdefault(task["condition_id"], []).append(cfg["train_setups"]["seed"])
        self.assertEqual(len(groups), 22)
        self.assertTrue(all(seeds == [2022, 2023, 2024] for seeds in groups.values()))

    def test_only_54_when_supplementary_disabled(self):
        with mock.patch.object(scheduler, "INCLUDE_SUPPLEMENTARY", False):
            self.assertEqual(len(scheduler.build_tasks()), 54)

    def test_baseline_is_reused_for_both_supplementary_groups(self):
        anchors = [t for t in scheduler.build_tasks() if len(t["categories"]) == 3]
        self.assertEqual(len(anchors), 3)
        for task in anchors:
            self.assertEqual(task["config"]["data_setups"]["dataset_name"], "cifar100")

    def test_hyperparameter_changes_invalidate_task_ids(self):
        original = {t["id"] for t in scheduler.build_tasks()}
        with mock.patch.object(scheduler, "OPTIMIZER", {"lr": 0.02, "momentum": 0.9, "weight_decay": 1e-5}):
            changed = {t["id"] for t in scheduler.build_tasks()}
        self.assertFalse(original & changed)

    def test_invalid_shards_rejected(self):
        with mock.patch.object(scheduler, "SHARDS_PER_CLIENT", [3]):
            with self.assertRaisesRegex(ValueError, "Invalid sharding"):
                scheduler.build_tasks()

    def test_task_config_objects_are_independent(self):
        tasks = scheduler.build_tasks()
        tasks[0]["config"]["train_setups"]["optimizer"]["params"]["lr"] = 99
        self.assertEqual(tasks[1]["config"]["train_setups"]["optimizer"]["params"]["lr"], 0.01)

    def test_failure_continues_and_success_is_skipped(self):
        tasks = scheduler.build_tasks()[:3]
        calls = []

        def fake_child(command, log_path, env):
            calls.append(command)
            cfg = json.loads(Path(command[-1]).read_text())
            self.assertEqual(env["PYTHONHASHSEED"], str(cfg["train_setups"]["seed"]))
            if len(calls) == 1:
                return 2
            scheduler.write_json(Path(log_path).parent / "summary.json",
                                 {"status": "success", "final_top1": 0.6})
            return 0

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            with mock.patch.object(scheduler, "run_child", side_effect=fake_child), \
                    contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(scheduler.execute(tasks, output), 1)
            self.assertEqual(len(calls), 3)
            self.assertEqual(json.loads((output / "queue_summary.json").read_text()),
                             {"success": 2, "failed": 1, "skipped": 0})
            self.assertIn(",2,3,", (output / "aggregate.csv").read_text())
            with mock.patch.object(scheduler, "run_child", side_effect=fake_child), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(scheduler.execute(tasks, output), 0)
            self.assertEqual(len(calls), 4)
            self.assertEqual(json.loads((output / "queue_summary.json").read_text())["skipped"], 2)

    def test_interrupt_stops_queue(self):
        with tempfile.TemporaryDirectory() as temporary:
            with mock.patch.object(scheduler, "run_child", side_effect=KeyboardInterrupt) as child, \
                    contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(KeyboardInterrupt):
                    scheduler.execute(scheduler.build_tasks()[:2], Path(temporary))
            self.assertEqual(child.call_count, 1)

    def test_real_non_training_child_exit_codes(self):
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()):
            log = Path(temporary) / "child.log"
            command = [sys.executable, "-c", "print('example error'); raise SystemExit(7)"]
            self.assertEqual(scheduler.run_child(command, log, None), 7)
            self.assertIn("example error", log.read_text())
            self.assertEqual(scheduler.run_child([sys.executable, "-c", "print('next task')"], log, None), 0)

    def test_duplicate_queue_lock(self):
        with tempfile.TemporaryDirectory() as temporary:
            with scheduler.queue_lock(Path(temporary)):
                with self.assertRaises(RuntimeError):
                    with scheduler.queue_lock(Path(temporary)):
                        pass


class DownloadTests(unittest.TestCase):
    def test_download_missing_tiny_without_network(self):
        # A tiny archive verifies preparation, not training or the real dataset.
        payload = io.BytesIO()
        with zipfile.ZipFile(payload, "w") as zipped:
            zipped.writestr("tiny-imagenet-200/wnids.txt", "class-a\n")
        payload.seek(0)
        with tempfile.TemporaryDirectory() as temporary:
            cfg = {"data_setups": {"root": temporary, "dataset_name": "tinyimagenet"},
                   "batch_protocol": {"download_if_missing": True}}

            def complete(root):
                return (Path(root) / "wnids.txt").is_file()

            with mock.patch.object(data_helper, "tiny_complete", side_effect=complete), \
                    mock.patch.object(data_helper.urllib.request, "urlopen", return_value=payload) as download, \
                    contextlib.redirect_stdout(io.StringIO()):
                data_helper.prepare_data(cfg)
                self.assertEqual(download.call_count, 1)
                self.assertTrue((Path(temporary) / "tinyimagenet/wnids.txt").is_file())
                data_helper.prepare_data(cfg)
                self.assertEqual(download.call_count, 1)  # Existing data is reused.

    def test_missing_data_with_download_disabled(self):
        with tempfile.TemporaryDirectory() as temporary:
            cfg = {"data_setups": {"root": temporary, "dataset_name": "tinyimagenet"},
                   "batch_protocol": {"download_if_missing": False}}
            with self.assertRaises(FileNotFoundError):
                data_helper.prepare_data(cfg)

    def test_zip_extraction_and_traversal_rejection(self):
        with tempfile.TemporaryDirectory() as temporary:
            archive = Path(temporary) / "tiny.zip"
            output = Path(temporary) / "output"
            with zipfile.ZipFile(archive, "w") as zipped:
                zipped.writestr("tiny-imagenet-200/wnids.txt", "class-a\n")
            data_helper.safe_extract(archive, output)
            self.assertEqual((output / "tiny-imagenet-200/wnids.txt").read_text(), "class-a\n")
            with zipfile.ZipFile(archive, "w") as zipped:
                zipped.writestr("tiny-imagenet-200/../../escape.txt", "bad")
            with self.assertRaises(ValueError):
                data_helper.safe_extract(archive, output)
            self.assertFalse((Path(temporary) / "escape.txt").exists())


if __name__ == "__main__":
    unittest.main()
