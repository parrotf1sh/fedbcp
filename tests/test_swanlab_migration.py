"""Historical formats, retry identity, cloud verification and read-only previews."""
import contextlib
import csv
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
spec = importlib.util.spec_from_file_location("migrate_swanlab_test", ROOT / "scripts/migrate_results_to_swanlab.py")
migration = importlib.util.module_from_spec(spec)
spec.loader.exec_module(migration)


def fixture(root, attempt="attempt-100", status="success", history="csv"):
    directory = root / "task1" / attempt
    directory.mkdir(parents=True)
    config = {"train_setups": {"scenario": {"n_rounds": 3}},
              "batch_protocol": {"task_id": "task1"},
              "wandb_setups": {"project": "Old", "entity": "old-team", "name": "fedhpro-test",
                               "group": "paired-seeds", "tags": ["seed-2022"]}}
    (directory / "config.json").write_text(json.dumps(config))
    (directory / "summary.json").write_text(json.dumps({"status": status, "final_top1": .7, "completed_rounds": 3}))
    rows = [{"round": 1, "global/top1": .5, "train/loss": 2.},
            {"round": 2, "train/loss": 1.5},
            {"round": 3, "global/top1": .7, "train/loss": 1.}]
    if history == "csv":
        with (directory / "metrics.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["round", "global/top1", "train/loss"])
            writer.writeheader(); writer.writerows(rows)
    elif history == "jsonl":
        (directory / "metrics.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (directory / "clients.jsonl").write_text('{"round":1,"clients":[0,2]}\n')
    return directory, rows


class FakeRun:
    def __init__(self, kwargs):
        self.id = kwargs["id"]
        self.url = "https://example.test/" + self.id
        self.path = kwargs["workspace"] + "/" + kwargs["project"] + "/" + self.id
        self.summary = {}
        self.logs = []
        self.files = []
        self.define_metric = mock.Mock()
        self.finish = mock.Mock()

    def log(self, data, step):
        self.logs.append((dict(data), step))

    def save(self, path, base_path):
        self.files.append(Path(path).name)


class FakeTracker:
    def __init__(self):
        self.calls = []
        self.runs = []

    def init(self, **kwargs):
        self.calls.append(kwargs)
        run = FakeRun(kwargs)
        self.runs.append(run)
        return run


class MigrationTests(unittest.TestCase):
    def test_csv_preview_preserves_sparse_values_and_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            directory, rows = fixture(root)
            before = {p: p.read_bytes() for p in root.rglob('*') if p.is_file()}
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(migration.main(['--source', str(root), '--dry-run']), 0)
            self.assertEqual(before, {p: p.read_bytes() for p in root.rglob('*') if p.is_file()})
            item = migration.load_result(directory)
            self.assertEqual(item['rows'], rows)
            self.assertNotIn('global/top1', item['rows'][1])

    def test_jsonl_preferred_over_derived_csv_and_nested_fedbtr(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory, rows = fixture(Path(tmp), history='jsonl')
            (directory / 'metrics.csv').write_text('invalid derived file')
            self.assertEqual(migration.load_result(directory)['rows'], rows)
            btr = Path(tmp) / 'btr'; btr.mkdir()
            (btr / 'experiment.json').write_text(json.dumps({'experiment': {'train_setups': {'scenario': {'n_rounds': 1}}}}))
            (btr / 'rounds.jsonl').write_text('{"round":1,"global":{"class_accuracy":[0.5,0.8]}}\n')
            (btr / 'final.json').write_text('{"best_round":1,"last_global_test":{"macro_accuracy":0.65}}')
            item = migration.load_result(btr)
            self.assertEqual(item['rows'][0]['global/class_accuracy/class_001'], .8)
            self.assertEqual(item['summary']['final/last_global_test/macro_accuracy'], .65)

    def test_latest_success_and_incomplete_opt_in_no_empty_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root, 'attempt-100')
            newest, _ = fixture(root, 'attempt-200')
            fixture(root, 'attempt-300', 'failed')
            fixture(root, 'attempt-400', 'failed', history='none')
            items, skipped, errors = migration.discover([root, root / 'task1'])
            self.assertFalse(errors)
            self.assertEqual([i['directory'] for i in items], [newest.resolve()])
            self.assertEqual(len(skipped), 3)
            items, _, _ = migration.discover([root], True)
            self.assertEqual(len(items), 3)

    def test_bad_rounds_nonfinite_and_truncated_rows_fail_before_upload(self):
        cases = ['round,x\n1,2\n1,3\n', 'round,x\n2,2\n1,3\n', 'round,x\n1.5,2\n',
                 'round,x\n1,nan\n', 'round,x\n1\n', 'round,round\n1,1\n']
        for content in cases:
            with self.subTest(content=content), tempfile.TemporaryDirectory() as tmp:
                directory, _ = fixture(Path(tmp))
                (directory / 'metrics.csv').write_text(content)
                _, _, errors = migration.discover([Path(tmp)])
                self.assertEqual(len(errors), 1)

    def test_successful_summary_cannot_hide_truncated_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory, _ = fixture(Path(tmp))
            (directory / 'metrics.csv').write_text('round,x\n1,2\n')
            with self.assertRaisesRegex(ValueError, 'last saved round'):
                migration.load_result(directory)

    def test_upload_retry_reuses_id_and_verified_run_is_skipped(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            root = Path(tmp); directory, rows = fixture(root)
            item = migration.load_result(directory); tracker = FakeTracker()
            api = types.SimpleNamespace(username='new-account')
            state = root / 'migration-state'
            original = {p: p.read_bytes() for p in directory.iterdir()}
            with mock.patch.object(migration, 'verify_cloud', side_effect=ValueError('not uploaded')):
                self.assertEqual(migration.migrate([item], state, 'Point1', tracker=tracker, api=api), 1)
            with mock.patch.object(migration, 'verify_cloud') as verify:
                self.assertEqual(migration.migrate([item], state, 'Point1', tracker=tracker, api=api), 0)
                self.assertEqual(migration.migrate([item], state, 'Point1', tracker=tracker, api=api), 0)
                verify.assert_called_once()
            self.assertEqual(len(tracker.calls), 2)
            self.assertEqual(tracker.calls[0]['id'], tracker.calls[1]['id'])
            self.assertEqual(tracker.calls[0]['workspace'], 'new-account')
            self.assertEqual(tracker.calls[0]['resume'], 'allow')
            self.assertEqual(tracker.runs[1].logs, [(row, row['round']) for row in rows])
            self.assertIn('clients.jsonl', tracker.runs[1].files)
            self.assertEqual(original, {p: p.read_bytes() for p in directory.iterdir()})
            api.username = 'another-account'
            with mock.patch.object(migration, 'verify_cloud'):
                migration.migrate([item], state, 'Point1', tracker=tracker, api=api)
            self.assertNotEqual(tracker.calls[1]['id'], tracker.calls[2]['id'])

    def test_changed_source_rejected_before_init(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            directory, _ = fixture(Path(tmp)); item = migration.load_result(directory)
            (directory / 'clients.jsonl').write_text('changed')
            tracker = FakeTracker()
            result = migration.migrate([item], Path(tmp) / 'state', 'Point1', workspace='test', mode='offline', tracker=tracker)
            self.assertEqual(result, 1)
            self.assertFalse(tracker.calls)

    def test_full_cloud_verification_rejects_missing_or_wrong_points(self):
        expected = {'accuracy': {1: .5, 3: .7}}
        good = {'list': [{'key':'accuracy', 'metrics':[{'step':1,'value':.5},{'step':3,'value':.7}]}]}
        migration.compare_series(expected, good)
        bad = {'list': [{'key':'accuracy', 'metrics':[{'step':1,'value':.5}]}]}
        with self.assertRaises(ValueError): migration.compare_series(expected, bad)
        good['list'][0]['metrics'][1]['value'] = .9
        with self.assertRaises(ValueError): migration.compare_series(expected, good)

    def test_summary_only_has_no_fabricated_round_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory, _ = fixture(Path(tmp), history='none')
            item = migration.load_result(directory)
            self.assertFalse(item['rows'])
            expected = migration.expected_series(item)
            self.assertNotIn('round', expected)
            self.assertEqual(expected['summary/final_top1'], {0:.7})


if __name__ == '__main__':
    unittest.main()
