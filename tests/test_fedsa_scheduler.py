"""FedSA queue checks: no torch, data download or training."""
import contextlib
import csv
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


queue = load_file('fedsa_queue_test', ROOT / 'scripts/run_fedsa_experiments.py')
config = load_file('fedsa_config_test', ROOT / 'src/algorithms/fedsa/config.py')


class FedSAQueueTests(unittest.TestCase):
    def test_matrix_and_pairing(self):
        tasks = queue.build_tasks('/data/paired')
        reference = [t for t in queue.queue.build_tasks('/data/paired')
                     if t['config']['train_setups']['seed'] in (2022, 2023)]
        self.assertEqual(len(tasks), 44)
        self.assertEqual(len({t['id'] for t in tasks}), 44)
        self.assertEqual(sum('main' in t['categories'] for t in tasks), 36)
        groups = {}
        for task, other in zip(tasks, reference):
            cfg, baseline = task['config'], other['config']
            self.assertEqual(cfg['data_setups'], baseline['data_setups'])
            for key in ('scenario', 'model', 'optimizer', 'scheduler', 'seed'):
                self.assertEqual(cfg['train_setups'][key], baseline['train_setups'][key])
            self.assertEqual(cfg['train_setups']['algo']['params'], config.resolve_config({}))
            self.assertFalse(cfg['batch_protocol']['save_checkpoints'])
            self.assertEqual(cfg['batch_protocol']['model_aggregation'], 'none_persistent_client_models')
            self.assertEqual(cfg['wandb_setups']['project'], 'Point1')
            self.assertIn('persistent-client-models', cfg['wandb_setups']['tags'])
            groups.setdefault(task['condition_id'], []).append(cfg['train_setups']['seed'])
        self.assertEqual(len(groups), 22)
        self.assertTrue(all(s == [2022, 2023] for s in groups.values()))

    def test_parameters_change_identity_and_do_not_mutate_fedavg(self):
        before = queue.queue.build_tasks('/data/paired')
        ids = {t['id'] for t in queue.build_tasks('/data/paired')}
        for key, value in [('LAMBDA_ANCHOR', 2.), ('ANCHOR_SEED', 1), ('EVAL_INTERVAL', 1),
                           ('ANCHOR_MOMENTUM', .9), ('LAMBDA_MCL', .5), ('LAMBDA_CC', 0.)]:
            with mock.patch.object(queue, key, value):
                self.assertFalse(ids & {t['id'] for t in queue.build_tasks('/data/paired')})
        self.assertEqual(before, queue.queue.build_tasks('/data/paired'))

    def test_reject_invalid_or_unsupported_batch_settings(self):
        for key, value in [('LAMBDA_ANCHOR', -1), ('ANCHOR_MOMENTUM', 1.1),
                           ('ANCHOR_SEED', True), ('EVAL_INTERVAL', 0), ('FINAL_EVAL_ROUNDS', 9)]:
            with mock.patch.object(queue, key, value), self.assertRaises(ValueError):
                queue.build_tasks()

    def test_schedule_covers_real_last_ten_training_rounds(self):
        schedule = load_file('fedsa_batch_schedule', ROOT / 'src/algorithms/fedsa/batch.py').evaluation_rounds
        expected = [1] + list(range(10, 291, 10)) + list(range(291, 301))
        self.assertEqual(schedule(300), expected)
        self.assertEqual(len(expected), 40)
        self.assertEqual(schedule(300, 1), list(range(1, 301)))
        self.assertEqual(schedule(3), [1, 2, 3])
        for task in queue.build_tasks():
            cfg = task['config']
            self.assertEqual(cfg['batch_protocol']['evaluation_rounds'], expected)
            self.assertEqual(cfg['train_setups']['algo']['params']['anchor_seed'], 0)
            self.assertTrue(cfg['batch_protocol']['write_metrics_files'])

    def test_main_only_and_shared_cache(self):
        with mock.patch.object(queue, 'INCLUDE_SUPPLEMENTARY', False):
            self.assertEqual(len(queue.build_tasks()), 36)
        prepare = load_file('prepare_fedsa_test', ROOT / 'scripts/prepare_partitions.py')
        ours = prepare.unique_configs(queue.build_tasks('/data/paired'))
        reference = prepare.unique_configs(queue.queue.build_tasks('/data/paired'))
        self.assertEqual(len(ours), 18)
        self.assertEqual([t['data_setups'] for t in ours], [t['data_setups'] for t in reference])

    def test_failure_continues_success_skips_and_two_seed_summary(self):
        tasks = queue.build_tasks()[:2]
        calls = []
        def child(command, log, env):
            cfg = json.loads(Path(command[-1]).read_text())
            self.assertEqual(cfg['train_setups']['algo']['name'], 'fedsa')
            calls.append(cfg)
            if len(calls) == 1:
                return 1
            queue.queue.write_json(Path(log).parent / 'summary.json', {'status': 'success', 'final_top1': .6})
            return 0
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(queue.queue, 'run_child', side_effect=child), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            output = Path(directory)
            self.assertEqual(queue.queue.execute(tasks, output), 1)
            self.assertEqual(queue.queue.execute(tasks, output), 0)
            self.assertEqual(len(calls), 3)
            with (output / 'aggregate.csv').open() as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(row['n_expected'], '2')
            self.assertEqual(row['n_success'], '2')


if __name__ == '__main__':
    unittest.main()
