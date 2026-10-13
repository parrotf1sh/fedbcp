"""FedNH queue checks: no torch, data download or training."""
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


queue = load_file('fednh_queue_test', ROOT / 'scripts/run_fednh_experiments.py')
config = load_file('fednh_config_test', ROOT / 'src/algorithms/fednh/config.py')


class FedNHQueueTests(unittest.TestCase):
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
            self.assertEqual(cfg['train_setups']['algo']['params'], config.resolve_config({'metrics_only': True}))
            self.assertFalse(cfg['batch_protocol']['save_checkpoints'])
            self.assertEqual(cfg['batch_protocol']['model_aggregation'], 'equal_participant_model_deltas')
            self.assertEqual(cfg['wandb_setups']['project'], 'Point1')
            self.assertIn('equal-client-aggregation', cfg['wandb_setups']['tags'])
            groups.setdefault(task['condition_id'], []).append(cfg['train_setups']['seed'])
        self.assertEqual(len(groups), 22)
        self.assertTrue(all(s == [2022, 2023] for s in groups.values()))

    def test_parameters_change_identity_and_do_not_mutate_fedavg(self):
        before = queue.queue.build_tasks('/data/paired')
        ids = {t['id'] for t in queue.build_tasks('/data/paired')}
        for key, value in [('SMOOTHING', .8), ('FIX_SCALING', True), ('SCALING_INIT', 2.),
                           ('MAX_GRAD_NORM', 5.), ('SERVER_LR', .5), ('SERVER_LR_DECAY', .99)]:
            with mock.patch.object(queue, key, value):
                changed = queue.build_tasks('/data/paired')
            self.assertFalse(ids & {t['id'] for t in changed})
        with mock.patch.object(queue, 'CLIENT_ADV_PROTOTYPE_AGG', True), \
                mock.patch.object(queue, 'SERVER_ADV_PROTOTYPE_AGG', True):
            self.assertFalse(ids & {t['id'] for t in queue.build_tasks('/data/paired')})
        self.assertEqual(before, queue.queue.build_tasks('/data/paired'))

    def test_reject_invalid_or_unsupported_batch_settings(self):
        for key, value in [('SMOOTHING', 1.1), ('SMOOTHING', float('nan')), ('FIX_SCALING', 1),
                           ('MAX_GRAD_NORM', 0), ('CLIENT_ADV_PROTOTYPE_AGG', True),
                           ('HEAD_INIT', 'uniform'), ('EXCLUDE', ['prototype']),
                           ('CLIENT_LR_SCHEDULER', 'diminishing')]:
            with mock.patch.object(queue, key, value), self.assertRaises(ValueError):
                queue.build_tasks()
        self.assertFalse(config.resolve_config({})['metrics_only'])
        self.assertEqual(config.resolve_config({'exclude': ['prototype']})['exclude'], ['prototype'])
        with self.assertRaises(ValueError):
            config.resolve_config({'metrics_only': 1})

    def test_main_only_and_shared_cache(self):
        with mock.patch.object(queue, 'INCLUDE_SUPPLEMENTARY', False):
            self.assertEqual(len(queue.build_tasks()), 36)
        prepare = load_file('prepare_fednh_test', ROOT / 'scripts/prepare_partitions.py')
        ours = prepare.unique_configs(queue.build_tasks('/data/paired'))
        reference = prepare.unique_configs(queue.queue.build_tasks('/data/paired'))
        self.assertEqual(len(ours), 18)
        self.assertEqual([t['data_setups'] for t in ours], [t['data_setups'] for t in reference])

    def test_failure_continues_success_skips_and_two_seed_summary(self):
        tasks = queue.build_tasks()[:2]
        calls = []
        def child(command, log, env):
            cfg = json.loads(Path(command[-1]).read_text())
            self.assertEqual(cfg['train_setups']['algo']['name'], 'fednh')
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
