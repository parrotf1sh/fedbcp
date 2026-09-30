"""FedProc queue/reporting checks using simulated objects, never model training."""
import contextlib
import csv
import importlib.util
import io
import json
import math
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


queue = load_file('fedproc_queue_test', ROOT / 'scripts/run_fedproc_experiments.py')
config = load_file('fedproc_config_test', ROOT / 'src/algorithms/fedproc/config.py')


class QueueTests(unittest.TestCase):
    def test_matrix_pairing_and_projected_configuration(self):
        tasks = queue.build_tasks('/data/paired')
        references = queue.queue.build_tasks('/data/paired')
        references = [t for t in references if t['config']['train_setups']['seed'] in (2022, 2023)]
        self.assertEqual(len(tasks), 44)
        self.assertEqual(len({t['id'] for t in tasks}), 44)
        self.assertEqual(sum('main' in t['categories'] for t in tasks), 36)
        groups = {}
        for task, reference in zip(tasks, references):
            cfg, ref = task['config'], reference['config']
            self.assertEqual(cfg['data_setups'], ref['data_setups'])
            for field in ('scenario', 'model', 'optimizer', 'scheduler', 'seed'):
                self.assertEqual(cfg['train_setups'][field], ref['train_setups'][field])
            params = cfg['train_setups']['algo']['params']
            self.assertEqual(params, config.resolve_config(dict(metrics_only=True, aggregation='sampled', use_project_head=True)))
            self.assertFalse(cfg['batch_protocol']['save_checkpoints'])
            self.assertEqual(cfg['wandb_setups']['project'], 'Point1')
            self.assertIn('sampled_proj256', task['name'])
            self.assertIn('proj256', cfg['wandb_setups']['tags'])
            groups.setdefault(task['condition_id'], []).append(cfg['train_setups']['seed'])
        self.assertEqual(len(groups), 22)
        self.assertTrue(all(seeds == [2022, 2023] for seeds in groups.values()))

    def test_parameter_changes_update_identity_without_mutating_other_algorithms(self):
        before = queue.queue.build_tasks('/data/paired')
        ids = {t['id'] for t in queue.build_tasks('/data/paired')}
        for key, value in [('AGGREGATION', 'source'), ('USE_PROJECT_HEAD', False), ('OUT_DIM', 128),
                           ('ALPHA_ROUNDS', 300), ('SERVER_MOMENTUM', 0.5)]:
            with mock.patch.object(queue, key, value):
                changed = queue.build_tasks('/data/paired')
            self.assertFalse(ids & {t['id'] for t in changed})
        self.assertEqual(before, queue.queue.build_tasks('/data/paired'))

    def test_invalid_settings_fail_before_any_training(self):
        for key, value in [('AGGREGATION', 'unknown'), ('USE_PROJECT_HEAD', 1), ('OUT_DIM', 0),
                           ('ALPHA_ROUNDS', 0), ('SERVER_MOMENTUM', float('nan'))]:
            with mock.patch.object(queue, key, value), self.assertRaises(ValueError):
                queue.build_tasks()
        with self.assertRaises(ValueError):
            config.resolve_config({'metrics_only': 1})
        with self.assertRaises(ValueError):
            config.resolve_config({'unexpected': True})

    def test_main_only_and_shared_partitions(self):
        with mock.patch.object(queue, 'INCLUDE_SUPPLEMENTARY', False):
            self.assertEqual(len(queue.build_tasks()), 36)
        prepare = load_file('prepare_fedproc_test', ROOT / 'scripts/prepare_partitions.py')
        ours = prepare.unique_configs(queue.build_tasks('/data/paired'))
        baseline = prepare.unique_configs(queue.queue.build_tasks('/data/paired'))
        self.assertEqual(len(ours), 18)
        self.assertEqual([c['data_setups'] for c in ours], [c['data_setups'] for c in baseline])

    def test_failure_continues_success_skips_and_two_seed_summary(self):
        tasks = queue.build_tasks()[:2]
        calls = []
        def child(command, log, env):
            cfg = json.loads(Path(command[-1]).read_text())
            self.assertEqual(cfg['train_setups']['algo']['name'], 'fedproc')
            calls.append(cfg)
            if len(calls) == 1:
                return 1
            queue.queue.write_json(Path(log).parent / 'summary.json', {'status': 'success', 'final_top1': 0.6})
            return 0
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(queue.queue, 'run_child', side_effect=child), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            output = Path(temporary)
            self.assertEqual(queue.queue.execute(tasks, output), 1)
            self.assertEqual(queue.queue.execute(tasks, output), 0)
            self.assertEqual(len(calls), 3)
            with (output / 'aggregate.csv').open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]['n_expected'], '2')
            self.assertEqual(rows[0]['n_success'], '2')
            self.assertEqual(float(rows[0]['final_top1_std']), 0)

    def test_effective_loss_weights_at_boundaries(self):
        for step, expected in [(0, (1, 0)), (1, (.01, .99)), (99, (.99, .01)), (100, (1, 0)), (299, (1, 0))]:
            actual = config.loss_weights(step, 100)
            for a, b in zip(actual, expected):
                self.assertAlmostEqual(a, b)


class Vector:
    def __init__(self, values):
        self.values = list(values)
    def clone(self):
        return Vector(self.values)
    def numel(self):
        return len(self.values)
    def element_size(self):
        return 4
    def __add__(self, other):
        return Vector([a + b for a, b in zip(self.values, other.values)])
    def __mul__(self, scalar):
        return Vector([v * scalar for v in self.values])
    def __truediv__(self, scalar):
        return self * (1 / scalar)


class Scalar:
    def __init__(self, value):
        self.value = value
    def __rmul__(self, weight):
        return Scalar(weight * self.value)
    def __add__(self, other):
        return Scalar(self.value + other.value)
    def detach(self):
        return self
    def item(self):
        return self.value
    def backward(self):
        pass  # Simulated scalar, no autograd or model execution.


def fake_torch():
    return types.SimpleNamespace(Tensor=Vector, no_grad=lambda: lambda fn: fn,
        zeros_like=lambda value: Vector([0] * value.numel()),
        isfinite=lambda value: Scalar(math.isfinite(value.value)))


class ClientMetricTests(unittest.TestCase):
    def test_batch_weighted_components_without_extra_local_evaluation(self):
        modules = {
            'torch': fake_torch(), 'algorithms.BaseClientTrainer': types.SimpleNamespace(BaseClientTrainer=object),
            'algorithms.fedproc.criterion': types.SimpleNamespace(PrototypeContrastiveLoss=object),
            'algorithms.fedproc.config': config,
        }
        with mock.patch.dict(sys.modules, modules):
            module = load_file('algorithms.fedproc.client_test', ROOT / 'src/algorithms/fedproc/ClientTrainer.py')
        client = module.ClientTrainer.__new__(module.ClientTrainer)
        client.model = mock.MagicMock()
        logits = mock.MagicMock()
        logits.detach.return_value.argmax.return_value.__eq__.return_value.sum.return_value.item.return_value = 1
        client.model.return_value = (logits, object())
        client.datasize, client.local_epochs, client.device = 3, 2, 'cpu'
        client.round_idx, client.alpha_rounds = 1, 100
        client.algo_params = {'metrics_only': True}
        client.global_prototypes = []
        client.optimizer = mock.Mock()
        client._get_local_stats = mock.Mock(side_effect=AssertionError('extra evaluation'))
        client.trainloader = []
        for size in (2, 1):
            data, target = mock.Mock(), mock.Mock()
            data.to.return_value = data
            target.to.return_value = target
            target.long.return_value = target
            target.numel.return_value = size
            client.trainloader.append((data, target))
        client.criterion = mock.Mock(side_effect=[Scalar(v) for v in (1, 4, 1, 4)])
        client.prototype_criterion = mock.Mock(side_effect=[Scalar(v) for v in (2, 5, 2, 5)])
        totals, size = client.train()
        self.assertEqual(size, 3)
        self.assertEqual(totals['seen'], 6)
        self.assertEqual(totals['correct'], 4)
        self.assertAlmostEqual(totals['ce_loss_sum'], 12)
        self.assertAlmostEqual(totals['prototype_loss_sum'], 18)
        self.assertAlmostEqual(totals['weighted_ce_loss_sum'], .12)
        self.assertAlmostEqual(totals['weighted_prototype_loss_sum'], 17.82)
        self.assertAlmostEqual(totals['loss_sum'], 17.94)
        self.assertEqual(client.model.call_count, 4)
        client._get_local_stats.assert_not_called()
        client.criterion.side_effect = [Scalar(float('nan'))]
        client.prototype_criterion.side_effect = [Scalar(1)]
        with self.assertRaises(FloatingPointError):
            client.train()


class FakeBase:
    def _aggregation(self, weights, sizes):
        result = weights[0]['w'] * (sizes[0] / sum(sizes))
        for weight, size in zip(weights[1:], sizes[1:]):
            result = result + weight['w'] * (size / sum(sizes))
        return {'w': result}
    def _results_updater(self, combined, local):
        for key, value in local.items():
            combined.setdefault(key, []).append(value)
        return combined


class ServerMetricTests(unittest.TestCase):
    def test_initialization_refresh_missing_classes_and_full_report_communication(self):
        torch_stub = fake_torch()
        torch_stub.nn = types.SimpleNamespace(functional=types.ModuleType('torch.nn.functional'))
        wandb = types.SimpleNamespace(run=types.SimpleNamespace(summary={}), define_metric=mock.Mock(), log=mock.Mock())
        modules = {
            'torch': torch_stub, 'torch.nn': torch_stub.nn,
            'torch.nn.functional': torch_stub.nn.functional, 'wandb': wandb,
            'algorithms.BaseServer': types.SimpleNamespace(BaseServer=FakeBase),
            'algorithms.fedproc.ClientTrainer': types.SimpleNamespace(ClientTrainer=object),
            'algorithms.fedproc.config': config,
            'algorithms.fedproc.model': types.SimpleNamespace(ModelWithFeatures=object, ModelWithProjection=object),
        }
        with mock.patch.dict(sys.modules, modules):
            utils = load_file('fedproc_utils_test', ROOT / 'src/algorithms/fedproc/utils.py')
            report = load_file('fedproc_reporting_test', ROOT / 'src/algorithms/fedavg/reporting.py')
            with mock.patch.dict(sys.modules, {'algorithms.fedproc.utils': utils, 'algorithms.fedavg.reporting': report}):
                module = load_file('algorithms.fedproc.server_test', ROOT / 'src/algorithms/fedproc/Server.py')
                server = module.Server.__new__(module.Server)
                server.cfg = config.resolve_config({'metrics_only': True, 'aggregation': 'sampled', 'use_project_head': True})
                server.num_classes = server.n_clients = 3
                server.device = 'cpu'
                server.global_prototypes = None
                server.prototype_seen_classes = set()
                server.prototype_total_bytes = server.prototype_total_samples = 0
                server.prototype_total_seconds = 0.0
                server.server_results = {'client_history': [], 'test_accuracy': []}
                server.model = mock.Mock()
                server.model.state_dict.return_value = {'w': Vector([1, 1])}
                server.model.parameters.return_value = [Vector([1, 1])]
                server.optimizer = mock.Mock()
                server.optimizer.state_dict.return_value = {}
                server.optimizer.param_groups = [{'lr': .01}]
                server.scheduler, server.testloader, server.n_rounds = None, None, 2
                events = []
                sizes = {0: 10, 1: 20, 2: 30}
                class Client:
                    model = mock.Mock()
                    def upload_prototypes(self):
                        events.append(('extract', server.selected))
                        size = sizes[server.selected]
                        return {'feature_sums': {server.selected: Vector([size, size * 2])},
                                'class_counts': {server.selected: size}}
                    def upload_local(self):
                        return {'w': Vector([server.selected + 1] * 2)}
                    def download_global(self, weights, optimizer, prototypes, step):
                        self.step = step
                    def train(self):
                        events.append(('train', server.selected))
                        seen = sizes[server.selected] * 3
                        ce, proto = config.loss_weights(self.step, 100)
                        return {'seen': seen, 'correct': seen, 'loss_sum': (ce + proto * 2) * seen,
                                'ce_loss_sum': seen, 'prototype_loss_sum': seen * 2,
                                'weighted_ce_loss_sum': ce * seen, 'weighted_prototype_loss_sum': proto * 2 * seen}, sizes[server.selected]
                    def reset(self):
                        pass
                server.client = Client()
                server._set_client_data = lambda idx: setattr(server, 'selected', idx)
                server._print_start = lambda: None
                server._client_sampling = lambda step: ([2, 0] if step == 0 else [1])
                metrics = {'global/top1': .5, 'global/loss': 1., 'global/macro_f1': .4, 'global/worst20_recall': .3}
                with tempfile.TemporaryDirectory() as directory, mock.patch.object(report, 'synchronize'), \
                        mock.patch.object(report, 'log_partition'), mock.patch.object(report, 'evaluate', return_value=metrics), \
                        contextlib.redirect_stdout(io.StringIO()):
                    server.experiment_config = {'batch_protocol': {'output_dir': directory},
                        'train_setups': {'seed': 2022, 'algo': {'name': 'fedproc'}}}
                    server.run()
                    with (Path(directory) / 'metrics.csv').open() as handle:
                        rows = list(csv.DictReader(handle))
                    self.assertEqual({p.suffix for p in Path(directory).iterdir()}, {'.csv', '.jsonl'})
                self.assertEqual(events, [('extract', 2), ('extract', 0), ('train', 2), ('train', 0),
                                          ('extract', 2), ('extract', 0), ('train', 1), ('extract', 1)])
                # Each sparse upload is 8 feature bytes + 16 label/count bytes.
                self.assertEqual(int(rows[0]['fedproc/prototype_upload_bytes']), 96)
                self.assertEqual(int(rows[0]['fedproc/prototype_download_bytes']), 48)
                self.assertEqual(int(rows[0]['fedproc/prototype_uninitialized_classes']), 1)
                self.assertEqual(int(rows[1]['fedproc/prototype_initialization_upload_bytes']), 0)
                self.assertEqual(int(rows[1]['fedproc/prototype_preserved_classes']), 2)
                self.assertEqual(int(rows[1]['fedproc/prototype_uninitialized_classes']), 0)
                self.assertEqual(server.prototype_total_samples, 100)
                self.assertEqual(server.prototype_total_bytes, 192)
                self.assertEqual(server.batch_summary['communication_bytes'], 240)
                self.assertEqual(server.batch_summary['aggregation'], 'sampled')
                self.assertTrue(server.batch_summary['use_project_head'])
                self.assertEqual(int(rows[1]['train/processed_samples']), 180)
                self.assertEqual([v.values for v in server.global_prototypes], [[1, 2]] * 3)
                first_weights = server.model.load_state_dict.call_args_list[0].args[0]['w'].values
                self.assertEqual(first_weights, [2.5, 2.5])  # Actual selected sizes 30 and 10.


if __name__ == '__main__':
    unittest.main()
