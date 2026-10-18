"""FedSA persistent-state and reporting checks with NumPy stand-ins, no training."""
import contextlib
import copy
import csv
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np
from test_fednh_metrics import Tensor, Base, fake_torch, load_file, ROOT


@contextlib.contextmanager
def modules():
    torch = fake_torch()
    run = types.SimpleNamespace(disabled=False, config={}, summary={}, define_metric=mock.Mock(), log=mock.Mock())
    config = load_file('fedsa_metrics_config', ROOT / 'src/algorithms/fedsa/config.py')
    logging = load_file('fedsa_logging_test', ROOT / 'src/experiment_logging.py')
    stubs = {'torch': torch, 'torch.nn': torch.nn, 'torch.nn.functional': torch.nn.functional,
             'wandb': types.SimpleNamespace(run=run), 'experiment_logging': logging,
             'longtail_metrics': types.SimpleNamespace(groups_from_counts=mock.Mock()),
             'algorithms.BaseServer': types.SimpleNamespace(BaseServer=Base),
             'algorithms.fedsa.ClientTrainer': types.SimpleNamespace(ClientTrainer=object),
             'algorithms.fedsa.config': config}
    with contextlib.ExitStack() as stack:
        stack.enter_context(mock.patch.dict(sys.modules, stubs))
        proto_utils = load_file('fedsa_proto_utils_test', ROOT / 'src/algorithms/fedproto/utils.py')
        stack.enter_context(mock.patch.dict(sys.modules, {'algorithms.fedproto.utils': proto_utils}))
        utils = load_file('algorithms.fedsa.utils_test', ROOT / 'src/algorithms/fedsa/utils.py')
        stack.enter_context(mock.patch.object(utils, 'preserve_evaluation_rng', lambda loaders: contextlib.nullcontext()))
        stack.enter_context(mock.patch.dict(sys.modules, {'algorithms.fedsa.utils': utils}))
        measures = load_file('algorithms.fedsa.measures_test', ROOT / 'src/algorithms/fedsa/measures.py')
        common = load_file('fedsa_common_reporting_test', ROOT / 'src/algorithms/fedavg/reporting.py')
        batch = load_file('algorithms.fedsa.batch_test', ROOT / 'src/algorithms/fedsa/batch.py')
        stack.enter_context(mock.patch.dict(sys.modules, {'algorithms.fedsa.measures': measures,
                                 'algorithms.fedavg.reporting': common, 'algorithms.fedsa.batch': batch}))
        server = load_file('algorithms.fedsa.server_test', ROOT / 'src/algorithms/fedsa/Server.py')
        reporting = load_file('algorithms.fedsa.reporting_test', ROOT / 'src/algorithms/fedsa/reporting.py')
        stack.enter_context(mock.patch.object(common, 'synchronize'))
        stack.enter_context(mock.patch.object(common, 'log_partition'))
        stack.enter_context(mock.patch.object(reporting, 'class_margins', lambda anchors: {'min': 1., 'mean': 1., 'max': 1.}))
        yield config, utils, server, reporting, batch, measures, run


def prediction(value, classifier=False):
    result = dict(accuracy=value, macro_f1=value, worst20_recall=value,
                  class_accuracy=[value, value], class_count=[1, 1],
                  samples=2, evaluated_classes=2)
    if classifier: result['cross_entropy'] = 1.
    return result


class FedSAMetricsTests(unittest.TestCase):
    def test_persistent_clients_equal_prototypes_and_all_client_evaluation(self):
        with modules() as (config, _, module, __, batch, ___, ____):
            server = module.Server.__new__(module.Server)
            server.cfg = config.resolve_config({})
            server.n_clients, server.num_classes, server.device = 3, 2, 'cpu'
            server.initial_weights = {'w': Tensor([0.])}
            server.local_weights, server.local_prototypes = {}, {}
            server.semantic_anchors = Tensor([[10., 0.], [0., 10.]])
            server.global_prototypes = None
            server.optimizer = mock.Mock(); server.optimizer.state_dict.return_value = {}
            server.batch_tracking = True; server.batch_eval_bytes = 0
            server.groups = {}; server.testloader = object()
            server.data_distributed = {'local': {i: {'datasize': size, 'test': None} for i, size in enumerate([1, 9, 3])}}
            downloads, histories, evaluated = [], [], []
            evaluation_state = {}
            class Client:
                model = mock.Mock()
                def download_global(self, weights, optimizer, anchors, history):
                    self.weights = copy.deepcopy(weights)
                    downloads.append((self.datasize, weights['w'].item()))
                    histories.append(copy.deepcopy(history))
                def train(self): return {'total_loss': 1.}, self.datasize
                def upload_local(self): return {'w': self.weights['w'] + self.datasize}
                def upload_prototypes(self):
                    label = 1 if self.datasize == 3 else 0
                    value = 1. if self.datasize == 1 else float(self.datasize)
                    return {label: Tensor([value, 0.])}
                def reset(self): pass
            server.client = Client()
            server.client.model.load_state_dict.side_effect = lambda weights, **kw: evaluation_state.update(weights)
            server._clients_training([0, 1])
            self.assertEqual(server.global_prototypes[0][0].item(), 5.)  # (1+9)/2, not 8.2.
            self.assertEqual(server.batch_round_upload_bytes, 32)
            np.testing.assert_allclose(server.semantic_anchors.a[0], [9.9995, 0.], atol=1e-6)
            np.testing.assert_array_equal(server.semantic_anchors.a[1], [0., 10.])
            server._clients_training([1, 2])
            self.assertEqual(downloads, [(1, 0.), (9, 0.), (9, 9.), (3, 0.)])
            self.assertIsNone(histories[0]); self.assertIsNone(histories[1]); self.assertIsNone(histories[3])
            self.assertEqual(histories[2][0][0].item(), 9.)
            self.assertEqual(server.local_weights[0]['w'].item(), 1.)
            self.assertEqual(server.local_weights[1]['w'].item(), 18.)
            saved = {i: w['w'].item() for i, w in server.local_weights.items()}
            def evaluate(model, loader, prototypes, *args):
                evaluated.append(evaluation_state['w'].item())
                self.assertEqual(set(prototypes), {0, 1})
                value = evaluation_state['w'].item() / 100
                return dict(prototype=prediction(value), classifier=prediction(value, True))
            with mock.patch.object(module, 'evaluate_client', side_effect=evaluate):
                result = server.evaluate_clients()
                self.assertEqual(server.evaluate_clients(local=True), [])
            self.assertEqual(evaluated, [1., 18., 3.])
            self.assertAlmostEqual(result['prototype']['accuracy'], 22 / 300)
            self.assertEqual(server.batch_eval_bytes, 3 * batch.prototype_bytes(server.global_prototypes))
            self.assertEqual(saved, {i: w['w'].item() for i, w in server.local_weights.items()})
            with self.assertRaises(NotImplementedError): server._aggregation([], [])
            server._clients_training([2])
            self.assertEqual(set(server.global_prototypes), {1})  # Missing class does not persist globally.

    def test_300_round_report_has_40_real_evaluations_and_no_stale_accuracy(self):
        with modules() as (config, _, __, reporting, batch, ___, run):
            server = types.SimpleNamespace(n_rounds=300, n_clients=3, num_classes=2, device='cpu', groups={},
                cfg=config.resolve_config({}), semantic_anchors=Tensor([[1., 0.], [0., 1.]]),
                initial_weights={'w': Tensor([0.])}, global_prototypes={0: Tensor([1., 0.])},
                server_results={'client_history': [], 'test_accuracy': [], 'classifier_test_accuracy': []})
            server.data_distributed = {'local': {i: {'train': types.SimpleNamespace(drop_last=False)} for i in range(3)}, 'global': {}}
            server.optimizer = types.SimpleNamespace(param_groups=[{'lr': .01}])
            server._client_sampling = lambda round_idx: [0, 1]
            server.step_scheduler = mock.Mock()
            def train(sampled):
                server.batch_round_upload_bytes = 32
                return {'total_loss': [1., 3.], 'local_gap_is_negative_infinity': [0., 1.]}
            server._clients_training = train
            eval_rounds = []
            def evaluate(loader=None, local=False):
                if local: return []
                current_round = len(server.server_results['client_history'])
                eval_rounds.append(current_round)
                server.batch_eval_bytes += 48
                value = current_round / 300
                return {'prototype': dict(prediction(value), client_accuracy_std=0.),
                        'classifier': prediction(value, True), 'clients': []}
            server.evaluate_clients = evaluate
            with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
                server.experiment_config = {'train_setups': {'seed': 2022}, 'batch_protocol': {
                    'fedsa_batch': True, 'write_metrics_files': True, 'save_checkpoints': False,
                    'output_dir': directory, 'eval_interval': 10, 'final_eval_rounds': 10,
                    'evaluation_rounds': batch.evaluation_rounds(300)}}
                summary = reporting.run_experiment(server)
                lines = [json.loads(line) for line in (Path(directory) / 'metrics.jsonl').read_text().splitlines()]
                with (Path(directory) / 'metrics.csv').open() as handle: rows = list(csv.DictReader(handle))
                self.assertEqual({p.suffix for p in Path(directory).iterdir()}, {'.json', '.csv', '.jsonl'})
            self.assertEqual(eval_rounds, batch.evaluation_rounds(300))
            self.assertEqual(len(eval_rounds), 40)
            self.assertEqual(summary['completed_rounds'], 300)
            self.assertEqual(summary['last10_training_rounds'], list(range(291, 301)))
            self.assertAlmostEqual(summary['last10_top1_mean'], 295.5 / 300)
            self.assertEqual(summary['communication_bytes'], 300 * (32 + 2 * 16))
            self.assertEqual(summary['evaluation_communication_bytes'], 40 * 48)
            self.assertEqual(summary['initial_model_distribution_bytes_estimate'], 12)
            self.assertEqual(len(lines), 300)
            self.assertEqual(rows[1]['global/top1'], '')
            self.assertNotIn('global/top1', lines[1])
            self.assertNotIn('test/prototype/accuracy', lines[1])
            self.assertNotIn('global/loss', lines[0])
            self.assertNotIn('final_loss', summary)
            self.assertEqual(summary['final_classifier_loss'], 1.)
            self.assertEqual(server.step_scheduler.call_count, 300)
            self.assertEqual(run.log.call_count, 300)
            self.assertEqual(lines[1]['train/total_loss'], 2.)  # Existing equal-client logging.

    def test_legacy_report_stays_every_round_and_wandb_only(self):
        with modules() as (config, _, __, reporting, ___, ____, run):
            server = types.SimpleNamespace(n_rounds=2, n_clients=1, num_classes=2, device='cpu', groups={},
                cfg=config.resolve_config({}), semantic_anchors=Tensor([[1., 0.], [0., 1.]]),
                global_prototypes={0: Tensor([1., 0.])}, server_results={'client_history': [], 'test_accuracy': [], 'classifier_test_accuracy': []},
                data_distributed={'local': {0: {'train': types.SimpleNamespace(drop_last=False)}}, 'global': {}},
                optimizer=types.SimpleNamespace(param_groups=[{'lr': .01}]))
            server._client_sampling = lambda i: [0]
            server._clients_training = lambda sampled: {'total_loss': [1.]}
            server.step_scheduler = mock.Mock()
            def evaluate(loader=None, local=False):
                if local: return []
                return {'prototype': dict(prediction(.5), client_accuracy_std=0.),
                        'classifier': prediction(.6, True), 'clients': []}
            server.evaluate_clients = mock.Mock(side_effect=evaluate)
            with contextlib.redirect_stdout(io.StringIO()): reporting.run_experiment(server)
            self.assertEqual(server.evaluate_clients.call_count, 3)  # Two full tests, final local test query.
            self.assertFalse(hasattr(server, 'batch_tracking'))
            self.assertNotIn('communication_bytes', server.batch_summary)

    def test_batch_rejects_invalid_schedule_and_incomplete_final_window(self):
        with modules() as (_, __, ___, ____, batch, _____, ______):
            with tempfile.TemporaryDirectory() as directory:
                protocol = {'output_dir': directory, 'eval_interval': 10, 'final_eval_rounds': 10,
                            'evaluation_rounds': [1], 'write_metrics_files': True}
                server = types.SimpleNamespace(n_rounds=300, n_clients=1, initial_weights={'w': Tensor([0.])},
                                              experiment_config={'batch_protocol': protocol})
                with self.assertRaisesRegex(ValueError, 'manifest'): batch.BatchReporter(server)
                protocol['evaluation_rounds'] = batch.evaluation_rounds(300)
                writer = batch.BatchReporter(server)
                writer.completed_rounds = 300
                writer.top1_by_round = {i: .5 for i in protocol['evaluation_rounds'] if i != 299}
                with self.assertRaisesRegex(ValueError, 'required'): writer.finish({}, 0.)


if __name__ == '__main__':
    unittest.main()
