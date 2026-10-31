"""SwanLab lifecycle stand-ins: opt-in JSON summary, no training or network."""
import contextlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]


@contextlib.contextmanager
def lifecycle():
    run = types.SimpleNamespace(id='test-id', url='https://example.test/run', summary={}, finish=mock.Mock())
    tracking = types.SimpleNamespace(__version__='test', init=mock.Mock(return_value=run))
    def write_json(path, value):
        Path(path).write_text(json.dumps(value, allow_nan=False))
    stubs = {'experiment_tracker': tracking, 'torch': types.SimpleNamespace(__version__='test', version=types.SimpleNamespace(cuda=None)),
             'algorithms.fedavg.reporting': types.SimpleNamespace(write_json=write_json),
             'fedavg_data': types.SimpleNamespace(prepare_data=mock.Mock(), TINY_URL='test-url')}
    with mock.patch.dict(sys.modules, stubs):
        spec = importlib.util.spec_from_file_location('fedsa_lifecycle_test', ROOT / 'src/fedavg_batch.py')
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        yield module, run


class LifecycleTests(unittest.TestCase):
    def config(self, directory, algorithm='fedsa', flags=True):
        protocol = {'output_dir': directory}
        if flags:
            protocol.update(fedsa_batch=True, write_metrics_files=True)
        return {'train_setups': {'algo': {'name': algorithm}}, 'batch_protocol': protocol, 'swanlab_setups': {}}

    def test_success_writes_summary_and_failure_preserves_error(self):
        with lifecycle() as (module, run), tempfile.TemporaryDirectory() as directory:
            module.run_tracked(self.config(directory), lambda cfg: {'final_top1': .6})
            summary = json.loads((Path(directory) / 'summary.json').read_text())
            self.assertEqual(summary['status'], 'success')
            self.assertEqual(summary['swanlab_run_id'], 'test-id')
            run.finish.assert_called_with(exit_code=0)
            with self.assertRaisesRegex(RuntimeError, 'original failure'):
                module.run_tracked(self.config(directory), mock.Mock(side_effect=RuntimeError('original failure')))
            summary = json.loads((Path(directory) / 'summary.json').read_text())
            self.assertEqual(summary['status'], 'failed')
            self.assertIn('original failure', summary['error'])
            run.finish.assert_called_with(exit_code=1)

    def test_tracking_only_algorithms_keep_existing_default(self):
        for algorithm in ('fedsa', 'fedproto', 'tinyproto'):
            with self.subTest(algorithm=algorithm), lifecycle() as (module, run), tempfile.TemporaryDirectory() as directory:
                module.run_tracked(self.config(directory, algorithm, flags=False), lambda cfg: {'final_top1': .6})
                self.assertEqual(list(Path(directory).iterdir()), [])
                self.assertEqual(run.summary['status'], 'success')

    def test_both_fedsa_flags_required_and_interrupt_is_reported(self):
        with lifecycle() as (module, run), tempfile.TemporaryDirectory() as directory:
            cfg = self.config(directory); del cfg['batch_protocol']['write_metrics_files']
            module.run_tracked(cfg, lambda cfg: {'final_top1': .6})
            self.assertEqual(list(Path(directory).iterdir()), [])
            with self.assertRaises(KeyboardInterrupt):
                module.run_tracked(self.config(directory), mock.Mock(side_effect=KeyboardInterrupt()))
            self.assertEqual(json.loads((Path(directory) / 'summary.json').read_text())['status'], 'interrupted')
            run.finish.assert_called_with(exit_code=130)


if __name__ == '__main__':
    unittest.main()
