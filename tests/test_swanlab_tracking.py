"""SwanLab API adaptation and unchanged batch identities across all queues."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import types
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import experiment_tracker as tracking
from tracking_config import tracking_settings


class TrackingTests(unittest.TestCase):
    def test_summary_maps_to_config_and_last_step_with_native_finish_states(self):
        for code, state in [(0,'success'), (1,'crashed'), (130,'aborted')]:
            native = types.SimpleNamespace(mode='offline', id='local', config={}, alive=True,
                define_metric=mock.Mock(), log=mock.Mock(), finish=mock.Mock())
            run = tracking.Run(native)
            run.define_metric('*', step_metric='round')
            native.define_metric.assert_called_with('*', x_axis='round')
            run.log({'round':5,'loss':.2}, step=5)
            run.summary.update(final_top1=.8, status='success', error='example')
            self.assertEqual(native.config['result_summary']['final_top1'], .8)
            run.finish(exit_code=code)
            native.log.assert_called_with({'summary/final_top1':.8}, step=5)
            native.finish.assert_called_once_with(state=state, error='example' if code else None)
            run.finish(exit_code=code)
            native.finish.assert_called_once()
            self.assertIsNone(run.url)

    def test_old_config_conversion_never_reuses_remote_entity_or_run(self):
        old = {'wandb_setups': {'name':'old-name','entity':'old-team','project':'Point1',
            'dir':'/tmp/old','force':True,'save_code':False,'id':'old-id','resume':'must'}}
        with mock.patch.dict(os.environ, {}, clear=True):
            result = tracking_settings(old)
        self.assertEqual(result, {'project':'Point1','experiment_name':'old-name','logdir':'/tmp/old','mode':'online'})
        self.assertIn('entity', old['wandb_setups'])
        with mock.patch.dict(os.environ, {'SWANLAB_MODE':'offline','SWANLAB_WORKSPACE':'new-team','SWANLAB_PROJ_NAME':'New'}, clear=True):
            result = tracking_settings(old)
        self.assertEqual((result['mode'],result['workspace'],result['project']), ('offline','new-team','New'))

    def test_every_queue_retains_pre_migration_task_ids(self):
        expected = json.loads((ROOT / 'tests/fixtures/tracking_task_ids.json').read_text())
        with mock.patch.dict(os.environ, {'SWANLAB_MODE':'offline','SWANLAB_WORKSPACE':'new-team','SWANLAB_PROJ_NAME':'New'}):
            for name, snapshot in expected.items():
                spec = importlib.util.spec_from_file_location(name + '_tracking_test', ROOT / 'scripts' / (name + '.py'))
                module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
                tasks = module.build_tasks('/root/data')
                ids = [task['id'] for task in tasks]
                self.assertEqual(len(ids), snapshot['count'])
                self.assertEqual(hashlib.sha256(json.dumps(ids).encode()).hexdigest(), snapshot['sha256'], name)
                for task in tasks:
                    cfg = task['config']['swanlab_setups']
                    self.assertEqual((cfg['mode'],cfg['workspace'],cfg['project']), ('offline','new-team','New'))
                    self.assertNotIn('entity',cfg)
                    self.assertNotIn('force',cfg)


if __name__ == '__main__':
    unittest.main()
