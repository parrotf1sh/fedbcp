"""FedRCL configuration regressions using the project's actual JSON loader."""
import importlib.util
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "src/config/fedrcl.json"


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


config = load_file("fedrcl_config_test", ROOT / "src/algorithms/fedrcl/config.py")
utils = load_file("fedrcl_utils_test", ROOT / "src/utils.py")


class FedRCLConfigTests(unittest.TestCase):
    def test_confloader_matches_plain_json_for_each_model_mapping(self):
        params = utils.ConfLoader(str(CONFIG_PATH)).opt.train_setups.algo.params
        plain = json.loads(CONFIG_PATH.read_text())["train_setups"]["algo"]["params"]
        self.assertIsInstance(params["rcl_loss"]["pairs"][0],
                              utils.ConfLoader.DictWithAttributeAccess)
        for num_layers in (4, 8):
            with self.subTest(num_layers=num_layers):
                resolved = config.resolve_config(params, num_layers)
                self.assertEqual(resolved, config.resolve_config(plain, num_layers))
                self.assertEqual(resolved["rcl_loss"]["branch_level"],
                                 list(range(num_layers - 1)))
                self.assertIs(type(resolved["rcl_loss"]["pairs"][0]), dict)

    def test_nested_pairs_and_lists_do_not_alias_input_or_defaults(self):
        params = utils.ConfLoader(str(CONFIG_PATH)).opt.train_setups.algo.params
        params["rcl_loss"]["branch_level"] = [0, 2]
        params["rcl_loss"]["pairs"][0]["branch_level"] = [0]
        before = json.dumps(params, sort_keys=True)
        defaults = json.dumps(config.DEFAULTS, sort_keys=True)
        resolved = config.resolve_config(params, 4)
        resolved["ce_loss"]["weight"] = 3.0
        resolved["rcl_loss"]["branch_level"].append(1)
        resolved["rcl_loss"]["pairs"][0]["branch_level"].append(2)
        resolved["rcl_loss"]["pairs"][1]["threshold"] = 0.8
        self.assertEqual(json.dumps(params, sort_keys=True), before)
        self.assertEqual(json.dumps(config.DEFAULTS, sort_keys=True), defaults)
        self.assertEqual(config.resolve_config(params, 4)["ce_loss"]["weight"], 1.0)

    def test_false_null_and_zero_overrides_keep_their_values(self):
        params = utils.ConfLoader(str(CONFIG_PATH)).opt.train_setups.algo.params
        params["prox_loss"] = None
        params["rcl_loss"]["branch_level"] = False
        params["rcl_loss"]["pairs"][0]["temp"] = 0
        resolved = config.resolve_config(params, 4)
        self.assertIsNone(resolved["prox_loss"])
        self.assertIs(resolved["rcl_loss"]["branch_level"], False)
        self.assertEqual(resolved["rcl_loss"]["pairs"][0]["temp"], 0)
        self.assertEqual(resolved["rcl_loss"]["temp"], 0.05)


if __name__ == "__main__":
    unittest.main()
