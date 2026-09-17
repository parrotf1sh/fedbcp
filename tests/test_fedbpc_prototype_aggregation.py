import importlib.util
from pathlib import Path
import unittest

import torch
import torch.nn.functional as F


PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "algorithms"
    / "fedbpc"
    / "utils.py"
)
SPEC = importlib.util.spec_from_file_location("fedbpc_utils", PATH)
fedbpc_utils = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fedbpc_utils)
aggregate_prototypes = fedbpc_utils.aggregate_prototypes


class PrototypeAggregationTests(unittest.TestCase):
    def _payload(self, prototype, confidence=1.0, count=10.0):
        return {
            "prototypes": torch.tensor([prototype], dtype=torch.float),
            "prototype_valid": torch.tensor([True]),
            "confidence": torch.tensor([confidence], dtype=torch.float),
            "class_counts": torch.tensor([count], dtype=torch.float),
        }

    def _aggregate(self, payloads, **overrides):
        options = {
            "prototype_payloads": payloads,
            "num_classes": 1,
            "global_prototypes": torch.tensor([[1.0, 0.0]]),
            "prototype_valid": torch.tensor([True]),
            "prototype_age": torch.tensor([2]),
            "proto_momentum": 0.5,
            "count_shrinkage": 0.0,
            "min_proto_contributors": 1,
            "max_prototype_age": 5,
            "singleton_update_enabled": True,
            "singleton_update_scale": 2.0 / 3.0,
        }
        options.update(overrides)
        return aggregate_prototypes(**options)

    def test_singleton_update_uses_reduced_step_and_resets_age(self):
        prototypes, valid, age, metrics = self._aggregate(
            [self._payload([0.0, 1.0])]
        )
        expected = F.normalize(torch.tensor([[2.0 / 3.0, 1.0 / 3.0]]), dim=1)

        self.assertTrue(torch.allclose(prototypes, expected, atol=1e-6))
        self.assertTrue(valid.item())
        self.assertEqual(age.item(), 0)
        self.assertAlmostEqual(
            metrics["diag_proto_class_000_effective_momentum"],
            2.0 / 3.0,
            places=6,
        )
        self.assertEqual(metrics["prototype_singleton_damped_count"], 1.0)
        self.assertEqual(metrics["diag_proto_class_000_singleton_damped"], 1.0)
        self.assertGreater(
            metrics["prototype_singleton_applied_update_cosine_mean"], 0.0
        )

    def test_multi_contributor_update_preserves_base_momentum(self):
        payloads = [
            self._payload([0.0, 1.0]),
            self._payload([0.0, 1.0]),
        ]
        adaptive, _, _, adaptive_metrics = self._aggregate(payloads)
        baseline, _, _, _ = self._aggregate(
            payloads,
            singleton_update_enabled=False,
        )

        self.assertTrue(torch.allclose(adaptive, baseline, atol=1e-7))
        self.assertAlmostEqual(
            adaptive_metrics["diag_proto_class_000_effective_momentum"],
            0.5,
            places=7,
        )
        self.assertEqual(adaptive_metrics["prototype_singleton_damped_count"], 0.0)
        self.assertEqual(adaptive_metrics["prototype_multi_update_count"], 1.0)

    def test_new_singleton_class_is_initialized_without_damping(self):
        prototypes, valid, age, metrics = self._aggregate(
            [self._payload([0.0, 1.0])],
            global_prototypes=torch.zeros(1, 2),
            prototype_valid=torch.tensor([False]),
            prototype_age=torch.tensor([0]),
        )

        self.assertTrue(torch.allclose(prototypes, torch.tensor([[0.0, 1.0]])))
        self.assertTrue(valid.item())
        self.assertEqual(age.item(), 0)
        self.assertEqual(metrics["diag_proto_class_000_effective_momentum"], 0.0)
        self.assertEqual(metrics["diag_proto_class_000_singleton_damped"], 0.0)

    def test_disabled_or_unit_scale_matches_original_update(self):
        payloads = [self._payload([0.0, 1.0])]
        disabled, _, _, _ = self._aggregate(
            payloads,
            singleton_update_enabled=False,
        )
        unit_scale, _, _, unit_scale_metrics = self._aggregate(
            payloads,
            singleton_update_scale=1.0,
        )
        expected = F.normalize(torch.tensor([[0.5, 0.5]]), dim=1)

        self.assertTrue(torch.allclose(disabled, expected, atol=1e-7))
        self.assertTrue(torch.allclose(unit_scale, expected, atol=1e-7))
        self.assertEqual(
            unit_scale_metrics["prototype_singleton_damped_count"], 0.0
        )

    def test_updated_prototype_remains_normalized(self):
        prototypes, _, _, metrics = self._aggregate(
            [self._payload([-0.3, 0.7], confidence=0.2, count=2.0)]
        )

        self.assertTrue(torch.allclose(torch.norm(prototypes, dim=1), torch.ones(1)))
        self.assertEqual(metrics["prototype_current_round_coverage"], 1.0)


if __name__ == "__main__":
    unittest.main()
