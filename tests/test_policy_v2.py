import tempfile
import unittest
from pathlib import Path

from phish_detector.policy_v2 import ThompsonSamplingPolicy


class TestPolicyV2(unittest.TestCase):
    def test_thompson_sampling_is_deterministic_with_seed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            policy_path = Path(tmp_dir) / "policy.json"
            policy_a = ThompsonSamplingPolicy(str(policy_path), seed=123)
            decision_a = policy_a.select_action(0.72, 12)

            policy_b = ThompsonSamplingPolicy(str(policy_path), seed=123)
            decision_b = policy_b.select_action(0.72, 12)

            self.assertEqual(decision_a.action, decision_b.action)
            self.assertEqual(decision_a.confidence, decision_b.confidence)


if __name__ == "__main__":
    unittest.main()
