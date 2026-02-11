import unittest

from phish_detector.scoring import binary_label_for_score, label_for_score


class TestScoring(unittest.TestCase):
    def test_label_boundaries(self) -> None:
        self.assertEqual(label_for_score(0), "green")
        self.assertEqual(label_for_score(25), "green")
        self.assertEqual(label_for_score(26), "yellow")
        self.assertEqual(label_for_score(60), "yellow")
        self.assertEqual(label_for_score(61), "red")

    def test_binary_label_boundary(self) -> None:
        self.assertEqual(binary_label_for_score(60), "legit")
        self.assertEqual(binary_label_for_score(61), "phish")


if __name__ == "__main__":
    unittest.main()
