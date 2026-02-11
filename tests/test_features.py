import unittest

from phish_detector.features import extract_features, get_feature_schema, load_suspicious_tlds, vectorize_features
from phish_detector.parsing import parse_url


class TestFeatures(unittest.TestCase):
    def test_vector_schema_length(self) -> None:
        parsed = parse_url("https://paypa1-login.example.com")
        features = extract_features(parsed, load_suspicious_tlds())
        vector = vectorize_features(features)
        self.assertEqual(len(vector), len(get_feature_schema()))

    def test_homoglyph_flag(self) -> None:
        parsed = parse_url("https://paypa1-login.example.com")
        features = extract_features(parsed, load_suspicious_tlds())
        self.assertTrue(features["has_homoglyph"])


if __name__ == "__main__":
    unittest.main()
