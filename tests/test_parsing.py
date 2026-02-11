import unittest

from phish_detector.parsing import parse_url


class TestParsing(unittest.TestCase):
    def test_parse_adds_scheme(self) -> None:
        parsed = parse_url("example.com/path")
        self.assertEqual(parsed.scheme, "http")
        self.assertEqual(parsed.host, "example.com")

    def test_shortener_detection(self) -> None:
        parsed = parse_url("https://bit.ly/test")
        self.assertTrue(parsed.is_shortener)


if __name__ == "__main__":
    unittest.main()
