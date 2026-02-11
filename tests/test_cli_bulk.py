import csv
import os
import tempfile
import unittest

from phish_detector.cli import main as cli_main


class TestCliBulk(unittest.TestCase):
    def test_bulk_processing_keeps_file_open(self) -> None:
        os.environ["GOJO_DISABLE_ENRICHMENT"] = "1"
        with tempfile.TemporaryDirectory() as tmp_dir:
            input_path = os.path.join(tmp_dir, "input.csv")
            output_path = os.path.join(tmp_dir, "output.csv")
            with open(input_path, "w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["url"])
                writer.writeheader()
                writer.writerow({"url": "example.com"})

            args = [
                "--input-csv",
                input_path,
                "--output",
                output_path,
                "--output-format",
                "csv",
                "--ml-mode",
                "none",
            ]
            self.assertEqual(cli_main(args), 0)

            with open(output_path, "r", encoding="utf-8", newline="") as handle:
                output_rows = list(csv.DictReader(handle))
            self.assertEqual(len(output_rows), 1)


if __name__ == "__main__":
    unittest.main()
