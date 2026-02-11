import os
import unittest

from webapp.app import app


class TestWebApp(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["GOJO_DISABLE_HEARTBEAT"] = "1"
        app.config["TESTING"] = True
        app.config["WTF_CSRF_ENABLED"] = False
        self.client = app.test_client()

    def test_health_endpoint(self) -> None:
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)

    def test_download_path_traversal_blocked(self) -> None:
        response = self.client.get("/download/../secrets.txt")
        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
