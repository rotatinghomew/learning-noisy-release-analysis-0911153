import unittest
from app.catalog import local_packet
from app.domain import ANALYSES, validate
from app.engine import analyze


class DomainTests(unittest.TestCase):
    def setUp(self):
        self.packet = local_packet("data/current.json")

    def result(self):
        return analyze(self.packet, next(iter(ANALYSES)))["result"]

    def test_negative_release_preserved(self):
        values = [row["released_value"] for release in self.result()["releases"] for row in release["measurements"]]
        self.assertTrue(any(value < 0 for value in values))

    def test_noise_covariance_is_not_lost_in_source(self):
        first, second = self.packet["releases"][:2]
        self.assertIn("shared", first["measurements"][0]["loading"])
        self.assertIn("shared", second["measurements"][0]["loading"])
