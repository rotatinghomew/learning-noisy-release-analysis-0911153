import unittest
from app.catalog import local_packet
from app.releases import release_profile as profile


class ContextTests(unittest.TestCase):
    def setUp(self):
        self.packet = local_packet("data/current.json")

    def test_covariance_retains_cross_release_dependency(self):
        result = profile(self.packet)
        matrix = result["known_noise_covariance"]
        self.assertTrue(result["correlated_cross_release_pairs"])
        self.assertEqual(matrix, [list(column) for column in zip(*matrix)])

    def test_superseded_release_is_not_current(self):
        self.assertEqual(profile(self.packet)["current_releases"], ["r1", "r3"])
