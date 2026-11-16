import copy
import json
import random
import unittest
from pathlib import Path

from app.catalog import local_packet
from app.domain import ANALYSES
from app.engine import analyze
from app.latent import latent_participation
from app.protocol import digest


def flat(result):
    """Flatten estimates and intervals for comparison with a numeric tolerance."""
    values = {}
    for period, block in result["periods"].items():
        for cell_id, report in block["cells"].items():
            interval = report["interval"] or {}
            values[f"cell/{period}/{cell_id}"] = (
                report["estimate"], interval.get("lower"), interval.get("upper"))
        for name, report in block["aggregates"].items():
            interval = report["interval"] or {}
            values[f"aggregate/{period}/{name}"] = (
                report["estimate"], interval.get("lower"), interval.get("upper"))
    for change in result["changes"]:
        interval = change["interval"] or {}
        values[f"change/{change['measure']}"] = (
            change["estimate"], interval.get("lower"), interval.get("upper"))
    return values


class LatentParticipationTests(unittest.TestCase):
    def setUp(self):
        self.packet = local_packet("data/current.json")

    def assert_flat_close(self, first, second, tolerance=1e-4):
        self.assertEqual(set(first), set(second))
        for key in first:
            for left, right in zip(first[key], second[key]):
                if left is None or right is None:
                    self.assertEqual(left, right, key)
                    continue
                self.assertAlmostEqual(left, right, delta=tolerance, msg=key)

    # -- registration and provenance ------------------------------------
    def test_analysis_is_registered_and_source_unchanged(self):
        before = copy.deepcopy(self.packet)
        envelope = analyze(self.packet, "latent_participation")
        self.assertEqual(self.packet, before)
        self.assertEqual(envelope["source_sha256"], digest(before))
        self.assertEqual(envelope["analysis"], "latent_participation")

    def test_raw_release_values_remain_queryable(self):
        result = latent_participation(self.packet, {})
        raw = {(row["release_id"], row["measurement"]): row["released_value"]
               for row in result["data"]["raw_release_values"]}
        for release in self.packet["releases"]:
            for measurement in release["measurements"]:
                self.assertEqual(raw[(release["release_id"], measurement["measurement"])],
                                 float(measurement["value"]))
        # A negative released value must survive into the audit trail.
        self.assertTrue(any(value < 0 for value in raw.values()))

    # -- correlated noise and duplicated releases -----------------------
    def test_duplicate_republication_adds_no_information(self):
        packet = local_packet("data/cases/boundary-2.json")
        plain = latent_participation(packet, {"release_scope": "ids",
                                              "release_ids": ["r1", "r3"]})
        copied = latent_participation(packet, {"release_scope": "ids",
                                               "release_ids": ["r1", "r1-copy", "r3"]})
        self.assert_flat_close(flat(plain), flat(copied))
        self.assertEqual(plain["fit"]["p_value"], copied["fit"]["p_value"])
        # The redundant rows are reported rather than silently double-counted.
        self.assertEqual(len(copied["noise"]["redundant_republished_measurements"]), 4)
        self.assertEqual(copied["fit"]["independent_noisy_observations"],
                         plain["fit"]["independent_noisy_observations"])

    def test_correction_sharing_noise_is_not_independent(self):
        """A revised release shares the shared/new perturbations with the old one."""
        result = latent_participation(self.packet, {"release_scope": "all"})
        pairs = result["noise"]["equal_weight_average"]["pairs"]
        self.assertTrue(pairs, "expected r2/r3 pairs to be compared")
        for pair in pairs:
            self.assertGreater(pair["shared_noise_covariance"], 0)
            self.assertGreater(pair["understatement_factor"], 1.0)

    def test_scope_options_change_which_releases_are_used(self):
        default = latent_participation(self.packet, {})
        wide = latent_participation(self.packet, {"release_scope": "all"})
        self.assertEqual(default["data"]["releases_used"], ["r1", "r3"])
        self.assertIn("r2", wide["data"]["releases_used"])
        self.assertTrue(any(entry["release_id"] == "r2"
                            for entry in default["data"]["releases_excluded"]))
        # Different evidence must produce a different fingerprint upstream.
        self.assertNotEqual(digest(default), digest(wide))

    # -- constraints -----------------------------------------------------
    def test_counts_respect_non_negativity_and_period_totals(self):
        result = latent_participation(self.packet, {})
        for period, block in result["periods"].items():
            total = sum(report["estimate"] for report in block["cells"].values())
            self.assertAlmostEqual(total, 120.0, delta=1e-6)
            for report in block["cells"].values():
                self.assertGreaterEqual(report["estimate"], -1e-9)

    def test_intersection_and_margins_are_mutually_consistent(self):
        result = latent_participation(self.packet, {})
        for period, block in result["periods"].items():
            cells = block["cells"]
            both = cells[f"{period}-both"]["estimate"]
            music_only = cells[f"{period}-music_only"]["estimate"]
            digital_only = cells[f"{period}-digital_only"]["estimate"]
            neither = cells[f"{period}-neither"]["estimate"]
            self.assertAlmostEqual(music_only + digital_only + both, block["aggregates"]["participating"]["estimate"], delta=1e-6)
            self.assertAlmostEqual(both + neither, 120.0 - music_only - digital_only, delta=1e-6)

    def test_declared_total_is_not_given_a_noise_interval(self):
        result = latent_participation(self.packet, {})
        for block in result["periods"].values():
            total = block["aggregates"]["total"]
            self.assertEqual(total["pinned_by"], "declared_constraints")
            self.assertIsNone(total["release_noise_standard_deviation"])
            self.assertEqual(total["interval"]["width"], 0.0)

    # -- conflict diagnosis ----------------------------------------------
    def test_conflicting_noiseless_releases_are_reported_not_repaired(self):
        packet = local_packet("data/cases/boundary-3.json")
        result = latent_participation(packet, {})
        self.assertFalse(result["fit"]["feasible"])
        self.assertFalse(result["fit"]["estimated"])
        contradictions = result["contradictions"]
        self.assertFalse(contradictions["structurally_feasible"])
        self.assertFalse(contradictions["jointly_consumable"])
        witness = contradictions["infeasibility_witness"]
        self.assertIsNotNone(witness)
        self.assertTrue(witness["relations"])
        # Nothing is estimated, so no tidy table is presented.
        for block in result["periods"].values():
            for report in block["cells"].values():
                self.assertIsNone(report["estimate"])
        # The offending raw value is still visible.
        raw = [row["released_value"] for row in result["data"]["raw_release_values"]]
        self.assertIn(500.0, raw)

    def test_joint_tension_lists_maximal_consistent_releases(self):
        result = latent_participation(self.packet, {"release_scope": "all",
                                                    "consistency_level": 0.5})
        contradictions = result["contradictions"]
        self.assertFalse(contradictions["jointly_consumable"])
        self.assertTrue(contradictions["maximal_consistent_release_sets"])
        self.assertTrue(contradictions["conflicting_groups"])
        for group in contradictions["conflicting_groups"]:
            self.assertIn(group["verdict"], ("statistically_tension", "structurally_inconsistent"))

    def test_duplicate_and_correction_do_not_fake_significance(self):
        packet = local_packet("data/cases/boundary-2.json")
        result = latent_participation(packet, {})
        self.assertTrue(result["contradictions"]["jointly_consumable"])

    # -- coverage vs noise uncertainty -----------------------------------
    def test_uncovered_population_is_not_filled_in(self):
        packet = copy.deepcopy(self.packet)
        packet["revision"] = "coverage-test"
        packet["releases"][0]["measurements"] = [
            row for row in packet["releases"][0]["measurements"]
            if row["measurement"] != "r1-neither"]
        packet["constraints"] = [row for row in packet["constraints"]
                                 if row["period"] != "before"]
        result = latent_participation(packet, {})
        uncovered = {row["cell_id"] for row in result["coverage"]["uncovered_cells"]}
        self.assertIn("before-neither", uncovered)
        report = result["periods"]["before"]["cells"]["before-neither"]
        self.assertFalse(report["identified"])
        self.assertIsNone(report["estimate"])
        self.assertIsNone(report["interval"])
        # The uncertainty of covered quantities is still available.
        self.assertIsNotNone(result["periods"]["before"]["cells"]["before-both"]["interval"])

    def test_uncovered_but_determined_cell_is_labelled(self):
        packet = local_packet("data/cases/boundary-1.json")
        packet["releases"][0]["measurements"] = [
            row for row in packet["releases"][0]["measurements"]
            if row["measurement"] != "r1-neither"]
        packet["constraints"].append({"kind": "linear",
                                      "coefficients": {"before-neither": 1},
                                      "value": 50})
        result = latent_participation(packet, {})
        entry = next(row for row in result["coverage"]["uncovered_cells"]
                     if row["cell_id"] == "before-neither")
        self.assertEqual(entry["status"], "determined_by_declared_constraints")
        self.assertEqual(entry["implied_by_constraints"], 50.0)

    def test_noiseless_case_is_exactly_recovered(self):
        packet = local_packet("data/cases/boundary-1.json")
        result = latent_participation(packet, {})
        expected = {
            "before-music_only": 20.0, "before-digital_only": 40.0,
            "before-both": 10.0, "before-neither": 50.0,
            "after-music_only": 30.0, "after-digital_only": 35.0,
            "after-both": 15.0, "after-neither": 40.0,
        }
        for period, block in result["periods"].items():
            for cell_id, report in block["cells"].items():
                self.assertAlmostEqual(report["estimate"], expected[cell_id], places=6)
                self.assertEqual(report["interval"]["width"], 0.0)

    # -- reproducibility -------------------------------------------------
    def test_analysis_is_deterministic(self):
        first = latent_participation(self.packet, {})
        second = latent_participation(self.packet, {})
        self.assertEqual(digest(first), digest(second))
        self.assertFalse(first["method"]["monte_carlo"])
        self.assertIsNone(first["method"]["random_seed"])

    def test_row_order_does_not_change_results(self):
        baseline = flat(latent_participation(self.packet, {"release_scope": "all"}))
        generator = random.Random(20261116)
        for _ in range(5):
            packet = copy.deepcopy(self.packet)
            generator.shuffle(packet["releases"])
            generator.shuffle(packet["cells"])
            for release in packet["releases"]:
                generator.shuffle(release["measurements"])
            candidate = latent_participation(packet, {"release_scope": "all"})
            self.assertEqual(candidate["estimand"]["periods"], ["before", "after"])
            self.assert_flat_close(baseline, flat(candidate))

    def test_changes_report_uncertainty_direction(self):
        result = latent_participation(self.packet, {})
        by_measure = {change["measure"]: change for change in result["changes"]}
        self.assertIn("participating", by_measure)
        participating = by_measure["participating"]
        self.assertIsNotNone(participating["interval"])
        self.assertIsNotNone(participating["direction_resolved"])
        # Direction resolution must agree with the interval, not the sign alone.
        interval = participating["interval"]
        expected = interval["lower"] > 0 or interval["upper"] < 0
        self.assertEqual(participating["direction_resolved"], expected)

    def test_unknown_option_is_rejected(self):
        with self.assertRaises(ValueError):
            latent_participation(self.packet, {"average_releases": True})

    def test_declared_linear_relation_is_enforced(self):
        packet = copy.deepcopy(self.packet)
        packet["constraints"].append({"kind": "linear",
                                      "coefficients": {"before-both": 1, "after-both": -1},
                                      "value": 0})
        result = latent_participation(packet, {})
        before = result["periods"]["before"]["cells"]["before-both"]["estimate"]
        after = result["periods"]["after"]["cells"]["after-both"]["estimate"]
        self.assertAlmostEqual(before, after, delta=1e-6)
        self.assertEqual(len(result["data"]["constraints_applied"]), 3)

    def test_unsupported_constraint_is_reported_not_dropped(self):
        packet = copy.deepcopy(self.packet)
        packet["constraints"].append({"kind": "mystery", "note": "unknown rule"})
        result = latent_participation(packet, {})
        unapplied = result["data"]["constraints_not_applied"]
        self.assertEqual(len(unapplied), 1)
        self.assertEqual(unapplied[0]["constraint"]["kind"], "mystery")

    def test_release_scope_ids_requires_known_release(self):
        with self.assertRaises(ValueError):
            latent_participation(self.packet, {"release_scope": "ids",
                                               "release_ids": ["does-not-exist"]})


if __name__ == "__main__":
    unittest.main()
