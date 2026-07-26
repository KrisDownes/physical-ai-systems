import unittest
from pathlib import Path
from statistics import mean
from tempfile import TemporaryDirectory

from robot_lab.experiment import run_experiment_matrix

class BehaviorTests(unittest.TestCase):
    def test_seeded_aggregate_relationships(self) -> None:
        with TemporaryDirectory() as directory:
            output_dir = Path(directory)
            results = run_experiment_matrix(output_dir)

            mild_v1 = [
                result
                for result in results
                if result.scenario_name == "mild_slip"
                and result.software_version == "v1"
            ]

            mild_v2 = [
                result
                for result in results
                if result.scenario_name == "mild_slip"
                and result.software_version == "v2"
            ]

            severe_v1 = [
                result
                for result in results
                if result.scenario_name == "severe_slip"
                and result.software_version == "v1"
            ]

            severe_v2 = [
                result
                for result in results
                if result.scenario_name == "severe_slip"
                and result.software_version == "v2"
            ]

            nominal_v2 = [
                result
                for result in results
                if result.scenario_name == "nominal"
                and result.software_version == "v2"
            ]

            self.assertEqual(len(mild_v1), 5)
            self.assertEqual(len(mild_v2), 5)
            self.assertEqual(len(severe_v1), 5)
            self.assertEqual(len(severe_v2), 5)
            self.assertEqual(len(nominal_v2), 5)

            mild_v1_success_rate = (
                sum(result.metrics.mission_success for result in mild_v1)
                / len(mild_v1)
            )

            mild_v2_success_rate = (
                sum(result.metrics.mission_success for result in mild_v2)
                / len(mild_v2)
            )

            self.assertGreater(
                mild_v2_success_rate,
                mild_v1_success_rate,
            )

            severe_v2_duration = (
                mean(result.metrics.duration_s for result in severe_v2)
            )

            nominal_v2_duration = (
                mean(result.metrics.duration_s for result in nominal_v2)
            )

            self.assertGreater(
                severe_v2_duration,
                nominal_v2_duration,
            )

            severe_v2_cte = (
                mean(result.metrics.cross_track_rmse_m for result in severe_v2)
            )

            nominal_v2_cte = (
                mean(result.metrics.cross_track_rmse_m for result in nominal_v2)
            )

            self.assertGreater(
                severe_v2_cte,
                nominal_v2_cte,
            )

            mild_v1_waypoints = (
                mean(result.metrics.waypoints_reached for result in mild_v1)
            )

            severe_v1_waypoints = (
                mean(result.metrics.waypoints_reached for result in severe_v1)
            )

            self.assertLess(
                severe_v1_waypoints,
                mild_v1_waypoints,
            )
