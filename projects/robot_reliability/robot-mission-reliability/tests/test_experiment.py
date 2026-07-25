import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from robot_lab.evaluation import evaluate_run
from robot_lab.replay import replay_events
from robot_lab.simulation import RunConfig, run_simulation
from robot_lab.experiment import run_one_case, run_experiment_matrix
from robot_lab.scenarios import SCENARIOS


class ExperimentTests(unittest.TestCase):
    def test_run_is_replayable_and_detects_slip(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "v1.jsonl"
            run_simulation(path, RunConfig(software_version="v1"))
            events = list(replay_events(path))
            metrics = evaluate_run(path)

            self.assertGreater(len(events), 100)
            self.assertGreater(metrics.telemetry_samples, 100)
            self.assertGreaterEqual(metrics.detector_alerts, 1)
            self.assertIsNotNone(metrics.detection_latency_s)
            self.assertEqual(metrics.log_sequence_gaps, 0)

    def test_v2_improves_localization_and_mission_outcome(self) -> None:
        with TemporaryDirectory() as directory:
            v1_path = Path(directory) / "v1.jsonl"
            v2_path = Path(directory) / "v2.jsonl"
            run_simulation(v1_path, RunConfig(software_version="v1"))
            run_simulation(v2_path, RunConfig(software_version="v2"))
            v1 = evaluate_run(v1_path)
            v2 = evaluate_run(v2_path)

            self.assertLess(v2.localization_rmse_m, v1.localization_rmse_m)
            self.assertFalse(v1.mission_success)
            self.assertTrue(v2.mission_success)
            self.assertIsNone(v1.completion_time_s)
            self.assertIsNone(v1.path_efficiency)
            self.assertGreater(v1.duration_s, 0.0)

            self.assertEqual(v2.completion_time_s, v2.duration_s)
            self.assertIsNotNone(v2.path_efficiency)

    def test_run_one_case_writes_and_evaluates_log(self) -> None:
        scenarios_by_name = {scenario.name: scenario for scenario in SCENARIOS}
        scenario = scenarios_by_name["nominal"]
        with TemporaryDirectory() as directory:
            output_dir = Path(directory)
            result = run_one_case(
                output_dir=output_dir,
                scenario=scenario,
                software_version="v1",
                seed=2,
            )
            expected_log_path = (
                output_dir
                / "runs"
                / "nominal"
                / "v1_seed_2.jsonl"
            )
            metrics = result.metrics
            self.assertEqual(
                metrics.completion_time_s,
                metrics.duration_s,
                )
            self.assertGreater(metrics.path_length_m, 0.0)
            self.assertIsNotNone(metrics.path_efficiency)
            self.assertAlmostEqual(metrics.path_efficiency, 1.0, places=6)
            self.assertAlmostEqual(metrics.cross_track_rmse_m, 0.0, places=6)
            self.assertAlmostEqual(
                metrics.cross_track_max_error_m,
                0.0,
                places=6,
                )
            self.assertEqual(result.scenario_name, scenario.name)
            self.assertEqual(result.software_version, "v1")
            self.assertEqual(result.seed, 2)
            self.assertEqual(result.log_path, expected_log_path)
            self.assertEqual(result.metrics.software_version, "v1")
            self.assertGreater(result.metrics.telemetry_samples, 0)
            self.assertTrue(result.log_path.exists())
            self.assertGreater(result.log_path.stat().st_size, 0)

    def test_run_experiment_matrix_covers_all_cases(self) -> None:
        software_versions = ("v1", "v2")
        scenarios = SCENARIOS
        with TemporaryDirectory() as directory:
            output_dir = Path(directory)
            results = run_experiment_matrix(
                output_dir=output_dir,
                scenarios=scenarios,
                software_versions=software_versions,
            )

            actual_cases = {
                (
                    result.scenario_name,
                    result.software_version,
                    result.seed,
                )
                for result in results
            }

            expected_cases = {
                (
                    scenario.name,
                    software_version,
                    seed,
                )
                for scenario in scenarios
                for seed in scenario.seeds
                for software_version in software_versions
            }

            self.assertEqual(len(results), 40)
            self.assertEqual(actual_cases, expected_cases)
            self.assertTrue(
                all(result.log_path.exists() for result in results)
            )