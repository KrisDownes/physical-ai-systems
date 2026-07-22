import unittest
from robot_lab.scenarios import SCENARIOS, make_run_config


class ScenarioTests(unittest.TestCase):
    def test_scenarios_names_are_unique(self) -> None:
        names = [scenario.name for scenario in SCENARIOS]
        self.assertEqual(len(names), len(set(names)))

    def test_catalog_contains_expected_scenarios(self) -> None:
        actual_names = {scenario.name for scenario in SCENARIOS}
        expected_names = {"nominal", "mild_slip", "severe_slip", "noisy_gps"}
        self.assertEqual(actual_names, expected_names)

    def test_all_scenarios_use_expected_seeds(self) -> None:
        expected_seeds = (0, 1, 2, 3, 4)
        for scenario in SCENARIOS:
            with self.subTest(scenario=scenario.name):
                self.assertTupleEqual(scenario.seeds, expected_seeds)

    def test_no_fault_scenarios_disable_slip(self) -> None:
        scenarios_by_name = {scenario.name: scenario for scenario in SCENARIOS}
        for name in ("nominal", "noisy_gps"):
            with self.subTest(scenario=name):
                scenario = scenarios_by_name[name]
                self.assertGreater(scenario.slip_start_s, scenario.max_time_s)
                self.assertGreater(scenario.slip_end_s, scenario.slip_start_s)
                self.assertEqual(scenario.left_wheel_traction, 1.0)

    def test_slip_scenarios_have_active_faults_and_ordered_severity(self) -> None:
        scenarios_by_name = {scenario.name: scenario for scenario in SCENARIOS}
        mild = scenarios_by_name["mild_slip"]
        severe = scenarios_by_name["severe_slip"]
        for scenario in (mild, severe):
            with self.subTest(scenario=scenario.name):
                self.assertLess(scenario.slip_start_s, scenario.max_time_s)
                self.assertGreater(scenario.slip_end_s, scenario.slip_start_s)
                self.assertLess(scenario.left_wheel_traction, 1.0)
                self.assertGreater(scenario.left_wheel_traction, 0.0)
                self.assertLessEqual(scenario.slip_end_s, scenario.max_time_s)
        self.assertLess(severe.left_wheel_traction, mild.left_wheel_traction)
        self.assertEqual(severe.slip_start_s, mild.slip_start_s)
        self.assertEqual(severe.slip_end_s, mild.slip_end_s)

    def test_make_run_config_copies_scenario_and_run_values(self) -> None:
        scenarios_by_name = {scenario.name: scenario for scenario in SCENARIOS}
        scenario = scenarios_by_name["mild_slip"]

        config = make_run_config(
            scenario=scenario,
            software_version="v2",
            seed=3,
        )

        self.assertEqual(config.software_version, "v2")
        self.assertEqual(config.seed, 3)

        self.assertEqual(config.max_time_s, scenario.max_time_s)
        self.assertEqual(config.gps_period_s, scenario.gps_period_s)
        self.assertEqual(config.gps_noise_std_m, scenario.gps_noise_std_m)
        self.assertEqual(
            config.waypoint_tolerance_m,
            scenario.waypoint_tolerance_m,
            )
        self.assertEqual(config.slip_start_s, scenario.slip_start_s)
        self.assertEqual(config.slip_end_s, scenario.slip_end_s)
        self.assertEqual(
            config.left_wheel_traction,
            scenario.left_wheel_traction,
        )
