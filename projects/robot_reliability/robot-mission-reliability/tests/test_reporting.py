import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from robot_lab.experiment import run_experiment_matrix
from robot_lab.reporting import summarize_results, scenario_comparison_markdown, write_scenario_comparison

class ReportingTests(unittest.TestCase):
    def test_summarize_results_creates_expected_groups(self) -> None:
         with TemporaryDirectory() as directory:
            output_dir = Path(directory)
            results = run_experiment_matrix(output_dir)
            summaries = summarize_results(results)
            markdown = scenario_comparison_markdown(summaries)

            summaries_by_key = {
                (
                    summary.scenario_name,
                    summary.software_version,
                ): summary
                for summary in summaries
            }

            self.assertEqual(len(summaries), 8)
            self.assertEqual(len(summaries_by_key), 8)
            for key, summary in summaries_by_key.items():
                with self.subTest(group=key):
                    self.assertEqual(summary.run_count, 5)
            

            nominal_v1 = summaries_by_key[("nominal", "v1")]
            mild_v1 = summaries_by_key[("mild_slip", "v1")]
            severe_v2 = summaries_by_key[("severe_slip", "v2")]

            self.assertEqual(nominal_v1.successful_runs, 5)
            self.assertEqual(nominal_v1.success_rate, 1.0)

            self.assertEqual(mild_v1.successful_runs, 0)
            self.assertIsNone(mild_v1.mean_completion_time_s)
            self.assertIsNone(mild_v1.mean_path_efficiency)

            self.assertEqual(severe_v2.successful_runs, 5)
            self.assertIsNotNone(severe_v2.mean_completion_time_s)

            data_rows = [
                line
                for line in markdown.splitlines()
                if line.startswith("| `")
            ]

            self.assertEqual(len(data_rows), 8)
            self.assertIn("# Scenario Comparison", markdown)
            self.assertIn("| `nominal` | `v1` |", markdown)
            self.assertIn("5/5 (100%)", markdown)
            self.assertIn("0/5 (0%)", markdown)
            self.assertIn("—", markdown)
            self.assertTrue(markdown.endswith("\n"))

            report_path = output_dir / "scenario_comparison.md"
            write_scenario_comparison(summaries, report_path)

            self.assertTrue(report_path.exists())
            self.assertEqual(
                report_path.read_text(encoding="utf-8"),
                markdown,
            )
            self.assertIn(
                "![Mission success rate](assets/success_rate.svg)",
                markdown,
            )
            self.assertIn(
                "![Mean cross-track RMSE](assets/cross_track_rmse.svg)",
                markdown
            )

    def test_summarize_results_rejects_empty_input(self) -> None:
        with self.assertRaises(ValueError):
            summarize_results([])
        
        with self.assertRaises(ValueError):
            scenario_comparison_markdown([])
