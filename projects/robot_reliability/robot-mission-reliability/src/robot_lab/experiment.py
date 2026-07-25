from dataclasses import dataclass, asdict
from pathlib import Path
import csv

from .evaluation import RunMetrics, evaluate_run
from .scenarios import Scenario, make_run_config, SCENARIOS
from .simulation import run_simulation

SOFTWARE_VERSIONS = ("v1", "v2")

RESULT_COLUMNS = (
    "run_id",
    "scenario_name",
    "software_version",
    "seed",
    "log_path",
    "max_time_s",
    "gps_period_s",
    "gps_noise_std_m",
    "waypoint_tolerance_m",
    "slip_start_s",
    "slip_end_s",
    "left_wheel_traction",
    "mission_success",
    "waypoints_reached",
    "duration_s",
    "completion_time_s",
    "telemetry_samples",
    "localization_rmse_m",
    "localization_max_error_m",
    "detector_alerts",
    "detector_false_positives",
    "detection_latency_s",
    "log_sequence_gaps",
    "path_length_m",
    "path_efficiency",
    "cross_track_rmse_m",
    "cross_track_max_error_m",
)

@dataclass(frozen=True)
class ExperimentResult:
    scenario: Scenario
    software_version: str
    seed: int
    log_path: Path
    metrics: RunMetrics

    @property
    def scenario_name(self) -> str:
        return self.scenario.name

def run_one_case(
    output_dir: Path,
    scenario: Scenario,
    software_version: str,
    seed: int,
) -> ExperimentResult:
    log_path = (
        output_dir
        / "runs"
        / scenario.name
        / f"{software_version}_seed_{seed}.jsonl"
    )
    config = make_run_config(
        scenario=scenario,
        software_version=software_version,
        seed=seed,
    )
    run_id = run_simulation(output_path=log_path, config=config)
    metrics = evaluate_run(path=log_path)
    if metrics.run_id != run_id:
        raise ValueError(
            f"Run ID mismatch: simulation={run_id} evaluation={metrics.run_id}"
        )
    return ExperimentResult(
        scenario=scenario,
        software_version=software_version,
        seed=seed,
        log_path=log_path,
        metrics=metrics,
    )


def run_experiment_matrix(
    output_dir: Path,
    scenarios: tuple[Scenario, ...] = SCENARIOS,
    software_versions: tuple[str, ...] = SOFTWARE_VERSIONS,
) -> list[ExperimentResult]:
    results: list[ExperimentResult] = []
    for scenario in scenarios:
        for seed in scenario.seeds:
            for software_version in software_versions:
                result = run_one_case(
                    output_dir=output_dir,
                    scenario=scenario,
                    software_version=software_version,
                    seed=seed,
                )
                results.append(result)
    return results


def result_to_row(
    result: ExperimentResult,
) -> dict[str, object]:
    row = asdict(result.metrics)
    row.update(
        {
            "scenario_name": result.scenario.name,
            "software_version": result.software_version,
            "seed": result.seed,
            "log_path": str(result.log_path),
            "max_time_s": result.scenario.max_time_s,
            "gps_period_s": result.scenario.gps_period_s,
            "gps_noise_std_m": result.scenario.gps_noise_std_m,
            "waypoint_tolerance_m": result.scenario.waypoint_tolerance_m,
            "slip_start_s": result.scenario.slip_start_s,
            "slip_end_s": result.scenario.slip_end_s,
            "left_wheel_traction": result.scenario.left_wheel_traction,
        }
    )
    return row

def write_results_csv(
    results: list[ExperimentResult],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=RESULT_COLUMNS)

        writer.writeheader()

        for result in results:
            row = result_to_row(result)
            writer.writerow(row)
