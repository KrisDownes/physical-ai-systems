from dataclasses import dataclass
from statistics import mean
from pathlib import Path

from .experiment import ExperimentResult

@dataclass(frozen=True)
class ScenarioSummary:
    scenario_name: str
    software_version: str
    run_count: int
    successful_runs: int
    success_rate: float
    mean_waypoints_reached: float
    mean_completion_time_s: float | None
    mean_localization_rmse_m: float
    mean_cross_track_rmse_m: float
    mean_path_efficiency: float | None

def summarize_results(
    results: list[ExperimentResult],
) -> list[ScenarioSummary]:
    if not results:
        raise ValueError("Cannot summarize an empty result list")

    groups: dict[
        tuple[str,str],
        list[ExperimentResult],
    ] = {}

    for result in results:
        key = (
            result.scenario_name,
            result.software_version,
        )

        if key not in groups:
            groups[key] = []
        
        groups[key].append(result)
    
    summaries: list[ScenarioSummary] = []
    for (scenario_name, software_version), group_results in groups.items():
        run_count = len(group_results)
        
        successful_results = [
            result
            for result in group_results
            if result.metrics.mission_success
        ]

        successful_runs = len(successful_results)
        success_rate = successful_runs / run_count

        mean_waypoints_reached = mean(
            result.metrics.waypoints_reached
            for result in group_results
        )

        completion_times = [
            result.metrics.completion_time_s
            for result in group_results
            if result.metrics.completion_time_s is not None
        ]

        mean_completion_time_s = (
            mean(completion_times)
            if completion_times
            else None
        )

        path_efficiencies = [
            result.metrics.path_efficiency
            for result in group_results
            if result.metrics.path_efficiency if not None
        ]

        mean_path_efficiency = (
            mean(path_efficiencies)
            if path_efficiencies
            else None
        )

        mean_localization_rmse_m = mean(
            result.metrics.localization_rmse_m
            for result in group_results
        )

        mean_cross_track_rmse_m = mean(
            result.metrics.cross_track_rmse_m
            for result in group_results
        )

        summary = ScenarioSummary(
            scenario_name=scenario_name,
            software_version=software_version,
            run_count=run_count,
            successful_runs=successful_runs,
            success_rate=success_rate,
            mean_waypoints_reached=mean_waypoints_reached,
            mean_completion_time_s=mean_completion_time_s,
            mean_localization_rmse_m=mean_localization_rmse_m,
            mean_cross_track_rmse_m=mean_cross_track_rmse_m,
            mean_path_efficiency=mean_path_efficiency,
        )
        summaries.append(summary)
    return summaries

def scenario_comparison_markdown(
    summaries: list[ScenarioSummary],
) -> str:
    if not summaries:
        raise ValueError("Cannot create table without summaries")

    lines: list[str] = [
        "# Scenario Comparison",
        "",
        "Results are means across five deterministic seeds.",
        "",
        "| Scenario | Version | Runs | Success | Mean waypoints | Mean completion time (s) | Localization RMSE (m) | Cross-track RMSE (m) | Path efficiency |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for summary in summaries:
        success = (
            f"{summary.successful_runs}/{summary.run_count} "
            f"({summary.success_rate:.0%})"
        )

        completion = (
            "—"
            if summary.mean_completion_time_s is None
            else f"{summary.mean_completion_time_s:.2f}"
        )

        path_efficiency = (
            "—"
            if summary.mean_path_efficiency is None
            else f"{summary.mean_path_efficiency:.3f}"
        )
        lines.append(
            f"| `{summary.scenario_name}` "
            f"| `{summary.software_version}` "
            f"| {summary.run_count} "
            f"| {success} "
            f"| {summary.mean_waypoints_reached:.1f} "
            f"| {completion} "
            f"| {summary.mean_localization_rmse_m:.3f} "
            f"| {summary.mean_cross_track_rmse_m:.3f} "
            f"| {path_efficiency} |"
        )
    
    lines.extend(
        [
            "",
            "## Plots",
            "",
            "### Mission Success Rate",
            "",
            "![Mission success rate](assets/success_rate.svg)",
            "",
            "### Cross-Track Error",
            "",
            "![Mean cross-track RMSE](assets/cross_track_rmse.svg)",
            "",
            "## Interpretation Notes",
            "",
            "- Completion time and path efficiency are calculated only from successful runs.",
            "- An em dash means the group contained no successful runs.",
            "- Cross-track RMSE measures lateral path deviation, not motion smoothness or control jerk.",
            "- A failed run can have lower cross-track error because it stops making meaningful route progress.",
            "- These are descriptive results across five fixed seeds, not claims about statistical generalization.",
        ]
    )

    return "\n".join(lines) + "\n"

def write_scenario_comparison(
    summaries: list[ScenarioSummary],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    markdown = scenario_comparison_markdown(summaries)

    with open(output_path, mode="w", encoding="utf-8") as file:
        file.write(markdown)