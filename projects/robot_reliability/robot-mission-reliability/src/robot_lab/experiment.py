from dataclasses import dataclass
from pathlib import Path

from .evaluation import RunMetrics,evaluate_run
from .scenarios import Scenario, make_run_config,SCENARIOS
from .simulation import run_simulation

SOFTWARE_VERSIONS = ("v1", "v2")

@dataclass(frozen=True)
class ExperimentResult:
    scenario_name: str
    software_version: str
    seed: int
    log_path: Path
    metrics: RunMetrics

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
        scenario_name=scenario.name,
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
