from dataclasses import dataclass
from .simulation import RunConfig

@dataclass(frozen=True)
class Scenario:
    name: str
    seeds: tuple[int, ...]

    max_time_s: float = 38.0
    gps_period_s: float = 0.5
    gps_noise_std_m: float = 0.06
    waypoint_tolerance_m: float = 0.22
    slip_start_s: float = 5.0
    slip_end_s: float = 9.0
    left_wheel_traction: float = 0.28


def make_run_config(
    scenario: Scenario,
    software_version: str,
    seed: int,
) -> RunConfig:
    return RunConfig(
        software_version=software_version,
        seed=seed,
        max_time_s=scenario.max_time_s,
        gps_period_s=scenario.gps_period_s,
        gps_noise_std_m=scenario.gps_noise_std_m,
        waypoint_tolerance_m=scenario.waypoint_tolerance_m,
        slip_start_s=scenario.slip_start_s,
        slip_end_s=scenario.slip_end_s,
        left_wheel_traction=scenario.left_wheel_traction,
    )

SEEDS = (0, 1, 2, 3, 4)

SCENARIOS = (
    Scenario(
        name="nominal",
        seeds=SEEDS,
        slip_start_s=100.0,
        slip_end_s=101.9,
        left_wheel_traction=1.0,
        ),
    Scenario(
        name="mild_slip",
        seeds=SEEDS,
        left_wheel_traction=0.65
    ),
    Scenario(
        name="severe_slip",
        seeds=SEEDS,
        left_wheel_traction=0.1
    ),
    Scenario(
        name="noisy_gps",
        seeds=SEEDS,
        gps_noise_std_m=0.12,
        left_wheel_traction=1.0,
        slip_start_s=100.0,
        slip_end_s=101.9,
        ),
)