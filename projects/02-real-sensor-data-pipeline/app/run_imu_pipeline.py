from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

APP_DIR = PROJECT_ROOT / "app"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

FETCH_SCRIPT = APP_DIR / "fetch_phyphox_run.py"
BUILD_SCRIPT = APP_DIR / "build_processed_imu_dataset.py"
ANALYSIS_SCRIPT = APP_DIR / "quaternion_experiment.py"
VALIDATION_SCRIPT = APP_DIR / "validate_motion_segments.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run full IMU pipeline: fetch phyphox data, build processed CSV, run quaternion/event analysis."
    )

    parser.add_argument(
        "--run-id",
        required=True,
        help="Run ID, e.g. imu_stationary_then_motion_002",
    )

    parser.add_argument(
        "--url",
        default=None,
        help="phyphox remote URL, e.g. http://192.168.50.75. Required unless --skip-fetch is used.",
    )

    parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="Recording duration in seconds.",
    )

    parser.add_argument(
        "--raw-out-root",
        type=Path,
        default=Path("/mnt/d/PhoneTelemetry/inbox"),
        help="Where raw phyphox CSVs are saved. Default: /mnt/d/PhoneTelemetry/inbox",
    )

    parser.add_argument(
        "--bias-window-sec",
        type=float,
        default=5.0,
        help="Initial stationary window used to estimate gyro bias.",
    )

    parser.add_argument(
        "--skip-fetch",
        action="store_true",
        help="Skip phyphox recording/fetch step and use existing raw files.",
    )

    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Skip processed CSV build step and use existing data/processed/<run-id>.csv.",
    )

    parser.add_argument(
        "--skip-analysis",
        action="store_true",
        help="Skip quaternion/event analysis step.",
    )

    parser.add_argument(
        "--protocol",
        type=Path,
        default=None,
        help="Optional protocol CSV for validating clean event segments.",
    )

    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip protocol validation even if --protocol is provided.",
    )

    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def run_command(command: list[str]) -> None:
    print("\n" + "=" * 80)
    print("Running:")
    print(" ".join(command))
    print("=" * 80)

    subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=True,
    )


def check_script_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing script: {path}")


def main() -> None:
    args = parse_args()

    run_id = args.run_id
    raw_out_root = resolve_path(args.raw_out_root)
    raw_run_dir = raw_out_root / run_id
    processed_csv = PROCESSED_DIR / f"{run_id}.csv"
    output_run_dir = OUTPUTS_DIR / run_id

    check_script_exists(FETCH_SCRIPT)
    check_script_exists(BUILD_SCRIPT)
    check_script_exists(ANALYSIS_SCRIPT)
    if args.protocol is not None and not args.skip_validation:
        check_script_exists(VALIDATION_SCRIPT)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    print("\nIMU pipeline")
    print(f"project_root:    {PROJECT_ROOT}")
    print(f"run_id:          {run_id}")
    print(f"raw_run_dir:     {raw_run_dir}")
    print(f"processed_csv:   {processed_csv}")
    print(f"output_run_dir:  {output_run_dir}")

    if not args.skip_fetch:
        if args.url is None:
            raise ValueError("--url is required unless --skip-fetch is used.")

        run_command(
            [
                sys.executable,
                str(FETCH_SCRIPT),
                "--url",
                args.url,
                "--run-id",
                run_id,
                "--duration",
                str(args.duration),
                "--out-root",
                str(raw_out_root),
            ]
        )
    else:
        print("\nSkipping fetch stage.")

    if not raw_run_dir.exists():
        raise FileNotFoundError(
            f"Raw run folder does not exist: {raw_run_dir}\n"
            "Either run without --skip-fetch or check --raw-out-root and --run-id."
        )

    if not args.skip_build:
        run_command(
            [
                sys.executable,
                str(BUILD_SCRIPT),
                "--run-id",
                run_id,
                "--raw-dir",
                str(raw_run_dir),
                "--output",
                str(processed_csv),
                "--bias-window-sec",
                str(args.bias_window_sec),
            ]
        )
    else:
        print("\nSkipping build stage.")

    if not processed_csv.exists():
        raise FileNotFoundError(
            f"Processed CSV does not exist: {processed_csv}\n"
            "Either run without --skip-build or check the build stage."
        )

    if not args.skip_analysis:
        run_command(
            [
                sys.executable,
                str(ANALYSIS_SCRIPT),
                "--input",
                str(processed_csv),
                "--experiment-id",
                run_id,
            ]
        )
    else:
        print("\nSkipping analysis stage.")
    
    if args.protocol is not None and not args.skip_validation:
        protocol_path = resolve_path(args.protocol)

        run_command(
            [
                sys.executable,
                str(VALIDATION_SCRIPT),
                "--run-id",
                run_id,
                "--protocol",
                str(protocol_path),
            ]
            )
    else:
        print("\nSkipping validation stage.")

    print("\n" + "=" * 80)
    print("Pipeline complete.")
    print(f"Raw files:       {raw_run_dir}")
    print(f"Processed CSV:   {processed_csv}")
    print(f"Outputs:         {output_run_dir}")

    if args.protocol is not None and not args.skip_validation:
        print(f"Validation:      {output_run_dir / 'validation'}")

    print("=" * 80)


if __name__ == "__main__":
    main()