from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INBOX_ROOT = PROJECT_ROOT / "data" / "inbox"


BUFFER_QUERY = (
    "acc_time=full&accX=full&accY=full&accZ=full"
    "&graT=full&graX=full&graY=full&graZ=full"
    "&gyro_time=full&gyroX=full&gyroY=full&gyroZ=full"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch a phyphox IMU run directly from the remote API."
    )

    parser.add_argument(
        "--url",
        required=True,
        help="phyphox remote URL, e.g. http://192.168.50.75",
    )

    parser.add_argument(
        "--run-id",
        required=True,
        help="Run ID, e.g. imu_labeled_motion_001",
    )

    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Optional recording duration in seconds. If provided, script clears, starts, waits, stops, then fetches.",
    )

    parser.add_argument(
        "--out-root",
        type=Path,
        default=DEFAULT_INBOX_ROOT,
        help="Output root folder. Default: data/inbox",
    )

    return parser.parse_args()


def phy_url(base_url: str, path: str) -> str:
    base_url = base_url.rstrip("/") + "/"
    return urljoin(base_url, path.lstrip("/"))


def send_control(base_url: str, command: str) -> None:
    url = phy_url(base_url, f"control?cmd={command}")
    response = requests.get(url, timeout=10)
    response.raise_for_status()

    data = response.json()
    if not data.get("result", False):
        raise RuntimeError(f"phyphox control command failed: {command}")

    print(f"control: {command}")


def fetch_buffers(base_url: str) -> dict:
    url = phy_url(base_url, f"get?{BUFFER_QUERY}")
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.json()


def get_buffer(data: dict, name: str) -> list[float]:
    return data["buffer"][name]["buffer"]


def make_sensor_frames(data: dict) -> dict[str, pd.DataFrame]:
    acc = pd.DataFrame(
        {
            "time_sec": get_buffer(data, "acc_time"),
            "acc_x": get_buffer(data, "accX"),
            "acc_y": get_buffer(data, "accY"),
            "acc_z": get_buffer(data, "accZ"),
        }
    )

    gravity = pd.DataFrame(
        {
            "time_sec": get_buffer(data, "graT"),
            "gravity_x": get_buffer(data, "graX"),
            "gravity_y": get_buffer(data, "graY"),
            "gravity_z": get_buffer(data, "graZ"),
        }
    )

    gyro = pd.DataFrame(
        {
            "time_sec": get_buffer(data, "gyro_time"),
            "gyro_x": get_buffer(data, "gyroX"),
            "gyro_y": get_buffer(data, "gyroY"),
            "gyro_z": get_buffer(data, "gyroZ"),
        }
    )

    return {
        "accelerometer": acc,
        "gravity": gravity,
        "gyroscope": gyro,
    }


def save_run(
    run_id: str,
    out_root: Path,
    raw_data: dict,
    frames: dict[str, pd.DataFrame],
) -> Path:
    run_dir = out_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    raw_json_path = run_dir / f"{run_id}_phyphox_buffers.json"

    with open(raw_json_path, "w", encoding="utf-8") as f:
        json.dump(raw_data, f, indent=2)

    for name, frame in frames.items():
        output_path = run_dir / f"{run_id}_{name}.csv"
        frame.to_csv(output_path, index=False)
        print(f"saved: {output_path} rows={len(frame)}")

    print(f"saved: {raw_json_path}")
    return run_dir


def print_counts(frames: dict[str, pd.DataFrame]) -> None:
    print("\nSample counts:")
    for name, frame in frames.items():
        if len(frame) == 0:
            duration = 0.0
        else:
            duration = frame["time_sec"].iloc[-1] - frame["time_sec"].iloc[0]

        print(f"{name:14s} rows={len(frame):6d} duration={duration:8.3f}s")


def main() -> None:
    args = parse_args()

    base_url = args.url
    run_id = args.run_id

    if args.duration is not None:
        send_control(base_url, "clear")
        send_control(base_url, "start")
        print(f"recording for {args.duration:.2f} seconds...")
        time.sleep(args.duration)
        send_control(base_url, "stop")

    raw_data = fetch_buffers(base_url)
    frames = make_sensor_frames(raw_data)

    print_counts(frames)

    if any(len(frame) == 0 for frame in frames.values()):
        raise RuntimeError("One or more sensor frames are empty. Check phyphox recording and sensor permissions.")

    save_run(
        run_id=run_id,
        out_root=args.out_root,
        raw_data=raw_data,
        frames=frames,
    )


if __name__ == "__main__":
    main()
