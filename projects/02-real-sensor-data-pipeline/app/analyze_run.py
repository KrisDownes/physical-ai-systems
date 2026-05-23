import argparse
import sqlite3
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

def load_run(db_path: Path, run_id: str) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        """
        SELECT timestamp_sec, sensor_name, value
        FROM measurements
        WHERE run_id = ?
        ORDER BY timestamp_sec ASC
        """,
        conn,
        params=(run_id,),
    )

    conn.close()

    if df.empty:
        raise ValueError(f"No measurements found for run_id={run_id}")

    return df

def analyze(df: pd.DataFrame, run_id: str, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Run ID: {run_id}")
    print(f"Samples: {len(df)}")
    print("\nSensors:")
    for sensor_name, group in df.groupby("sensor_name"):
        mean = group["value"].mean()
        std = group["value"].std()
        min_v = group["value"].min()
        max_v = group["value"].max()
        print(f"  {sensor_name}: mean={mean:.3f}, std={std:.3f}, min={min_v:.3f}, max={max_v:.3f}")
    
    for sensor_name, group in df.groupby("sensor_name"):
        plt.figure(figsize=(12, 5))
        plt.plot(group["timestamp_sec"], group["value"])
        plt.xlabel("Time (sec)")
        plt.ylabel(sensor_name)
        plt.title(f"{run_id}: {sensor_name}")
        plt.tight_layout()

        safe_name = (
            sensor_name
            .replace("/", "_")
            .replace("\\", "_")
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
        )

        path = output_dir / f"{run_id}_{safe_name}.png"
        plt.savefig(path)
        plt.close()

    print(f"\nSaved plots to {output_dir}")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--db",
        default="/mnt/d/PhoneTelemetry/db/telemetry.db",
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--output-dir",
        default="projects/02-real-sensor-data-pipeline/plots",
    )
    args = parser.parse_args()

    df = load_run(Path(args.db), args.run_id)
    analyze(df, args.run_id, Path(args.output_dir))


if __name__ == "__main__":
    main()