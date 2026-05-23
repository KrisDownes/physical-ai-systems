import argparse
import hashlib
import json
import sqlite3
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd

def file_hash(path: Path) -> str:
    """Calculate the SHA256 hash of a file"""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def create_tables(conn: sqlite3.Connection) -> None:
    """Create tables in the SQLite db"""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS runs (
            run_id TEXT PRIMARY KEY,
            source_file TEXT NOT NULL,
            file_hash TEXT NOT NULL,
            imported_at TEXT NOT NULL,
            row_count INTEGER NOT NULL,
            columns_json TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS measurements (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            timestamp_sec REAL NOT NULL,
            sensor_name TEXT NOT NULL,
            value REAL NOT NULL,
            FOREIGN KEY (run_id) REFERENCES runs(run_id)
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_measurements_run_time
        ON measurements(run_id, timestamp_sec)
    """)

    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_measurements_sensor
        ON measurements(sensor_name)
    """)

    conn.commit()

def find_time_column(df: pd.DataFrame) -> str:
    candidates = []
    for col in df.columns:
        name = str(col).lower()
        if "time" in name or name.strip() in {"t", "seconds", "sec"}:
            candidates.append(col)
    if candidates:
        return candidates[0]
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if numeric_cols:
        return numeric_cols[0]
    raise ValueError("Could not find a time or numeric column in the CSV")

def clean_column_name(col:str) -> str:
    return (
        str(col)
        .strip()
        .replace("\n", " ")
        .replace("\r", " ")
    )
def ingest_csv(csv_path: Path, db_path: Path, run_id: str | None = None) -> str:
    csv_path = csv_path.resolve()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    # phyphox CSV files may vary. pandas usually handles csv exports
    df = pd.read_csv(csv_path)

    if df.empty:
        raise ValueError(f"CSV is empty: {csv_path}")
    
    df.columns = [clean_column_name(col) for col in df.columns]
    time_col = find_time_column(df)

    #Convert numeric columns. Non numeric values become NaN
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=[time_col])

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    sensor_cols = [col for col in numeric_cols if col != time_col]
    if not sensor_cols:
        raise ValueError(f"No numeric sensor columns found in the CSV: {csv_path}")
    if run_id is None:
        stem =csv_path.stem.replace(" ", "_")
        run_id = f"{stem}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    
    h = file_hash(csv_path)
    conn = sqlite3.connect(db_path)
    create_tables(conn)

    existing = conn.execute(
        "SELECT run_id FROM runs WHERE file_hash = ?",
        (h,),
    ).fetchone()
    if existing:
        print(f"File already ingested as run_id={existing[0]}")
        conn.close()
        return existing[0]
    
    imported_at = datetime.now(timezone.utc).isoformat()
    conn.execute(
        """
        INSERT INTO runs (
            run_id, source_file, file_hash, imported_at, row_count, columns_json
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            run_id,
            str(csv_path),
            h,
            imported_at,
            len(df),
            json.dumps(list(df.columns)),
        ),
    )

    records = []
    for _, row in df.iterrows():
        timestamp = float(row[time_col])

        for sensor_col in sensor_cols:
            value = row[sensor_col]
            if pd.isna(value):
                continue

            records.append(
                (
                    run_id,
                    timestamp,
                    sensor_col,
                    float(value),
                )
            )

    conn.executemany(
        """
        INSERT INTO measurements (
            run_id, timestamp_sec, sensor_name, value
        )
        VALUES (?, ?, ?, ?)
        """,
        records,
    )

    conn.commit()
    conn.close()

    print(f"Ingested {csv_path}")
    print(f"Run ID: {run_id}")
    print(f"Rows: {len(df)}")
    print(f"Sensor columns: {sensor_cols}")
    print(f"Measurements inserted: {len(records)}")

    return run_id

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to phyphox CSV file")
    parser.add_argument(
        "--db",
        default="/mnt/d/PhoneTelemetry/db/telemetry.db",
        help="SQLite database path",
    )
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()

    ingest_csv(
        csv_path=Path(args.csv),
        db_path=Path(args.db),
        run_id=args.run_id,
    )


if __name__ == "__main__":
    main()