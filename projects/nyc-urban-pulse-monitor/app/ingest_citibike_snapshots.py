from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest raw Citi Bike GBFS JSON snapshots into DuckDB."
    )

    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "citibike",
        help="Directory containing raw Citi Bike snapshot JSON files.",
    )

    parser.add_argument(
        "--db",
        type=Path,
        default=PROJECT_ROOT / "data" / "db" / "urban_pulse.duckdb",
        help="DuckDB database path.",
    )

    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def parse_iso_timestamp(value: str | None) -> datetime | None:
    if value is None:
        return None

    # Python accepts +00:00 but not always bare Z in older situations.
    value = value.replace("Z", "+00:00")
    dt = datetime.fromisoformat(value)

    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)

    return dt


def epoch_to_utc(epoch_value: Any) -> datetime | None:
    if epoch_value in (None, "", 0):
        return None

    try:
        return datetime.fromtimestamp(int(epoch_value), tz=timezone.utc).replace(tzinfo=None)
    except Exception:
        return None


def get_station_rows(doc: dict) -> list[dict]:
    return doc.get("payload", {}).get("data", {}).get("stations", [])


def get_payload(doc: dict) -> dict:
    return doc.get("payload", {})


def read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def iter_snapshot_files(raw_dir: Path) -> list[Path]:
    files: list[Path] = []

    for subdir in ["station_information", "station_status"]:
        folder = raw_dir / subdir
        if folder.exists():
            files.extend(sorted(folder.glob("*.json")))

    return sorted(files)


def classify_station_status(row: dict) -> str:
    is_installed = row.get("is_installed")
    is_renting = row.get("is_renting")
    is_returning = row.get("is_returning")

    bikes = row.get("num_bikes_available")
    docks = row.get("num_docks_available")

    if is_installed == 0:
        return "not_installed"

    if is_renting == 0 and is_returning == 0:
        return "offline"

    if bikes == 0 and is_renting == 1:
        return "empty"

    if docks == 0 and is_returning == 1:
        return "full"

    return "available"


def create_tables(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS raw_citibike_snapshots (
            snapshot_id VARCHAR,
            source_name VARCHAR,
            source_url VARCHAR,
            observed_at_utc TIMESTAMP,
            fetched_at_utc TIMESTAMP,
            source_last_updated_epoch BIGINT,
            source_last_updated_utc TIMESTAMP,
            source_ttl_sec INTEGER,
            source_version VARCHAR,
            raw_file_path VARCHAR,
            station_count INTEGER,
            ingested_at_utc TIMESTAMP
        );
        """
    )

    con.execute(
        """
        CREATE TABLE IF NOT EXISTS dim_citibike_stations (
            station_id VARCHAR,
            name VARCHAR,
            short_name VARCHAR,
            lat DOUBLE,
            lon DOUBLE,
            capacity INTEGER,
            region_id VARCHAR,
            rental_methods_json VARCHAR,
            station_type VARCHAR,
            has_kiosk BOOLEAN,
            external_id VARCHAR,
            last_seen_at_utc TIMESTAMP,
            source_snapshot_id VARCHAR,
            raw_file_path VARCHAR
        );
        """
    )

    con.execute(
        """
        CREATE TABLE IF NOT EXISTS fact_citibike_station_status (
            snapshot_id VARCHAR,
            observed_at_utc TIMESTAMP,
            source_last_updated_utc TIMESTAMP,
            station_id VARCHAR,
            num_bikes_available INTEGER,
            num_ebikes_available INTEGER,
            num_bikes_disabled INTEGER,
            num_docks_available INTEGER,
            num_docks_disabled INTEGER,
            is_installed INTEGER,
            is_renting INTEGER,
            is_returning INTEGER,
            last_reported_epoch BIGINT,
            last_reported_utc TIMESTAMP,
            status_class VARCHAR,
            raw_file_path VARCHAR
        );
        """
    )


def upsert_raw_snapshot(
    con: duckdb.DuckDBPyConnection,
    *,
    snapshot_id: str,
    doc: dict,
    path: Path,
    station_count: int,
) -> None:
    payload = get_payload(doc)

    row = {
        "snapshot_id": snapshot_id,
        "source_name": doc.get("source_name"),
        "source_url": doc.get("source_url"),
        "observed_at_utc": parse_iso_timestamp(doc.get("observed_at_utc")),
        "fetched_at_utc": parse_iso_timestamp(doc.get("fetched_at_utc")),
        "source_last_updated_epoch": payload.get("last_updated"),
        "source_last_updated_utc": epoch_to_utc(payload.get("last_updated")),
        "source_ttl_sec": payload.get("ttl"),
        "source_version": str(payload.get("version")) if payload.get("version") is not None else None,
        "raw_file_path": str(path),
        "station_count": station_count,
        "ingested_at_utc": datetime.now(timezone.utc).replace(tzinfo=None),
    }

    df = pd.DataFrame([row])
    con.register("tmp_raw_snapshot", df)

    con.execute(
        "DELETE FROM raw_citibike_snapshots WHERE snapshot_id = ?",
        [snapshot_id],
    )

    con.execute(
        """
        INSERT INTO raw_citibike_snapshots
        SELECT
            snapshot_id,
            source_name,
            source_url,
            observed_at_utc,
            fetched_at_utc,
            source_last_updated_epoch,
            source_last_updated_utc,
            source_ttl_sec,
            source_version,
            raw_file_path,
            station_count,
            ingested_at_utc
        FROM tmp_raw_snapshot;
        """
    )


def upsert_station_information(
    con: duckdb.DuckDBPyConnection,
    *,
    snapshot_id: str,
    doc: dict,
    path: Path,
) -> int:
    observed_at_utc = parse_iso_timestamp(doc.get("observed_at_utc"))
    rows = []

    for station in get_station_rows(doc):
        rows.append(
            {
                "station_id": str(station.get("station_id")),
                "name": station.get("name"),
                "short_name": station.get("short_name"),
                "lat": station.get("lat"),
                "lon": station.get("lon"),
                "capacity": station.get("capacity"),
                "region_id": str(station.get("region_id")) if station.get("region_id") is not None else None,
                "rental_methods_json": json.dumps(station.get("rental_methods")) if station.get("rental_methods") is not None else None,
                "station_type": station.get("station_type"),
                "has_kiosk": station.get("has_kiosk"),
                "external_id": station.get("external_id"),
                "last_seen_at_utc": observed_at_utc,
                "source_snapshot_id": snapshot_id,
                "raw_file_path": str(path),
            }
        )

    if not rows:
        return 0

    df = pd.DataFrame(rows)
    con.register("tmp_station_information", df)

    con.execute(
        """
        DELETE FROM dim_citibike_stations
        WHERE station_id IN (
            SELECT station_id FROM tmp_station_information
        );
        """
    )

    con.execute(
        """
        INSERT INTO dim_citibike_stations
        SELECT
            station_id,
            name,
            short_name,
            lat,
            lon,
            capacity,
            region_id,
            rental_methods_json,
            station_type,
            has_kiosk,
            external_id,
            last_seen_at_utc,
            source_snapshot_id,
            raw_file_path
        FROM tmp_station_information;
        """
    )

    return len(rows)


def upsert_station_status(
    con: duckdb.DuckDBPyConnection,
    *,
    snapshot_id: str,
    doc: dict,
    path: Path,
) -> int:
    observed_at_utc = parse_iso_timestamp(doc.get("observed_at_utc"))
    payload = get_payload(doc)
    source_last_updated_utc = epoch_to_utc(payload.get("last_updated"))

    rows = []

    for station in get_station_rows(doc):
        last_reported_epoch = station.get("last_reported")

        rows.append(
            {
                "snapshot_id": snapshot_id,
                "observed_at_utc": observed_at_utc,
                "source_last_updated_utc": source_last_updated_utc,
                "station_id": str(station.get("station_id")),
                "num_bikes_available": station.get("num_bikes_available"),
                "num_ebikes_available": station.get("num_ebikes_available"),
                "num_bikes_disabled": station.get("num_bikes_disabled"),
                "num_docks_available": station.get("num_docks_available"),
                "num_docks_disabled": station.get("num_docks_disabled"),
                "is_installed": station.get("is_installed"),
                "is_renting": station.get("is_renting"),
                "is_returning": station.get("is_returning"),
                "last_reported_epoch": last_reported_epoch,
                "last_reported_utc": epoch_to_utc(last_reported_epoch),
                "status_class": classify_station_status(station),
                "raw_file_path": str(path),
            }
        )

    if not rows:
        return 0

    df = pd.DataFrame(rows)
    con.register("tmp_station_status", df)

    con.execute(
        "DELETE FROM fact_citibike_station_status WHERE source_last_updated_utc = ?",
        [source_last_updated_utc],
    )

    con.execute(
        """
        INSERT INTO fact_citibike_station_status
        SELECT
            snapshot_id,
            observed_at_utc,
            source_last_updated_utc,
            station_id,
            num_bikes_available,
            num_ebikes_available,
            num_bikes_disabled,
            num_docks_available,
            num_docks_disabled,
            is_installed,
            is_renting,
            is_returning,
            last_reported_epoch,
            last_reported_utc,
            status_class,
            raw_file_path
        FROM tmp_station_status;
        """
    )

    return len(rows)


def main() -> None:
    args = parse_args()

    raw_dir = resolve_path(args.raw_dir)
    db_path = resolve_path(args.db)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw directory not found: {raw_dir}")

    files = iter_snapshot_files(raw_dir)

    if not files:
        raise FileNotFoundError(f"No snapshot JSON files found under: {raw_dir}")

    con = duckdb.connect(str(db_path))
    create_tables(con)

    print("Citi Bike snapshot ingest")
    print(f"raw_dir: {raw_dir}")
    print(f"db:      {db_path}")
    print(f"files:   {len(files)}")

    total_info_rows = 0
    total_status_rows = 0

    for path in files:
        doc = read_json(path)
        source_name = doc.get("source_name")
        snapshot_id = path.stem
        station_count = len(get_station_rows(doc))

        upsert_raw_snapshot(
            con,
            snapshot_id=snapshot_id,
            doc=doc,
            path=path,
            station_count=station_count,
        )

        if source_name == "station_information":
            n = upsert_station_information(
                con,
                snapshot_id=snapshot_id,
                doc=doc,
                path=path,
            )
            total_info_rows += n
            print(f"station_information: {path.name} rows={n}")

        elif source_name == "station_status":
            n = upsert_station_status(
                con,
                snapshot_id=snapshot_id,
                doc=doc,
                path=path,
            )
            total_status_rows += n
            print(f"station_status:      {path.name} rows={n}")

        else:
            print(f"skipping unknown source_name={source_name}: {path}")

    print("\nIngest complete.")
    print(f"station information rows processed: {total_info_rows}")
    print(f"station status rows processed:      {total_status_rows}")

    print("\nWarehouse counts:")
    print(
        con.execute(
            """
            SELECT source_name, COUNT(*) AS snapshot_count
            FROM raw_citibike_snapshots
            GROUP BY source_name
            ORDER BY source_name
            """
        ).fetchdf()
    )

    print(
        con.execute(
            """
            SELECT status_class, COUNT(*) AS row_count
            FROM fact_citibike_station_status
            GROUP BY status_class
            ORDER BY row_count DESC
            """
        ).fetchdf()
    )


if __name__ == "__main__":
    main()