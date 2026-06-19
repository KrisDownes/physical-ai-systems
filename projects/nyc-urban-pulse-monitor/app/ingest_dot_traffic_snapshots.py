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
        description="Ingest raw NYC DOT traffic speed snapshots into DuckDB."
    )

    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "dot" / "traffic_speeds",
    )

    parser.add_argument(
        "--db",
        type=Path,
        default=PROJECT_ROOT / "data" / "db" / "urban_pulse.duckdb",
    )

    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_iso_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None

    value = value.replace("Z", "+00:00")
    dt = datetime.fromisoformat(value)

    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)

    return dt


def safe_float(value: Any) -> float | None:
    if value in (None, ""):
        return None

    try:
        return float(value)
    except Exception:
        return None


def safe_int(value: Any) -> int | None:
    if value in (None, ""):
        return None

    try:
        return int(float(value))
    except Exception:
        return None


def parse_link_points(value: str | None) -> tuple[float | None, float | None, float | None, float | None, int]:
    """
    link_points is a string like:
      "40.744,-73.771 40.745,-73.769 ..."

    We store start/end coordinates and point count for a simple spatial approximation.
    Later, if needed, we can decode the full polyline.
    """
    if not value:
        return None, None, None, None, 0

    points: list[tuple[float, float]] = []

    for token in value.split():
        try:
            lat_str, lon_str = token.split(",", 1)
            points.append((float(lat_str), float(lon_str)))
        except Exception:
            continue

    if not points:
        return None, None, None, None, 0

    start_lat, start_lon = points[0]
    end_lat, end_lon = points[-1]

    return start_lat, start_lon, end_lat, end_lon, len(points)


def create_tables(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS raw_dot_traffic_speed_snapshots (
            snapshot_id VARCHAR,
            source_name VARCHAR,
            source_url VARCHAR,
            observed_at_utc TIMESTAMP,
            fetched_at_utc TIMESTAMP,
            raw_file_path VARCHAR,
            row_count INTEGER,
            ingested_at_utc TIMESTAMP
        );
        """
    )

    con.execute(
        """
        CREATE TABLE IF NOT EXISTS dim_dot_traffic_segments (
            link_id VARCHAR,
            link_name VARCHAR,
            borough VARCHAR,
            owner VARCHAR,
            transcom_id VARCHAR,
            link_points VARCHAR,
            encoded_poly_line VARCHAR,
            encoded_poly_line_lvls VARCHAR,
            start_lat DOUBLE,
            start_lon DOUBLE,
            end_lat DOUBLE,
            end_lon DOUBLE,
            point_count INTEGER,
            last_seen_at_utc TIMESTAMP,
            source_snapshot_id VARCHAR,
            raw_file_path VARCHAR
        );
        """
    )

    con.execute(
        """
        CREATE TABLE IF NOT EXISTS fact_dot_traffic_speeds (
            snapshot_id VARCHAR,
            observed_at_utc TIMESTAMP,
            data_as_of TIMESTAMP,
            link_id VARCHAR,
            speed_mph DOUBLE,
            travel_time_sec INTEGER,
            status INTEGER,
            borough VARCHAR,
            link_name VARCHAR,
            owner VARCHAR,
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
) -> None:
    row = {
        "snapshot_id": snapshot_id,
        "source_name": doc.get("source_name"),
        "source_url": doc.get("source_url"),
        "observed_at_utc": parse_iso_timestamp(doc.get("observed_at_utc")),
        "fetched_at_utc": parse_iso_timestamp(doc.get("fetched_at_utc")),
        "raw_file_path": str(path),
        "row_count": doc.get("row_count"),
        "ingested_at_utc": datetime.now(timezone.utc).replace(tzinfo=None),
    }

    df = pd.DataFrame([row])
    con.register("tmp_raw_dot_snapshot", df)

    con.execute(
        "DELETE FROM raw_dot_traffic_speed_snapshots WHERE snapshot_id = ?",
        [snapshot_id],
    )

    con.execute(
        """
        INSERT INTO raw_dot_traffic_speed_snapshots
        SELECT
            snapshot_id,
            source_name,
            source_url,
            observed_at_utc,
            fetched_at_utc,
            raw_file_path,
            row_count,
            ingested_at_utc
        FROM tmp_raw_dot_snapshot;
        """
    )


def upsert_dim_segments(
    con: duckdb.DuckDBPyConnection,
    *,
    snapshot_id: str,
    doc: dict,
    path: Path,
) -> int:
    observed_at_utc = parse_iso_timestamp(doc.get("observed_at_utc"))
    rows = []

    latest_by_link: dict[str, dict] = {}

    for raw in doc.get("payload", []):
        link_id = str(raw.get("link_id")) if raw.get("link_id") is not None else None
        if not link_id:
            continue

        data_as_of = parse_iso_timestamp(raw.get("data_as_of"))
        previous = latest_by_link.get(link_id)

        if previous is None:
            latest_by_link[link_id] = raw
            continue

        previous_time = parse_iso_timestamp(previous.get("data_as_of"))

        if previous_time is None or (data_as_of is not None and data_as_of > previous_time):
            latest_by_link[link_id] = raw

    for raw in latest_by_link.values():
        start_lat, start_lon, end_lat, end_lon, point_count = parse_link_points(
            raw.get("link_points")
        )

        rows.append(
            {
                "link_id": str(raw.get("link_id")),
                "link_name": raw.get("link_name"),
                "borough": raw.get("borough"),
                "owner": raw.get("owner"),
                "transcom_id": str(raw.get("transcom_id")) if raw.get("transcom_id") is not None else None,
                "link_points": raw.get("link_points"),
                "encoded_poly_line": raw.get("encoded_poly_line"),
                "encoded_poly_line_lvls": raw.get("encoded_poly_line_lvls"),
                "start_lat": start_lat,
                "start_lon": start_lon,
                "end_lat": end_lat,
                "end_lon": end_lon,
                "point_count": point_count,
                "last_seen_at_utc": observed_at_utc,
                "source_snapshot_id": snapshot_id,
                "raw_file_path": str(path),
            }
        )

    if not rows:
        return 0

    df = pd.DataFrame(rows)
    con.register("tmp_dot_segments", df)

    con.execute(
        """
        DELETE FROM dim_dot_traffic_segments
        WHERE link_id IN (
            SELECT link_id FROM tmp_dot_segments
        );
        """
    )

    con.execute(
        """
        INSERT INTO dim_dot_traffic_segments
        SELECT
            link_id,
            link_name,
            borough,
            owner,
            transcom_id,
            link_points,
            encoded_poly_line,
            encoded_poly_line_lvls,
            start_lat,
            start_lon,
            end_lat,
            end_lon,
            point_count,
            last_seen_at_utc,
            source_snapshot_id,
            raw_file_path
        FROM tmp_dot_segments;
        """
    )

    return len(rows)


def upsert_fact_speeds(
    con: duckdb.DuckDBPyConnection,
    *,
    snapshot_id: str,
    doc: dict,
    path: Path,
) -> int:
    observed_at_utc = parse_iso_timestamp(doc.get("observed_at_utc"))
    rows = []

    for raw in doc.get("payload", []):
        link_id = str(raw.get("link_id")) if raw.get("link_id") is not None else None
        data_as_of = parse_iso_timestamp(raw.get("data_as_of"))

        if not link_id or data_as_of is None:
            continue

        rows.append(
            {
                "snapshot_id": snapshot_id,
                "observed_at_utc": observed_at_utc,
                "data_as_of": data_as_of,
                "link_id": link_id,
                "speed_mph": safe_float(raw.get("speed")),
                "travel_time_sec": safe_int(raw.get("travel_time")),
                "status": safe_int(raw.get("status")),
                "borough": raw.get("borough"),
                "link_name": raw.get("link_name"),
                "owner": raw.get("owner"),
                "raw_file_path": str(path),
            }
        )

    if not rows:
        return 0

    df = pd.DataFrame(rows)
    con.register("tmp_dot_speed_facts", df)

    # Partition-replace by the data_as_of values present in this file.
    # Natural analytical key is link_id + data_as_of.
    con.execute(
        """
        DELETE FROM fact_dot_traffic_speeds AS existing
        WHERE EXISTS (
            SELECT 1
            FROM tmp_dot_speed_facts AS incoming
            WHERE
                existing.link_id = incoming.link_id
                AND existing.data_as_of = incoming.data_as_of
        );
        """
    )

    con.execute(
        """
        INSERT INTO fact_dot_traffic_speeds
        SELECT
            snapshot_id,
            observed_at_utc,
            data_as_of,
            link_id,
            speed_mph,
            travel_time_sec,
            status,
            borough,
            link_name,
            owner,
            raw_file_path
        FROM tmp_dot_speed_facts;
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

    files = sorted(raw_dir.glob("dot_traffic_speeds_*.json"))

    if not files:
        raise FileNotFoundError(f"No DOT traffic speed snapshots found in: {raw_dir}")

    con = duckdb.connect(str(db_path))
    create_tables(con)

    print("DOT traffic speeds ingest")
    print(f"raw_dir: {raw_dir}")
    print(f"db:      {db_path}")
    print(f"files:   {len(files)}")

    total_dim_rows = 0
    total_fact_rows = 0

    for path in files:
        doc = read_json(path)
        snapshot_id = path.stem

        upsert_raw_snapshot(con, snapshot_id=snapshot_id, doc=doc, path=path)

        dim_rows = upsert_dim_segments(con, snapshot_id=snapshot_id, doc=doc, path=path)
        fact_rows = upsert_fact_speeds(con, snapshot_id=snapshot_id, doc=doc, path=path)

        total_dim_rows += dim_rows
        total_fact_rows += fact_rows

        print(f"{path.name}: dim_segments={dim_rows} fact_rows={fact_rows}")

    print("\nIngest complete.")
    print(f"segment rows processed: {total_dim_rows}")
    print(f"speed rows processed:   {total_fact_rows}")

    print("\nWarehouse counts:")
    print(
        con.execute(
            """
            SELECT COUNT(*) AS segment_count
            FROM dim_dot_traffic_segments
            """
        ).fetchdf()
    )

    print(
        con.execute(
            """
            SELECT
                COUNT(*) AS speed_fact_rows,
                COUNT(DISTINCT link_id) AS unique_links,
                COUNT(DISTINCT data_as_of) AS unique_data_as_of,
                MIN(data_as_of) AS first_data_as_of,
                MAX(data_as_of) AS last_data_as_of
            FROM fact_dot_traffic_speeds
            """
        ).fetchdf()
    )

    print("\nSample latest rows:")
    print(
        con.execute(
            """
            SELECT
                data_as_of,
                borough,
                link_name,
                speed_mph,
                travel_time_sec,
                status
            FROM fact_dot_traffic_speeds
            ORDER BY data_as_of DESC, link_id
            LIMIT 10
            """
        ).fetchdf()
    )


if __name__ == "__main__":
    main()
