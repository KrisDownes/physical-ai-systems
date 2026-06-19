from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DOT_TRAFFIC_SPEEDS_ENDPOINT = "https://data.cityofnewyork.us/resource/i4gi-tjb9.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch a raw NYC DOT traffic speeds snapshot from NYC Open Data."
    )

    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "dot" / "traffic_speeds",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=50000,
        help="Maximum number of rows to fetch from the Socrata API.",
    )

    parser.add_argument(
        "--timeout-sec",
        type=float,
        default=30.0,
    )

    return parser.parse_args()


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def timestamp_slug(ts: datetime) -> str:
    return ts.strftime("%Y%m%dT%H%M%SZ")


def fetch_json(limit: int, timeout_sec: float) -> list[dict]:
    params = {
        "$limit": str(limit),
        "$order": "data_as_of DESC",
    }

    response = requests.get(
        DOT_TRAFFIC_SPEEDS_ENDPOINT,
        params=params,
        timeout=timeout_sec,
        headers={"User-Agent": "nyc-urban-pulse-monitor/0.1"},
    )

    response.raise_for_status()
    return response.json()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def main() -> None:
    args = parse_args()

    observed_at_utc = utc_now()
    slug = timestamp_slug(observed_at_utc)

    rows = fetch_json(limit=args.limit, timeout_sec=args.timeout_sec)

    output_path = args.out_dir / f"dot_traffic_speeds_{slug}.json"

    wrapped_payload = {
        "source_name": "dot_traffic_speeds",
        "source_url": DOT_TRAFFIC_SPEEDS_ENDPOINT,
        "observed_at_utc": observed_at_utc.isoformat(),
        "fetched_at_utc": utc_now().isoformat(),
        "query": {
            "$limit": args.limit,
            "$order": "data_as_of DESC",
        },
        "row_count": len(rows),
        "payload": rows,
    }

    write_json(output_path, wrapped_payload)

    print(f"saved: {output_path}")
    print(f"rows:  {len(rows)}")

    if rows:
        print("\nFirst row keys:")
        print(sorted(rows[0].keys()))

        print("\nFirst row:")
        print(json.dumps(rows[0], indent=2))


if __name__ == "__main__":
    main()