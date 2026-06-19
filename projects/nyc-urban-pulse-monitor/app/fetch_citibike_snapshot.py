from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEEDS = {
    "station_information": {
        "url": "https://gbfs.citibikenyc.com/gbfs/en/station_information.json",
        "subdir": "station_information",
    },
    "station_status": {
        "url": "https://gbfs.citibikenyc.com/gbfs/en/station_status.json",
        "subdir": "station_status",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch raw Citi Bike GBFS snapshots."
    )

    parser.add_argument(
        "--feed",
        choices=["both", "station_information", "station_status"],
        default="both",
        help=(
            "Which Citi Bike GBFS feed to fetch. "
            "'both' fetches station_information and station_status."
        ),
    )

    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "citibike",
    )

    parser.add_argument(
        "--timeout-sec",
        type=float,
        default=20.0,
    )

    return parser.parse_args()


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def timestamp_slug(ts: datetime) -> str:
    return ts.strftime("%Y%m%dT%H%M%SZ")


def fetch_json(url: str, timeout_sec: float) -> dict:
    response = requests.get(
        url,
        timeout=timeout_sec,
        headers={"User-Agent": "nyc-urban-pulse-monitor/0.1"},
    )
    response.raise_for_status()
    return response.json()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def selected_feeds(feed_arg: str) -> list[str]:
    if feed_arg == "both":
        return ["station_information", "station_status"]

    return [feed_arg]


def main() -> None:
    args = parse_args()

    observed_at_utc = utc_now()
    slug = timestamp_slug(observed_at_utc)

    manifest_rows = []

    for feed_name in selected_feeds(args.feed):
        feed = FEEDS[feed_name]

        payload = fetch_json(feed["url"], timeout_sec=args.timeout_sec)

        output_path = (
            args.out_dir
            / feed["subdir"]
            / f"{feed_name}_{slug}.json"
        )

        wrapped_payload = {
            "source_name": feed_name,
            "source_url": feed["url"],
            "observed_at_utc": observed_at_utc.isoformat(),
            "fetched_at_utc": utc_now().isoformat(),
            "payload": payload,
        }

        write_json(output_path, wrapped_payload)

        station_count = len(payload.get("data", {}).get("stations", []))

        manifest_rows.append(
            {
                "source_name": feed_name,
                "source_url": feed["url"],
                "observed_at_utc": observed_at_utc.isoformat(),
                "output_path": str(output_path),
                "station_count": station_count,
                "source_last_updated": payload.get("last_updated"),
                "source_ttl_sec": payload.get("ttl"),
            }
        )

        print(f"saved: {output_path}")
        print(f"feed: {feed_name}")
        print(f"stations: {station_count}")
        print(f"source_last_updated: {payload.get('last_updated')}")
        print(f"source_ttl_sec: {payload.get('ttl')}")

    manifest_path = args.out_dir / f"manifest_{args.feed}_{slug}.json"
    write_json(
        manifest_path,
        {
            "feed_request": args.feed,
            "observed_at_utc": observed_at_utc.isoformat(),
            "snapshots": manifest_rows,
        },
    )

    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
