from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]

BASE_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download one month of NYC TLC trip data as Parquet."
    )

    parser.add_argument(
        "--taxi-type",
        default="yellow",
        choices=["yellow", "green", "fhv", "fhvhv"],
        help="TLC trip type.",
    )

    parser.add_argument(
        "--year",
        type=int,
        required=True,
        help="Year, e.g. 2024",
    )

    parser.add_argument(
        "--month",
        type=int,
        required=True,
        help="Month number, 1-12",
    )

    parser.add_argument(
        "--out-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "tlc",
        help="Raw data output root.",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if file already exists.",
    )

    return parser.parse_args()


def build_filename(taxi_type: str, year: int, month: int) -> str:
    return f"{taxi_type}_tripdata_{year}-{month:02d}.parquet"


def build_url(taxi_type: str, year: int, month: int) -> str:
    filename = build_filename(taxi_type, year, month)
    return f"{BASE_URL}/{filename}"


def download_file(url: str, output_path: Path, force: bool = False) -> dict:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not force:
        return {
            "status": "skipped_existing",
            "path": str(output_path),
            "bytes": output_path.stat().st_size,
        }

    tmp_path = output_path.with_suffix(output_path.suffix + ".part")

    start = time.time()

    with requests.get(url, stream=True, timeout=60) as response:
        if response.status_code == 404:
            raise FileNotFoundError(f"TLC file not found: {url}")

        response.raise_for_status()

        total_bytes = 0

        with open(tmp_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue

                f.write(chunk)
                total_bytes += len(chunk)

    tmp_path.replace(output_path)

    elapsed = time.time() - start

    return {
        "status": "downloaded",
        "path": str(output_path),
        "bytes": total_bytes,
        "elapsed_sec": elapsed,
        "mb_per_sec": (total_bytes / 1_000_000) / elapsed if elapsed > 0 else None,
    }


def write_metadata(
    metadata_path: Path,
    taxi_type: str,
    year: int,
    month: int,
    url: str,
    file_info: dict,
) -> None:
    metadata = {
        "dataset": "nyc_tlc_trip_records",
        "taxi_type": taxi_type,
        "year": year,
        "month": month,
        "source_url": url,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "file": file_info,
    }

    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def main() -> None:
    args = parse_args()

    filename = build_filename(args.taxi_type, args.year, args.month)
    url = build_url(args.taxi_type, args.year, args.month)

    raw_dir = args.out_root / args.taxi_type / f"{args.year}" / f"{args.month:02d}"
    output_path = raw_dir / filename
    metadata_path = raw_dir / f"{filename}.metadata.json"

    print("TLC download")
    print(f"taxi_type: {args.taxi_type}")
    print(f"year:      {args.year}")
    print(f"month:     {args.month:02d}")
    print(f"url:       {url}")
    print(f"output:    {output_path}")

    file_info = download_file(url=url, output_path=output_path, force=args.force)

    write_metadata(
        metadata_path=metadata_path,
        taxi_type=args.taxi_type,
        year=args.year,
        month=args.month,
        url=url,
        file_info=file_info,
    )

    print("\nDone.")
    print(json.dumps(file_info, indent=2))
    print(f"metadata: {metadata_path}")


if __name__ == "__main__":
    main()