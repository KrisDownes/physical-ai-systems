from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


EXPECTED_LABELS = {
    "calibration_stationary": {"stationary"},
    "stationary": {"stationary"},
    "small_motion": {"small_motion"},
    "rotation": {"rotation"},
    "active_motion": {"active_motion"},
    "motion": {"active_motion", "rotation", "small_motion"},
    "shake": {"active_motion", "impact_or_shake", "rotation"},
    "active_shake": {"active_motion", "impact_or_shake", "rotation"},
    "rotation_or_motion": {"rotation", "active_motion"},
    "impact": {"impact_or_shake"},
    "impact_or_shake": {"impact_or_shake"},
    "marker": {"impact_or_shake"},
    "transition": None,
    "ignore": None,
}


BRIEF_EVENT_LABELS = {
    "impact",
    "impact_or_shake",
    "marker",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate cleaned IMU motion segments against an expected protocol timeline."
    )

    parser.add_argument(
        "--run-id",
        required=True,
        help="Run ID, e.g. imu_stationary_then_motion_002",
    )

    parser.add_argument(
        "--protocol",
        type=Path,
        required=True,
        help="CSV with start_time_sec,end_time_sec,expected_label,notes.",
    )

    parser.add_argument(
        "--segments",
        type=Path,
        default=None,
        help="Optional path to clean group segments CSV. Defaults to outputs/<run-id>/csv/<run-id>_clean_group_segments.csv.",
    )

    parser.add_argument(
        "--min-match-fraction",
        type=float,
        default=0.50,
        help="Minimum fraction of a protocol window that must match expected labels.",
    )

    parser.add_argument(
        "--min-brief-event-overlap-sec",
        type=float,
        default=0.01,
        help="Minimum overlap needed for brief events like impact/marker.",
    )

    return parser.parse_args()


def resolve_project_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def default_segments_path(run_id: str) -> Path:
    return (
        OUTPUTS_DIR
        / run_id
        / "csv"
        / f"{run_id}_clean_group_segments.csv"
    )


def validate_protocol_schema(protocol: pd.DataFrame) -> None:
    required_cols = {
        "start_time_sec",
        "end_time_sec",
        "expected_label",
    }

    missing = required_cols - set(protocol.columns)
    if missing:
        raise ValueError(f"Protocol CSV missing columns: {sorted(missing)}")

    if "notes" not in protocol.columns:
        protocol["notes"] = ""

    if (protocol["end_time_sec"] <= protocol["start_time_sec"]).any():
        raise ValueError("Every protocol row must have end_time_sec > start_time_sec.")

    unknown_labels = sorted(
        set(protocol["expected_label"]) - set(EXPECTED_LABELS)
    )

    if unknown_labels:
        raise ValueError(
            "Unknown expected_label values:\n"
            + "\n".join(unknown_labels)
            + "\n\nAllowed labels:\n"
            + "\n".join(sorted(EXPECTED_LABELS))
        )


def validate_segments_schema(segments: pd.DataFrame) -> None:
    required_cols = {
        "start_time_sec",
        "end_time_sec",
        "event_group_clean",
    }

    missing = required_cols - set(segments.columns)
    if missing:
        raise ValueError(f"Segments CSV missing columns: {sorted(missing)}")


def interval_overlap(
    a_start: float,
    a_end: float,
    b_start: float,
    b_end: float,
) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def summarize_detected_overlap(
    expected_start: float,
    expected_end: float,
    segments: pd.DataFrame,
) -> dict[str, float]:
    overlaps: dict[str, float] = {}

    for _, segment in segments.iterrows():
        detected_label = str(segment["event_group_clean"])

        overlap = interval_overlap(
            expected_start,
            expected_end,
            float(segment["start_time_sec"]),
            float(segment["end_time_sec"]),
        )

        if overlap <= 0:
            continue

        overlaps[detected_label] = overlaps.get(detected_label, 0.0) + overlap

    return overlaps


def validate_one_row(
    row: pd.Series,
    segments: pd.DataFrame,
    min_match_fraction: float,
    min_brief_event_overlap_sec: float,
) -> dict:
    expected_start = float(row["start_time_sec"])
    expected_end = float(row["end_time_sec"])
    expected_label = str(row["expected_label"])
    notes = str(row.get("notes", ""))

    expected_duration = expected_end - expected_start
    allowed_detected_labels = EXPECTED_LABELS[expected_label]

    overlaps = summarize_detected_overlap(
        expected_start=expected_start,
        expected_end=expected_end,
        segments=segments,
    )

    if overlaps:
        dominant_detected_label = max(overlaps, key=overlaps.get)
        dominant_overlap_sec = overlaps[dominant_detected_label]
    else:
        dominant_detected_label = "none"
        dominant_overlap_sec = 0.0

    if allowed_detected_labels is None:
        allowed_overlap_sec = 0.0
        match_fraction = None
        passed = True
        status = "ignored"
    else:
        allowed_overlap_sec = sum(
            overlap
            for label, overlap in overlaps.items()
            if label in allowed_detected_labels
        )

        match_fraction = (
            allowed_overlap_sec / expected_duration
            if expected_duration > 0
            else 0.0
        )

        if expected_label in BRIEF_EVENT_LABELS:
            passed = allowed_overlap_sec >= min_brief_event_overlap_sec
        else:
            passed = match_fraction >= min_match_fraction

        status = "pass" if passed else "fail"

    return {
        "start_time_sec": expected_start,
        "end_time_sec": expected_end,
        "duration_sec": expected_duration,
        "expected_label": expected_label,
        "allowed_detected_labels": (
            ",".join(sorted(allowed_detected_labels))
            if allowed_detected_labels is not None
            else "ignored"
        ),
        "dominant_detected_label": dominant_detected_label,
        "dominant_overlap_sec": dominant_overlap_sec,
        "allowed_overlap_sec": allowed_overlap_sec,
        "match_fraction": match_fraction,
        "status": status,
        "notes": notes,
        "overlap_detail_json": json.dumps(overlaps),
    }


def validate_protocol(
    protocol: pd.DataFrame,
    segments: pd.DataFrame,
    min_match_fraction: float,
    min_brief_event_overlap_sec: float,
) -> pd.DataFrame:
    rows = []

    for _, row in protocol.iterrows():
        rows.append(
            validate_one_row(
                row=row,
                segments=segments,
                min_match_fraction=min_match_fraction,
                min_brief_event_overlap_sec=min_brief_event_overlap_sec,
            )
        )

    return pd.DataFrame(rows)


def build_summary(results: pd.DataFrame) -> dict:
    evaluated = results[results["status"] != "ignored"]
    ignored = results[results["status"] == "ignored"]

    if len(evaluated) == 0:
        pass_rate = None
    else:
        pass_rate = float((evaluated["status"] == "pass").mean())

    return {
        "protocol_rows": int(len(results)),
        "evaluated_rows": int(len(evaluated)),
        "ignored_rows": int(len(ignored)),
        "passed_rows": int((evaluated["status"] == "pass").sum()),
        "failed_rows": int((evaluated["status"] == "fail").sum()),
        "pass_rate": pass_rate,
    }


def save_outputs(
    run_id: str,
    results: pd.DataFrame,
    summary: dict,
) -> None:
    validation_dir = OUTPUTS_DIR / run_id / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)

    results_path = validation_dir / f"{run_id}_protocol_validation.csv"
    summary_path = validation_dir / f"{run_id}_protocol_validation_summary.json"

    results.to_csv(results_path, index=False)

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nsaved: {results_path}")
    print(f"saved: {summary_path}")


def print_results(results: pd.DataFrame, summary: dict) -> None:
    display_cols = [
        "start_time_sec",
        "end_time_sec",
        "expected_label",
        "dominant_detected_label",
        "allowed_overlap_sec",
        "match_fraction",
        "status",
    ]

    printable = results[display_cols].copy()

    if "match_fraction" in printable:
        printable["match_fraction"] = printable["match_fraction"].apply(
            lambda x: "" if pd.isna(x) else f"{x:.3f}"
        )

    printable["allowed_overlap_sec"] = printable["allowed_overlap_sec"].apply(
        lambda x: f"{x:.3f}"
    )

    print("\nProtocol validation:")
    print(printable.to_string(index=False))

    print("\nSummary:")
    print(json.dumps(summary, indent=2))


def main() -> None:
    args = parse_args()

    protocol_path = resolve_project_path(args.protocol)

    if args.segments is None:
        segments_path = default_segments_path(args.run_id)
    else:
        segments_path = resolve_project_path(args.segments)

    if not protocol_path.exists():
        raise FileNotFoundError(f"Protocol file not found: {protocol_path}")

    if not segments_path.exists():
        raise FileNotFoundError(f"Segments file not found: {segments_path}")

    protocol = pd.read_csv(protocol_path)
    segments = pd.read_csv(segments_path)

    validate_protocol_schema(protocol)
    validate_segments_schema(segments)

    results = validate_protocol(
        protocol=protocol,
        segments=segments,
        min_match_fraction=args.min_match_fraction,
        min_brief_event_overlap_sec=args.min_brief_event_overlap_sec,
    )

    summary = build_summary(results)

    print_results(results, summary)
    save_outputs(args.run_id, results, summary)


if __name__ == "__main__":
    main()
