"""Summarize three completed author-example runs, without claiming replication."""
import argparse
import json
from pathlib import Path


def value(data, key):
    sections = data[key]
    if len(sections) != 1:
        raise ValueError(f"expected one section for {key}")
    conditions = next(iter(sections.values()))
    if len(conditions) != 1:
        raise ValueError(f"expected one condition for {key}")
    samples = next(iter(conditions.values()))
    if len(samples) != 1:
        raise ValueError(f"expected one completed episode for {key}")
    return samples[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--flair", type=Path, required=True)
    parser.add_argument("--driver", type=Path, required=True)
    parser.add_argument("--nominal-driver", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    files = {"flair": args.flair, "driver": args.driver, "driver_nominal": args.nominal_driver}
    data = {k: json.loads(v.read_text()) for k, v in files.items()}
    result = {"warning": "One unpaired completed run per condition; seed unknown; author metric has unresolved angular conversion and nominal simulation-clock discrepancy. Not a paper reproduction or physical result.",
              "files": {k: str(v) for k, v in files.items()}, "metrics": {}}
    for metric in ("tracking_errors", "completion_times"):
        v = {k: value(x, metric) for k, x in data.items()}
        denominator = v["driver"] - v["driver_nominal"]
        result["metrics"][metric] = {"values": v, "denominator": denominator,
                                     "one_run_recovered_overhead_percent": (100 * (1 - (v["flair"] - v["driver_nominal"]) / denominator)) if abs(denominator) > 1e-12 else None}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
