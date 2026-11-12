"""Summarize the author-notebook table and optional altered analyses."""
import argparse
import json
from pathlib import Path
from statistics import median


def summarize(path):
    data = json.loads(path.read_text())
    result = {"analysis": str(path), "angular_mode": data.get("angular_mode", "authors"),
              "repetition_ids": data["repetition_ids"]}
    for metric in ("tracking_errors", "completion_times"):
        values = {}
        for name, entry in data["summary"].items():
            groups = entry[metric]
            if len(groups) != 1 or len(next(iter(groups.values()))) != 1:
                raise ValueError(f"unexpected groups in {name} {metric}")
            samples = next(iter(next(iter(groups.values())).values()))
            if len(samples) != len(data["repetition_ids"]):
                raise ValueError(f"missing repetitions in {name} {metric}")
            values[name] = median(samples)
        baseline = values["driver_nominal"]
        driver = (values["driver"] - baseline) / baseline * 100
        flair = (values["flair"] - baseline) / baseline * 100
        result[metric] = {"medians": values, "driver_overhead_percent": driver,
                          "flair_overhead_percent": flair,
                          "recovered_overhead_percent": (1 - flair / driver) * 100}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {"warning": "Re-analysis of published data, not a new reproduction. Non-authors angular modes are sensitivity checks and change the published metric.",
              "analyses": [summarize(path) for path in args.inputs]}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
