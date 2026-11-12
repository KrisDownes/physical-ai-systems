"""Run selected, unmodified author notebook functions on published raw data.

Each CSV is restricted to specified rep IDs before notebook processing. This
is analysis of authors' recorded data, not a newly simulated reproduction.
Only audited cells containing imports, settings and functions are executed.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


FILES = {
    "flair": ("hexapod_main_flair.csv", "4d86df193d6d3825fc02e89a6ab8a2cc"),
    "flair_nominal": ("hexapod_main_flair_no_perturb.csv", "2855ecc0a03f66624a67eead9180ea31"),
    "driver": ("hexapod_main_no_flair.csv", "dec32648076694b89c1e72b7d2bfc6af"),
    "driver_nominal": ("hexapod_main_no_flair_no_perturb.csv", "7e96eb7574acd7791e2f667ffdcc3e29"),
}


def md5(path):
    digest = hashlib.md5()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--notebook", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reps", type=int, nargs="+", default=[0])
    parser.add_argument("--angular-mode", choices=("authors", "raw_simulator", "negative_raw"), default="authors")
    args = parser.parse_args()
    if md5(args.notebook) != "f4625a2bcdb3ac84b28aaa198392530c":
        raise ValueError("unexpected authors' notebook checksum")
    notebook = json.loads(args.notebook.read_text())
    namespace = {}
    for index in (1, 3, 6, 9, 10, 12, 13, 14):
        cell = "".join(notebook["cells"][index]["source"])
        if index == 10 and args.angular_mode != "authors":
            original = 'sensor_z_buffer = np.asarray(-sensor_df["wz"].values * np.pi / 180)'
            if cell.count(original) != 1:
                raise ValueError("unexpected notebook angular conversion")
            replacement = 'sensor_z_buffer = np.asarray(sensor_df["wz"].values)' if args.angular_mode == "raw_simulator" else 'sensor_z_buffer = np.asarray(-sensor_df["wz"].values)'
            cell = cell.replace(original, replacement, 1)
        exec(compile(cell, f"{args.notebook}:cell{index}", "exec"), namespace)
    read_csv = pd.read_csv
    summaries = {}
    for name, (filename, expected) in FILES.items():
        path = args.data_dir / filename
        if md5(path) != expected:
            raise ValueError(f"unexpected checksum for {filename}")
        def filtered_read_csv(*read_args, **read_kwargs):
            data = read_csv(*read_args, **read_kwargs)
            return data[data["Sections_index"].isin(args.reps)].copy()
        pd.read_csv = filtered_read_csv
        try:
            _, _, times, _, errors = namespace["process_data"](
                str(path), namespace["no_perturb_no_flair"], namespace["no_perturb_flair"],
                namespace["no_flair"], namespace["flair"], copy_chicane=False)
        finally:
            pd.read_csv = read_csv
        flatten = lambda nested: {section: {label: np.asarray(values).ravel().tolist() for label, values in algorithms.items()} for section, algorithms in nested.items()}
        summaries[name] = {"tracking_errors": flatten(errors), "completion_times": flatten(times)}
        print("finished", name, flush=True)
    result = {"source": "authors' Zenodo notebook selected cells; published raw data", "angular_mode": args.angular_mode,
              "notebook_md5": md5(args.notebook), "repetition_ids": args.reps,
              "warning": "authors' recorded data, not a new simulation; non-authors angular modes alter the analysis and are not the published metric",
              "files": {name: checksum for name, (_, checksum) in FILES.items()}, "summary": summaries}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
