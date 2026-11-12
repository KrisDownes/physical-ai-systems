"""Apply the published hexapod analysis to a completed author example run."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--notebook", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--run-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    notebook_bytes = args.notebook.read_bytes()
    if hashlib.md5(notebook_bytes).hexdigest() != "f4625a2bcdb3ac84b28aaa198392530c":
        raise ValueError("unexpected notebook checksum")
    log = args.run_log.read_text()
    if "Driver is done." not in log or "Done with rep 1 / 1" not in log or "Failed rep" in log:
        raise ValueError("no evidence of a completed, successful author example episode")
    notebook = json.loads(notebook_bytes)
    namespace = {}
    for index in (1, 3, 6, 9, 10, 12, 13, 14):
        exec(compile("".join(notebook["cells"][index]["source"]), f"{args.notebook}:cell{index}", "exec"), namespace)
    raw, _, times, _, errors = namespace["process_data"](
        str(args.csv), namespace["no_perturb_no_flair"], namespace["no_perturb_flair"],
        namespace["no_flair"], namespace["flair"], copy_chicane=False)
    flatten = lambda nested: {section: {label: np.asarray(values).ravel().tolist() for label, values in alg.items()} for section, alg in nested.items()}
    result = {"warning": "One completed author-example evaluation, not a 20-repetition paper reproduction. Author analysis applies -wz*pi/180; its suitability for these simulation values remains unresolved.",
              "source_csv": str(args.csv), "source_log": str(args.run_log), "data_rows": len(raw),
              "tracking_errors": flatten(errors), "completion_times": flatten(times)}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
