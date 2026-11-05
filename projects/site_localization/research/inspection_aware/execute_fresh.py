"""Run the frozen fresh connected-layout full missions sequentially."""
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT.parents[1]
OUT = ROOT / "results/fresh_inspection_aware_20261028"


def load(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def main():
    pipeline = load(OUT / "pipeline_freeze_v2.json")
    scenario = load(OUT / "scenario_freeze_v2.json")
    if sha(OUT / "pipeline_freeze_v2.json") != scenario["pipeline_freeze_sha256"]:
        raise RuntimeError("Pipeline manifest changed")
    for case, arm in scenario["execution_order"]:
        for name, expected in pipeline["files"].items():
            if sha(REPO / name) != expected:
                raise RuntimeError("Frozen method changed: " + name)
        for name, expected in scenario["templates"].items():
            if sha(REPO / name) != expected:
                raise RuntimeError("Frozen scenario changed: " + name)
        template = OUT / "templates" / (case + "_" + arm)
        output = OUT / ("mission_" + case + "_" + arm)
        if output.exists():
            raise FileExistsError(output)
        command = ["/usr/bin/python3", str(ROOT / "research/inspection_aware/run_continuous.py"),
                   "--template", str(template), "--out", str(output),
                   "--slam-params", str(ROOT / "research/constraint_repair/slam_params_radius_1p5.yaml"),
                   "--seed", "7601", "--wall-limit", "800"]
        if arm == "aware":
            command += ["--inspector-script", str(ROOT / "research/inspection_aware/inspection.py")]
        with (OUT / "execution.jsonl").open("a") as ledger:
            ledger.write(json.dumps(dict(event="start", case=case, arm=arm, utc=utc(), command=command)) + "\n")
        with (OUT / f"{case}_{arm}.stdout").open("x") as log:
            code = subprocess.call(command, stdout=log, stderr=subprocess.STDOUT)
        result = load(output / "outcome.json") if (output / "outcome.json").exists() else {}
        with (OUT / "execution.jsonl").open("a") as ledger:
            ledger.write(json.dumps(dict(event="finish", case=case, arm=arm, utc=utc(), returncode=code,
                                         complete_contract=result.get("complete_contract"),
                                         infrastructure_failure=result.get("infrastructure_failure"))) + "\n")
        print(case, arm, code, result.get("complete_contract"), result.get("inspection_reason"), flush=True)
        if code or result.get("infrastructure_failure"):
            raise SystemExit(2)


if __name__ == "__main__":
    main()
