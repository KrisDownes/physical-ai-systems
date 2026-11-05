"""Read-only planned-cost reserve diagnostic, without evaluator truth."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/matched_visibility_20261027"
RESERVE_M = 4.9


def load(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def evaluate(path):
    policy = load(path / "inspection/public/policy.json")
    limit = policy["max_path_m"]
    assert limit == 18
    plans = [event for event in rows(path / "inspection/method.jsonl")
             if event["event"] == "plan" and event.get("path") is not None]
    cumulative = 0.
    previous_station = None
    decisions = []
    for event in plans:
        length = float(event["path"]["length_m"])
        new_station = event["purpose"] in ("old_view", "search") and (
            previous_station is None or event["goal"][:2] != previous_station)
        if new_station:
            admission_cost = cumulative + length + RESERVE_M
            decisions.append(dict(time_s=event["time"], station=event["goal"][:2],
                                  planned_before_m=cumulative, planned_leg_m=length,
                                  assumed_reserve_m=RESERVE_M,
                                  total_with_reserve_m=admission_cost,
                                  admitted=admission_cost<=limit))
            previous_station = event["goal"][:2]
        cumulative += length
    first_reject = next((item for item in decisions if not item["admitted"]), None)
    return dict(output=str(path), limit_m=limit, assumed_reserve_m=RESERVE_M,
                decisions=decisions, first_rejected_station=None if first_reject is None else first_reject["station"],
                scope="Offline planned-cost gate; no physical task outcome inferred",
                truth_used=False)


def main():
    nearest = evaluate(OUT / "mission_wrong_asset_nearest_infra_retry02")
    gain = evaluate(OUT / "mission_wrong_asset_gain")
    result = dict(protocol="docs/post_repair_connected/budget_replay_protocol.md",
                  nearest=nearest, visibility_gain=gain)
    (OUT / "budget_replay.json").write_text(json.dumps(result, indent=2) + "\n")
    for name, arm in (("nearest", nearest), ("gain", gain)):
        print(name, [(x["station"],round(x["total_with_reserve_m"],2),x["admitted"]) for x in arm["decisions"]])


if __name__ == "__main__":
    main()
