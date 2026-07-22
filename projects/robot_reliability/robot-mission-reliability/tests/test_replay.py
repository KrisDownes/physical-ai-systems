import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from robot_lab.replay import replay_events
from robot_lab.simulation import RunConfig, run_simulation


class ReplayTests(unittest.TestCase):
    def test_duplicate_sequences_are_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "run.jsonl"
            run_simulation(path, RunConfig(software_version="v1", max_time_s=1.0))
            events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            events[1]["sequence"] = events[0]["sequence"]
            path.write_text(
                "\n".join(json.dumps(event) for event in events) + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "duplicate sequence"):
                list(replay_events(path))
