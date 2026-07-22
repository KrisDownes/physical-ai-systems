import unittest

from robot_lab.contracts import EventEnvelope, validate_event


class ContractTests(unittest.TestCase):
    def test_valid_event_has_no_errors(self) -> None:
        event = EventEnvelope.create(
            run_id="run",
            mission_id="mission",
            robot_id="robot",
            software_version="v1",
            event_type="robot.telemetry",
            sequence=0,
            event_time_s=0.0,
            payload={},
        )
        self.assertEqual(validate_event(event.to_dict()), [])

    def test_invalid_event_reports_multiple_errors(self) -> None:
        errors = validate_event({"sequence": "zero", "payload": []})
        self.assertGreaterEqual(len(errors), 3)
