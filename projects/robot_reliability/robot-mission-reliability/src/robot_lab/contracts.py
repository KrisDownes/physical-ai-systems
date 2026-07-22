from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class EventEnvelope:
    """Common metadata required to correlate every event from one robot run."""

    event_id: str
    run_id: str
    mission_id: str
    robot_id: str
    software_version: str
    event_type: str
    sequence: int
    event_time_s: float
    recorded_at: str
    schema_version: str
    payload: dict[str, Any]

    @classmethod
    def create(
        cls,
        *,
        run_id: str,
        mission_id: str,
        robot_id: str,
        software_version: str,
        event_type: str,
        sequence: int,
        event_time_s: float,
        payload: dict[str, Any],
    ) -> "EventEnvelope":
        return cls(
            event_id=str(uuid4()),
            run_id=run_id,
            mission_id=mission_id,
            robot_id=robot_id,
            software_version=software_version,
            event_type=event_type,
            sequence=sequence,
            event_time_s=round(event_time_s, 6),
            recorded_at=datetime.now(timezone.utc).isoformat(),
            schema_version=SCHEMA_VERSION,
            payload=payload,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


REQUIRED_FIELDS = {
    "event_id",
    "run_id",
    "mission_id",
    "robot_id",
    "software_version",
    "event_type",
    "sequence",
    "event_time_s",
    "recorded_at",
    "schema_version",
    "payload",
}


def validate_event(event: dict[str, Any]) -> list[str]:
    """Return validation errors rather than throwing away diagnostic context."""

    errors: list[str] = []
    missing = REQUIRED_FIELDS - event.keys()
    if missing:
        errors.append(f"missing fields: {sorted(missing)}")
    if not isinstance(event.get("sequence"), int):
        errors.append("sequence must be an integer")
    if not isinstance(event.get("event_time_s"), (int, float)):
        errors.append("event_time_s must be numeric")
    if not isinstance(event.get("payload"), dict):
        errors.append("payload must be an object")
    if event.get("event_type") not in {
        "robot.telemetry",
        "software.health",
        "mission.event",
        "fault.injection",
    }:
        errors.append("event_type is unknown")
    return errors
