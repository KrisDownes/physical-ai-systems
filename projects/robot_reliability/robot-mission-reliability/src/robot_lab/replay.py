from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from .contracts import validate_event


def read_events(path: Path) -> Iterator[dict]:
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            event = json.loads(line)
            errors = validate_event(event)
            if errors:
                raise ValueError(f"invalid event at line {line_number}: {'; '.join(errors)}")
            yield event


def replay_events(path: Path) -> Iterator[dict]:
    """Yield events in robot event-time order and reject ambiguous ordering."""

    events = list(read_events(path))
    sequences = [event["sequence"] for event in events]
    if len(sequences) != len(set(sequences)):
        raise ValueError("duplicate sequence number in run")
    yield from sorted(events, key=lambda event: (event["event_time_s"], event["sequence"]))
