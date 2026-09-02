"""Scoped failure and visited memory for one exploration mission."""

from dataclasses import dataclass
import math


@dataclass
class FailureRecord:
    """Lifetime failure evidence and its current exclusion state."""

    x: float
    y: float
    count: int
    blocked_until_s: float
    permanent_radius_m: float | None = None


class FrontierMemory:
    """Own all spatial failure and visited state for one mission."""

    def __init__(self):
        """Start with no spatial evidence."""
        self.failures = []
        self.visited = []

    def register_failure(
        self,
        x,
        y,
        now_s,
        match_radius_m,
        cooldown_s,
        promotion_failures,
        permanent_radius_m,
    ):
        """Record a failure and return its lifecycle outcome."""
        record = next(
            (
                item for item in self.failures
                if math.hypot(x - item.x, y - item.y) <= match_radius_m
            ),
            None,
        )
        if record is None:
            self.failures.append(FailureRecord(
                x=x,
                y=y,
                count=1,
                blocked_until_s=now_s + cooldown_s,
            ))
            return 'new'

        record.count += 1
        if record.count >= promotion_failures:
            record.blocked_until_s = float('inf')
            record.permanent_radius_m = permanent_radius_m
            return 'promoted'

        record.blocked_until_s = now_s + cooldown_s
        return 'cooldown_renewed'

    def prune(self, now_s):
        """Expire cooldowns without erasing lifetime failure counts."""
        for record in self.failures:
            if (
                record.permanent_radius_m is None
                and now_s >= record.blocked_until_s
            ):
                record.blocked_until_s = 0.0

    def exclusion_reason(
        self,
        x,
        y,
        now_s,
        temporary_radius_m,
        visited_radius_m,
    ):
        """Return ``permanent``, ``temporary``, ``visited``, or ``None``."""
        for record in self.failures:
            if record.permanent_radius_m is not None and math.hypot(
                x - record.x, y - record.y
            ) <= record.permanent_radius_m:
                return 'permanent'

        for record in self.failures:
            if (
                record.permanent_radius_m is None
                and now_s < record.blocked_until_s
                and math.hypot(x - record.x, y - record.y)
                <= temporary_radius_m
            ):
                return 'temporary'

        if any(
            math.hypot(x - visited_x, y - visited_y) <= visited_radius_m
            for visited_x, visited_y in self.visited
        ):
            return 'visited'
        return None

    def active_cooldowns(self, now_s):
        """Return active temporary failure records."""
        return [
            record for record in self.failures
            if record.permanent_radius_m is None
            and now_s < record.blocked_until_s
        ]

    @property
    def permanent_failures(self):
        """Return records promoted to permanent scoped exclusion."""
        return [
            record for record in self.failures
            if record.permanent_radius_m is not None
        ]
