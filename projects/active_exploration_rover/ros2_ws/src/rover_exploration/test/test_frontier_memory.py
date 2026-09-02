"""Tests for the one authoritative frontier-memory representation."""

from rover_exploration.frontier_memory import FrontierMemory


def fail(memory, x=1.0, y=1.0, now_s=0.0):
    return memory.register_failure(
        x=x,
        y=y,
        now_s=now_s,
        match_radius_m=0.75,
        cooldown_s=30.0,
        promotion_failures=2,
        permanent_radius_m=0.20,
    )


def excluded(memory, x=1.0, y=1.0, now_s=0.0):
    return memory.exclusion_reason(
        x=x,
        y=y,
        now_s=now_s,
        temporary_radius_m=0.75,
        visited_radius_m=0.60,
    )


def test_new_failure_has_temporary_cooldown():
    memory = FrontierMemory()
    assert fail(memory) == 'new'
    assert excluded(memory, now_s=29.9) == 'temporary'
    assert excluded(memory, now_s=30.0) is None


def test_cooldown_expiry_keeps_lifetime_count_for_promotion():
    memory = FrontierMemory()
    fail(memory, now_s=0.0)
    memory.prune(30.0)

    assert memory.failures[0].count == 1
    assert fail(memory, now_s=40.0) == 'promoted'
    assert memory.failures[0].count == 2


def test_second_nearby_failure_promotes_same_record():
    memory = FrontierMemory()
    fail(memory, now_s=0.0)
    assert fail(memory, x=1.5, now_s=10.0) == 'promoted'
    assert len(memory.failures) == 1
    assert len(memory.permanent_failures) == 1


def test_permanent_exclusion_is_scoped_to_failed_approach():
    memory = FrontierMemory()
    fail(memory)
    fail(memory, now_s=31.0)

    assert excluded(memory, x=1.19, now_s=1000.0) == 'permanent'
    assert excluded(memory, x=1.21, now_s=1000.0) is None


def test_prune_never_resurrects_permanent_failure():
    memory = FrontierMemory()
    fail(memory)
    fail(memory, now_s=31.0)
    memory.prune(1_000_000.0)

    assert excluded(memory, now_s=1_000_000.0) == 'permanent'


def test_overlapping_active_cooldown_excludes_candidate():
    memory = FrontierMemory()
    fail(memory, x=0.4, now_s=0.0)
    fail(memory, x=1.2, now_s=20.0)

    assert excluded(memory, x=0.8, now_s=35.0) == 'temporary'


def test_visited_memory_is_distinct_from_failure_memory():
    memory = FrontierMemory()
    memory.visited.append((2.0, 3.0))

    assert excluded(memory, x=2.5, y=3.0) == 'visited'
    assert not memory.failures


def test_active_cooldown_and_permanent_views_are_derived():
    memory = FrontierMemory()
    fail(memory, x=1.0, now_s=0.0)
    fail(memory, x=3.0, now_s=0.0)
    fail(memory, x=3.0, now_s=1.0)

    assert [record.x for record in memory.active_cooldowns(2.0)] == [1.0]
    assert [record.x for record in memory.permanent_failures] == [3.0]
