# Spatial goal memory for frontier failures.
#
# Two distinct concepts are stored separately:
#
# * Lifetime FAILURE RECORDS: one per spatial region, holding
#   ``failure_count``. They never expire while the node runs and
#   are the promotion evidence. Pruning cooldowns must not erase
#   them.
# * ACTIVE COOLDOWNS: each record carries ``blocked_until_s``.
#   A region is temporarily excluded only while
#   ``now_s < blocked_until_s``. Expiry clears eligibility but
#   leaves the record and its count untouched.

import math


def find_record_index(records, x, y, radius_m):
    """Index of the first record within radius_m of (x, y)."""
    for index, record in enumerate(records):
        distance = math.hypot(
            x - record['x'], y - record['y']
        )

        if distance <= radius_m:
            return index

    return None


def record_failure(
    failure_records,
    permanent_regions,
    x,
    y,
    now_s,
    match_radius_m,
    blacklist_duration_s,
    promotion_failures,
):
    """Register a stuck/path failure near (x, y)."""
    # Returns 'promoted', 'cooldown_renewed', or 'new'.
    #
    # The lifetime record is matched spatially within
    # ``match_radius_m`` so SLAM drift cannot reset the count. Its
    # ``failure_count`` increments on every hit; reaching
    # ``promotion_failures`` promotes the region to the permanent
    # blacklist. Independently, ``blocked_until_s`` is always pushed
    # to ``now_s + blacklist_duration_s`` so an active or renewed
    # failure cools down again.

    index = find_record_index(
        failure_records, x, y, match_radius_m
    )

    if index is None:
        failure_records.append({
            'x': x,
            'y': y,
            'failure_count': 1,
            'blocked_until_s': (
                now_s + blacklist_duration_s
            ),
        })
        return 'new'

    record = failure_records[index]
    record['failure_count'] += 1

    blocked_until = now_s + blacklist_duration_s

    if record['failure_count'] >= promotion_failures:
        already_permanent = find_region_index(
            permanent_regions,
            record['x'], record['y'],
            match_radius_m,
        )

        if already_permanent is None:
            permanent_regions.append(
                (record['x'], record['y'])
            )

        record['blocked_until_s'] = float('inf')
        return 'promoted'

    record['blocked_until_s'] = blocked_until
    return 'cooldown_renewed'


def prune_expired_cooldowns(failure_records, now_s):
    # Clear expired cooldowns without deleting any record.
    # Lifetime failure counts survive pruning by design: a pruned
    # cooldown still counts toward later promotion.

    for record in failure_records:
        if (
            record['blocked_until_s'] != float('inf')
            and now_s >= record['blocked_until_s']
        ):
            record['blocked_until_s'] = 0.0


def find_region_index(regions, x, y, radius_m):
    """Index of the first (x, y) tuple region within radius."""
    for index, region in enumerate(regions):
        distance = math.hypot(
            x - region[0], y - region[1]
        )

        if distance <= radius_m:
            return index

    return None


def is_excluded(
    x,
    y,
    failure_records,
    permanent_regions,
    visited_regions,
    now_s=0.0,
    exclusion_radius_m=0.75,
    visited_radius_m=0.60,
):
    """Classify why a candidate is excluded."""
    # Returns one of None, 'permanent', 'temporary', 'visited'.
    # Permanent is checked first so no amount of elapsed time can
    # resurrect a permanently failed region.
    #
    # Every failure record within exclusion_radius_m is inspected:
    # if ANY carries an active cooldown the candidate is temporarily
    # excluded, even when an expired record also matches. Expired
    # records never mask an active neighbour's exclusion.

    if find_region_index(
        permanent_regions, x, y, exclusion_radius_m
    ) is not None:
        return 'permanent'

    for record in failure_records:
        distance = math.hypot(
            x - record['x'], y - record['y']
        )

        if distance > exclusion_radius_m:
            continue

        if (
            record['blocked_until_s'] == float('inf')
            or now_s < record['blocked_until_s']
        ):
            return 'temporary'

    if find_region_index(
        visited_regions, x, y, visited_radius_m
    ) is not None:
        return 'visited'

    return None
