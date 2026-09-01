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
    promote_radius_m=None,
):
    """Register a stuck/path failure near (x, y)."""
    # Returns 'promoted', 'cooldown_renewed', or 'new'.
    #
    # The lifetime record is matched spatially within ``match_radius_m``
    # so SLAM drift cannot reset the count. Its ``failure_count``
    # increments on every hit; reaching ``promotion_failures`` promotes
    # the region to the permanent blacklist. Independently,
    # ``blocked_until_s`` is always pushed to ``now_s +
    # blacklist_duration_s`` so an active or renewed failure cools
    # down again.
    #
    # ``promote_radius_m`` bounds the PERMANENT exclusion footprint:
    # where a failure was scoped to the actual failed approach, this is
    # much smaller than ``match_radius_m`` (the lifetime-record grouping
    # radius), so promoting a large frontier cluster's failed cell does
    # not blanket the whole cluster's remaining approach cells. When
    # ``None`` the call is a legacy promotion and stores a bare (x, y)
    # region (infinite footprint) for compatibility.

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
            # Scope the permanent exclusion to the failed approach, not
            # the whole cluster centroid. When promote_radius_m is given
            # we store (x, y, radius); the legacy call (None) keeps the
            # bare (x, y) form (infinite footprint) for compatibility.
            if promote_radius_m is None:
                permanent_regions.append(
                    (record['x'], record['y'])
                )
            else:
                permanent_regions.append(
                    (record['x'], record['y'], promote_radius_m)
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
    """
    Index of the first region within ``radius_m`` of (x, y).

    A region may be a bare ``(x, y)`` tuple (legacy, treated as an
    infinite footprint) or ``(x, y, exclude_radius_m)`` (the V16.4
    scoped footprint). The effective footprint is the minimum of the
    stored radius and the query ``radius_m`` so a small-scoped permanent
    region never shadows a larger grouping radius.
    """
    for index, region in enumerate(regions):
        if len(region) >= 3:
            exclude_radius_m = region[2]
        else:
            exclude_radius_m = float('inf')

        effective = min(exclude_radius_m, radius_m)

        distance = math.hypot(
            x - region[0], y - region[1]
        )

        if distance <= effective:
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
    temporary_radius_m=None,
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
    #
    # ``temporary_radius_m`` bounds how far a promoted (infinite-cooldown)
    # failure record still blocks: scoping it to the actual failed
    # approach (not the lifetime-grouping radius) lets an alternative
    # approach in the same cluster survive. Defaults to
    # ``exclusion_radius_m`` for legacy behaviour.

    if temporary_radius_m is None:
        temporary_radius_m = exclusion_radius_m

    if find_region_index(
        permanent_regions, x, y, exclusion_radius_m
    ) is not None:
        return 'permanent'

    for record in failure_records:
        distance = math.hypot(
            x - record['x'], y - record['y']
        )

        record_radius = (
            temporary_radius_m
            if record['blocked_until_s'] == float('inf')
            else exclusion_radius_m
        )
        if distance > record_radius:
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
