"""State and direct transitions for one autonomous exploration mission."""

from collections import deque
from dataclasses import dataclass, field
import math

from rover_exploration.frontier_memory import FrontierMemory
from rover_exploration.frontier_selection import build_reachable_candidates
from rover_exploration.frontier_selection import filter_candidates
from rover_exploration.frontier_selection import find_reachable_approach
from rover_exploration.frontier_selection import grid_cell_center
from rover_exploration.frontier_selection import select_weighted_goal
from rover_exploration.frontier_selection import world_point_to_grid_cell
from rover_exploration.grid_planning import compute_reachable_component
from rover_exploration.grid_planning import find_escape_path
from rover_exploration.grid_planning import is_traversable_grid_cell
from rover_exploration.grid_planning import reconstruct_grid_path
from rover_exploration.stuck_detection import is_stuck


@dataclass(frozen=True)
class PolicyConfig:
    goal_reached_distance_m: float
    maximum_goal_path_failures: int
    maximum_fresh_approaches_per_target: int
    stuck_window_s: float
    stuck_progress_threshold_m: float
    stuck_alignment_threshold_rad: float
    blacklist_radius_m: float
    blacklist_duration_s: float
    permanent_after_failures: int
    visited_radius_m: float
    permanent_exclusion_radius_m: float
    distance_slack_m: float
    completion_debounce_period_s: float
    approach_search_radius_m: float


@dataclass
class MissionCounters:
    goals_assigned: int = 0
    goals_reached: int = 0
    failure_events: int = 0
    temporary_failure_events: int = 0
    recovery_requests: int = 0


@dataclass
class TargetState:
    anchor: tuple[int, int]
    goal_world: tuple[float, float] | None
    attempted_cells: set[tuple[int, int]] = field(default_factory=set)
    attempted_paths: set[tuple[tuple[int, int], ...]] = field(
        default_factory=set
    )
    fresh_approaches: int = 0
    path_failures: int = 0


@dataclass(frozen=True)
class TerminalDecision:
    outcome: str
    blocked_reason: str | None
    frontier_cells: int
    frontier_clusters: int
    reachable_candidates: int
    eligible_candidates: int
    temporary_rejected: int
    permanent_rejected: int
    retry_exhausted: int


@dataclass
class PendingTerminal:
    started_s: float
    decision: TerminalDecision


@dataclass
class CycleStats:
    approach_cells: list[tuple[int, int]] = field(default_factory=list)
    visited_rejected: int = 0
    temporary_rejected: int = 0
    permanent_rejected: int = 0
    retry_exhausted: int = 0
    too_close: int = 0
    duplicates: int = 0
    unreachable_clusters: int = 0
    eligible: int = 0
    selected_cluster_size: int = 0


@dataclass
class PlanUpdate:
    path: list[tuple[int, int]] | None = None
    selected_cell: tuple[int, int] | None = None
    stats: CycleStats = field(default_factory=CycleStats)
    completed_now: bool = False
    failure: tuple[str, float, float] | None = None
    goal_reached: tuple[float, float] | None = None
    goal_assigned: tuple[float, float] | None = None
    cooldown_hold_started: bool = False
    debounce_started: bool = False


@dataclass(frozen=True)
class StuckEvent:
    goal_x: float
    goal_y: float
    failure_outcome: str


class ExplorationPolicy:
    """Own the small amount of state that survives between map cycles."""

    def __init__(self, config):
        self.config = config
        self.memory = FrontierMemory()
        self.counters = MissionCounters()
        self.target = None
        self.progress_samples = deque()
        self.recovery_state = 'idle'
        self.pending_terminal = None
        self.terminal = None
        self.cooldown_hold = False

    @property
    def complete(self):
        return self.terminal is not None

    def recovery_status(self, active):
        """Apply obstacle-guard status; return whether a cycle ended."""
        if active:
            self.recovery_state = 'active'
            return False
        ended = self.recovery_state != 'idle'
        self.recovery_state = 'idle'
        if ended:
            self.progress_samples.clear()
        return ended

    def observe_pose(self, now_s, pose):
        """Register progress and return a newly detected stuck event."""
        if (
            self.complete
            or self.recovery_state != 'idle'
            or self.target is None
            or self.target.goal_world is None
        ):
            return None

        x, y, yaw = pose
        self.progress_samples.append((now_s, x, y, yaw))
        minimum_time = now_s - self.config.stuck_window_s
        while self.progress_samples and self.progress_samples[0][0] < minimum_time:
            self.progress_samples.popleft()

        if not is_stuck(
            progress_samples=self.progress_samples,
            goal_position=self.target.goal_world,
            minimum_window_s=self.config.stuck_window_s - 1.5,
            progress_threshold_m=self.config.stuck_progress_threshold_m,
            alignment_threshold_rad=self.config.stuck_alignment_threshold_rad,
        ):
            return None

        goal_x, goal_y = self.target.goal_world
        outcome = self._record_failure(goal_x, goal_y, now_s)
        self.target.goal_world = None
        if outcome == 'promoted':
            self.target = None
        self.progress_samples.clear()
        self.pending_terminal = None
        self.recovery_state = 'requested'
        self.counters.recovery_requests += 1
        return StuckEvent(goal_x, goal_y, outcome)

    def update(
        self,
        *,
        raw_data,
        planning_data,
        width,
        height,
        resolution,
        origin_x,
        origin_y,
        frontier_cells,
        frontier_clusters,
        robot_cell,
        robot_world,
        now_s,
    ):
        """Advance planning once and return the path/publication facts."""
        update = PlanUpdate()
        if self.complete or self.recovery_state != 'idle':
            return update
        if robot_cell is None or robot_world is None:
            return update

        bfs = self._reachable_tree(
            raw_data, planning_data, width, height, robot_cell
        )
        if bfs is None:
            return update

        active_anchor = self.target.anchor if self.target else None
        candidates = build_reachable_candidates(
            raw_data=raw_data,
            planning_data=planning_data,
            width=width,
            height=height,
            clusters=frontier_clusters,
            bfs=bfs,
            max_search_radius_cells=max(
                1,
                int(round(self.config.approach_search_radius_m / resolution)),
            ),
            active_anchor=active_anchor,
            attempted_cells=(
                self.target.attempted_cells if self.target else None
            ),
        )
        update.stats.duplicates = candidates.duplicates
        update.stats.unreachable_clusters = candidates.unreachable_clusters
        self.memory.prune(now_s)

        existing = self._update_existing_target(
            update,
            raw_data,
            planning_data,
            width,
            height,
            resolution,
            origin_x,
            origin_y,
            frontier_clusters,
            bfs,
            robot_world,
            now_s,
        )
        if existing:
            return update

        filtered = filter_candidates(
            candidates=candidates,
            memory=self.memory,
            resolution=resolution,
            origin_x=origin_x,
            origin_y=origin_y,
            robot_world=robot_world,
            now_s=now_s,
            temporary_radius_m=self.config.blacklist_radius_m,
            visited_radius_m=self.config.visited_radius_m,
            goal_reached_distance_m=self.config.goal_reached_distance_m,
            active_anchor=(self.target.anchor if self.target else None),
            fresh_approaches=(
                self.target.fresh_approaches if self.target else 0
            ),
            maximum_fresh_approaches=(
                self.config.maximum_fresh_approaches_per_target
            ),
        )
        self._copy_filter_stats(update.stats, candidates, filtered)

        selection = select_weighted_goal(
            bfs=bfs,
            candidates=candidates,
            eligible=filtered.eligible,
            distance_slack_cells=max(
                1, int(round(self.config.distance_slack_m / resolution))
            ),
        )
        if selection is not None:
            self._assign_selection(
                update,
                selection,
                candidates,
                resolution,
                origin_x,
                origin_y,
            )
            return update

        self._hold_or_complete(
            update,
            frontier_cells=len(frontier_cells),
            frontier_clusters=len(frontier_clusters),
            candidates=len(candidates.sizes),
            now_s=now_s,
        )
        return update

    @staticmethod
    def _open_escape_corridor(raw_data, planning_data, width, height, start):
        escape = find_escape_path(
            raw_data, planning_data, width, height, start
        )
        if escape is None:
            return False
        for row, column in escape[:-1]:
            planning_data[row * width + column] = 0
        return True

    def _reachable_tree(self, raw_data, planning_data, width, height, start):
        if not is_traversable_grid_cell(
            planning_data, width, height, start[0], start[1]
        ) and not self._open_escape_corridor(
            raw_data, planning_data, width, height, start
        ):
            return None
        return compute_reachable_component(
            planning_data, width, height, start
        )

    def _hold_or_complete(
        self, update, frontier_cells, frontier_clusters, candidates, now_s
    ):
        if update.stats.temporary_rejected:
            update.cooldown_hold_started = not self.cooldown_hold
            self.cooldown_hold = True
            self.pending_terminal = None
            return

        self.cooldown_hold = False
        decision = self._terminal_decision(
            frontier_cells, frontier_clusters, candidates, update.stats
        )
        if self.pending_terminal is None:
            self.pending_terminal = PendingTerminal(now_s, decision)
            update.debounce_started = True
            return

        self.pending_terminal.decision = decision
        if now_s - self.pending_terminal.started_s < (
            self.config.completion_debounce_period_s
        ):
            return
        self.terminal = decision
        self.target = None
        self.progress_samples.clear()
        self.pending_terminal = None
        update.completed_now = True

    @staticmethod
    def _copy_filter_stats(stats, candidates, filtered):
        stats.approach_cells = list(candidates.sizes)
        stats.visited_rejected = filtered.visited
        stats.temporary_rejected = filtered.temporary
        stats.permanent_rejected = filtered.permanent
        stats.retry_exhausted = filtered.retry_exhausted
        stats.too_close = filtered.too_close
        stats.eligible = len(filtered.eligible)

    def result_values(self, completion_time_s):
        """Return schema fields for the latched terminal decision."""
        if self.terminal is None:
            raise RuntimeError('mission result requested before terminal decision')
        terminal = self.terminal
        counters = self.counters
        return {
            'outcome': terminal.outcome,
            'blocked_reason': terminal.blocked_reason,
            'completion_time_s': completion_time_s,
            'goals_assigned': counters.goals_assigned,
            'goals_reached': counters.goals_reached,
            'failure_events': counters.failure_events,
            'temporary_failure_events': counters.temporary_failure_events,
            'permanent_failed_regions': len(self.memory.permanent_failures),
            'recovery_requests': counters.recovery_requests,
            'visited_regions': len(self.memory.visited),
            'frontier_cells': terminal.frontier_cells,
            'frontier_clusters': terminal.frontier_clusters,
            'geometric_frontier_cells': terminal.frontier_cells,
            'geometric_frontier_clusters': terminal.frontier_clusters,
            'reachable_candidate_clusters': terminal.reachable_candidates,
            'post_exclusion_eligible': terminal.eligible_candidates,
        }

    def _record_failure(self, x, y, now_s):
        outcome = self.memory.register_failure(
            x=x,
            y=y,
            now_s=now_s,
            match_radius_m=self.config.blacklist_radius_m,
            cooldown_s=self.config.blacklist_duration_s,
            promotion_failures=self.config.permanent_after_failures,
            permanent_radius_m=self.config.permanent_exclusion_radius_m,
        )
        self.counters.failure_events += 1
        if outcome != 'promoted':
            self.counters.temporary_failure_events += 1
        return outcome

    def _update_existing_target(
        self,
        update,
        raw_data,
        planning_data,
        width,
        height,
        resolution,
        origin_x,
        origin_y,
        clusters,
        bfs,
        robot_world,
        now_s,
    ):
        if self.target is None or self.target.goal_world is None:
            return False
        goal_x, goal_y = self.target.goal_world
        if math.hypot(goal_x - robot_world[0], goal_y - robot_world[1]) <= (
            self.config.goal_reached_distance_m
        ):
            self.memory.visited.append((goal_x, goal_y))
            self.counters.goals_reached += 1
            update.goal_reached = (goal_x, goal_y)
            self.target = None
            self.progress_samples.clear()
            return False

        goal_cell = world_point_to_grid_cell(
            goal_x, goal_y, resolution, origin_x, origin_y
        )
        path = None
        if goal_cell in bfs['cost']:
            path = reconstruct_grid_path(bfs['came_from'], goal_cell)
        if path is not None:
            update.selected_cell = goal_cell
            update.path = path
            return True

        return self._handle_invalid_target(
            update,
            raw_data,
            planning_data,
            width,
            height,
            resolution,
            origin_x,
            origin_y,
            clusters,
            bfs,
            goal_x,
            goal_y,
            now_s,
        )

    def _handle_invalid_target(
        self,
        update,
        raw_data,
        planning_data,
        width,
        height,
        resolution,
        origin_x,
        origin_y,
        clusters,
        bfs,
        goal_x,
        goal_y,
        now_s,
    ):
        self.target.path_failures += 1
        if self.target.path_failures < self.config.maximum_goal_path_failures:
            return True

        fresh = self._fresh_approach(
            raw_data,
            planning_data,
            width,
            height,
            resolution,
            clusters,
            bfs,
        )
        if fresh is not None:
            cell, path = fresh
            self.target.goal_world = grid_cell_center(
                cell[0], cell[1], resolution, origin_x, origin_y
            )
            self.target.attempted_cells.add(cell)
            self.target.attempted_paths.add(tuple(path))
            self.target.fresh_approaches += 1
            self.counters.goals_assigned += 1
            self.progress_samples.clear()
            self.pending_terminal = None
            update.selected_cell = cell
            update.path = path
            update.goal_assigned = self.target.goal_world
            return True

        outcome = self._record_failure(goal_x, goal_y, now_s)
        update.failure = (outcome, goal_x, goal_y)
        self.target.goal_world = None
        if outcome == 'promoted':
            self.target = None
        self.progress_samples.clear()
        return False

    def _fresh_approach(
        self,
        raw_data,
        planning_data,
        width,
        height,
        resolution,
        clusters,
        bfs,
    ):
        target = self.target
        if target.fresh_approaches >= (
            self.config.maximum_fresh_approaches_per_target
        ):
            return None
        cluster = next((item for item in clusters if target.anchor in item), None)
        if cluster is None:
            return None

        excluded = set(target.attempted_cells)
        radius = max(
            1, int(round(self.config.approach_search_radius_m / resolution))
        )
        while len(excluded) < width * height:
            cell = find_reachable_approach(
                raw_data=raw_data,
                planning_data=planning_data,
                width=width,
                height=height,
                cluster=cluster,
                bfs=bfs,
                max_search_radius_cells=radius,
                excluded_cells=excluded,
            )
            if cell is None:
                return None
            path = reconstruct_grid_path(bfs['came_from'], cell)
            if path is not None and tuple(path) not in target.attempted_paths:
                return cell, path
            excluded.add(cell)
        return None

    def _assign_selection(
        self,
        update,
        selection,
        candidates,
        resolution,
        origin_x,
        origin_y,
    ):
        cell, path = selection
        anchor = candidates.anchors[cell]
        goal_world = grid_cell_center(
            cell[0], cell[1], resolution, origin_x, origin_y
        )
        if self.target is not None and anchor == self.target.anchor:
            self.target.goal_world = goal_world
            self.target.attempted_cells.add(cell)
            self.target.attempted_paths.add(tuple(path))
            self.target.fresh_approaches += 1
        else:
            self.target = TargetState(
                anchor=anchor,
                goal_world=goal_world,
                attempted_cells={cell},
                attempted_paths={tuple(path)},
            )
        self.counters.goals_assigned += 1
        self.progress_samples.clear()
        self.pending_terminal = None
        self.cooldown_hold = False
        update.path = path
        update.selected_cell = cell
        update.goal_assigned = goal_world
        update.stats.selected_cluster_size = candidates.sizes[cell]

    @staticmethod
    def _terminal_decision(frontier_cells, frontier_clusters, candidates, stats):
        if frontier_clusters == 0:
            return TerminalDecision(
                'success', None, frontier_cells, 0, candidates, stats.eligible,
                stats.temporary_rejected, stats.permanent_rejected,
                stats.retry_exhausted,
            )
        if stats.retry_exhausted:
            reason = (
                'geometric frontier remains (>= 5 cells in a component) '
                'but the active target exhausted its fresh-approach retry cap'
            )
        elif stats.permanent_rejected:
            reason = (
                'geometric frontier remains (>= 5 cells in a component) '
                'but all candidates are permanently blacklisted; bounded '
                'recovery exhausted'
            )
        else:
            reason = (
                'geometric frontier remains (>= 5 cells in a component) but '
                'no candidate is eligible; remaining rejects are visited / '
                'too-close cells, which are stable and do not resolve on retry'
            )
        return TerminalDecision(
            'blocked', reason, frontier_cells, frontier_clusters, candidates,
            stats.eligible, stats.temporary_rejected, stats.permanent_rejected,
            stats.retry_exhausted,
        )
