"""Destination selection only. No selector publishes velocity or plans a path."""
import math
from rover_exploration.frontier_selection import select_weighted_goal, grid_cell_center
from rover_exploration.grid_planning import reconstruct_grid_path


class ClassicalSelector:
    state = 'executing'

    def select(self, *, context, **selection):
        return select_weighted_goal(**selection)


class GoalExecutive:
    """Single outstanding request, fail-closed cancellation and current-grid validation.

    Time arguments are monotonic wall seconds. Geometry is online-map meters.
    The transport calls respond(); only select() on the ROS thread commits goals.
    """
    def __init__(self, emit, request, debounce_s=1.0, response_timeout_s=60.0):
        self.emit, self.request = emit, request
        self.debounce_s, self.response_timeout_s = debounce_s, response_timeout_s
        self.state = 'awaiting_goal'; self.pending = None; self.response = None
        self.generation = 0; self.goal_version = 0; self.last_request = -math.inf
        self.reason = 'startup'; self.closed = False; self.safe = True
        self.outcomes = []; self.request_count = 0

    def event(self, reason, **details):
        self.reason = reason
        self.outcomes.append({'reason': reason, 'goal_version': self.goal_version, **details})
        self.outcomes = self.outcomes[-12:]

    def cancel(self, reason, complete=False):
        if self.closed: return
        self.generation += 1; self.pending = None; self.response = None
        self.reason = reason
        self.closed = True; self.state = 'complete' if complete else 'stopped'
        self.emit('cancel', reason=reason, generation=self.generation)

    def safety(self, safe):
        if safe == self.safe: return
        self.safe = safe
        if not safe:
            self.generation += 1; self.pending = None; self.response = None
            self.state = 'stopped'; self.emit('safety_stop')
        elif not self.closed:
            self.state = 'reconsidering'; self.reason = 'safety_recovered_goal_invalid'
            self.emit('safety_recovered')

    def respond(self, request_id, destination=None, error=None):
        if self.closed or not self.pending or self.pending['request_id'] != request_id:
            self.emit('response_discarded', request_id=request_id, reason='canceled_or_superseded')
            return
        self.response = (destination, error)

    def select(self, *, context, bfs, candidates, eligible, distance_slack_cells):
        now = context['wall_s']
        if self.closed or not self.safe: return None
        if self.pending:
            if now-self.pending['wall_s'] > self.response_timeout_s:
                self.cancel('decision_timeout'); return None
            if self.response is None: return None
            destination, error = self.response
            pending = self.pending; self.pending = None; self.response = None
            if error:
                self.cancel(error); return None
            offered = {c['id']: c for c in pending['candidates']}
            candidate = offered.get(destination)
            # Revalidate the frozen world point against current eligible approaches.
            match = None
            if candidate:
                for cell in eligible:
                    x,y = grid_cell_center(*cell,context['resolution'],context['origin_x'],context['origin_y'])
                    if math.hypot(x-candidate['x_m'],y-candidate['y_m']) <= context['resolution']*.75:
                        match=cell; break
            if match is None or match not in bfs['cost']:
                self.event('stale_or_invalid_destination'); self.state='reconsidering'
                self.emit('response_discarded',request_id=pending['request_id'],reason=self.reason,
                          request_map_version=pending['map_version'],current_map_version=context['map_version'])
                return None
            self.goal_version += 1; self.state='executing'
            self.emit('goal_selected',request_id=pending['request_id'],goal_version=self.goal_version,
                      candidate=candidate,accepted_destination_m=grid_cell_center(*match,context['resolution'],context['origin_x'],context['origin_y']),validated_map_version=context['map_version'],
                      request_to_response_s=now-pending['wall_s'])
            return match,reconstruct_grid_path(bfs['came_from'],match)
        if not eligible or now-self.last_request < self.debounce_s: return None
        self.generation += 1; self.request_count += 1
        choices=[]
        for i,cell in enumerate(sorted(eligible)):
            x,y=grid_cell_center(*cell,context['resolution'],context['origin_x'],context['origin_y'])
            choices.append(dict(id=f'A{i+1}',x_m=x,y_m=y,cells=candidates.sizes[cell]))
        self.pending=dict(request_id=self.generation,goal_version=self.goal_version,
                          map_version=context['map_version'],wall_s=now,sim_s=context['now_s'],
                          reason=self.reason,candidates=choices,recent_goal_outcomes=list(self.outcomes))
        self.last_request=now; self.state='awaiting_goal'
        self.emit('decision_requested',**self.pending)
        self.request(dict(self.pending))
        return None
