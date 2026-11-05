"""Finite Bayesian visual search over an acquired 2-D map.

Engineering adaptation of Ye and Tsotsos (1999): a discrete object-pose
prior, camera action detection likelihood, negative-image update, and
expected detection per travel cost. No scenario or evaluator input.
"""
import math
import time

import numpy as np


def wrap(a):
    return np.arctan2(np.sin(a), np.cos(a))


class CameraBeliefSearch:
    def __init__(self, planner, map_path, coordinates):
        self.planner = planner
        with open(map_path) as stream:
            stream.readline()
            self.raw_grid = np.loadtxt(stream, dtype=int)
        # Every fourth collision-clear map cell, with four possible panel faces.
        yy, xx = np.nonzero(planner.free[::4, ::4])
        xy = np.c_[planner.ox + 4*xx*planner.res,
                   planner.oy + 4*yy*planner.res]
        self.xy = np.repeat(xy, 4, axis=0)
        self.face = np.tile(np.arange(4)*math.pi/2, len(xy))
        self.weight = np.full(len(self.xy), 1/max(len(self.xy), 1))
        self.stations = [(float(x), float(y)) for x in coordinates for y in coordinates
                         if planner.index((x, y)) is not None]
        self.los_cache = {}
        self.last_update_id = -1

    def los(self, station):
        key = tuple(station[:2])
        if key in self.los_cache:
            return self.los_cache[key]
        origin = np.asarray(key)
        distance = np.linalg.norm(self.xy-origin, axis=1)
        selected = np.flatnonzero((distance >= .6) & (distance <= 5.))
        visible = np.zeros(len(self.xy), bool)
        if len(selected):
            # An acquired 2-D occupancy ray is only a proxy for the renderer's
            # three-dimensional obstruction test.
            ray = origin[None, None, :] + np.linspace(0, 1, 101)[None, :, None] * (
                self.xy[selected, None, :] - origin[None, None, :])
            ix = np.rint((ray[:, :, 0]-self.planner.ox)/self.planner.res).astype(int)
            iy = np.rint((ray[:, :, 1]-self.planner.oy)/self.planner.res).astype(int)
            inside = (ix >= 0) & (ix < self.planner.w) & (iy >= 0) & (iy < self.planner.h)
            ix = np.clip(ix, 0, self.planner.w-1)
            iy = np.clip(iy, 0, self.planner.h-1)
            visible[selected] = np.all(inside & (self.raw_grid[iy, ix] == -1), axis=1)
        self.los_cache[key] = visible
        return visible

    def detection_probability(self, pose):
        station = np.asarray(pose[:2])
        delta = self.xy-station
        range_m = np.linalg.norm(delta, axis=1)
        bearing = np.arctan2(delta[:, 1], delta[:, 0])
        # Actual renderer: 1920 px, 60 degree horizontal FOV, 0.16 m marker.
        focal_px = 960/math.tan(math.radians(30))
        front = np.cos(self.face-bearing)
        marker_edge_px = focal_px*.16*np.maximum(front, 0)/np.maximum(range_m, .1)
        camera = (abs(wrap(bearing-pose[2])) <= math.radians(25)) & (front >= .35)
        return .75 * (camera & (marker_edge_px >= 16) & self.los(station)).astype(float)

    def update_negative(self, capture):
        if capture['id'] <= self.last_update_id:
            return None
        self.last_update_id = capture['id']
        ids = capture['vision'].get('observed_ids', [])
        if 7 in ids or capture['vision'].get('target') is not None:
            return dict(capture_id=capture['id'], negative=False)
        likelihood = self.detection_probability(capture['associated_pose'])
        self.weight *= 1-likelihood
        total = self.weight.sum()
        if total > 0:
            self.weight /= total
        return dict(capture_id=capture['id'], negative=True,
                    expected_detection_probability=float(np.dot(self.weight, likelihood)),
                    support_count=int(np.count_nonzero(self.weight > 1e-9)))

    def choose(self, start, visited, path_so_far, max_path):
        began = time.monotonic()
        options = []
        for station in self.stations:
            if any(np.linalg.norm(np.asarray(station)-np.asarray(v)) < .5 for v in visited):
                continue
            route = self.planner.path(start, station)
            if route is None:
                continue
            length = float(route['length_m'])
            # Same eight orientations/captures as the conventional policies.
            probability = np.zeros(len(self.xy))
            for k in range(8):
                p = self.detection_probability([*station, k*math.pi/4])
                probability = 1-(1-probability)*(1-p)
            # Budget check over pose hypotheses, not a hidden target pose.
            outward = -np.c_[np.cos(self.face), np.sin(self.face)]
            stand_off = self.xy + 1.1*outward
            approach_lower_bound = np.linalg.norm(stand_off-np.asarray(station), axis=1)
            remaining = max_path-path_so_far-length
            actionable = remaining >= approach_lower_bound+1.4
            mass = float(np.dot(self.weight, probability))
            useful = float(np.dot(self.weight, probability*actionable))
            options.append(dict(station=list(station), path_m=length,
                                expected_detection_mass=mass,
                                budget_feasible_detection_mass=useful,
                                remaining_path_m=remaining,
                                score=useful/(length+1.)))
        if not options:
            return None, dict(rule='no_reachable_unvisited_station', options=options,
                              elapsed_wall_s=time.monotonic()-began)
        chosen = min(options, key=lambda q: (-q['score'], -q['budget_feasible_detection_mass'],
                                              q['path_m'], *q['station']))
        if chosen['budget_feasible_detection_mass'] <= 0:
            return None, dict(rule='no_budget_feasible_detection_station', options=options,
                              elapsed_wall_s=time.monotonic()-began)
        return [*chosen['station'], 0.], dict(rule='discrete_bayesian_expected_detection_per_cost',
            selected=chosen['station'], options=options, hypotheses=len(self.xy),
            elapsed_wall_s=time.monotonic()-began, truth_used=False,
            limitation='2-D map occlusion and uncalibrated detection likelihood; not a reader predictor')
