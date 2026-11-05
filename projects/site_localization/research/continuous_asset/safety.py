"""Conventional estimated-motion rollout and conservative route-budget reserve."""
import math
import numpy as np

def arc(pose,v,w,horizon=1.,step=.05):
    x,y,a=pose;points=[]
    for t in np.arange(0,horizon+step/2,step):
        if abs(w)<1e-8:points.append([x+v*t*math.cos(a),y+v*t*math.sin(a)])
        else:points.append([x+v/w*(math.sin(a+w*t)-math.sin(a)),y-v/w*(math.cos(a+w*t)-math.cos(a))])
    return points

def admissible(grid,pose,v,w):
    return all(grid.index(p) is not None for p in arc(pose,v,w))

def reserve(travelled,leg,home,now,leg_turns=1):
    # Explicit engineering allowance, not a probabilistic guarantee. Recheck
    # after map/pose changes; stopping/replanning cannot consume the reserve silently.
    distance_required=1.25*(leg+home)+.5
    time_required=1.25*(leg+home)/.18+8*leg_turns+15
    return dict(admit=distance_required<=20-travelled and time_required<=240-now,
        required_distance_m=distance_required,remaining_distance_m=20-travelled,
        required_time_s=time_required,remaining_time_s=240-now,
        path_leg_m=leg,path_return_m=home,factor=1.25,stopping_margin_m=.5)
