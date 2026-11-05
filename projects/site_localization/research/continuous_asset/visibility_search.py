"""Conventional map-only visibility gain per travel effort for fixed search stations.

This estimates possible coverage, not actual marker detection or absence.
It reads the acquired occupancy grid and visited stations, never the scene.
"""
import math
from pathlib import Path
import time

import numpy as np


class VisibilitySearch:
    def __init__(self,planner,map_path,coordinates):
        self.planner=planner
        with Path(map_path).open() as stream:
            stream.readline()
            self.grid=np.loadtxt(stream,dtype=int)
        yy,xx=np.nonzero(self.grid[::2,::2]==-1)
        self.cells=np.c_[2*xx,2*yy]
        self.points=np.c_[planner.ox+self.cells[:,0]*planner.res,
                          planner.oy+self.cells[:,1]*planner.res]
        self.area=(2*planner.res)**2
        self.station_masks={}
        self.stations=[]
        for x in coordinates:
            for y in coordinates:
                point=(float(x),float(y))
                if planner.index(point) is not None:
                    self.stations.append(point)

    def visibility(self,station):
        station=tuple(float(v) for v in station[:2])
        if station in self.station_masks:return self.station_masks[station]
        origin=np.asarray(station)
        distance=np.linalg.norm(self.points-origin,axis=1)
        selected=np.flatnonzero((distance>=.6)&(distance<=5.))
        result=np.zeros(len(self.points),bool)
        if len(selected):
            # 201 samples resolve a 5 m ray at 2.5 cm, finer than the map cell.
            ray=origin[None,None,:]+np.linspace(0,1,201)[None,:,None]*(
                self.points[selected,None,:]-origin[None,None,:])
            ix=np.rint((ray[:,:,0]-self.planner.ox)/self.planner.res).astype(int)
            iy=np.rint((ray[:,:,1]-self.planner.oy)/self.planner.res).astype(int)
            inside=(ix>=0)&(ix<self.planner.w)&(iy>=0)&(iy<self.planner.h)
            ix=np.clip(ix,0,self.planner.w-1);iy=np.clip(iy,0,self.planner.h-1)
            result[selected]=np.all(inside&(self.grid[iy,ix]==-1),axis=1)
        self.station_masks[station]=result
        return result

    def choose(self,start,visited):
        beginning=time.monotonic()
        seen=np.zeros(len(self.points),bool)
        for point in visited:seen|=self.visibility(point)
        options=[]
        for point in self.stations:
            if any(np.linalg.norm(np.asarray(point)-np.asarray(v))<.5 for v in visited):continue
            route=self.planner.path(start,point)
            if route is None:continue
            gain=float(np.count_nonzero(self.visibility(point)&~seen)*self.area)
            length=float(route['length_m'])
            options.append(dict(station=list(point),new_visible_area_m2=gain,path_m=length,
                                weighted_graph_cost=float(route['graph_cost']),
                                gain_per_effort=gain/(length+1.)))
        if not options:
            return None,dict(options=[],elapsed_wall_s=time.monotonic()-beginning,truth_used=False)
        if any(q['new_visible_area_m2']>0 for q in options):
            chosen=min(options,key=lambda q:(-q['gain_per_effort'],q['weighted_graph_cost'],*q['station']))
            rule='max_new_map_visible_area_per_path_plus_1m'
        else:
            chosen=min(options,key=lambda q:(q['weighted_graph_cost'],*q['station']))
            rule='no_predicted_gain_nearest_reachable'
        return [*chosen['station'],0.],dict(rule=rule,options=options,selected=chosen['station'],
                                            sampled_map_cells=len(self.points),
                                            elapsed_wall_s=time.monotonic()-beginning,
                                            truth_used=False,
                                            limitation='2D map upper bound; no elevated occlusion, marker orientation or readability')
