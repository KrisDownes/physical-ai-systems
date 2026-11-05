"""Conservative routing with a bounded, hard-clearance-checked start connector."""
import numpy as np
from planning import GridPlanner

class BufferedPlanner(GridPlanner):
    def __init__(self,path):
        super().__init__(path,.52);self.hard=GridPlanner(path,.45)

    def connector(self,start):
        start=np.asarray(start[:2]);index=super().index(start)
        if index is not None:return start
        if self.hard.index(start) is None:return None
        choices=[]
        for y,x in zip(*np.nonzero(self.free)):
            point=self.point(y*self.w+x);distance=float(np.linalg.norm(point-start))
            if distance<=.3:choices.append((distance,y*self.w+x))
        for _,i in sorted(choices):
            point=self.point(i)
            if self.hard.segment_free(start,point):return point
        return None

    def start_index(self,start):
        point=self.connector(start)
        return None if point is None else super().index(point)

    def paths_from(self,start):
        point=self.connector(start)
        return (None,None) if point is None else super().paths_from(point)

    def path(self,start,goal):
        point=self.connector(start)
        if point is None:return None
        path=super().path(point,goal)
        if path is None:return None
        distance=float(np.linalg.norm(point-np.asarray(start[:2])))
        if distance>1e-8:
            path['points']=[list(start[:2]),*path['points']];path['length_m']+=distance;path['graph_cost']+=distance
        path['start_connector_m']=distance
        return path
