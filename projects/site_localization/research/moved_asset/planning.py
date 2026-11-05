"""Occupancy adapter around SciPy Dijkstra, not a new search/planning algorithm."""
import math
from pathlib import Path
import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra


class GridPlanner:
    def __init__(self,path,inflation=.45):
        with Path(path).open() as f:
            w,h,self.res,self.ox,self.oy=np.fromstring(f.readline(),sep=' ')
            self.w,self.h=int(w),int(h);cells=np.loadtxt(f,dtype=int).reshape(self.h,self.w)
        distance=distance_transform_edt(np.pad(cells==-1,1,constant_values=False))[1:-1,1:-1]*self.res
        self.free=distance>=inflation+self.res/math.sqrt(2)
        self.distance=distance
        # Standard 8-connected lattice with explicit no-corner-cut adjacency.
        src=[];dst=[];cost=[]
        for dy,dx in [(0,1),(1,0),(1,1),(1,-1)]:
            for y,x in zip(*np.nonzero(self.free)):
                v,u=y+dy,x+dx
                if not(0<=v<self.h and 0<=u<self.w and self.free[v,u]):continue
                if dx and dy and not(self.free[y,u] and self.free[v,x]):continue
                a,b=y*self.w+x,v*self.w+u
                weight=self.res*math.hypot(dx,dy)*(1+.1/min(distance[y,x],distance[v,u]))
                src.extend([a,b]);dst.extend([b,a]);cost.extend([weight,weight])
        self.graph=csr_matrix((cost,(src,dst)),shape=(self.w*self.h,self.w*self.h))

    def index(self,point):
        x,y=np.rint((np.asarray(point[:2])-[self.ox,self.oy])/self.res).astype(int)
        if not(0<=x<self.w and 0<=y<self.h and self.free[y,x]):return None
        return int(y*self.w+x)

    def point(self,index):return np.array([self.ox+(index%self.w)*self.res,self.oy+(index//self.w)*self.res])

    def segment_free(self,a,b):
        return all(self.index(p) is not None for p in np.linspace(a,b,max(2,int(np.ceil(np.linalg.norm(np.asarray(b)-a)/(self.res/4)))+1)))

    def paths_from(self,start):
        index=self.index(start)
        if index is None:return None,None
        return dijkstra(self.graph,directed=False,indices=index,return_predecessors=True)

    def path(self,start,goal):
        a,b=self.index(start),self.index(goal)
        if a is None or b is None:return None
        distance,previous=self.paths_from(start)
        if not np.isfinite(distance[b]):return None
        nodes=[b]
        while nodes[-1]!=a:
            parent=int(previous[nodes[-1]])
            if parent<0:return None
            nodes.append(parent)
        points=[np.asarray(start[:2]),*[self.point(i) for i in nodes[::-1][1:-1]],np.asarray(goal[:2])]
        # Conventional collision-checked line-of-sight simplification.
        reduced=[points[0]];i=0
        while i<len(points)-1:
            j=len(points)-1
            while j>i+1 and not self.segment_free(points[i],points[j]):j-=1
            if not self.segment_free(points[i],points[j]):return None
            reduced.append(points[j]);i=j
        return dict(points=[p.tolist() for p in reduced],graph_cost=float(distance[b]),
                    length_m=float(sum(np.linalg.norm(b-a) for a,b in zip(reduced,reduced[1:]))))

    def next_search(self,start,visited,coordinates):
        distance,_=self.paths_from(start)
        if distance is None:return None
        options=[]
        for x in coordinates:
            for y in coordinates:
                if any(np.linalg.norm(np.array([x,y])-p)<.5 for p in visited):continue
                index=self.index([x,y])
                if index is not None and np.isfinite(distance[index]):options.append((float(distance[index]),x,y))
        if not options:return None
        _,x,y=min(options)
        return [x,y,0.]
