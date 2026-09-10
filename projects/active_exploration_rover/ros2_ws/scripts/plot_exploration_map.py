"""Evaluator-only final SLAM map and raw frontier figure from a recorded bag."""
import argparse
import json
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/rover-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import rosbag2_py
from nav_msgs.msg import OccupancyGrid
from rclpy.serialization import deserialize_message
from rover_exploration.mission_evaluator import find_frontier_cells, cluster_frontier_cells

parser=argparse.ArgumentParser()
parser.add_argument('episode',type=Path)
args=parser.parse_args()
reader=rosbag2_py.SequentialReader()
reader.open(rosbag2_py.StorageOptions(uri=str(args.episode/'bag'),storage_id=''),
            rosbag2_py.ConverterOptions('',''))
reader.set_filter(rosbag2_py.StorageFilter(topics=['/map']))
last=None
while reader.has_next():
    _,data,_=reader.read_next();last=deserialize_message(data,OccupancyGrid)
if last is None:raise SystemExit('No map in recording')
w,h=last.info.width,last.info.height
cells=find_frontier_cells(data=last.data,width=w,height=h)
clusters=cluster_frontier_cells(cells,min_cluster_size=1)
fig,ax=plt.subplots(figsize=(8,7))
grid=np.asarray(last.data).reshape(h,w)
cmap=plt.get_cmap('gray_r').copy();cmap.set_bad('#9ba7b3')
ax.imshow(np.ma.masked_where(grid<0,grid),origin='lower',cmap=cmap,vmin=0,vmax=100,interpolation='nearest')
if cells:
    ax.scatter([p[1] for p in cells],[p[0] for p in cells],s=5,c='tomato',label='Raw frontier cells')
    ax.legend()
ax.set(title='Evaluator-only final SLAM map; raw frontiers in red',xlabel='Grid column',ylabel='Grid row')
fig.tight_layout();fig.savefig(args.episode/'final-map.png',dpi=150)
(args.episode/'frontier-diagnostic.json').write_text(json.dumps({
    'component_sizes':sorted([len(c) for c in clusters],reverse=True),
    'width':w,'height':h,'resolution_m':last.info.resolution,
    'frame_id':last.header.frame_id,
    'note':'No reachable-area filtering. This data is never sent to the driver.'},indent=2))
