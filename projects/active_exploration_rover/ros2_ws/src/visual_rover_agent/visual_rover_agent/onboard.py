"""Onboard sensor presentation only: no destination selection or ground truth."""
import base64
from io import BytesIO
import math
import time
from PIL import Image, ImageDraw
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import LaserScan
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from tf2_ros import Buffer, TransformListener, TransformException
from rover_exploration.frontier_selection import (
    find_frontier_cells, cluster_frontier_cells, representative_frontier_cell)


def stamp(msg):
    return msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9


def yaw(q):
    return math.atan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z))


def clearances(scan):
    from visual_rover_agent.node import sector_minimum
    # The executor's validated sector helper treats +inf as no return.
    result = {}
    for name, angle in [('front', 0), ('left', math.pi/2),
                        ('rear', math.pi), ('right', -math.pi/2)]:
        value = sector_minimum(scan, angle, math.radians(30))
        result[name] = {'minimum_m': value if value is not None and math.isfinite(value) else None,
                        'state': 'missing_or_invalid' if value is None else
                                 'no_return_within_range' if math.isinf(value) else 'valid'}
    return result


def map_view(grid, pose):
    w, h, r = grid.info.width, grid.info.height, grid.info.resolution
    if w <= 0 or h <= 0 or r <= 0 or len(grid.data) != w*h:
        raise ValueError('invalid_map_layout')
    origin = grid.info.origin
    angle = yaw(origin.orientation)
    def xy(row, col):
        x, y = (col+.5)*r, (row+.5)*r
        return (origin.position.x+math.cos(angle)*x-math.sin(angle)*y,
                origin.position.y+math.sin(angle)*x+math.cos(angle)*y)
    clusters = cluster_frontier_cells(find_frontier_cells(grid.data, w, h), 5)
    candidates = []
    # Every qualifying component, deterministic grid order, no utility ranking.
    for i, cluster in enumerate(clusters):
        row, col = representative_frontier_cell(cluster)
        x, y = xy(row, col)
        candidates.append(dict(id=f'F{i+1}', x_m=round(x,3), y_m=round(y,3), cells=len(cluster)))
    im = Image.new('RGB', (w,h))
    im.putdata([(145,145,145) if v < 0 else (250,250,250) if v == 0 else (25,25,25)
                for v in grid.data])
    im = im.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    scale = min(640/w, 640/h)
    im = im.resize((max(1,round(w*scale)),max(1,round(h*scale))), Image.Resampling.NEAREST)
    canvas = Image.new('RGB', (max(640,im.width), im.height+76), 'white')
    canvas.paste(im,(0,60)); draw=ImageDraw.Draw(canvas)
    draw.text((8,3),'Online SLAM: white free / black occupied / gray unknown',fill='black')
    draw.text((8,18),f'{grid.header.frame_id}: meters; grid +y up, +x right; origin yaw {math.degrees(angle):.1f} deg',fill='black')
    draw.text((8,33),f'Origin ({origin.position.x:.2f}, {origin.position.y:.2f}); {r:.3f} m/cell; blue rover, red frontiers',fill='black')
    def pixel(x,y):
        dx,dy=x-origin.position.x,y-origin.position.y
        return ((math.cos(angle)*dx+math.sin(angle)*dy)/r*scale,
                60+im.height-(-math.sin(angle)*dx+math.cos(angle)*dy)/r*scale)
    for c in candidates:
        x,y=pixel(c['x_m'],c['y_m']);draw.ellipse((x-3,y-3,x+3,y+3),fill='red');draw.text((x+4,y),c['id'],fill='red')
    if pose.get('state') == 'valid':
        x,y=pixel(pose['x_m'],pose['y_m']); a=pose['yaw_rad']-angle
        draw.ellipse((x-5,y-5,x+5,y+5),fill='blue')
        draw.line((x,y,x+22*math.cos(a),y-22*math.sin(a)),fill='blue',width=4)
    data=BytesIO();canvas.save(data,format='PNG')
    return candidates, base64.b64encode(data.getvalue()).decode()


class Onboard:
    def __init__(self, node):
        self.node=node; self.grid=None; self.scan=None
        self.tf=Buffer(); self.listener=TransformListener(self.tf,node)
        node.create_subscription(OccupancyGrid,'/map',lambda m:self.set_sensor('grid',m),10)
        node.create_subscription(LaserScan,'/scan',lambda m:self.set_sensor('scan',m),qos_profile_sensor_data)

    def set_sensor(self,name,msg):
        setattr(self,name,(msg,time.monotonic()))

    def snapshot(self, camera_sim):
        def state(item,limit):
            if item is None: return {'state':'missing'}
            msg,received=item; age=time.monotonic()-received; sim_age=camera_sim-stamp(msg)
            return dict(state='stale' if age>limit or sim_age>limit else 'valid',
                        captured_sim_time_s=stamp(msg),received_monotonic_s=received,
                        age_wall_s=age,age_relative_to_camera_sim_s=sim_age,stale_after_s=limit,
                        frame=msg.header.frame_id)
        scan,grid=self.scan,self.grid
        lidar=state(scan,.5)
        if scan:
            lidar.update(sector_half_width_deg=30,range_max_m=scan[0].range_max if math.isfinite(scan[0].range_max) else None,
                         convention='scan frame: front +x, left +y; CCW positive',sectors=clearances(scan[0]))
        mapping=state(grid,5.0); pose={'state':'missing'}; image=None
        if grid:
            try:
                tf=self.tf.lookup_transform(grid[0].header.frame_id,'base_footprint',Time())
                age=camera_sim-stamp(tf)
                pose=dict(state='valid' if abs(age)<=.5 else 'stale',captured_sim_time_s=stamp(tf),
                          age_relative_to_camera_sim_s=age,stale_after_sim_s=.5,
                          x_m=tf.transform.translation.x,y_m=tf.transform.translation.y,
                          yaw_rad=yaw(tf.transform.rotation),frame=grid[0].header.frame_id)
            except TransformException:
                pass
            try:
                candidates,image=map_view(grid[0],pose)
                mapping.update(resolution_m=grid[0].info.resolution,frontiers=candidates,
                    frontier_definition='free cell cardinally adjacent to unknown; 8-connected components >=5 cells; representative cell, no reachability filtering or ranking; IDs local to this observation')
            except ValueError as error:
                mapping.update(state='invalid',error=str(error))
        return dict(lidar=lidar,slam_map=mapping,estimated_pose=pose,
                    coordinates='meters in online map frame; yaw radians from +x, positive counterclockwise; no simulator coordinates'),image
