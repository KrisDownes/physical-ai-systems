"""Unmodified commissioned controller with the common acquisition budget guard."""
from pathlib import Path
import json
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'research'))
from controller import Mission
from geometry_msgs.msg import Twist
import rclpy

class Survey(Mission):
    def __init__(self,out):
        super().__init__(out);self.distance=0.;self.previous=None;self.finished=False
    def tick(self):
        super().tick()
        if self.finished:return
        if self.odom:
            xy=self.odom[-1][1][:2]
            if self.previous is not None:self.distance+=float(np.linalg.norm(xy-self.previous))
            self.previous=xy.copy()
        if self.done or self.sim_time>=240 or self.distance>=20:
            reason='commissioned_route_completed' if self.done else 'total_acquisition_budget'
            self.finished=True;self.done=True;self.cmd.publish(Twist())
            (self.out/'acquisition_done.json').write_text(json.dumps(dict(reason=reason,time=self.sim_time,wheel_distance_m=self.distance),indent=2)+'\n')

if __name__=='__main__':
    rclpy.init();node=Survey(Path(sys.argv[1]))
    try:rclpy.spin(node)
    except KeyboardInterrupt:pass
    finally:node.close();node.destroy_node();rclpy.try_shutdown()
