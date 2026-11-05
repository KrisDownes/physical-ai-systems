"""Separate outcome observer: simulator truth/contacts never feed controller."""
import json
from pathlib import Path
import sys
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from nav_msgs.msg import Odometry
from ros_gz_interfaces.msg import Contacts
from std_msgs.msg import String


class Judge(Node):
    def __init__(self,path):
        super().__init__('site_outcome_observer');self.file=(path/'judge.jsonl').open('w')
        self.create_subscription(Odometry,'/judge/truth',self.truth,qos_profile_sensor_data)
        self.create_subscription(Contacts,'/site/contacts',self.contact,qos_profile_sensor_data)
        self.create_subscription(Contacts,'/judge/wheel_contacts',self.wheel,qos_profile_sensor_data)
        self.create_subscription(String,'/site/events',self.event,10)
    def write(self,row):self.file.write(json.dumps(row)+'\n');self.file.flush()
    def truth(self,m):
        p=m.pose.pose;q=p.orientation
        self.write(dict(type='truth',time=m.header.stamp.sec+m.header.stamp.nanosec*1e-9,
                        x=p.position.x,y=p.position.y,q=[q.x,q.y,q.z,q.w]))
    def contact(self,m):
        self.write(dict(type='contacts',time=m.header.stamp.sec+m.header.stamp.nanosec*1e-9,
                        count=len(m.contacts),pairs=[[c.collision1.name,c.collision2.name] for c in m.contacts]))
    def wheel(self,m):
        self.write(dict(type='wheel_contacts',time=m.header.stamp.sec+m.header.stamp.nanosec*1e-9,count=len(m.contacts)))
    def event(self,m):self.write(dict(type='event',event=json.loads(m.data)))


def main():
    rclpy.init();node=Judge(Path(sys.argv[1]))
    try:rclpy.spin(node)
    except KeyboardInterrupt:pass
    finally:node.file.close();node.destroy_node();rclpy.try_shutdown()


if __name__=='__main__':main()
