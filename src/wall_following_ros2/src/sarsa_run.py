#!/usr/bin/env python3
import os, math
from pathlib import Path
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist

SAVE_PATH = Path(os.path.expanduser('~/.ros/wf_sarsa.npy'))

class SarsaRunner(Node):
    def __init__(self):
        super().__init__('sarsa_runner')
        self.max_linear = 0.28
        self.setup_state_action_space()
        self.q_table={s:[0.0,0.0,0.0,0.0] for s in self.all_states}
        if not self.try_load_qtable():
            self.get_logger().error(f"No learned SARSA table at {SAVE_PATH}. Exiting.")
            rclpy.shutdown(); return

        self.scan_sub=self.create_subscription(LaserScan,'/scan',self.scan_cb,qos_profile_sensor_data)
        self.cmd_pub=self.create_publisher(Twist,'/cmd_vel',10)
        self.get_logger().info("SARSA runner ready (greedy).")

    def setup_state_action_space(self):
        self.L_STATES=['CLOSE','FAR']; self.F_STATES=['TOO_CLOSE','CLOSE','MEDIUM','FAR']
        self.RF_STATES=['CLOSE','FAR']; self.R_STATES=['TOO_CLOSE','CLOSE','MEDIUM','FAR','TOO_FAR']
        self.bounds={'TOO_CLOSE':(0.0,0.3),'CLOSE':(0.3,0.6),'MEDIUM':(0.6,1.2),'FAR':(1.2,2.5),'TOO_FAR':(2.5,float('inf'))}
        self.L_SECTOR=(80,100); self.F_SECTOR=(355,5); self.RF_SECTOR=(310,320); self.R_SECTOR=(260,280)
        self.actions={'FORWARD':(0.25,0.0),'LEFT':(0.22,0.785),'RIGHT':(0.22,-0.785),'STOP':(0.0,0.0)}
        self.action_names=list(self.actions.keys())
        self.all_states=[(l,f,rf,r) for l in self.L_STATES for f in self.F_STATES for rf in self.RF_STATES for r in self.R_STATES]

    def try_load_qtable(self):
        if SAVE_PATH.exists():
            try:
                data=np.load(SAVE_PATH,allow_pickle=True).item()
                if isinstance(data,dict): self.q_table.update(data); self.get_logger().info(f"Loaded SARSA table from {SAVE_PATH}"); return True
            except Exception as e: self.get_logger().warn(f"Load failed: {e}")
        return False

    def get_sector_avg(self,ranges,a,b):
        n=len(ranges); vals=(list(ranges[int(a*n/360):])+list(ranges[:int(b*n/360)])) if a>b else list(ranges[int(a*n/360):int(b*n/360)])
        vals=[v for v in vals if v!=float('inf') and not math.isnan(v)]
        return sum(vals)/len(vals) if vals else float('inf')

    def dist_to_state(self,d,t):
        B=self.bounds
        avail=['TOO_CLOSE','CLOSE','MEDIUM','FAR'] if t=='front' else (['TOO_CLOSE','CLOSE','MEDIUM','FAR','TOO_FAR'] if t=='right' else ['CLOSE','FAR'])
        for s in avail:
            lo,hi=B[s]
            if lo<=d<hi:return s
        return 'TOO_FAR' if t=='right' else 'FAR'

    def determine_state(self,scan):
        L=self.get_sector_avg(scan,*self.L_SECTOR); F=self.get_sector_avg(scan,*self.F_SECTOR)
        RF=self.get_sector_avg(scan,*self.RF_SECTOR); R=self.get_sector_avg(scan,*self.R_SECTOR)
        return (self.dist_to_state(L,'left'), self.dist_to_state(F,'front'),
                self.dist_to_state(RF,'right_front'), self.dist_to_state(R,'right'))

    def execute(self,name):
        lin,ang=self.actions[name]
        tw=Twist(); tw.linear.x=float(max(min(lin,self.max_linear),-self.max_linear)); tw.angular.z=float(ang)
        self.cmd_pub.publish(tw)

    def scan_cb(self,msg: LaserScan):
        s=self.determine_state(msg.ranges)
        a=int(np.argmax(self.q_table[s]))
        self.execute(self.action_names[a])

def main():
    rclpy.init()
    node=SarsaRunner()
    if rclpy.ok():
        try: rclpy.spin(node)
        except KeyboardInterrupt: pass
        finally: node.destroy_node(); rclpy.shutdown()

if __name__=='__main__':
    main()
