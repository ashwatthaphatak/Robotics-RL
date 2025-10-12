#!/usr/bin/env python3
import os, sys, math, csv, time
from pathlib import Path
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist, Point, Quaternion
from tf2_ros import Buffer, TransformListener
from ros_gz_interfaces.srv import SetEntityPose

ENTITY_NAME_CANDIDATES = [
    'turtlebot3_burger', 'turtlebot3_burger_0', 'burger', 'turtlebot3'
]


class WallFollowerAgent(Node):
    def __init__(self, mode='train', algorithm='q_learning',
                 table_path='~/.ros/wf_qtable.npy',
                 log_path='~/.ros/wf_train_log.csv',
                 reward_mode='sparse', goal_x=2.6, goal_y=3.1, goal_r=0.5,
                 episodes=999999, steps_per_episode=1500,
                 alpha=0.3, gamma=0.95, epsilon=0.30,
                 epsilon_decay=0.997, epsilon_min=0.05):
        super().__init__('wall_follower_agent')

        self.mode = mode
        self.algorithm = algorithm
        self.table_path = Path(os.path.expanduser(table_path))
        self.log_path = Path(os.path.expanduser(log_path))
        self.reward_mode = reward_mode
        self.goal = (goal_x, goal_y, goal_r)
        self.episodes = episodes
        self.steps_per_episode = steps_per_episode
        self.alpha, self.gamma = alpha, gamma
        self.epsilon, self.epsilon_decay, self.epsilon_min = epsilon, epsilon_decay, epsilon_min

        self.collision_thresh = 0.15
        self.step_penalty = -0.05
        self.max_linear = 0.28

        self.setup_state_action_space()
        self.q_table = {s: [0.0, 0.0, 0.0, 0.0] for s in self.all_states}

        if self.mode == 'test':
            if not self.try_load_qtable():
                self.get_logger().error(f"[TEST] No learned Q-table at {self.table_path}")
                rclpy.shutdown()
                return
        else:
            self.try_load_qtable()

        # ROS
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_cb, qos_profile_sensor_data)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.reset_cli = self.create_client(SetEntityPose, '/world/default/set_pose')
        self.reset_cli.wait_for_service(timeout_sec=3.0)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.training = (self.mode == 'train')
        self.ep = 0
        self.step = 0
        self.ep_return = 0.0
        self.prev_state = None
        self.prev_action = None
        self.first_random = True
        self.last_cmd = Twist()

        # keepalive to prevent Gazebo sleep
        self.keepalive_timer = self.create_timer(0.10, self._keepalive_tick)

        self.scenarios = [
            {'x': (-2.5, -1.5), 'y': (-3.5, -2.5), 'yaw': (-math.pi, math.pi)},
            {'x': (-1.0,  0.0), 'y': (-3.5, -2.5), 'yaw': (-math.pi, math.pi)},
            {'x': ( 0.5,  1.5), 'y': (-2.0, -1.0), 'yaw': (-math.pi, math.pi)},
            {'x': ( 1.0,  2.0), 'y': ( 0.0,  1.0), 'yaw': (-math.pi, math.pi)},
            {'x': (-2.0, -1.0), 'y': ( 1.5,  2.5), 'yaw': (-math.pi, math.pi)},
        ]

        self.entity_name = self.resolve_entity_name()
        if self.training:
            self.start_episode()
        self.get_logger().info(f"Agent ready | mode={self.mode} | algo={self.algorithm} | reward={self.reward_mode}")

    # ---------- Setup ----------
    def setup_state_action_space(self):
        self.L_STATES = ['CLOSE', 'FAR']
        self.F_STATES = ['TOO_CLOSE', 'CLOSE', 'MEDIUM', 'FAR']
        self.RF_STATES = ['CLOSE', 'FAR']
        self.R_STATES = ['TOO_CLOSE', 'CLOSE', 'MEDIUM', 'FAR', 'TOO_FAR']
        self.bounds = {
            'TOO_CLOSE': (0.0, 0.3), 'CLOSE': (0.3, 0.6),
            'MEDIUM': (0.6, 1.2), 'FAR': (1.2, 2.5), 'TOO_FAR': (2.5, float('inf'))
        }
        self.L_SECTOR=(80,100); self.F_SECTOR=(355,5)
        self.RF_SECTOR=(310,320); self.R_SECTOR=(260,280)
        self.actions={
            'FORWARD':(0.25,0.0),
            'LEFT':(0.22,0.785),
            'RIGHT':(0.22,-0.785),
            'STOP':(0.0,0.0)
        }
        self.action_names=list(self.actions.keys())
        self.all_states=[(l,f,rf,r)
            for l in self.L_STATES for f in self.F_STATES
            for rf in self.RF_STATES for r in self.R_STATES]

    def try_load_qtable(self):
        p=self.table_path
        if p.exists():
            try:
                data=np.load(p,allow_pickle=True).item()
                self.q_table.update(data)
                self.get_logger().info(f"Loaded Q-table from {p}")
                return True
            except Exception as e:
                self.get_logger().warn(f"Failed to load table: {e}")
        return False

    # ---------- Gazebo helpers ----------
    def resolve_entity_name(self):
        if not self.reset_cli.service_is_ready():
            self.get_logger().warn("SetEntityPose service not ready.")
            return None
        for cand in ENTITY_NAME_CANDIDATES:
            if self.try_set_pose_probe(cand):
                self.get_logger().info(f"Using Gazebo entity: {cand}")
                return cand
        self.get_logger().warn("Could not resolve Gazebo entity.")
        return None

    def try_set_pose_probe(self, name):
        try:
            req=SetEntityPose.Request()
            req.entity.name=name
            req.pose.position=Point(x=0.0,y=0.0,z=0.02)
            req.pose.orientation=Quaternion(x=0.0,y=0.0,z=0.0,w=1.0)
            fut=self.reset_cli.call_async(req)
            rclpy.spin_until_future_complete(self,fut,timeout_sec=0.1)
            return fut.done() and fut.result() is not None
        except Exception:
            return False

    def teleport(self,x,y,yaw):
        if not self.entity_name:
            self.get_logger().warn("No valid entity for teleport — skipping reset.")
            return
        try:
            req=SetEntityPose.Request()
            req.entity.name=self.entity_name
            req.pose.position=Point(x=float(x),y=float(y),z=0.02)
            qz,qw=math.sin(yaw/2.0),math.cos(yaw/2.0)
            req.pose.orientation=Quaternion(x=0.0,y=0.0,z=qz,w=qw)
            self.reset_cli.call_async(req)
        except Exception:
            pass

    # ---------- Core RL ----------
    def get_sector_avg(self,ranges,a,b):
        n=len(ranges)
        if a>b:
            vals=list(ranges[int(a*n/360):])+list(ranges[:int(b*n/360)])
        else:
            vals=list(ranges[int(a*n/360):int(b*n/360)])
        vals=[v for v in vals if v!=float('inf') and not math.isnan(v)]
        return sum(vals)/len(vals) if vals else float('inf')

    def dist_to_state(self,d,t):
        B=self.bounds
        if t=='front':avail=['TOO_CLOSE','CLOSE','MEDIUM','FAR']
        elif t=='right':avail=['TOO_CLOSE','CLOSE','MEDIUM','FAR','TOO_FAR']
        else:avail=['CLOSE','FAR']
        for s in avail:
            lo,hi=B[s]
            if lo<=d<hi:return s
        return 'TOO_FAR' if t=='right' else 'FAR'

    def determine_state(self,scan):
        L=self.get_sector_avg(scan,*self.L_SECTOR)
        F=self.get_sector_avg(scan,*self.F_SECTOR)
        RF=self.get_sector_avg(scan,*self.RF_SECTOR)
        R=self.get_sector_avg(scan,*self.R_SECTOR)
        s=(self.dist_to_state(L,'left'),
           self.dist_to_state(F,'front'),
           self.dist_to_state(RF,'right_front'),
           self.dist_to_state(R,'right'))
        return s,(L,F,RF,R)

    def execute(self,name):
        lin,ang=self.actions[name]
        tw=Twist(); tw.linear.x=float(max(min(lin,self.max_linear),-self.max_linear)); tw.angular.z=float(ang)
        self.cmd_pub.publish(tw)
        self.last_cmd=tw

    def _keepalive_tick(self):
        self.cmd_pub.publish(self.last_cmd)

    def at_goal(self):
        if self.reward_mode!='sparse':return False
        try:
            tf=self.tf_buffer.lookup_transform('map','base_footprint',rclpy.time.Time())
            x=tf.transform.translation.x; y=tf.transform.translation.y
            gx,gy,gr=self.goal; dx,dy=x-gx,y-gy
            return dx*dx+dy*dy<=gr*gr
        except Exception:return False

    def eps_greedy(self,s):
        if np.random.rand()<self.epsilon:
            return np.random.randint(len(self.action_names))
        return int(np.argmax(self.q_table[s]))

    def reward(self,dists,s,a):
        front=dists[1] if not math.isinf(dists[1]) else 10.0
        if front<self.collision_thresh or s[1]=='TOO_CLOSE':return -100.0
        if self.reward_mode=='sparse':return self.step_penalty
        r_state=s[3]
        band={'MEDIUM':1.0,'CLOSE':0.4,'FAR':0.4,'TOO_CLOSE':-0.8,'TOO_FAR':-0.8}[r_state]
        progress=0.6 if self.action_names[a]=='FORWARD' else 0.1
        return band+progress-0.15

    def td_update(self,s,a,r,s2,a2=None):
        q=self.q_table[s]
        target=r+self.gamma*(self.q_table[s2][a2] if self.algorithm=='sarsa' and a2 is not None else max(self.q_table[s2]))
        q[a]+=self.alpha*(target-q[a])

    # ---------- Episodes ----------
    def start_episode(self):
        self.ep+=1; self.step=0; self.ep_return=0.0; self.prev_state=None; self.prev_action=None; self.first_random=True
        i=(self.ep-1)%len(self.scenarios); box=self.scenarios[i]
        x=np.random.uniform(*box['x']); y=np.random.uniform(*box['y']); yaw=np.random.uniform(*box['yaw'])
        self.teleport(x,y,yaw); time.sleep(1.0); self.execute('FORWARD')
        self.epsilon=max(self.epsilon*self.epsilon_decay,self.epsilon_min)
        self.get_logger().info(f"[TRAIN] Episode {self.ep} start | ε={self.epsilon:.3f}")

    def end_episode(self,reason='timeout'):
        self.get_logger().info(f"[TRAIN] Ep{self.ep} end | steps={self.step} return={self.ep_return:.1f} | {reason}")
        try:
            np.save(self.table_path,self.q_table)
            new=not self.log_path.exists()
            with open(self.log_path,'a',newline='') as f:
                w=csv.writer(f)
                if new:w.writerow(['episode','return','steps','reason','algorithm'])
                w.writerow([self.ep,round(self.ep_return,2),self.step,reason,self.algorithm])
        except Exception:pass
        if self.ep<self.episodes:self.start_episode()
        else:self.execute('STOP')

    # ---------- Main callback ----------
    def scan_cb(self,msg):
        s,d=self.determine_state(msg.ranges)
        if not self.training:
            a=int(np.argmax(self.q_table[s])); self.execute(self.action_names[a]); return
        if self.reward_mode=='sparse' and self.at_goal():
            self.ep_return+=1000.0; self.end_episode('success'); return
        if self.first_random:
            a=np.random.randint(len(self.action_names)); self.first_random=False
        else:
            a=self.eps_greedy(s)
        self.execute(self.action_names[a])
        r=self.reward(d,s,a); self.ep_return+=r
        if self.prev_state is not None and self.prev_action is not None:
            if self.algorithm=='sarsa':
                a2=self.eps_greedy(s); self.td_update(self.prev_state,self.prev_action,r,s,a2)
            else:self.td_update(self.prev_state,self.prev_action,r,s)
        self.prev_state,self.prev_action=s,a; self.step+=1
        if r<=-100.0:self.end_episode('collision');return
        if self.step>=self.steps_per_episode:self.end_episode('timeout');return
        if self.step%500==0:
            np.save(self.table_path,self.q_table)
            self.get_logger().info(f"[TRAIN] autosave step {self.step} ret={self.ep_return:.1f}")
