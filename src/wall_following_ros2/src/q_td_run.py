#!/usr/bin/env python3
import os, sys, math, csv, time, argparse
from pathlib import Path
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist, Point, Quaternion
from tf2_ros import Buffer, TransformListener
from ros_gz_interfaces.srv import SetEntityPose

# Candidates for the Gazebo entity name (auto-detect at startup)
ENTITY_NAME_CANDIDATES = [
    'turtlebot3_burger', 'turtlebot3_burger_0', 'burger', 'turtlebot3'
]

# Where to persist learned Q-table and logs
SAVE_PATH = Path.home() / '.ros' / 'wf_qtable.npy'
LOG_PATH  = Path.home() / '.ros' / 'wf_train_log.csv'


class QLearningWallFollower(Node):
    def __init__(self,
                 mode='train',
                 algorithm='q_learning',          # q_learning | sarsa
                 reward_mode='sparse',            # sparse | shaped
                 goal_x=2.6, goal_y=3.1, goal_r=0.5,
                 episodes=999999, steps_per_episode=1500,
                 alpha=0.3, gamma=0.95,
                 epsilon=0.30, epsilon_decay=0.997, epsilon_min=0.05):
        super().__init__('q_learning_wall_follower')

        # ----- Config -----
        self.mode, self.algorithm = mode, algorithm
        self.reward_mode = reward_mode
        self.goal = (float(goal_x), float(goal_y), float(goal_r))
        self.episodes = int(episodes)
        self.steps_per_episode = int(steps_per_episode)
        self.alpha, self.gamma = float(alpha), float(gamma)
        self.epsilon, self.epsilon_decay, self.epsilon_min = float(epsilon), float(epsilon_decay), float(epsilon_min)

        # Motion & reward knobs
        self.collision_thresh = 0.15   # front distance considered collision
        self.step_penalty = -0.05      # per-step cost (sparse)
        self.max_linear = 0.28         # cap linear velocity

        # ----- State/action space -----
        self.setup_state_action_space()

        # ----- Q-table -----
        self.q_table = {s: [0.0, 0.0, 0.0, 0.0] for s in self.all_states}

        if self.mode == 'test':
            if not self.try_load_qtable():
                self.get_logger().error(f"[TEST] No learned Q-table at {SAVE_PATH}. Aborting.")
                rclpy.shutdown()
                return
            else:
                self.get_logger().info(f"[TEST] Loaded learned Q-table from {SAVE_PATH}")
        else:
            # In TRAIN, try to warm-start if a table exists
            self.try_load_qtable()

        # ----- ROS I/O -----
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_cb, 10)
        self.cmd_pub  = self.create_publisher(Twist, '/cmd_vel', 10)

        # Teleport service (episode reset). If not available, training still runs.
        self.reset_cli = self.create_client(SetEntityPose, '/world/default/set_pose')
        self.reset_cli.wait_for_service(timeout_sec=3.0)

        # TF Buffer for goal check (sparse reward)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ----- Bookkeeping -----
        self.training = (self.mode == 'train')
        self.ep = 0
        self.step = 0
        self.ep_return = 0.0
        self.prev_state = None
        self.prev_action = None

        # Movement “always moves” fix: force first action to random
        self.first_random = True

        # Five scenario regions (adjust if you have specific starts)
        self.scenarios = [
            {'x': (-2.5, -1.5), 'y': (-3.5, -2.5), 'yaw': (-math.pi, math.pi)},
            {'x': (-1.0,  0.0), 'y': (-3.5, -2.5), 'yaw': (-math.pi, math.pi)},
            {'x': ( 0.5,  1.5), 'y': (-2.0, -1.0), 'yaw': (-math.pi, math.pi)},
            {'x': ( 1.0,  2.0), 'y': ( 0.0,  1.0), 'yaw': (-math.pi, math.pi)},
            {'x': (-2.0, -1.0), 'y': ( 1.5,  2.5), 'yaw': (-math.pi, math.pi)},
        ]

        # Resolve the Gazebo entity name once
        self.entity_name = self.resolve_entity_name()

        # Start training episodes
        if self.training:
            self.start_episode()

        self.get_logger().info(f"RL Node ready | mode={self.mode} | algo={self.algorithm} | reward={self.reward_mode}")

    # ====================== Setup helpers ======================
    def setup_state_action_space(self):
        self.L_STATES = ['CLOSE', 'FAR']
        self.F_STATES = ['TOO_CLOSE', 'CLOSE', 'MEDIUM', 'FAR']
        self.RF_STATES = ['CLOSE', 'FAR']
        self.R_STATES = ['TOO_CLOSE', 'CLOSE', 'MEDIUM', 'FAR', 'TOO_FAR']

        self.bounds = {
            'TOO_CLOSE': (0.0, 0.3),
            'CLOSE':     (0.3, 0.6),
            'MEDIUM':    (0.6, 1.2),
            'FAR':       (1.2, 2.5),
            'TOO_FAR':   (2.5, float('inf')),
        }

        # LiDAR sectors (deg)
        self.L_SECTOR  = (80, 100)
        self.F_SECTOR  = (355, 5)
        self.RF_SECTOR = (310, 320)
        self.R_SECTOR  = (260, 280)

        # ACTIONS — ensure FORWARD moves (movement fix #1)
        self.actions = {
            'FORWARD': (0.25, 0.0),     # > 0.0 linear speed
            'LEFT':    (0.22, 0.785),
            'RIGHT':   (0.22, -0.785),
            'STOP':    (0.0, 0.0)
        }
        self.action_names = list(self.actions.keys())

        # Precompute all states
        self.all_states = []
        for l in self.L_STATES:
            for f in self.F_STATES:
                for rf in self.RF_STATES:
                    for r in self.R_STATES:
                        self.all_states.append((l, f, rf, r))

    def try_load_qtable(self):
        if SAVE_PATH.exists():
            try:
                data = np.load(SAVE_PATH, allow_pickle=True).item()
                if isinstance(data, dict):
                    self.q_table.update(data)
                    self.get_logger().info(f"Loaded Q-table from {SAVE_PATH}")
                    return True
            except Exception as e:
                self.get_logger().warn(f"Failed to load Q-table: {e}")
        return False

    # ====================== Gazebo teleport ======================
    def resolve_entity_name(self):
        if not self.reset_cli.service_is_ready():
            self.get_logger().warn("SetEntityPose service not ready; teleports may be skipped.")
            return None
        for cand in ENTITY_NAME_CANDIDATES:
            if self.try_set_pose_probe(cand):
                self.get_logger().info(f"Using Gazebo entity: {cand}")
                return cand
        self.get_logger().warn("Could not resolve Gazebo entity; teleports disabled.")
        return None

    def try_set_pose_probe(self, name: str) -> bool:
        try:
            req = SetEntityPose.Request()
            req.entity.name = name
            req.pose.position = Point(x=0.0, y=0.0, z=0.02)
            req.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
            fut = self.reset_cli.call_async(req)
            rclpy.spin_until_future_complete(self, fut, timeout_sec=0.15)
            return fut.done() and (fut.result() is not None)
        except Exception:
            return False

    def teleport(self, x: float, y: float, yaw: float):
        # Movement fix #3: skip teleport cleanly if no entity resolved
        if not self.entity_name:
            self.get_logger().warn("No valid entity for teleport — skipping episode reset.")
            return
        try:
            req = SetEntityPose.Request()
            req.entity.name = self.entity_name
            req.pose.position = Point(x=float(x), y=float(y), z=0.02)
            qz, qw = math.sin(yaw / 2.0), math.cos(yaw / 2.0)
            req.pose.orientation = Quaternion(x=0.0, y=0.0, z=qz, w=qw)
            self.reset_cli.call_async(req)
        except Exception:
            pass

    # ====================== State / Action utils ======================
    def get_sector_avg(self, ranges, a_deg, b_deg):
        n = len(ranges)
        if a_deg > b_deg:
            a = int(a_deg * n / 360.0)
            vals = list(ranges[a:]) + list(ranges[:int(b_deg * n / 360.0)])
        else:
            a = int(a_deg * n / 360.0)
            b = int(b_deg * n / 360.0)
            vals = list(ranges[a:b])
        vals = [v for v in vals if v != float('inf') and not math.isnan(v)]
        return sum(vals) / len(vals) if vals else float('inf')

    def dist_to_state(self, d, t):
        B = self.bounds
        if t == 'front':
            avail = ['TOO_CLOSE', 'CLOSE', 'MEDIUM', 'FAR']
        elif t == 'right':
            avail = ['TOO_CLOSE', 'CLOSE', 'MEDIUM', 'FAR', 'TOO_FAR']
        else:
            avail = ['CLOSE', 'FAR']
        for s in avail:
            lo, hi = B[s]
            if lo <= d < hi:
                return s
        return 'TOO_FAR' if t == 'right' else 'FAR'

    def determine_state(self, scan):
        L  = self.get_sector_avg(scan, *self.L_SECTOR)
        F  = self.get_sector_avg(scan, *self.F_SECTOR)
        RF = self.get_sector_avg(scan, *self.RF_SECTOR)
        R  = self.get_sector_avg(scan, *self.R_SECTOR)
        s = (
            self.dist_to_state(L, 'left'),
            self.dist_to_state(F, 'front'),
            self.dist_to_state(RF, 'right_front'),
            self.dist_to_state(R, 'right'),
        )
        return s, (L, F, RF, R)

    def execute(self, name: str):
        lin, ang = self.actions[name]
        tw = Twist()
        tw.linear.x  = float(max(min(lin, self.max_linear), -self.max_linear))
        tw.angular.z = float(ang)
        self.cmd_pub.publish(tw)

    def at_goal(self) -> bool:
        if self.reward_mode != 'sparse':
            return False
        try:
            tf = self.tf_buffer.lookup_transform('map', 'base_footprint', rclpy.time.Time())
            x = tf.transform.translation.x
            y = tf.transform.translation.y
            gx, gy, gr = self.goal
            dx, dy = x - gx, y - gy
            return (dx * dx + dy * dy) <= (gr * gr)
        except Exception:
            return False

    # ====================== RL core ======================
    def eps_greedy(self, s):
        # Exploration
        if np.random.rand() < self.epsilon:
            return np.random.randint(len(self.action_names))
        # Exploitation
        return int(np.argmax(self.q_table[s]))

    def reward(self, dists, s, a_idx):
        # Collision penalty
        front = dists[1] if not math.isinf(dists[1]) else 10.0
        if front < self.collision_thresh or s[1] == 'TOO_CLOSE':
            return -100.0

        if self.reward_mode == 'sparse':
            return self.step_penalty  # only step cost; success handled separately

        # shaped reward (optional mode)
        r_state = s[3]
        band = {'MEDIUM': 1.0, 'CLOSE': 0.4, 'FAR': 0.4, 'TOO_CLOSE': -0.8, 'TOO_FAR': -0.8}[r_state]
        progress = 0.6 if self.action_names[a_idx] == 'FORWARD' else 0.1
        return band + progress + (-0.15)

    def td_update(self, s, a, r, s2, a2=None):
        q = self.q_table[s]
        if self.algorithm == 'sarsa' and a2 is not None:
            target = r + self.gamma * self.q_table[s2][a2]
        else:
            target = r + self.gamma * max(self.q_table[s2])
        q[a] += self.alpha * (target - q[a])

    # ====================== Episodes ======================
    def start_episode(self):
        self.ep += 1
        self.step = 0
        self.ep_return = 0.0
        self.prev_state = None
        self.prev_action = None
        self.first_random = True  # movement fix #2: ensure first action is random

        # Pick scenario i and teleport (skip cleanly if no entity)
        i = (self.ep - 1) % len(self.scenarios)
        box = self.scenarios[i]
        x = np.random.uniform(*box['x'])
        y = np.random.uniform(*box['y'])
        yaw = np.random.uniform(*box['yaw'])
        self.teleport(x, y, yaw)

        # Let Gazebo settle and nudge forward once (movement fix #3)
        time.sleep(1.0)
        self.execute('FORWARD')

        # Decay epsilon each episode (keeps exploration for coverage)
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
        self.get_logger().info(f"[TRAIN] Episode {self.ep} start | ε={self.epsilon:.3f}")

    def end_episode(self, reason='timeout'):
        self.get_logger().info(f"[TRAIN] Ep{self.ep} end | steps={self.step} return={self.ep_return:.1f} | {reason}")
        # Save table and log
        try:
            np.save(SAVE_PATH, self.q_table)
            new = not LOG_PATH.exists()
            with open(LOG_PATH, 'a', newline='') as f:
                w = csv.writer(f)
                if new:
                    w.writerow(['episode', 'return', 'steps', 'reason', 'algorithm'])
                w.writerow([self.ep, round(self.ep_return, 2), self.step, reason, self.algorithm])
        except Exception:
            pass

        # Next episode (run indefinitely until you Ctrl-C)
        if self.ep < self.episodes:
            self.start_episode()
        else:
            self.execute('STOP')

    # ====================== Main callback ======================
    def scan_cb(self, msg: LaserScan):
        s, d = self.determine_state(msg.ranges)

        # TEST: greedy action from learned Q
        if not self.training:
            a = int(np.argmax(self.q_table[s]))
            self.execute(self.action_names[a])
            return

        # TRAIN:
        # Goal reached in sparse mode -> success
        if self.reward_mode == 'sparse' and self.at_goal():
            self.ep_return += 1000.0
            self.end_episode('success')
            return

        # Movement fix #2: force the very first action to random so the bot moves
        if self.first_random:
            a = np.random.randint(len(self.action_names))
            self.first_random = False
        else:
            a = self.eps_greedy(s)

        self.execute(self.action_names[a])

        # Reward + TD update
        r = self.reward(d, s, a)
        self.ep_return += r

        if self.prev_state is not None and self.prev_action is not None:
            if self.algorithm == 'sarsa':
                # On-policy: pick next action to compute target
                a_next = self.eps_greedy(s)
                self.td_update(self.prev_state, self.prev_action, r, s, a2=a_next)
            else:
                # Off-policy Q-learning
                self.td_update(self.prev_state, self.prev_action, r, s, a2=None)

        self.prev_state, self.prev_action = s, a
        self.step += 1

        # Episode termination conditions
        if r <= -100.0:
            self.end_episode('collision')
            return
        if self.step >= self.steps_per_episode:
            self.end_episode('timeout')
            return

        # Periodically save progress mid-episode
        if self.step % 500 == 0:
            try:
                np.save(SAVE_PATH, self.q_table)
                self.get_logger().info(f"[TRAIN] autosave at step {self.step}, return={self.ep_return:.1f}")
            except Exception:
                pass


# ====================== CLI / Main ======================
def parse_args(argv):
    p = argparse.ArgumentParser()
    p.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    p.add_argument('--algorithm', type=str, default='q_learning', choices=['q_learning', 'sarsa'])
    p.add_argument('--reward_mode', type=str, default='sparse', choices=['sparse', 'shaped'])
    p.add_argument('--goal_x', type=float, default=2.6)
    p.add_argument('--goal_y', type=float, default=3.1)
    p.add_argument('--goal_r', type=float, default=0.5)
    p.add_argument('--episodes', type=int, default=999999)
    p.add_argument('--steps_per_episode', type=int, default=1500)
    p.add_argument('--alpha', type=float, default=0.30)
    p.add_argument('--gamma', type=float, default=0.95)
    p.add_argument('--epsilon', type=float, default=0.30)
    p.add_argument('--epsilon_decay', type=float, default=0.997)
    p.add_argument('--epsilon_min', type=float, default=0.05)
    args, _ = p.parse_known_args(argv[1:])
    return args


def main(argv=None):
    rclpy.init(args=argv)
    args = parse_args(sys.argv)

    node = QLearningWallFollower(
        mode=args.mode,
        algorithm=args.algorithm,
        reward_mode=args.reward_mode,
        goal_x=args.goal_x, goal_y=args.goal_y, goal_r=args.goal_r,
        episodes=args.episodes, steps_per_episode=args.steps_per_episode,
        alpha=args.alpha, gamma=args.gamma,
        epsilon=args.epsilon, epsilon_decay=args.epsilon_decay, epsilon_min=args.epsilon_min,
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
