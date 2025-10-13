#!/usr/bin/env python3
import os, sys, math, csv, time, argparse
from pathlib import Path
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist, TwistStamped, Point, Quaternion
from tf2_ros import Buffer, TransformListener
from ros_gz_interfaces.srv import SetEntityPose

# -------- Persistence --------
SAVE_PATH = Path.home() / '.ros' / 'wf_qtable.npy'
LOG_PATH  = Path.home() / '.ros' / 'wf_train_log.csv'

# -------- Possible Gazebo entity names (auto-detect) --------
ENTITY_NAME_CANDIDATES = [
    'turtlebot3_burger',
    'turtlebot3_burger_0',
    'burger',
    'turtlebot3'
]

class QLearningWallFollower(Node):
    def __init__(self,
                 mode='train',
                 algorithm='q_learning',          # q_learning | sarsa
                 reward_mode='sparse',            # sparse | shaped
                 reset_mode='once',               # none | once | episode
                 goal_x=2.6, goal_y=3.1, goal_r=0.5,
                 episodes=999999, steps_per_episode=1500,
                 alpha=0.3, gamma=0.95,
                 epsilon=0.30, epsilon_decay=0.997, epsilon_min=0.05):
        super().__init__('sarsa_run')
        mode = 'test'
        algorithm = 'sarsa'

        # ---- Config ----
        self.mode          = mode
        self.algorithm     = algorithm
        self.reward_mode   = reward_mode
        self.reset_mode    = reset_mode
        self.goal          = (float(goal_x), float(goal_y), float(goal_r))
        self.episodes      = int(episodes)
        self.steps_per_episode = int(steps_per_episode)
        self.alpha         = float(alpha)
        self.gamma         = float(gamma)
        self.epsilon       = float(epsilon)
        self.epsilon_decay = float(epsilon_decay)
        self.epsilon_min   = float(epsilon_min)

        # ---- Motion & reward knobs ----
        self.collision_thresh = 0.10   # Laser front threshold for collision
        self.max_linear       = 0.28   # velocity cap
        self.step_penalty     = -0.05  # used in both sparse & shaped
        self.collision_penalty = -200.0
        self.forward_bonus     = 1.5
        self.spin_penalty      = -20.0
        self.spin_threshold    = 4
        self._spin_count       = 0
        self._last_turn_dir    = 0.0

        # ---- Discretization & actions ----
        self.setup_state_action_space()

        # ---- Q-table ----
        self.q_table = {s:[0.0, 0.0, 0.0, 0.0] for s in self.all_states}
        # Forward bias so early exploration prefers moving
        self.q_table_forward_idx = self.action_names.index('FORWARD')
        for s in self.all_states:
            self.q_table[s][self.q_table_forward_idx] += 0.5

        # Try warm-start
        self.try_load_qtable()

        # ---- ROS I/O ----
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_cb, 10)
        # Dual-topic publishers to satisfy either bridge variant
        self.cmd_pub  = self.create_publisher(Twist, '/cmd_vel', 10)
        self.cmd_pub_stamped = self.create_publisher(TwistStamped, '/cmd_vel_stamped', 10)

        # TF for optional goal checks (kept available)
        self.tf_buffer   = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Optional Gazebo reset service (only used if reset_mode != 'none')
        self.reset_cli = self.create_client(SetEntityPose, '/world/default/set_pose')
        self.reset_cli.wait_for_service(timeout_sec=3.0)

        # ---- Bookkeeping ----
        self.training     = (self.mode == 'train')
        self.ep           = 0
        self.step         = 0
        self.ep_return    = 0.0
        self.prev_state   = None
        self.prev_action  = None
        self.first_random = True

        # Start pose (used when reset_mode != 'none')
        self.start_pose = {'x': -2.0, 'y': -3.2, 'yaw': 0.0}

        # Try to resolve the Gazebo entity name (used only if we actually reset)
        self.entity_name = None
        if self.reset_mode in ('once', 'episode'):
            self.entity_name = self.resolve_entity_name()

        # Kickstart nudge to help Gazebo get going (tiny, harmless)
        self.kickstart_nudge()

        # Begin training
        if self.training:
            self.start_episode()

        self.get_logger().info(
            f"RL ready | mode={self.mode} | algo={self.algorithm} | reward={self.reward_mode} | reset_mode={self.reset_mode}"
        )

    # ====================== Setup helpers ======================
    def setup_state_action_space(self):
        # States
        self.L_STATES  = ['CLOSE', 'FAR']
        self.F_STATES  = ['TOO_CLOSE', 'CLOSE', 'MEDIUM', 'FAR']
        self.RF_STATES = ['CLOSE', 'FAR']
        self.R_STATES  = ['TOO_CLOSE', 'CLOSE', 'MEDIUM', 'FAR', 'TOO_FAR']

        # Distance bins
        self.bounds = {
            'TOO_CLOSE': (0.0, 0.4),
            'CLOSE':     (0.4, 0.7),
            'MEDIUM':    (0.7, 1.2),
            'FAR':       (1.2, 2.5),
            'TOO_FAR':   (2.5, float('inf')),
        }

        # LiDAR sectors (degrees)
        self.L_SECTOR  = (80, 100)
        self.F_SECTOR  = (355, 5)
        self.RF_SECTOR = (310, 320)
        self.R_SECTOR  = (260, 280)

        # Discrete actions
        self.actions = {
            'FORWARD': (0.25, 0.0),
            'LEFT':    (0.22, 0.785),
            'RIGHT':   (0.22, -0.785),
            'STOP':    (0.0, 0.0)
        }
        self.action_names = list(self.actions.keys())

        # Enumerate all states
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
                    self.get_logger().info(f"[Q] Warm-start from {SAVE_PATH}")
                    return True
            except Exception as e:
                self.get_logger().warn(f"[Q] Failed to load table: {e}")
        return False

    # ====================== Gazebo reset helpers (optional) ======================
    def resolve_entity_name(self):
        if not self.reset_cli.service_is_ready():
            self.get_logger().warn("SetEntityPose service not ready; resets may be skipped.")
            return None
        for cand in ENTITY_NAME_CANDIDATES:
            if self.try_set_pose_probe(cand):
                self.get_logger().info(f"[RESET] Using Gazebo entity: {cand}")
                return cand
        self.get_logger().warn("[RESET] Could not resolve entity; resets disabled.")
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
        """Synchronous, safe teleport (used only if reset_mode != 'none')."""
        if self.reset_mode == 'none':
            return
        if not self.reset_cli.service_is_ready():
            self.get_logger().warn("Teleport skipped: service not ready.")
            return

        # Re-resolve entity periodically
        if not self.entity_name or (self.ep % 50 == 0):
            self.entity_name = self.resolve_entity_name()
            if not self.entity_name:
                self.get_logger().warn("Teleport skipped: no valid entity.")
                return
        try:
            req = SetEntityPose.Request()
            req.entity.name = self.entity_name
            req.pose.position = Point(x=float(x), y=float(y), z=0.05)
            qz, qw = math.sin(yaw/2.0), math.cos(yaw/2.0)
            req.pose.orientation = Quaternion(x=0.0, y=0.0, z=qz, w=qw)
            fut = self.reset_cli.call_async(req)
            rclpy.spin_until_future_complete(self, fut, timeout_sec=2.0)
            time.sleep(1.0)  # settle
        except Exception as e:
            self.get_logger().warn(f"Teleport exception: {e}")

    # ====================== LiDAR / state helpers ======================
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

    # ====================== Action execution ======================
    def execute(self, name: str):
        lin, ang = self.actions[name]
        tw = Twist()
        tw.linear.x  = float(max(min(lin, self.max_linear), -self.max_linear))
        tw.angular.z = float(ang)
        self.cmd_pub.publish(tw)

        tws = TwistStamped()
        tws.header.stamp = self.get_clock().now().to_msg()
        tws.twist = tw
        self.cmd_pub_stamped.publish(tws)

    def kickstart_nudge(self):
        """Send a couple of small forward commands to ensure Gazebo starts moving."""
        try:
            tw = Twist()
            tw.linear.x = 0.12; tw.angular.z = 0.0
            for _ in range(2):
                self.cmd_pub.publish(tw)
                tws = TwistStamped()
                tws.header.stamp = self.get_clock().now().to_msg()
                tws.twist = tw
                self.cmd_pub_stamped.publish(tws)
                time.sleep(0.15)
        except Exception:
            pass

    # ====================== RL core ======================
    def eps_greedy(self, s):
        if np.random.rand() < self.epsilon:
            return np.random.randint(len(self.action_names))
        return int(np.argmax(self.q_table[s]))

    def reward(self, dists, s, a_idx):
        # Collision check
        front = dists[1] if not math.isinf(dists[1]) else 10.0
        if front < self.collision_thresh or s[1] == 'TOO_CLOSE':
            return self.collision_penalty

        if self.reward_mode == 'sparse':
            return self.step_penalty

        # Shaped reward
        r_state = s[3]
        band = {
            'MEDIUM': 1.0, 'CLOSE': 0.4, 'FAR': 0.4,
            'TOO_CLOSE': -1.0, 'TOO_FAR': -0.8
        }[r_state]

        is_forward = (self.action_names[a_idx] == 'FORWARD')
        progress = 0.6 + (self.forward_bonus if is_forward else 0.0)

        # Anti-spin
        ang = self.actions[self.action_names[a_idx]][1]
        if abs(ang) > 0.6:
            dir_sign = math.copysign(1, ang)
            if dir_sign == self._last_turn_dir:
                self._spin_count += 1
            else:
                self._spin_count = 1
            self._last_turn_dir = dir_sign
        else:
            self._spin_count = 0
            self._last_turn_dir = 0.0

        spin_term = self.spin_penalty if self._spin_count > self.spin_threshold else 0.0

        return band + progress + self.step_penalty + spin_term

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
        self.first_random = True

        # Optional reset policy
        if self.reset_mode == 'episode' or (self.reset_mode == 'once' and self.ep == 1):
            p = self.start_pose
            self.teleport(p['x'], p['y'], p['yaw'])

        # Decay epsilon each episode
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

        self.get_logger().info(
            f"[TRAIN] Episode {self.ep} start | ε={self.epsilon:.3f} | reset_mode={self.reset_mode}"
        )

    def end_episode(self, reason='timeout'):
        self.get_logger().info(
            f"[TRAIN] Ep{self.ep} end | steps={self.step} return={self.ep_return:.1f} | {reason}"
        )
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

        if self.ep < self.episodes:
            self.start_episode()
        else:
            self.execute('STOP')

    # ====================== Main callback ======================
    def scan_cb(self, msg: LaserScan):
        s, d = self.determine_state(msg.ranges)

        # TEST mode: greedy execution only
        if not self.training:
            a = int(np.argmax(self.q_table[s]))
            self.execute(self.action_names[a])
            return

        # TRAIN mode:
        # Select action
        if self.first_random:
            a = np.random.randint(len(self.action_names))
            self.first_random = False
        else:
            a = self.eps_greedy(s)

        # Execute
        self.execute(self.action_names[a])

        # Reward + TD
        r = self.reward(d, s, a)
        self.ep_return += r

        if self.prev_state is not None and self.prev_action is not None:
            if self.algorithm == 'sarsa':
                a_next = self.eps_greedy(s)
                self.td_update(self.prev_state, self.prev_action, r, s, a2=a_next)
            else:
                self.td_update(self.prev_state, self.prev_action, r, s)

        self.prev_state, self.prev_action = s, a
        self.step += 1

        # Termination
        if r <= self.collision_penalty:
            self.end_episode('collision')
            return
        if self.step >= self.steps_per_episode:
            self.end_episode('timeout')
            return

        # Autosave
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
    p.add_argument('--reset_mode', type=str, default='none', choices=['none', 'once', 'episode'])
    p.add_argument('--goal_x', type=float, default=-2.5)
    p.add_argument('--goal_y', type=float, default=-2.5)
    p.add_argument('--goal_r', type=float, default=0.5)
    p.add_argument('--episodes', type=int, default=999999)
    p.add_argument('--steps_per_episode', type=int, default=2000)
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
        reset_mode=args.reset_mode,
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
