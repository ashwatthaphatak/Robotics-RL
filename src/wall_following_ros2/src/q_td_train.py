#!/usr/bin/env python3
import os, sys, math, csv, time, argparse
from pathlib import Path
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist, TwistStamped, Point, Quaternion
from tf2_ros import Buffer, TransformListener
from ros_gz_interfaces.srv import SetEntityPose
from nav_msgs.msg import Odometry

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

# ====================== Helpers: yaw + smoothing ======================

def quat_to_yaw(q: Quaternion) -> float:
    """Convert quaternion to yaw (radians in [-pi, pi])."""
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)

class EMA:
    def __init__(self, alpha=0.3, init=None):
        self.alpha = alpha
        self.s = init
    def update(self, x):
        self.s = x if self.s is None else self.alpha * x + (1 - self.alpha) * self.s
        return self.s

# ====================== Turn controller (macro-action) ======================
class TurnController:
    """
    Commit to a target yaw delta (e.g., +90°, -90°, 180°) with speed ramp-down.
    Prevents RL from being interrupted by frequent LiDAR updates during the turn.
    """
    def __init__(self, yaw_provider, ang_speed=0.7, tol_deg=4.0, min_ticks=8, max_ticks=120):
        self._yaw = yaw_provider
        self.base_speed = ang_speed
        self.tol = math.radians(tol_deg)
        self.min_ticks = min_ticks     # ensure we don’t exit too early
        self.max_ticks = max_ticks     # safety bound
        self.active = False
        self._target = None
        self._ticks = 0

    @staticmethod
    def _wrap(a):
        return math.atan2(math.sin(a), math.cos(a))

    @staticmethod
    def _ang_err(cur, target):
        # shortest signed distance target - cur in [-pi, pi]
        e = math.atan2(math.sin(target - cur), math.cos(target - cur))
        return e

    def start(self, delta_deg):
        if self.active:
            return
        cur = self._yaw()
        self._target = self._wrap(cur + math.radians(delta_deg))
        self._ticks = 0
        self.active = True

    def cancel(self):
        self.active = False
        self._target = None
        self._ticks = 0

    def step(self):
        """Return a Twist command when active, else None."""
        if not self.active:
            return None
        self._ticks += 1
        cur = self._yaw()
        err = self._ang_err(cur, self._target)
        at_goal = abs(err) < self.tol and self._ticks >= self.min_ticks
        timeout = self._ticks >= self.max_ticks
        stop = Twist()
        stop.linear.x = 0.0
        stop.angular.z = 0.0
        if at_goal or timeout:
            self.cancel()
            return stop
        # Speed ramps down near target to reduce overshoot.
        k = max(0.25, min(1.0, abs(err) / (math.pi / 4)))  # scale in [0.25, 1]
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = self.base_speed * k * (1.0 if err > 0 else -1.0)
        return cmd

# ====================== Supervisor (left/UTurn triggers) ======================
class JunctionSupervisor:
    """
    Lightweight rule layer to trigger left or U-turn when appropriate while
    policy follows the right wall during normal motion.
    """
    def __init__(self, min_consec=3):
        self.front_ema = EMA(0.3)
        self.left_ema  = EMA(0.3)
        self.right_ema = EMA(0.3)
        self._open_left_count = 0
        self._dead_end_count  = 0
        self._min_consec = min_consec

    def update(self, L, F, R):
        f = self.front_ema.update(F)
        l = self.left_ema.update(L)
        r = self.right_ema.update(R)
        return f, l, r

    def maybe_turn(self, f, l, r, turn_ctrl: TurnController,
                   front_block=0.45, side_block=0.40, left_open=0.9):
        # Count consecutive frames to avoid flicker
        if f < front_block and l > left_open:
            self._open_left_count += 1
        else:
            self._open_left_count = 0

        if f < front_block and l < side_block and r < side_block:
            self._dead_end_count += 1
        else:
            self._dead_end_count = 0

        if not turn_ctrl.active:
            if self._dead_end_count >= self._min_consec:
                turn_ctrl.start(180.0)
                return 'u_turn'
            if self._open_left_count >= self._min_consec:
                turn_ctrl.start(90.0)
                return 'left_90'
        return None


class QLearningWallFollower(Node):
    def __init__(self,
                 mode='train',
                 algorithm='q_learning',          # q_learning | sarsa
                 reward_mode='sparse',            # sparse | shaped
                 reset_mode='once',               # none | once | episode
                 goal_x=-2.6, goal_y=3.1, goal_r=0.5,
                 episodes=999999, steps_per_episode=1500,
                 alpha=0.3, gamma=0.95,
                 epsilon=0.30, epsilon_decay=0.997, epsilon_min=0.05):
        super().__init__('q_td_train')
        mode = 'train'
        algorithm = 'q_learning'

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

        # ---- Turn controller & supervisor params ----
        self.turn_ang_speed = 0.7
        self.turn_tol_deg   = 4.0
        self.front_block    = 0.45
        self.side_block     = 0.40
        self.left_open      = 0.90
        self.min_consec     = 3

        # ---- STUCK watchdog params ----
        self.stuck_timeout_s = 5.0      # seconds stationary before reset
        self.stuck_pos_eps   = 0.03     # meters movement to count as progress
        self._stuck_ref_pose = None     # (x, y)
        self._last_move_time = None     # wall time in ROS clock seconds

        # ---- Goal stopping knobs ----
        self.stop_on_goal   = True
        self.goal_reward    = 1000.0
        self.goal_frames    = ['base_footprint', 'base_link']  # TF frames to try

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

        # Odometry for yaw during macro-turns & stuck detection
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_cb, 20)
        self._last_odom = None

        # TF for goal checks
        self.tf_buffer   = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Optional Gazebo reset service (only used if reset_mode != 'none'… unless forced)
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

        # Sticky action for small turns (keeps LEFT/RIGHT for a few ticks)
        self.sticky_steps_left = 0
        self.last_action_idx   = None

        # Start pose (used when reset_mode != 'none')
        self.start_pose = {'x': -2.0, 'y': -3.2, 'yaw': 0.0}

        # Turn controller + supervisor
        self.turn_ctrl = TurnController(self.current_yaw,
                                        ang_speed=self.turn_ang_speed,
                                        tol_deg=self.turn_tol_deg,
                                        min_ticks=8, max_ticks=120)
        self.supervisor = JunctionSupervisor(min_consec=self.min_consec)

        # Cached dists
        self.left_d = self.front_d = self.right_d = float('inf')

        # Try to resolve the Gazebo entity name (used only if we actually reset)
        self.entity_name = None
        if self.reset_mode in ('once', 'episode'):
            self.entity_name = self.resolve_entity_name()

        # Kickstart nudge to help Gazebo get going (tiny, harmless)
        self.kickstart_nudge()

        # Init stuck timer
        self._last_move_time = self.now_sec()

        # Begin training
        if self.training:
            self.start_episode()

        self.get_logger().info(
            f"RL ready | mode={self.mode} | algo={self.algorithm} | reward={self.reward_mode} | reset_mode={self.reset_mode}"
        )

    # ====================== Time / Odom helpers ======================
    def now_sec(self) -> float:
        return float(self.get_clock().now().nanoseconds) * 1e-9

    def odom_cb(self, msg: Odometry):
        self._last_odom = msg

    def current_yaw(self) -> float:
        if self._last_odom is None:
            return 0.0
        return quat_to_yaw(self._last_odom.pose.pose.orientation)

    def current_xy(self):
        if self._last_odom is None:
            return None
        p = self._last_odom.pose.pose.position
        return (float(p.x), float(p.y))

    # ====================== Goal check ======================
    def at_goal(self):
        """Return (bool, dist) — is robot within goal radius? Uses TF, falls back to odom."""
        gx, gy, gr = self.goal
        # Try TF first
        for base in self.goal_frames:
            try:
                t = self.tf_buffer.lookup_transform('map', base, Time())
                x = float(t.transform.translation.x)
                y = float(t.transform.translation.y)
                d = math.hypot(x - gx, y - gy)
                return (d <= gr), d
            except Exception:
                continue
        # Fallback: odom (assuming map≈odom due to static map->odom)
        if self._last_odom is not None:
            x = float(self._last_odom.pose.pose.position.x)
            y = float(self._last_odom.pose.pose.position.y)
            d = math.hypot(x - gx, y - gy)
            return (d <= gr), d
        return (False, float('inf'))

    # ====================== Setup helpers ======================
    def setup_state_action_space(self):
        # States
        self.L_STATES  = ['CLOSE', 'FAR']
        self.F_STATES  = ['TOO_CLOSE', 'CLOSE', 'MEDIUM', 'FAR']
        self.RF_STATES = ['CLOSE', 'FAR']
        self.R_STATES  = ['TOO_CLOSE', 'CLOSE', 'MEDIUM', 'FAR', 'TOO_FAR']

        # Distance bins
        self.bounds = {
            'TOO_CLOSE': (0.0, 0.6),
            'CLOSE':     (0.6, 0.9),
            'MEDIUM':    (0.9, 1.5),
            'FAR':       (1.5, 2.5),
            'TOO_FAR':   (2.5, float('inf')),
        }

        # LiDAR sectors (degrees)
        self.L_SECTOR  = (80, 100)
        self.F_SECTOR  = (355, 5)
        self.RF_SECTOR = (310, 320)
        self.R_SECTOR  = (260, 280)

        # Discrete actions
        self.actions = {
            'FORWARD': (0.50, 0.0),
            'LEFT':    (0.22, 0.985),
            'RIGHT':   (0.22, -0.985),
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
                self.get_logger().warning(f"[Q] Failed to load table: {e}")
        return False

    # ====================== Gazebo reset helpers (optional) ======================
    def resolve_entity_name(self):
        if not self.reset_cli.service_is_ready():
            self.get_logger().warning("SetEntityPose service not ready; resets may be skipped.")
            return None
        for cand in ENTITY_NAME_CANDIDATES:
            if self.try_set_pose_probe(cand):
                self.get_logger().info(f"[RESET] Using Gazebo entity: {cand}")
                return cand
        self.get_logger().warning("[RESET] Could not resolve entity; resets disabled.")
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

    def teleport(self, x: float, y: float, yaw: float, force: bool = False):
        """Synchronous, safe teleport. If force=True, ignore reset_mode and try anyway."""
        if (not force) and self.reset_mode == 'none':
            return
        if not self.reset_cli.service_is_ready():
            self.get_logger().warning("Teleport skipped: service not ready.")
            return

        # Re-resolve entity periodically or if unknown
        if not self.entity_name or (self.ep % 50 == 0):
            self.entity_name = self.resolve_entity_name()
            if not self.entity_name:
                self.get_logger().warning("Teleport skipped: no valid entity.")
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
            self.get_logger().warning(f"Teleport exception: {e}")

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

    def determine_state(self, ranges):
        L  = self.get_sector_avg(ranges, *self.L_SECTOR)
        F  = self.get_sector_avg(ranges, *self.F_SECTOR)
        RF = self.get_sector_avg(ranges, *self.RF_SECTOR)
        R  = self.get_sector_avg(ranges, *self.R_SECTOR)
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

    def publish_twist(self, tw: Twist):
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
                self.publish_twist(tw)
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
            # Back up slightly before next episode start
            twist = Twist()
            twist.linear.x = -0.15
            twist.angular.z = 0.0
            for _ in range(5):
                self.publish_twist(twist)
                time.sleep(0.1)
            return -200.0 

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
        self.sticky_steps_left = 0
        self.last_action_idx = None

        # Optional reset policy
        if self.reset_mode == 'episode' or (self.reset_mode == 'once' and self.ep == 1):
            p = self.start_pose
            self.teleport(p['x'], p['y'], p['yaw'])

        # Decay epsilon each episode
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

        # Reset stuck timers per episode
        self._stuck_ref_pose = None
        self._last_move_time = self.now_sec()

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

    # ====================== STUCK detection ======================
    def _stuck_check_and_handle(self) -> bool:
        """
        Returns True if a stuck reset was performed (and episode ended).
        """
        # If turning in place, don't consider stuck; refresh timer.
        if self.turn_ctrl.active:
            self._stuck_ref_pose = self.current_xy() or self._stuck_ref_pose
            self._last_move_time = self.now_sec()
            return False

        cur = self.current_xy()
        if cur is None:
            return False  # no odom yet

        now = self.now_sec()
        if self._stuck_ref_pose is None:
            self._stuck_ref_pose = cur
            self._last_move_time = now
            return False

        dx = cur[0] - self._stuck_ref_pose[0]
        dy = cur[1] - self._stuck_ref_pose[1]
        dist = math.hypot(dx, dy)

        if dist > self.stuck_pos_eps:
            # Progress made -> reset timer/anchor
            self._stuck_ref_pose = cur
            self._last_move_time = now
            return False

        # No progress; check timeout
        if (now - self._last_move_time) >= self.stuck_timeout_s:
            self.get_logger().warn(f"[STUCK] No XY progress > {self.stuck_timeout_s:.1f}s "
                                   f"(≈{dist:.3f}m). Teleporting to start and ending episode.")
            # Force teleport even if reset_mode == 'none'
            p = self.start_pose
            self.teleport(p['x'], p['y'], p['yaw'], force=True)
            self.end_episode('stuck')
            return True

        return False

    # ====================== Main callback ======================
    def scan_cb(self, msg: LaserScan):
        # --- Goal check FIRST (prevents any post-goal learning/pollution) ---
        if self.training and self.stop_on_goal:
            reached, dist = self.at_goal()
            if reached:
                # Optional: one terminal backprop to credit the last (s,a)
                if self.prev_state is not None and self.prev_action is not None:
                    # Terminal target = R_goal (no bootstrap)
                    q = self.q_table[self.prev_state]
                    q[self.prev_action] += self.alpha * (self.goal_reward - q[self.prev_action])
                self.ep_return += self.goal_reward
                # Prevent new episodes after success
                self.episodes = self.ep
                self.get_logger().info(f"[SUCCESS] Goal reached (d≈{dist:.2f} m). Stopping training.")
                self.end_episode('success')
                return

        # Update discrete state & raw distances
        s, d = self.determine_state(msg.ranges)
        L, F, RF, R = d
        self.left_d, self.front_d, self.right_d = L, F, R

        # If we are mid-turn, keep turning regardless of new observations
        tcmd = self.turn_ctrl.step()
        if tcmd is not None:
            self.publish_twist(tcmd)
            return

        # STUCK watchdog (only when not turning)
        if self._stuck_check_and_handle():
            return  # episode already ended & restarted

        # Let supervisor trigger left/UTurn if needed (works in train & test)
        f, l, r = self.supervisor.update(L, F, R)
        event = self.supervisor.maybe_turn(f, l, r, self.turn_ctrl,
                                           front_block=self.front_block,
                                           side_block=self.side_block,
                                           left_open=self.left_open)
        if event is not None:
            # Optional immediate stop before committing the turn
            stop = Twist(); stop.linear.x = 0.0; stop.angular.z = 0.0
            self.publish_twist(stop)
            return

        # TEST path (kept for completeness)
        if not self.training:
            a = int(np.argmax(self.q_table[s]))
            if self.sticky_steps_left > 0 and self.last_action_idx is not None:
                a = self.last_action_idx
                self.sticky_steps_left -= 1
            else:
                if self.action_names[a] in ('LEFT', 'RIGHT'):
                    self.sticky_steps_left = 6
                    self.last_action_idx = a
            self.execute(self.action_names[a])
            return

        # TRAIN path
        if self.first_random:
            a = np.random.randint(len(self.action_names))
            self.first_random = False
        else:
            if self.sticky_steps_left > 0 and self.last_action_idx is not None:
                a = self.last_action_idx
                self.sticky_steps_left -= 1
            else:
                a = self.eps_greedy(s)
                if self.action_names[a] in ('LEFT', 'RIGHT'):
                    self.sticky_steps_left = 6
                    self.last_action_idx = a

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
