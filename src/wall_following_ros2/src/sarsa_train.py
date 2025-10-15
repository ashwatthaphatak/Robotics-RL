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

# -------- Persistence (SARSA) --------
SAVE_PATH = Path.home() / '.ros' / 'wf_sarsa.numpy'           # keep this exact filename
LOG_PATH  = Path.home() / '.ros' / 'wf_sarsa_train_log.csv'

# -------- Possible Gazebo entity names (auto-detect) --------
ENTITY_NAME_CANDIDATES = ['turtlebot3_burger', 'turtlebot3_burger_0', 'burger', 'turtlebot3']

# ====================== Helpers: yaw + smoothing ======================
def quat_to_yaw(q: Quaternion) -> float:
    """Convert quaternion to yaw (radians in [-pi, pi])."""
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)

class EMA:
    def __init__(self, alpha=0.3, init=None):
        self.alpha = alpha; self.s = init
    def update(self, x):
        self.s = x if self.s is None else self.alpha * x + (1 - self.alpha) * self.s
        return self.s

# ====================== Turn controller (macro-action) ======================
class TurnController:
    """Commit to a target yaw delta (e.g., +90°, -90°, 180°) with ramp-down."""
    def __init__(self, yaw_provider, ang_speed=0.7, tol_deg=4.0, min_ticks=8, max_ticks=120):
        self._yaw = yaw_provider
        self.base_speed = ang_speed
        self.tol = math.radians(tol_deg)
        self.min_ticks = min_ticks
        self.max_ticks = max_ticks
        self.active = False
        self._target = None
        self._ticks = 0
    @staticmethod
    def _wrap(a): return math.atan2(math.sin(a), math.cos(a))
    @staticmethod
    def _ang_err(cur, target): return math.atan2(math.sin(target - cur), math.cos(target - cur))
    def start(self, delta_deg):
        if self.active: return
        cur = self._yaw()
        self._target = self._wrap(cur + math.radians(delta_deg))
        self._ticks = 0; self.active = True
    def cancel(self):
        self.active = False; self._target = None; self._ticks = 0
    def step(self):
        if not self.active: return None
        self._ticks += 1
        cur = self._yaw()
        err = self._ang_err(cur, self._target)
        at_goal = abs(err) < self.tol and self._ticks >= self.min_ticks
        timeout = self._ticks >= self.max_ticks
        stop = Twist(); stop.linear.x = 0.0; stop.angular.z = 0.0
        if at_goal or timeout:
            self.cancel(); return stop
        k = max(0.25, min(1.0, abs(err) / (math.pi/4)))
        cmd = Twist(); cmd.linear.x = 0.0
        cmd.angular.z = self.base_speed * k * (1.0 if err > 0 else -1.0)
        return cmd

# ====================== Supervisor (left/UTurn triggers) ======================
class JunctionSupervisor:
    """Debounced triggers for left-90 and 180° at dead-ends; normal motion follows right wall."""
    def __init__(self, min_consec=3):
        self.front_ema = EMA(0.3); self.left_ema = EMA(0.3); self.right_ema = EMA(0.3)
        self._open_left_count = 0; self._dead_end_count = 0; self._min_consec = min_consec
    def update(self, L, F, R):
        return self.front_ema.update(F), self.left_ema.update(L), self.right_ema.update(R)
    def maybe_turn(self, f, l, r, turn_ctrl: TurnController,
                   front_block=0.45, side_block=0.40, left_open=0.90):
        if f < front_block and l > left_open: self._open_left_count += 1
        else: self._open_left_count = 0
        if f < front_block and l < side_block and r < side_block: self._dead_end_count += 1
        else: self._dead_end_count = 0
        if not turn_ctrl.active:
            if self._dead_end_count >= self._min_consec:
                turn_ctrl.start(180.0); return 'u_turn'
            if self._open_left_count >= self._min_consec:
                turn_ctrl.start(90.0); return 'left_90'
        return None

# ====================== SARSA Trainer ======================
class SarsaWallFollower(Node):
    def __init__(self,
                 mode='train',
                 algorithm='sarsa',                # q_learning | sarsa
                 reward_mode='sparse',             # sparse | shaped
                 reset_mode='once',                # none | once | episode
                 goal_x=-2.6, goal_y=3.1, goal_r=0.5,
                 episodes=999999, steps_per_episode=1500,
                 alpha=0.3, gamma=0.95,
                 epsilon=0.30, epsilon_decay=0.997, epsilon_min=0.05):
        super().__init__('sarsa_train')
        mode = 'train'
        algorithm = 'sarsa'

        # ---- Config ----
        self.mode, self.algorithm = mode, algorithm
        self.reward_mode = reward_mode
        self.reset_mode  = reset_mode
        self.goal        = (float(goal_x), float(goal_y), float(goal_r))
        self.episodes    = int(episodes)
        self.steps_per_episode = int(steps_per_episode)
        self.alpha, self.gamma = float(alpha), float(gamma)
        self.epsilon, self.epsilon_decay, self.epsilon_min = float(epsilon), float(epsilon_decay), float(epsilon_min)

        # ---- Motion & reward knobs ----
        self.collision_thresh  = 0.10
        self.max_linear        = 0.28
        self.step_penalty      = -0.05
        self.collision_penalty = -200.0
        self.forward_bonus     = 1.5
        self.spin_penalty      = -20.0
        self.spin_threshold    = 4
        self._spin_count       = 0
        self._last_turn_dir    = 0.0

        # ---- Macro-turn & supervisor params ----
        self.turn_ang_speed = 0.7
        self.turn_tol_deg   = 4.0
        self.front_block    = 0.45
        self.side_block     = 0.40
        self.left_open      = 0.90
        self.min_consec     = 3

        # ---- Stop-on-goal (avoid post-goal pollution) ----
        self.stop_on_goal = True
        self.goal_reward  = 1000.0
        self.goal_frames  = ['base_footprint', 'base_link']

        # ---- STUCK watchdog ----
        self.stuck_timeout_s = 5.0
        self.stuck_pos_eps   = 0.03
        self._stuck_ref_pose = None
        self._last_move_time = None

        # ---- Discretization & actions (match q_td_train) ----
        self.setup_state_action_space()

        # ---- Q-table ----
        self.q_table = {s:[0.0, 0.0, 0.0, 0.0] for s in self.all_states}
        # Forward bias for early exploration
        fwd = self.action_names.index('FORWARD')
        for s in self.all_states:
            self.q_table[s][fwd] += 0.5

        # Warm start
        self.try_load_qtable()

        # ---- ROS I/O ----
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_cb, 10)
        self.cmd_pub  = self.create_publisher(Twist, '/cmd_vel', 10)
        self.cmd_pub_stamped = self.create_publisher(TwistStamped, '/cmd_vel_stamped', 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_cb, 20)
        self._last_odom = None

        # TF for goal checks
        self.tf_buffer   = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Gazebo reset service
        self.reset_cli = self.create_client(SetEntityPose, '/world/default/set_pose')
        self.reset_cli.wait_for_service(timeout_sec=3.0)

        # ---- Bookkeeping ----
        self.training     = True
        self.ep           = 0
        self.step         = 0
        self.ep_return    = 0.0
        self.prev_state   = None
        self.prev_action  = None
        self.first_random = True
        self.sticky_steps_left = 0
        self.last_action_idx   = None

        self.start_pose = {'x': -2.0, 'y': -3.2, 'yaw': 0.0}
        self.turn_ctrl = TurnController(self.current_yaw,
                                        ang_speed=self.turn_ang_speed,
                                        tol_deg=self.turn_tol_deg,
                                        min_ticks=8, max_ticks=120)
        self.supervisor = JunctionSupervisor(min_consec=self.min_consec)

        # cache dists
        self.left_d = self.front_d = self.right_d = float('inf')

        self.entity_name = self.resolve_entity_name()
        self.kickstart_nudge()
        self._last_move_time = self.now_sec()
        self.start_episode()

        self.get_logger().info(f"RL ready | mode={self.mode} | algo={self.algorithm} | reward={self.reward_mode} | reset_mode={self.reset_mode}")

    # ====================== Time / Odom ======================
    def now_sec(self) -> float:
        return float(self.get_clock().now().nanoseconds) * 1e-9
    def odom_cb(self, msg: Odometry): self._last_odom = msg
    def current_yaw(self) -> float:
        if self._last_odom is None: return 0.0
        return quat_to_yaw(self._last_odom.pose.pose.orientation)
    def current_xy(self):
        if self._last_odom is None: return None
        p = self._last_odom.pose.pose.position
        return (float(p.x), float(p.y))

    # ---------- Goal check ----------
    def at_goal(self):
        gx, gy, gr = self.goal
        for base in self.goal_frames:
            try:
                t = self.tf_buffer.lookup_transform('map', base, Time())
                x = float(t.transform.translation.x); y = float(t.transform.translation.y)
                d = math.hypot(x - gx, y - gy)
                return (d <= gr), d
            except Exception:
                continue
        if self._last_odom is not None:
            x = float(self._last_odom.pose.pose.position.x)
            y = float(self._last_odom.pose.pose.position.y)
            d = math.hypot(x - gx, y - gy)
            return (d <= gr), d
        return (False, float('inf'))

    # ====================== Setup helpers ======================
    def setup_state_action_space(self):
        self.L_STATES  = ['CLOSE', 'FAR']
        self.F_STATES  = ['TOO_CLOSE', 'CLOSE', 'MEDIUM', 'FAR']
        self.RF_STATES = ['CLOSE', 'FAR']
        self.R_STATES  = ['TOO_CLOSE', 'CLOSE', 'MEDIUM', 'FAR', 'TOO_FAR']
        self.bounds = {
            'TOO_CLOSE': (0.0, 0.6),
            'CLOSE':     (0.6, 0.9),
            'MEDIUM':    (0.9, 1.5),
            'FAR':       (1.5, 2.5),
            'TOO_FAR':   (2.5, float('inf')),
        }
        self.L_SECTOR  = (80, 100)
        self.F_SECTOR  = (355, 5)
        self.RF_SECTOR = (310, 320)
        self.R_SECTOR  = (260, 280)
        self.actions = {
            'FORWARD': (0.50, 0.0),
            'LEFT':    (0.22, 0.985),
            'RIGHT':   (0.22, -0.985),
            'STOP':    (0.0,  0.0)
        }
        self.action_names = list(self.actions.keys())
        self.all_states = []
        for l in self.L_STATES:
            for f in self.F_STATES:
                for rf in self.RF_STATES:
                    for r in self.R_STATES:
                        self.all_states.append((l, f, rf, r))

    # ====================== Q-table I/O ======================
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
    def save_qtable(self):
        try:
            SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(SAVE_PATH, 'wb') as f:
                np.save(f, self.q_table)
        except Exception as e:
            self.get_logger().warning(f"[Q] Failed to save table: {e}")

    # ====================== Gazebo reset helpers ======================
    def resolve_entity_name(self):
        if not self.reset_cli.service_is_ready():
            self.get_logger().warning("SetEntityPose service not ready; teleports may be skipped.")
            return None
        for cand in ENTITY_NAME_CANDIDATES:
            if self.try_set_pose_probe(cand):
                self.get_logger().info(f"[RESET] Using Gazebo entity: {cand}")
                return cand
        self.get_logger().warning("[RESET] Could not resolve entity; teleports might fail.")
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
        if (not force) and self.reset_mode == 'none':
            return
        if not self.reset_cli.service_is_ready():
            self.get_logger().warning("Teleport skipped: service not ready.")
            return
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
            time.sleep(1.0)
        except Exception as e:
            self.get_logger().warning(f"Teleport exception: {e}")

    # ====================== State helpers ======================
    def get_sector_avg(self, ranges, a_deg, b_deg):
        n = len(ranges)
        if a_deg > b_deg:
            a = int(a_deg * n / 360.0)
            vals = list(ranges[a:]) + list(ranges[:int(b_deg * n / 360.0)])
        else:
            a = int(a_deg * n / 360.0); b = int(b_deg * n / 360.0)
            vals = list(ranges[a:b])
        vals = [v for v in vals if v != float('inf') and not math.isnan(v)]
        return sum(vals) / len(vals) if vals else float('inf')
    def dist_to_state(self, d, t):
        B = self.bounds
        if t == 'front': avail = ['TOO_CLOSE','CLOSE','MEDIUM','FAR']
        elif t == 'right': avail = ['TOO_CLOSE','CLOSE','MEDIUM','FAR','TOO_FAR']
        else: avail = ['CLOSE','FAR']
        for s in avail:
            lo, hi = B[s]
            if lo <= d < hi: return s
        return 'TOO_FAR' if t == 'right' else 'FAR'
    def determine_state(self, scan):
        L  = self.get_sector_avg(scan, *self.L_SECTOR)
        F  = self.get_sector_avg(scan, *self.F_SECTOR)
        RF = self.get_sector_avg(scan, *self.RF_SECTOR)
        R  = self.get_sector_avg(scan, *self.R_SECTOR)
        s = ( self.dist_to_state(L,'left'),
              self.dist_to_state(F,'front'),
              self.dist_to_state(RF,'right_front'),
              self.dist_to_state(R,'right') )
        return s, (L, F, RF, R)

    # ====================== Action exec ======================
    def publish_twist(self, tw: Twist):
        self.cmd_pub.publish(tw)
        tws = TwistStamped()
        tws.header.stamp = self.get_clock().now().to_msg()
        tws.twist = tw
        self.cmd_pub_stamped.publish(tws)
    def execute(self, name: str):
        lin, ang = self.actions[name]
        tw = Twist()
        tw.linear.x  = float(max(min(lin, self.max_linear), -self.max_linear))
        tw.angular.z = float(ang)
        self.publish_twist(tw)
    def kickstart_nudge(self):
        try:
            tw = Twist(); tw.linear.x = 0.12; tw.angular.z = 0.0
            for _ in range(2):
                self.publish_twist(tw); time.sleep(0.15)
        except Exception:
            pass

    # ====================== RL core ======================
    def eps_greedy(self, s):
        if np.random.rand() < self.epsilon:
            return np.random.randint(len(self.action_names))
        return int(np.argmax(self.q_table[s]))
    def collided(self, dists, s) -> bool:
        front = dists[1] if not math.isinf(dists[1]) else 10.0
        return (front < self.collision_thresh) or (s[1] == 'TOO_CLOSE')
    def reward(self, dists, s, a_idx):
        if self.collided(dists, s):
            return self.collision_penalty
        if self.reward_mode == 'sparse':
            return self.step_penalty
        r_state = s[3]
        band = {'MEDIUM':1.0,'CLOSE':0.4,'FAR':0.4,'TOO_CLOSE':-1.0,'TOO_FAR':-0.8}[r_state]
        is_forward = (self.action_names[a_idx] == 'FORWARD')
        progress = 0.6 + (self.forward_bonus if is_forward else 0.0)
        ang = self.actions[self.action_names[a_idx]][1]
        if abs(ang) > 0.6:
            dir_sign = math.copysign(1, ang)
            if dir_sign == self._last_turn_dir: self._spin_count += 1
            else: self._spin_count = 1
            self._last_turn_dir = dir_sign
        else:
            self._spin_count = 0; self._last_turn_dir = 0.0
        spin_term = self.spin_penalty if self._spin_count > self.spin_threshold else 0.0
        return band + progress + self.step_penalty + spin_term
    def td_update_sarsa(self, s, a, r, s2, a2):
        q = self.q_table[s][a]
        target = r + self.gamma * self.q_table[s2][a2]
        self.q_table[s][a] += self.alpha * (target - q)
    def td_update_terminal(self, s, a, r):
        self.q_table[s][a] += self.alpha * (r - self.q_table[s][a])

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
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
        self._stuck_ref_pose = None
        self._last_move_time = self.now_sec()
        self.get_logger().info(f"[TRAIN] Episode {self.ep} start | ε={self.epsilon:.3f} | reset_mode={self.reset_mode}")
    def end_episode(self, reason='timeout'):
        self.get_logger().info(f"[TRAIN] Ep{self.ep} end | steps={self.step} return={self.ep_return:.1f} | {reason}")
        try:
            self.save_qtable()
            new = not LOG_PATH.exists()
            with open(LOG_PATH, 'a', newline='') as f:
                w = csv.writer(f)
                if new:
                    w.writerow(['episode','return','steps','reason','algorithm'])
                w.writerow([self.ep, round(self.ep_return,2), self.step, reason, self.algorithm])
        except Exception:
            pass
        if self.ep < self.episodes:
            self.start_episode()
        else:
            self.execute('STOP')

    # ====================== STUCK detection ======================
    def _stuck_check_and_handle(self) -> bool:
        if self.turn_ctrl.active:
            self._stuck_ref_pose = self.current_xy() or self._stuck_ref_pose
            self._last_move_time = self.now_sec()
            return False
        cur = self.current_xy()
        if cur is None: return False
        now = self.now_sec()
        if self._stuck_ref_pose is None:
            self._stuck_ref_pose = cur; self._last_move_time = now; return False
        dx = cur[0]-self._stuck_ref_pose[0]; dy = cur[1]-self._stuck_ref_pose[1]
        dist = math.hypot(dx, dy)
        if dist > self.stuck_pos_eps:
            self._stuck_ref_pose = cur; self._last_move_time = now; return False
        if (now - self._last_move_time) >= self.stuck_timeout_s:
            self.get_logger().warn(f"[STUCK] No XY progress > {self.stuck_timeout_s:.1f}s (≈{dist:.3f}m). Teleporting and ending episode.")
            p = self.start_pose
            self.teleport(p['x'], p['y'], p['yaw'], force=True)
            self.end_episode('stuck')
            return True
        return False

    # ====================== Main callback ======================
    def scan_cb(self, msg: LaserScan):
        # Stop training immediately on goal (avoid post-goal pollution)
        if self.training and self.stop_on_goal:
            reached, dist = self.at_goal()
            if reached:
                if self.prev_state is not None and self.prev_action is not None:
                    self.td_update_terminal(self.prev_state, self.prev_action, self.goal_reward)
                self.ep_return += self.goal_reward
                self.episodes = self.ep
                self.get_logger().info(f"[SUCCESS] Goal reached (d≈{dist:.2f} m). Stopping training.")
                self.end_episode('success')
                return

        # Build discrete state & distances
        s, d = self.determine_state(msg.ranges)
        L, F, RF, R = d
        self.left_d, self.front_d, self.right_d = L, F, R

        # Macro turn in progress?
        tcmd = self.turn_ctrl.step()
        if tcmd is not None:
            self.publish_twist(tcmd); return

        # STUCK watchdog
        if self._stuck_check_and_handle():
            return

        # Junction supervisor may trigger a committed turn
        f, l, r = self.supervisor.update(L, F, R)
        event = self.supervisor.maybe_turn(f, l, r, self.turn_ctrl,
                                           front_block=self.front_block,
                                           side_block=self.side_block,
                                           left_open=self.left_open)
        if event is not None:
            stop = Twist(); stop.linear.x = 0.0; stop.angular.z = 0.0
            self.publish_twist(stop); return

        # --- Select action (SARSA policy) ---
        if self.first_random:
            a = np.random.randint(len(self.action_names))
            self.first_random = False
        else:
            if self.sticky_steps_left > 0 and self.last_action_idx is not None:
                a = self.last_action_idx
                self.sticky_steps_left -= 1
            else:
                a = self.eps_greedy(s)
                if self.action_names[a] in ('LEFT','RIGHT'):
                    self.sticky_steps_left = 6
                    self.last_action_idx = a

        # Execute chosen action
        self.execute(self.action_names[a])

        # Reward
        r = self.reward(d, s, a)
        self.ep_return += r

        # ---- Collision: terminal update + reset episode ----
        if self.collided(d, s):
            if self.prev_state is not None and self.prev_action is not None:
                self.td_update_terminal(self.prev_state, self.prev_action, r)
            stop = Twist(); stop.linear.x = 0.0; stop.angular.z = 0.0
            self.publish_twist(stop)
            p = self.start_pose
            self.teleport(p['x'], p['y'], p['yaw'], force=True)
            self.end_episode('collision')
            return

        # ---- Normal SARSA (on-policy) update ----
        # Choose a' from s' according to current policy (ε-greedy)
        a_next = self.eps_greedy(s)
        if self.prev_state is not None and self.prev_action is not None:
            self.td_update_sarsa(self.prev_state, self.prev_action, r, s, a_next)

        self.prev_state, self.prev_action = s, a
        self.step += 1

        # Timeout?
        if self.step >= self.steps_per_episode:
            self.end_episode('timeout'); return

        # Autosave
        if self.step % 500 == 0:
            try:
                self.save_qtable()
                self.get_logger().info(f"[TRAIN] autosave at step {self.step}, return={self.ep_return:.1f}")
            except Exception:
                pass

# ====================== CLI / Main ======================
def parse_args(argv):
    p = argparse.ArgumentParser()
    p.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    p.add_argument('--algorithm', type=str, default='sarsa', choices=['q_learning', 'sarsa'])
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

    node = SarsaWallFollower(
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
