#!/usr/bin/env python3
import os, sys, math, csv, time, argparse
from pathlib import Path
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist, Point, Quaternion
from tf2_ros import Buffer, TransformListener
from ros_gz_interfaces.srv import SetEntityPose

# -------- Persistence (SARSA) --------
SAVE_PATH = Path.home() / '.ros' / 'wf_sarsa.numpy'   # keep this exact filename (saved via file handle)
LOG_PATH  = Path.home() / '.ros' / 'wf_sarsa_train_log.csv'

# -------- Gazebo guesses (same approach as q_td_train) --------
ENTITY_NAME_CANDIDATES = [
    'turtlebot3_burger_0',
    'turtlebot3_burger',
    'burger',
    'turtlebot3',
]
WORLD_NAME_CANDIDATES = ['default', 'empty']

def clamp(v, lo, hi): return lo if v < lo else hi if v > hi else v

def quat_to_yaw(q: Quaternion) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)

# ====================== Gentle turn controller (fixed-duration via dummy yaw) ======================
class TurnController:
    def __init__(self, get_yaw, ang_speed=0.35, tol_deg=6.0, min_ticks=6, max_ticks=150):
        self._yaw = get_yaw
        self.base_speed = float(ang_speed)
        self.tol = math.radians(float(tol_deg))
        self.min_ticks = int(min_ticks)
        self.max_ticks = int(max_ticks)
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
        self._ticks = 0
        self.active = True
    def cancel(self): self.active = False; self._target = None; self._ticks = 0
    def step(self):
        if not self.active: return None
        self._ticks += 1
        cur = self._yaw()
        err = self._ang_err(cur, self._target)
        at_goal = (abs(err) < self.tol) and (self._ticks >= self.min_ticks)
        timeout = self._ticks >= self.max_ticks
        tw = Twist()
        if at_goal or timeout:
            self.cancel(); tw.linear.x = 0.0; tw.angular.z = 0.0; return tw
        k = clamp(abs(err) / (math.pi / 3.0), 0.25, 1.0)
        tw.linear.x = 0.0; tw.angular.z = self.base_speed * k * (1.0 if err > 0 else -1.0)
        return tw

# ====================== SARSA Trainer (on-policy; right-wall; LiDAR-only) ======================
class SarsaWallFollower(Node):
    """
    Follow the RIGHT wall; take LEFT at real corners; take U-TURN at dead ends.
    On collision (from LiDAR), teleport to start and start a new episode.
    Uses Expected SARSA with a delayed update (uses next scan as s').
    """
    def __init__(self,
                 mode='train',
                 algorithm='sarsa',
                 reward_mode='right_wall',
                 reset_mode='once',
                 goal_x=2.6, goal_y=3.1, goal_r=0.5,
                 episodes=999999, steps_per_episode=1800,
                 alpha=0.30, gamma=0.95,
                 epsilon=0.30, epsilon_decay=0.997, epsilon_min=0.05):
        super().__init__('sarsa_train')

        # ---- Config ----
        self.mode          = 'train'
        self.algorithm     = 'sarsa'
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

        # ---- Motion & safety ----
        self.max_linear = 0.50
        self.actions = {
            'FORWARD': (0.45,  0.0),
            'LEFT':    (0.22,  0.50),   # gentler
            'RIGHT':   (0.22, -0.50),   # only for trimming back to right wall
            'STOP':    (0.0,   0.0),
        }
        self.action_names = list(self.actions.keys())

        # LiDAR thresholds
        self.collision_front_thresh = 0.20
        self.front_block_thresh     = 0.40
        self.side_detect_thresh     = 2.00
        self.side_block_thresh      = 0.40
        self.left_open_thresh       = 0.90

        # Reward (LiDAR-only right-wall band)
        self.step_penalty       = -0.02
        self.collision_penalty  = -350.0
        self.forward_bonus      = 0.10
        self.right_medium_lo    = 0.70
        self.right_medium_hi    = 1.20

        # straight-first phase
        self.startup_forward_sec   = 2.5
        self._startup_end_walltime = time.time() + self.startup_forward_sec

        # control-loop / deadman
        self._last_cmd = 'STOP'
        self._have_scan = False
        self._last_scan_time = 0.0

        # ---- Discretization ----
        self.setup_state_space()

        # ---- Q-table ----
        self.q_table = {s: [0.0, 0.0, 0.0, 0.0] for s in self.all_states}
        f_idx = self.action_names.index('FORWARD')
        for s in self.all_states:
            self.q_table[s][f_idx] += 0.25
        self.try_load_qtable()

        # ---- ROS I/O ----
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_cb, qos_profile_sensor_data)
        self.cmd_pub  = self.create_publisher(Twist, '/cmd_vel', 10)

        # TF only for optional goal-stop
        self.tf_buffer   = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ---- Teleport clients (probe worlds) ----
        self._pose_clients = []
        for wname in WORLD_NAME_CANDIDATES:
            cli = self.create_client(SetEntityPose, f'/world/{wname}/set_pose')
            cli.wait_for_service(timeout_sec=0.5)
            self._pose_clients.append((wname, cli))
        self._active_world = None
        self._active_client = None
        self.entity_name = None

        # start pose
        self.start_pose = {'x': -2.0, 'y': -3.2, 'yaw': 0.0}

        # Gentle turn controller (dummy yaw -> fixed-duration turns)
        self.turn_ctrl = TurnController(self._dummy_yaw, ang_speed=0.35, tol_deg=6.0, min_ticks=6, max_ticks=150)

        # 20 Hz control loop
        self.control_timer = self.create_timer(0.05, self.control_tick)

        # SARSA delayed-update buffers
        self.prev_state  = None
        self.prev_action = None

        # Begin
        self.start_episode()
        self.get_logger().info("SARSA trainer: follow RIGHT wall; LEFT at corners; U-TURN at dead ends (LiDAR-only).")

    # ====================== State space ======================
    def setup_state_space(self):
        self.L_STATES  = ['CLOSE', 'FAR']
        self.F_STATES  = ['TOO_CLOSE', 'CLOSE', 'MEDIUM', 'FAR']
        self.RF_STATES = ['CLOSE', 'FAR']
        self.R_STATES  = ['TOO_CLOSE', 'CLOSE', 'MEDIUM', 'FAR', 'TOO_FAR']
        self.bounds = {
            'TOO_CLOSE': (0.0, 0.60),
            'CLOSE':     (0.60, 0.90),
            'MEDIUM':    (0.90, 1.50),
            'FAR':       (1.50, 2.50),
            'TOO_FAR':   (2.50, float('inf')),
        }
        self.L_SECTOR  = (80, 100)
        self.F_SECTOR  = (355, 5)
        self.RF_SECTOR = (310, 320)
        self.R_SECTOR  = (260, 280)
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
                with open(SAVE_PATH, 'rb') as f:
                    data = np.load(f, allow_pickle=True).item()
                if isinstance(data, dict):
                    for k, v in data.items():
                        if k in self.q_table and len(v) == 4:
                            self.q_table[k] = list(map(float, v))
                    self.get_logger().info(f"[Q] Warm-start from {SAVE_PATH}")
                    return True
            except Exception as e:
                self.get_logger().warning(f"[Q] Failed to load table: {e}")
        return False
    def save_qtable(self):
        try:
            SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(SAVE_PATH, 'wb') as f:
                np.save(f, self.q_table)   # keep exact filename
        except Exception as e:
            self.get_logger().warning(f"[Q] Failed to save table: {e}")

    # ====================== Gazebo teleport (robust) ======================
    def _call_set_pose(self, client, name: str, x: float, y: float, yaw: float):
        req = SetEntityPose.Request()
        req.entity.name = name
        req.pose.position = Point(x=float(x), y=float(y), z=0.05)
        qz, qw = math.sin(yaw/2.0), math.cos(yaw/2.0)
        req.pose.orientation = Quaternion(x=0.0, y=0.0, z=qz, w=qw)
        fut = client.call_async(req)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=2.0)
        if not fut.done(): return False, "service call timeout"
        res = fut.result()
        if res is None:   return False, "no response"
        return bool(getattr(res, 'success', False)), getattr(res, 'status_message', '')
    def resolve_entity_and_world(self):
        if self._active_client and self.entity_name: return True
        ready = [(w, c) for (w, c) in self._pose_clients if c.service_is_ready()]
        if not ready:
            self.get_logger().warning("[RESET] No /world/*/set_pose services are ready.")
            return False
        for name in ENTITY_NAME_CANDIDATES:
            for (w, cli) in ready:
                ok, _ = self._call_set_pose(cli, name, 0.0, 0.0, 0.0)
                if ok:
                    self._active_world = w; self._active_client = cli; self.entity_name = name
                    self.get_logger().info(f"[RESET] Using world='{w}', entity='{name}'")
                    return True
        self.get_logger().warning("[RESET] Could not find valid entity in any world.")
        return False
    def teleport(self, x: float, y: float, yaw: float, force=False):
        if (not force) and self.reset_mode == 'none': return
        if not self.resolve_entity_and_world():
            self.get_logger().warning("[RESET] Teleport skipped: unresolved entity/world.")
            return
        ok, msg = self._call_set_pose(self._active_client, self.entity_name, x, y, yaw)
        if not ok:
            self._active_client = None; self.entity_name = None
            if self.resolve_entity_and_world():
                ok2, msg2 = self._call_set_pose(self._active_client, self.entity_name, x, y, yaw)
                if not ok2: self.get_logger().warning(f"[RESET] Teleport failed after re-probe: {msg2}")
            else:
                self.get_logger().warning(f"[RESET] Teleport failed: {msg}")
        else:
            time.sleep(0.6)  # settle
        # after a teleport, ensure forward drive restarts immediately
        self._startup_end_walltime = time.time() + self.startup_forward_sec
        self._last_cmd = 'FORWARD'
        self.prev_state = None
        self.prev_action = None

    # ====================== LiDAR helpers ======================
    def _sector_avg(self, ranges, a_deg, b_deg):
        n = len(ranges)
        if n == 0: return float('inf')
        if a_deg > b_deg:
            a = int(a_deg * n / 360.0)
            vals = list(ranges[a:]) + list(ranges[:int(b_deg * n / 360.0)])
        else:
            a = int(a_deg * n / 360.0); b = int(b_deg * n / 360.0)
            vals = list(ranges[a:b])
        vals = [v for v in vals if v != float('inf') and not math.isnan(v)]
        return float(sum(vals) / len(vals)) if vals else float('inf')
    def _sector_min(self, ranges, a_deg, b_deg):
        n = len(ranges)
        if n == 0: return float('inf')
        if a_deg > b_deg:
            a = int(a_deg * n / 360.0)
            vals = list(ranges[a:]) + list(ranges[:int(b_deg * n / 360.0)])
        else:
            a = int(a_deg * n / 360.0); b = int(b_deg * n / 360.0)
            vals = list(ranges[a:b])
        vals = [v for v in vals if not math.isnan(v)]
        return float(min(vals)) if vals else float('inf')
    def _dist_to_state(self, d, which):
        B = self.bounds
        if which == 'front':
            avail = ['TOO_CLOSE', 'CLOSE', 'MEDIUM', 'FAR']
        elif which == 'right':
            avail = ['TOO_CLOSE', 'CLOSE', 'MEDIUM', 'FAR', 'TOO_FAR']
        else:
            avail = ['CLOSE', 'FAR']
        for s in avail:
            lo, hi = B[s]
            if lo <= d < hi: return s
        return 'TOO_FAR' if which == 'right' else 'FAR'
    def determine_state(self, ranges):
        L  = self._sector_avg(ranges, *self.L_SECTOR)
        F  = self._sector_avg(ranges, *self.F_SECTOR)
        RF = self._sector_avg(ranges, *self.RF_SECTOR)
        R  = self._sector_avg(ranges, *self.R_SECTOR)
        s = ( self._dist_to_state(L, 'left'),
              self._dist_to_state(F, 'front'),
              self._dist_to_state(RF,'right_front'),
              self._dist_to_state(R, 'right') )
        return s, (L, F, RF, R)

    # ====================== Exec & policy ======================
    def publish_action(self, name: str):
        lin, ang = self.actions[name]
        tw = Twist()
        tw.linear.x  = clamp(lin, -self.max_linear, self.max_linear)
        tw.angular.z = float(ang)
        self.cmd_pub.publish(tw)
        self._last_cmd = name

    def eps_greedy_masked(self, s, allowed_mask):
        idxs = [i for i, ok in enumerate(allowed_mask) if ok]
        if not idxs:
            return self.action_names.index('FORWARD')
        if np.random.rand() < self.epsilon:
            return int(np.random.choice(idxs))
        qvals = self.q_table[s]
        return int(max(idxs, key=lambda i: qvals[i]))

    def build_action_mask(self, s, d, Fmin):
        """
        Follow-right policy:
          - FORWARD allowed unless front is blocked.
          - LEFT allowed when right is too close (move away) OR at committed corner (handled separately).
          - RIGHT allowed only when right is far/too-far (trim back to wall).
        """
        L, F, RF, R = d
        left_state, front_state, _, right_state = s

        front_blocked  = (Fmin < self.front_block_thresh) or (front_state in ('TOO_CLOSE','CLOSE'))

        allow = {name: True for name in self.action_names}
        allow['FORWARD'] = not front_blocked

        # trim away from left wall if already close
        if left_state in ('TOO_CLOSE','CLOSE'):
            allow['LEFT'] = False  # we'll go forward until corner/U-turn

        # stay off the right wall
        if right_state in ('TOO_CLOSE','CLOSE'):
            allow['RIGHT'] = False
        # only use RIGHT to come back to the wall
        if right_state not in ('FAR','TOO_FAR'):
            allow['RIGHT'] = False

        # allow LEFT only as gentle correction when the right wall is too close
        allow['LEFT'] = (right_state in ('TOO_CLOSE','CLOSE')) and (not front_blocked)

        return [allow[a] for a in self.action_names]

    # ====================== Reward (LiDAR-only; same idea as q_td_train) ======================
    def wall_follow_reward(self, dists, s, a_idx):
        Fmin = self._sector_min_last
        if Fmin < self.collision_front_thresh or s[1] == 'TOO_CLOSE':
            return self.collision_penalty
        r_state = s[3]
        band = {
            'MEDIUM':  +1.0,
            'CLOSE':   +0.3,
            'FAR':     +0.3,
            'TOO_CLOSE': -0.9,
            'TOO_FAR':   -0.6,
        }[r_state]
        progress = self.forward_bonus if self.action_names[a_idx] == 'FORWARD' else 0.0
        return band + progress + self.step_penalty

    # ====================== Expected SARSA TD (on-policy) ======================
    def td_expected_sarsa(self, s_prev, a_prev, r, s_cur, allowed_mask_s_cur):
        idxs = [i for i, ok in enumerate(allowed_mask_s_cur) if ok]
        if not idxs:
            exp_q = 0.0
        else:
            qvals = self.q_table[s_cur]
            best = max(idxs, key=lambda i: qvals[i])
            nA = len(idxs)
            pi = np.zeros(len(self.action_names), dtype=np.float64)
            for i in idxs:
                pi[i] = self.epsilon / nA
            pi[best] += (1.0 - self.epsilon)
            exp_q = float(np.dot(pi, qvals))
        qsa = self.q_table[s_prev][a_prev]
        self.q_table[s_prev][a_prev] = qsa + self.alpha * (r + self.gamma * exp_q - qsa)

    def td_terminal(self, s_prev, a_prev, r):
        qsa = self.q_table[s_prev][a_prev]
        self.q_table[s_prev][a_prev] = qsa + self.alpha * (r - qsa)

    # ====================== Episodes ======================
    def start_episode(self):
        self.ep = getattr(self, 'ep', 0) + 1
        self.step = 0
        self.ep_return = 0.0
        self.prev_state = None
        self.prev_action = None
        self._startup_end_walltime = time.time() + self.startup_forward_sec
        if self.reset_mode == 'episode' or (self.reset_mode == 'once' and self.ep == 1):
            p = self.start_pose
            self.teleport(p['x'], p['y'], p['yaw'])
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
        self.get_logger().info(f"[TRAIN] Episode {self.ep} start | ε={self.epsilon:.3f}")

    def end_episode(self, reason='timeout'):
        self.get_logger().info(f"[TRAIN] Ep{self.ep} end | steps={self.step} return={self.ep_return:.1f} | {reason}")
        try:
            self.save_qtable()
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
            self.publish_action('STOP')

    # ====================== Optional goal stop (same as q_td_train) ======================
    def at_goal(self) -> bool:
        try:
            tf = self.tf_buffer.lookup_transform('map', 'base_footprint', rclpy.time.Time())
            x = tf.transform.translation.x
            y = tf.transform.translation.y
            gx, gy, gr = self.goal
            return (x - gx) ** 2 + (y - gy) ** 2 <= gr ** 2
        except Exception:
            return False

    def _dummy_yaw(self) -> float:
        return 0.0

    # ====================== CONTROL LOOP ======================
    def control_tick(self):
        now = time.time()
        tcmd = self.turn_ctrl.step()
        if tcmd is not None:
            self.cmd_pub.publish(tcmd)
            self._last_cmd = 'TURNING'
            return
        if now < self._startup_end_walltime:
            self.publish_action('FORWARD'); return
        if (not self._have_scan) or (now - self._last_scan_time) > 0.6:
            self.publish_action('FORWARD'); return
        self.publish_action(self._last_cmd if self._last_cmd != 'TURNING' else 'FORWARD')

    # ====================== Main callback: delayed Expected SARSA + supervisor ======================
    def scan_cb(self, msg: LaserScan):
        self._have_scan = True
        self._last_scan_time = time.time()
        self._sector_min_last = self._sector_min(msg.ranges, 345, 15)

        # If we're mid-turn, keep turning and don't learn (prevents corrupt updates)
        tcmd = self.turn_ctrl.step()
        if tcmd is not None:
            self.cmd_pub.publish(tcmd)
            self._last_cmd = 'TURNING'
            return

        # Current observation
        s_cur, d_cur = self.determine_state(msg.ranges)
        Fmin = self._sector_min_last

        # ---- (A) compute reward for the *previous* transition and update
        if self.prev_action is not None:
            r = self.wall_follow_reward(d_cur, s_cur, self.prev_action)
            # collision terminal?
            if (Fmin < self.collision_front_thresh) or (s_cur[1] == 'TOO_CLOSE'):
                self.td_terminal(self.prev_state, self.prev_action, r)
                self.ep_return += r
                self.get_logger().warn("[COLLISION] detected — teleporting to start and starting new episode.")
                p = self.start_pose
                self.teleport(p['x'], p['y'], p['yaw'], force=True)
                self.end_episode('collision')
                return
            allowed_mask_cur = self.build_action_mask(s_cur, d_cur, Fmin)
            self.td_expected_sarsa(self.prev_state, self.prev_action, r, s_cur, allowed_mask_cur)
            self.ep_return += r
            self.step += 1
            if self.step >= self.steps_per_episode:
                self.end_episode('timeout'); return

        # ---- (B) supervisor: take LEFT at real corners / U-TURN at dead ends
        front_blocked = (Fmin < self.front_block_thresh) or (s_cur[1] in ('TOO_CLOSE','CLOSE'))
        right_present = (d_cur[3] < self.side_detect_thresh) and not math.isinf(d_cur[3])
        left_present  = (d_cur[0] < self.side_detect_thresh) and not math.isinf(d_cur[0])
        left_open     = (d_cur[0] > self.left_open_thresh) or math.isinf(d_cur[0])

        dead_end = front_blocked and (d_cur[0] < self.side_block_thresh) and (d_cur[3] < self.side_block_thresh)
        left_corner = front_blocked and left_open and right_present

        if dead_end:
            self.publish_action('STOP')
            self.turn_ctrl.start(180.0)
            self.prev_state = None; self.prev_action = None
            return
        if left_corner:
            self.publish_action('STOP')
            self.turn_ctrl.start(90.0)
            self.prev_state = None; self.prev_action = None
            return

        # ---- (C) choose next on-policy action under mask (follow-right)
        allowed_mask_cur = self.build_action_mask(s_cur, d_cur, Fmin)
        a_cur = self.eps_greedy_masked(s_cur, allowed_mask_cur)
        self.publish_action(self.action_names[a_cur])

        # set prev for next scan
        self.prev_state  = s_cur
        self.prev_action = a_cur

        # Autosave occasionally
        if (self.step % 500) == 0 and self.step > 0:
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
    p.add_argument('--reward_mode', type=str, default='right_wall', choices=['right_wall'])
    p.add_argument('--reset_mode', type=str, default='once', choices=['none', 'once', 'episode'])
    p.add_argument('--goal_x', type=float, default=2.6)
    p.add_argument('--goal_y', type=float, default=3.1)
    p.add_argument('--goal_r', type=float, default=0.5)
    p.add_argument('--episodes', type=int, default=999999)
    p.add_argument('--steps_per_episode', type=int, default=1800)
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
        try: node.destroy_node()
        except Exception: pass
        rclpy.shutdown()

if __name__ == '__main__':
    main()
