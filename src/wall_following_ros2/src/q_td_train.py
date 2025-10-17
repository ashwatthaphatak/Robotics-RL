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

# -------- Persistence --------
SAVE_PATH = Path.home() / '.ros' / 'wf_qtable.npy'
LOG_PATH  = Path.home() / '.ros' / 'wf_train_log.csv'

# -------- Gazebo guesses --------
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

# ====================== Gentle turn controller ======================
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

# ====================== Trainer ======================
class QLearningWallFollower(Node):
    """
    Q-Learning trainer that rewards following the RIGHT wall (LiDAR-only).
    On collision (from LiDAR), it teleports to start and starts a new episode.
    A control timer keeps sending commands even before the first scan arrives.
    """
    def __init__(self,
                 mode='train',
                 algorithm='q_learning',
                 reward_mode='right_wall',
                 reset_mode='once',
                 goal_x=2.6, goal_y=3.1, goal_r=0.5,
                 episodes=999999, steps_per_episode=1800,
                 alpha=0.30, gamma=0.95,
                 epsilon=0.30, epsilon_decay=0.997, epsilon_min=0.05):
        super().__init__('q_td_train')

        # ---- Config ----
        self.mode          = 'train'
        self.algorithm     = 'q_learning'
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
            'LEFT':    (0.22,  0.60),
            'RIGHT':   (0.22, -0.60),
            'STOP':    (0.0,   0.0),
        }
        self.action_names = list(self.actions.keys())

        # LiDAR thresholds
        self.collision_front_thresh = 0.20
        self.front_block_thresh     = 0.40
        self.side_detect_thresh     = 2.00

        # reward (LiDAR-only)
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

        # ---- Teleport clients (probe both worlds) ----
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

        # Turn controller (gentle)
        self.turn_ctrl = TurnController(self._dummy_yaw, ang_speed=0.35, tol_deg=6.0, min_ticks=6, max_ticks=150)

        # 20 Hz control loop (keeps publishing even before scans arrive)
        self.control_timer = self.create_timer(0.05, self.control_tick)

        # Begin
        self.start_episode()
        self.get_logger().info("Q-Learning trainer ready: follow RIGHT wall (LiDAR-only reward).")

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
                data = np.load(SAVE_PATH, allow_pickle=True).item()
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
            np.save(SAVE_PATH, self.q_table)
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

    def eps_greedy(self, s):
        if np.random.rand() < self.epsilon:
            return np.random.randint(len(self.action_names))
        return int(np.argmax(self.q_table[s]))

    # ====================== Reward (LiDAR-only) ======================
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

    # ====================== TD update ======================
    def td_update(self, s, a, r, s2):
        qsa = self.q_table[s][a]
        target = r + self.gamma * max(self.q_table[s2])
        self.q_table[s][a] = qsa + self.alpha * (target - qsa)

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

    # ====================== Optional goal stop ======================
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

        # keep turning if in a macro-turn
        tcmd = self.turn_ctrl.step()
        if tcmd is not None:
            self.cmd_pub.publish(tcmd)
            self._last_cmd = 'TURNING'
            return

        # straight-first: drive forward even if no scans yet
        if now < self._startup_end_walltime:
            self.publish_action('FORWARD')
            return

        # deadman: if we haven't seen a scan recently, keep moving forward
        if (not self._have_scan) or (now - self._last_scan_time) > 0.6:
            self.publish_action('FORWARD')
            return

        # otherwise, scan_cb has already chosen an action and published it;
        # repeat the last command so Gazebo keeps moving smoothly
        self.publish_action(self._last_cmd if self._last_cmd != 'TURNING' else 'FORWARD')

    # ====================== Main callback ======================
    def scan_cb(self, msg: LaserScan):
        self._have_scan = True
        self._last_scan_time = time.time()
        self._sector_min_last = self._sector_min(msg.ranges, 345, 15)

        # discrete state
        s, d = self.determine_state(msg.ranges)

        # collision => immediate teleport + new episode
        if (self._sector_min_last < self.collision_front_thresh) or (s[1] == 'TOO_CLOSE'):
            self.ep_return += self.collision_penalty
            self.get_logger().warn("[COLLISION] detected — teleporting to start and starting new episode.")
            p = self.start_pose
            self.teleport(p['x'], p['y'], p['yaw'], force=True)
            self.end_episode('collision')
            return

        # optional goal stop
        if self.at_goal():
            self.get_logger().info("[GOAL] reached — stopping training.")
            self.end_episode('goal')
            return

        # policy step
        a = self.eps_greedy(s)
        self.publish_action(self.action_names[a])

        # reward + TD
        r = self.wall_follow_reward(d, s, a)
        self.ep_return += r
        s2, _ = self.determine_state(msg.ranges)
        self.td_update(s, a, r, s2)
        self.prev_state, self.prev_action = s, a
        self.step += 1

        # timeout
        if self.step >= self.steps_per_episode:
            self.end_episode('timeout')
            return

        # autosave
        if self.step % 500 == 0:
            self.save_qtable()
            self.get_logger().info(f"[TRAIN] autosave at step {self.step}, return={self.ep_return:.1f}")

# ====================== CLI / Main ======================
def parse_args(argv):
    p = argparse.ArgumentParser()
    p.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    p.add_argument('--algorithm', type=str, default='q_learning', choices=['q_learning', 'sarsa'])
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
        try: node.destroy_node()
        except Exception: pass
        rclpy.shutdown()

if __name__ == '__main__':
    main()
