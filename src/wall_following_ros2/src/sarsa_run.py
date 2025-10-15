#!/usr/bin/env python3
import sys, math, time, argparse
from pathlib import Path
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time

from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist, TwistStamped, Quaternion
from nav_msgs.msg import Odometry
from tf2_ros import Buffer, TransformListener

# -------- Persistence --------
SAVE_SARSA = Path.home() / '.ros' / 'wf_sarsa.numpy'   # primary for SARSA
SAVE_Q     = Path.home() / '.ros' / 'wf_qtable.npy'    # fallback (Q-learning)

# ====================== Helpers ======================
def quat_to_yaw(q: Quaternion) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)

class EMA:
    def __init__(self, alpha=0.25, init=None):
        self.alpha = alpha; self.s = init
    def update(self, x):
        self.s = x if self.s is None else self.alpha * x + (1 - self.alpha) * self.s
        return self.s

class TurnController:
    """Macro 90°/180° turns, so we don’t get interrupted by LiDAR every tick."""
    def __init__(self, yaw_provider, ang_speed=2.0, tol_deg=8.0, min_ticks=4, max_ticks=120):
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
        stop = Twist(); stop.linear.x = 0.0; stop.angular.z = 0.0
        if (abs(err) < self.tol and self._ticks >= self.min_ticks) or (self._ticks >= self.max_ticks):
            self.cancel(); return stop
        k = max(0.25, min(1.0, abs(err) / (math.pi/4)))
        cmd = Twist(); cmd.linear.x = 0.0
        cmd.angular.z = self.base_speed * k * (1.0 if err > 0 else -1.0)
        return cmd

class JunctionSupervisor:
    """Trigger a left-90 at junctions and U-turn at dead-ends (debounced)."""
    def __init__(self, min_consec=2):
        self.front_ema = EMA(0.25); self.left_ema = EMA(0.25); self.right_ema = EMA(0.25)
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

# ====================== Run (greedy) node ======================
class SarsaRun(Node):
    def __init__(self,
                 reward_mode='shaped',
                 max_linear=0.50):
        super().__init__('sarsa_run')

        # --- Motion & discretization (match your trainers) ---
        self.max_linear = float(max_linear)
        self.actions = {
            'FORWARD': (0.50,  0.0),
            'LEFT':    (0.22,  0.985),
            'RIGHT':   (0.22, -0.985),
            'STOP':    (0.0,   0.0),
        }
        self.action_names = list(self.actions.keys())

        self.L_STATES  = ['CLOSE', 'FAR']
        self.F_STATES  = ['TOO_CLOSE', 'CLOSE', 'MEDIUM', 'FAR']
        self.RF_STATES = ['CLOSE', 'FAR']
        self.R_STATES  = ['TOO_CLOSE', 'CLOSE', 'MEDIUM', 'FAR', 'TOO_FAR']
        self.bounds = {  # same as q_td_train.py & sarsa_train.py (harmonized)
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

        # Build full state space
        self.all_states = []
        for l in self.L_STATES:
            for f in self.F_STATES:
                for rf in self.RF_STATES:
                    for r in self.R_STATES:
                        self.all_states.append((l, f, rf, r))

        # --- Q-table load: prefer SARSA file; fallback to Q-learning; else zeros ---
        self.q_table = {s:[0.0, 0.0, 0.0, 0.0] for s in self.all_states}
        loaded_path = None
        for path in (SAVE_SARSA, SAVE_Q):
            if path.exists():
                try:
                    data = np.load(path, allow_pickle=True).item()
                    if isinstance(data, dict):
                        self.q_table.update(data)
                        loaded_path = str(path)
                        break
                except Exception as e:
                    self.get_logger().warn(f"[Q] Failed loading {path}: {e}")
        if loaded_path:
            self.get_logger().info(f"[Q] Loaded table: {loaded_path}")
        else:
            self.get_logger().warn("[Q] No Q-table found; running with zeros (will mostly pick FORWARD fallback).")

        # --- ROS I/O ---
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_cb, qos_profile_sensor_data)
        self.cmd_pub  = self.create_publisher(Twist, '/cmd_vel', 10)
        self.cmd_pub_stamped = self.create_publisher(TwistStamped, '/cmd_vel_stamped', 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_cb, 20)

        # Odometry + helpers (for macro turns)
        self._last_odom = None
        self.turn_ctrl = TurnController(self.current_yaw, ang_speed=2.0, tol_deg=8.0, min_ticks=4, max_ticks=120)
        self.supervisor = JunctionSupervisor(min_consec=2)

        # Sticky small turns to avoid dithering
        self.sticky_steps_left = 0
        self.last_action_idx   = None

        # Startup forward drive to overcome initial idle states
        self.STARTUP_DRIVE_SECS = 2.5
        self.STARTUP_MIN_SCANS  = 3
        self._startup_end_time  = time.time() + self.STARTUP_DRIVE_SECS
        self._got_scan_count    = 0
        self._startup_timer = self.create_timer(0.05, self._startup_tick)  # 20 Hz

        self._last_debug = 0.0
        self.get_logger().info("[RUN] SARSA greedy run ready (no teleportation in test mode).")

    # ---------- Odometry ----------
    def odom_cb(self, msg: Odometry):
        self._last_odom = msg
    def current_yaw(self) -> float:
        if self._last_odom is None: return 0.0
        return quat_to_yaw(self._last_odom.pose.pose.orientation)

    # ---------- Startup open-loop ----------
    def _startup_tick(self):
        if (self._got_scan_count >= self.STARTUP_MIN_SCANS) or (time.time() >= self._startup_end_time):
            self._startup_timer.cancel()
            self.get_logger().info("[RUN] Startup forward drive finished; handing to greedy policy.")
            return
        tw = Twist(); tw.linear.x = 0.30; tw.angular.z = 0.0
        self.publish_twist(tw)

    # ---------- Utils ----------
    def publish_twist(self, tw: Twist):
        self.cmd_pub.publish(tw)
        tws = TwistStamped()
        tws.header.stamp = self.get_clock().now().to_msg()
        tws.twist = tw
        self.cmd_pub_stamped.publish(tws)

    # ---------- State helpers ----------
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
        if t == 'front':  avail = ['TOO_CLOSE','CLOSE','MEDIUM','FAR']
        elif t == 'right': avail = ['TOO_CLOSE','CLOSE','MEDIUM','FAR','TOO_FAR']
        else:             avail = ['CLOSE','FAR']
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

    # ---------- Main ----------
    def scan_cb(self, msg: LaserScan):
        self._got_scan_count += 1

        # Continue committed macro turn if active
        tcmd = self.turn_ctrl.step()
        if tcmd is not None:
            self.publish_twist(tcmd); return

        # Build state
        s, d = self.determine_state(msg.ranges)
        L, F, RF, R = d

        # Junction supervisor may start a turn
        f, l, r = self.supervisor.update(L, F, R)
        event = self.supervisor.maybe_turn(f, l, r, self.turn_ctrl,
                                           front_block=0.45, side_block=0.40, left_open=0.90)
        if event is not None:
            stop = Twist(); stop.linear.x = 0.0; stop.angular.z = 0.0
            self.publish_twist(stop)
            return

        # Greedy choice with safe fallback (if all Q's equal, go forward)
        qvals = self.q_table.get(s, [0.0, 0.0, 0.0, 0.0])
        if (max(qvals) - min(qvals)) < 1e-6:
            a = self.action_names.index('FORWARD')
        else:
            a = int(np.argmax(qvals))

        # Sticky turns for a few ticks to avoid chattering
        if self.sticky_steps_left > 0 and self.last_action_idx is not None:
            a = self.last_action_idx
            self.sticky_steps_left -= 1
        else:
            name = self.action_names[a]
            if name in ('LEFT', 'RIGHT'):
                self.sticky_steps_left = 6
                self.last_action_idx = a

        # Execute
        name = self.action_names[a]
        tw = Twist()
        tw.linear.x  = float(max(min(self.actions[name][0], self.max_linear), -self.max_linear))
        tw.angular.z = float(self.actions[name][1])
        self.publish_twist(tw)

        # Debug ~2 Hz
        now = time.time()
        if now - self._last_debug > 0.5:
            self._last_debug = now
            self.get_logger().info(f"[RUN] state={s} q={[round(float(x),2) for x in qvals]} -> {name}")

# ====================== CLI / Main ======================
def parse_args(argv):
    p = argparse.ArgumentParser()
    p.add_argument('--reward_mode', type=str, default='shaped', choices=['sparse','shaped'])
    p.add_argument('--max_linear', type=float, default=0.50)
    args, _ = p.parse_known_args(argv[1:])
    return args

def main(argv=None):
    rclpy.init(args=argv)
    args = parse_args(sys.argv)
    node = SarsaRun(reward_mode=args.reward_mode, max_linear=args.max_linear)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass

if __name__ == '__main__':
    main()
