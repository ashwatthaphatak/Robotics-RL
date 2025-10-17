#!/usr/bin/env python3
import sys, math, time, argparse
from pathlib import Path
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist

# -------- Persistence (SARSA only) --------
SAVE_PATH = Path.home() / '.ros' / 'wf_sarsa.numpy'

def clamp(v, lo, hi): return lo if v < lo else hi if v > hi else v

class SarsaRun(Node):
    """
    Inference-only runner that follows the right wall using the SARSA Q-table.
    - No learning, no odom/teleport/TF.
    - Same action set / discretization as training.
    - Corner logic forces LEFT; RIGHT only on true right openings.
    - Linear slowdown near obstacles; angular stays strong for turns.
    """
    def __init__(self, max_linear=0.50, debug=False):
        super().__init__('sarsa_run')

        self.debug = bool(debug)

        # ---- Motion & actions (match training magnitudes) ----
        self.max_linear = float(max_linear)
        self.actions = {
            'FORWARD': (0.45,  0.00),
            'LEFT':    (0.20,  0.45),
            'RIGHT':   (0.20, -0.45),
            'STOP':    (0.00,  0.00),
        }
        self.action_names = list(self.actions.keys())

        # ---- LiDAR thresholds ----
        self.collision_front_thresh = 0.22
        self.slowdown_front_thresh  = 0.60
        # earlier corner detection so we commit LEFT in time
        self.front_block_thresh     = 0.60
        self.side_detect_thresh     = 2.00

        # ---- Startup push (same idea as training) ----
        self.startup_forward_sec   = 1.0
        self._startup_end_walltime = time.time() + self.startup_forward_sec

        # ---- Discretization (must match training) ----
        self.setup_state_space()

        # ---- Load Q-table (SARSA) ----
        self.q_table = {s: [0.0, 0.0, 0.0, 0.0] for s in self.all_states}
        self.try_load_qtable()

        # ---- ROS I/O ----
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_cb, qos_profile_sensor_data)
        self.cmd_pub  = self.create_publisher(Twist, '/cmd_vel', 10)

        # ---- Runtime state ----
        self._sector_min_last   = float('inf')
        self._have_scan         = False
        self._last_scan_time    = 0.0
        self._last_cmd_name     = 'STOP'
        self.sticky_steps_left  = 0
        self.sticky_action_idx  = None

        # 20 Hz keep-alive / startup push
        self.control_timer = self.create_timer(0.05, self.control_tick)

        self.get_logger().info("[RUN] SARSA inference ready.")

    # ====================== Discretization ======================
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
        if not SAVE_PATH.exists():
            self.get_logger().warn(f"[Q] {SAVE_PATH} not found. Running with zeros.")
            return False
        try:
            with open(SAVE_PATH, 'rb') as f:
                data = np.load(f, allow_pickle=True).item()
            if isinstance(data, dict):
                updated = 0
                for k, v in data.items():
                    if (k in self.q_table) and (isinstance(v, (list, np.ndarray))) and (len(v) == 4):
                        self.q_table[k] = list(map(float, v))
                        updated += 1
                self.get_logger().info(f"[Q] Loaded SARSA table from {SAVE_PATH} ({updated} states).")
                return True
        except Exception as e:
            self.get_logger().warn(f"[Q] Failed to load {SAVE_PATH}: {e}")
        return False

    # ====================== LiDAR helpers ======================
    def _sector(self, ranges, a_deg, b_deg):
        n = len(ranges)
        if n == 0: return []
        if a_deg > b_deg:
            a = int(a_deg * n / 360.0)
            return list(ranges[a:]) + list(ranges[:int(b_deg * n / 360.0)])
        a = int(a_deg * n / 360.0); b = int(b_deg * n / 360.0)
        return list(ranges[a:b])

    def _sector_avg(self, ranges, a_deg, b_deg):
        vals = [v for v in self._sector(ranges, a_deg, b_deg) if v != float('inf') and not math.isnan(v)]
        return float(sum(vals) / len(vals)) if vals else float('inf')

    def _sector_min(self, ranges, a_deg, b_deg):
        vals = [v for v in self._sector(ranges, a_deg, b_deg) if not math.isnan(v)]
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
        s = ( self._dist_to_state(L,'left'),
              self._dist_to_state(F,'front'),
              self._dist_to_state(RF,'right_front'),
              self._dist_to_state(R,'right') )
        return s, (L, F, RF, R)

    # ====================== Policy helpers ======================
    def build_action_mask(self, s, d, Fmin):
        L, F, RF, R = d
        left_state, front_state, _, right_state = s

        front_blocked  = (Fmin < self.front_block_thresh) or (front_state in ('TOO_CLOSE','CLOSE'))
        right_present  = (R < self.side_detect_thresh) and not math.isinf(R)
        # opening only if front is CLEAR
        right_open     = (not right_present) and (right_state in ('FAR','TOO_FAR')) and (not front_blocked)

        left_too_close  = (left_state in ('TOO_CLOSE','CLOSE')) or (L < 0.55)
        right_too_close = (right_state == 'TOO_CLOSE') or (R < 0.55)

        allow = {n: True for n in self.action_names}

        corner = front_blocked and right_present
        if corner:
            # FORCE a left turn at real corners
            allow['LEFT'] = True
            allow['RIGHT'] = False
        else:
            # away from corners, don’t scrape left or right
            if left_too_close:
                allow['LEFT'] = False
            # RIGHT only for real openings and not scraping
            if (not right_open) or right_too_close:
                allow['RIGHT'] = False

        # If very close ahead, block FORWARD
        if Fmin < (self.collision_front_thresh + 0.05):
            allow['FORWARD'] = False

        return [allow[a] for a in self.action_names], corner

    def greedy_masked(self, s, allowed_mask):
        idxs = [i for i, ok in enumerate(allowed_mask) if ok]
        if not idxs:
            return self.action_names.index('STOP')
        qvals = self.q_table.get(s, [0.0, 0.0, 0.0, 0.0])
        best = max(idxs, key=lambda i: qvals[i])
        # prefer FORWARD if basically tied
        if (max(qvals[i] for i in idxs) - min(qvals[i] for i in idxs)) < 1e-9:
            if self.action_names.index('FORWARD') in idxs:
                return self.action_names.index('FORWARD')
        return int(best)

    # ====================== Exec ======================
    def publish_action(self, name: str, lin_scale: float = 1.0):
        base_lin, base_ang = self.actions[name]
        lin = clamp(base_lin * lin_scale, -self.max_linear, self.max_linear)
        ang = base_ang  # keep angular strong for turning near obstacles
        tw = Twist(); tw.linear.x = float(lin); tw.angular.z = float(ang)
        self.cmd_pub.publish(tw)
        self._last_cmd_name = name

    # ====================== Keepalive / Startup ======================
    def control_tick(self):
        now = time.time()
        if now < self._startup_end_walltime:
            self.publish_action('FORWARD')
            return
        if (not self._have_scan) or (now - self._last_scan_time) > 0.6:
            self.publish_action('FORWARD')
            return
        self.publish_action(self._last_cmd_name)

    # ====================== Main callback ======================
    def scan_cb(self, msg: LaserScan):
        self._have_scan = True
        self._last_scan_time = time.time()
        self._sector_min_last = self._sector_min(msg.ranges, 345, 15)

        s, d = self.determine_state(msg.ranges)
        L, F, RF, R = d

        # --- Collision reflex: gentle LEFT arc rather than STOP ---
        if (self._sector_min_last < self.collision_front_thresh) or (s[1] == 'TOO_CLOSE'):
            self.sticky_steps_left = 10
            self.sticky_action_idx = self.action_names.index('LEFT')
            self.publish_action('LEFT', lin_scale=0.25)
            if self.debug: self.get_logger().info("[RUN] Reflex LEFT (front hit).")
            return

        # --- Mask actions & detect corner ---
        allowed_mask, is_corner = self.build_action_mask(s, d, self._sector_min_last)

        # --- Continue sticky if active ---
        if self.sticky_steps_left > 0 and self.sticky_action_idx is not None:
            a = self.sticky_action_idx
            self.sticky_steps_left -= 1
        else:
            # Force LEFT at “real” right corner regardless of Q
            if is_corner and allowed_mask[self.action_names.index('LEFT')]:
                a = self.action_names.index('LEFT')
                self.sticky_action_idx = a
                self.sticky_steps_left = 8
                if self.debug: self.get_logger().info("[RUN] Commit LEFT (corner).")
            else:
                # Right opening → short RIGHT arc to stick with wall
                right_present = (R < self.side_detect_thresh) and not math.isinf(R)
                right_open = (not right_present) and (s[3] in ('FAR','TOO_FAR')) and (s[1] in ('MEDIUM','FAR'))
                if right_open and allowed_mask[self.action_names.index('RIGHT')]:
                    a = self.action_names.index('RIGHT')
                    self.sticky_action_idx = a
                    self.sticky_steps_left = 6
                    if self.debug: self.get_logger().info("[RUN] Commit RIGHT (right opening).")
                else:
                    a = self.greedy_masked(s, allowed_mask)

        # --- Slowdown linear near obstacles (keep angular unchanged) ---
        Fmin = self._sector_min_last
        lin_scale = 1.0 if Fmin >= self.slowdown_front_thresh else clamp((Fmin - 0.25) / (self.slowdown_front_thresh - 0.25), 0.2, 1.0)

        # Execute
        name = self.action_names[a]
        self.publish_action(name, lin_scale=lin_scale)

        if self.debug:
            q = [round(float(x), 3) for x in self.q_table.get(s, [0.0]*4)]
            self.get_logger().info(f"[RUN] state={s} q={q} -> {name} (lin_scale={lin_scale:.2f})")

# ====================== CLI / Main ======================
def parse_args(argv):
    p = argparse.ArgumentParser()
    p.add_argument('--max_linear', type=float, default=0.50)
    p.add_argument('--debug', action='store_true')
    args, _ = p.parse_known_args(argv[1:])
    return args

def main(argv=None):
    rclpy.init(args=argv)
    args = parse_args(sys.argv)
    node = SarsaRun(max_linear=args.max_linear, debug=args.debug)
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
