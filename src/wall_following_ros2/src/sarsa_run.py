#!/usr/bin/env python3
import sys, math, time, argparse
from pathlib import Path
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist, TwistStamped

SAVE_SARSA = Path.home() / '.ros' / 'wf_sarsa.numpy'   # trained SARSA table

def clamp(v, lo, hi): return lo if v < lo else hi if v > hi else v

# ---------- gentle, time-bounded turn controller (no odom/yaw needed) ----------
class TurnController:
    def __init__(self, ang_speed=0.35, min_ticks=8, max_ticks=140):
        self.base_speed = float(ang_speed)
        self.min_ticks = int(min_ticks)
        self.max_ticks = int(max_ticks)
        self.active = False
        self._dir = 0.0
        self._ticks = 0
    def start(self, deg):  # +90, -90, 180
        if self.active: return
        self._dir  = 1.0 if deg > 0 else -1.0
        self._ticks = 0
        self.active = True
    def cancel(self): self.active = False; self._dir = 0.0; self._ticks = 0
    def step(self):
        if not self.active: return None
        self._ticks += 1
        tw = Twist()
        if self._ticks >= self.max_ticks:
            self.cancel(); return tw  # stop
        k = 1.0 if self._ticks < (0.6 * self.max_ticks) else 0.55
        tw.linear.x = 0.0
        tw.angular.z = self.base_speed * k * self._dir
        return tw

# ---------- tiny EMA for junction debounce ----------
class EMA:
    def __init__(self, alpha=0.25, init=None):
        self.alpha = alpha; self.s = init
    def update(self, x):
        self.s = x if self.s is None else self.alpha * x + (1 - self.alpha) * self.s
        return self.s

class JunctionSupervisor:
    """Left-90 at corners; 180 at dead-ends (debounced)."""
    def __init__(self, min_consec=2):
        self.front_ema = EMA(0.25); self.left_ema = EMA(0.25); self.right_ema = EMA(0.25)
        self._open_left_count = 0; self._dead_end_count = 0; self._min_consec = int(min_consec)
    def update(self, L, F, R):
        return self.front_ema.update(F), self.left_ema.update(L), self.right_ema.update(R)
    def maybe_turn(self, f, l, r, turn_ctrl: TurnController,
                   front_block=0.40, side_block=0.40, left_open=0.90):
        # corner: front blocked and left open (we prefer left turns)
        if f < front_block and l > left_open: self._open_left_count += 1
        else: self._open_left_count = 0
        # dead-end: front blocked and both sides blocked
        if f < front_block and l < side_block and r < side_block: self._dead_end_count += 1
        else: self._dead_end_count = 0
        if not turn_ctrl.active:
            if self._dead_end_count >= self._min_consec:
                turn_ctrl.start(180.0); return 'u_turn'
            if self._open_left_count >= self._min_consec:
                turn_ctrl.start(90.0);  return 'left_90'
        return None

# ====================== SARSA greedy runner ======================
class SarsaRun(Node):
    def __init__(self, max_linear=0.50):
        super().__init__('sarsa_run')

        # --- actions (match trainer) ---
        self.max_linear = float(max_linear)
        self.actions = {
            'FORWARD': (0.45,  0.0),
            'LEFT':    (0.22,  0.60),
            'RIGHT':   (0.22, -0.60),
            'STOP':    (0.0,   0.0),
        }
        self.action_names = list(self.actions.keys())

        # --- lidar discretization (match trainer) ---
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

        # thresholds (mirror trainer)
        self.collision_front_thresh = 0.20
        self.front_block_thresh     = 0.40
        self.side_detect_thresh     = 2.00
        self.right_scrape_thresh    = 0.55  # avoid turning right when already close

        # build state space
        self.all_states = []
        for l in self.L_STATES:
            for f in self.F_STATES:
                for rf in self.RF_STATES:
                    for r in self.R_STATES:
                        self.all_states.append((l, f, rf, r))

        # --- load SARSA table only ---
        if not SAVE_SARSA.exists():
            raise FileNotFoundError(f"No trained SARSA table at {SAVE_SARSA}. Run training first.")
        with open(SAVE_SARSA, 'rb') as f:
            data = np.load(f, allow_pickle=True).item()
        self.q_table = {s:[0.0,0.0,0.0,0.0] for s in self.all_states}
        if isinstance(data, dict):
            for k, v in data.items():
                if k in self.q_table and len(v) == 4:
                    self.q_table[k] = list(map(float, v))
        self.get_logger().info(f"[Q] Loaded SARSA policy from {SAVE_SARSA}")

        # --- ROS I/O ---
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_cb, qos_profile_sensor_data)
        self.cmd_pub  = self.create_publisher(Twist, '/cmd_vel', 10)
        self.cmd_pub_stamped = self.create_publisher(TwistStamped, '/cmd_vel_stamped', 10)

        # junction handling + time-bounded macro turns (no odom dependency)
        self.turn_ctrl = TurnController(ang_speed=0.35, min_ticks=8, max_ticks=140)
        self.supervisor = JunctionSupervisor(min_consec=2)

        # control/deadman
        self._have_scan = False
        self._last_scan_time = 0.0
        self._last_cmd = 'STOP'
        self._startup_end = time.time() + 0.25  # tiny nudge only
        self.create_timer(0.05, self._control_tick)  # 20 Hz

        self._last_debug = 0.0
        self.get_logger().info("[RUN] SARSA runner ready (right-wall, left-at-corner, U-turn at dead-end).")

    # ---------- control loop (keeps robot moving smoothly) ----------
    def _control_tick(self):
        now = time.time()
        tcmd = self.turn_ctrl.step()
        if tcmd is not None:
            self._publish_twist(tcmd); self._last_cmd = 'TURNING'; return
        if now < self._startup_end:
            self._publish_action('FORWARD'); return
        if (not self._have_scan) or (now - self._last_scan_time) > 0.6:
            # deadman: keep rolling forward gently until scans flow
            self._publish_action('FORWARD'); return
        # otherwise, repeat last command set by scan_cb
        if self._last_cmd != 'TURNING':
            self._publish_action(self._last_cmd)

    # ---------- pub helpers ----------
    def _publish_twist(self, tw: Twist):
        self.cmd_pub.publish(tw)
        ts = TwistStamped()
        ts.header.stamp = self.get_clock().now().to_msg()
        ts.twist = tw
        self.cmd_pub_stamped.publish(ts)
    def _publish_action(self, name: str):
        lin, ang = self.actions[name]
        tw = Twist()
        tw.linear.x  = clamp(lin, -self.max_linear, self.max_linear)
        tw.angular.z = float(ang)
        self._publish_twist(tw)
        self._last_cmd = name

    # ---------- state helpers ----------
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
        if which == 'front':  avail = ['TOO_CLOSE','CLOSE','MEDIUM','FAR']
        elif which == 'right': avail = ['TOO_CLOSE','CLOSE','MEDIUM','FAR','TOO_FAR']
        else: avail = ['CLOSE','FAR']
        for s in avail:
            lo, hi = B[s]
            if lo <= d < hi: return s
        return 'TOO_FAR' if which == 'right' else 'FAR'

    def _determine_state(self, scan):
        L  = self._sector_avg(scan, *self.L_SECTOR)
        F  = self._sector_avg(scan, *self.F_SECTOR)
        RF = self._sector_avg(scan, *self.RF_SECTOR)
        R  = self._sector_avg(scan, *self.R_SECTOR)
        s = ( self._dist_to_state(L,'left'),
              self._dist_to_state(F,'front'),
              self._dist_to_state(RF,'right_front'),
              self._dist_to_state(R,'right') )
        return s, (L, F, RF, R)

    # ---------- action mask (mirrors trainer) ----------
    def _build_action_mask(self, s, d, front_blocked):
        L, F, RF, R = d
        left_state, front_state, _, right_state = s

        right_present  = (R < self.side_detect_thresh) and not math.isinf(R)
        left_present   = (L < self.side_detect_thresh) and not math.isinf(L)

        left_too_close  = (left_state in ('TOO_CLOSE','CLOSE')) or (L < 0.55)
        right_too_close = (right_state == 'TOO_CLOSE') or (R < self.right_scrape_thresh)

        # only allow LEFT at a real corner: front blocked AND right wall present
        corner = front_blocked and right_present

        allow = {name: True for name in self.action_names}
        # 1) block FORWARD whenever front is blocked
        if front_blocked:
            allow['FORWARD'] = False
        # 2) allow LEFT only at a corner; never if left is already close
        if not corner or left_too_close:
            allow['LEFT'] = False
        # 3) avoid RIGHT if right wall is already very close (prevent scraping)
        if right_too_close:
            allow['RIGHT'] = False

        return [allow[a] for a in self.action_names]

    # ---------- greedy over allowed actions ----------
    def _greedy_masked(self, s, allowed_mask):
        idxs = [i for i, ok in enumerate(allowed_mask) if ok]
        if not idxs:
            return self.action_names.index('STOP'), self.q_table[s]
        qvals = self.q_table[s]
        best = max(idxs, key=lambda i: qvals[i])
        return int(best), qvals

    # ---------- main callback ----------
    def scan_cb(self, msg: LaserScan):
        self._have_scan = True
        self._last_scan_time = time.time()

        # macro-turn in progress?
        tcmd = self.turn_ctrl.step()
        if tcmd is not None:
            self._publish_twist(tcmd); self._last_cmd = 'TURNING'; return

        # build state
        s, d = self._determine_state(msg.ranges)
        L, F, RF, R = d
        Fmin = self._sector_min(msg.ranges, 345, 15)   # narrow front min for reflex

        # junction management (left at corner, u-turn at dead-end)
        f_s, l_s, r_s = self.supervisor.update(L, F, R)
        event = self.supervisor.maybe_turn(f_s, l_s, r_s, self.turn_ctrl,
                                           front_block=self.front_block_thresh,
                                           side_block=0.40,
                                           left_open=0.90)
        if event is not None:
            self._publish_action('STOP'); return

        # determine "front blocked" for masking/reflex
        front_blocked = (Fmin < self.front_block_thresh) or (s[1] in ('TOO_CLOSE','CLOSE'))

        # if front is blocked and no macro-turn started, do a simple reflex:
        if front_blocked and not self.turn_ctrl.active:
            # dead-end reflex
            if (L < 0.40 and R < 0.40):
                self.turn_ctrl.start(180.0)
                self._publish_action('STOP'); return
            # corner-left reflex (prefer left turns)
            if (L > 0.90) and (R < self.side_detect_thresh):
                self.turn_ctrl.start(90.0)
                self._publish_action('STOP'); return

        # choose best allowed action from SARSA table
        allowed = self._build_action_mask(s, d, front_blocked)
        a, qvals = self._greedy_masked(s, allowed)
        name = self.action_names[a]
        self._publish_action(name)

        # minimal debug ~2 Hz
        now = time.time()
        if now - self._last_debug > 0.5:
            self._last_debug = now
            self.get_logger().info(f"[RUN] state={s} q={[round(float(x),2) for x in qvals]} allowed={allowed} -> {name}")

# ---------- CLI ----------
def parse_args(argv):
    p = argparse.ArgumentParser()
    p.add_argument('--max_linear', type=float, default=0.50)
    args, _ = p.parse_known_args(argv[1:])
    return args

def main(argv=None):
    rclpy.init(args=argv)
    args = parse_args(sys.argv)
    node = SarsaRun(max_linear=args.max_linear)
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
