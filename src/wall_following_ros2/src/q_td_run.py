#!/usr/bin/env python3
import time, math
from pathlib import Path
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist, TwistStamped, Quaternion
from nav_msgs.msg import Odometry

# -------- Persistence --------
SAVE_PATH = Path.home() / '.ros' / 'wf_qtable.npy'

# ====================== Helpers ======================
def quat_to_yaw(q: Quaternion) -> float:
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
    """Commit to a target yaw delta (e.g., +90°, 180°); ignores LiDAR until done."""
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

# ====================== Supervisor (junctions) ======================
class JunctionSupervisor:
    """Trigger left or U-turns at junctions/dead-ends; otherwise let greedy policy act."""
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

# ====================== RUN node (greedy from learned Q) ======================
class QRun(Node):
    def __init__(self):
        super().__init__('q_td_run')

        # Fixed action ordering (do NOT rely on dict key order)
        self.action_names = ('FORWARD','LEFT','RIGHT','STOP')
        self.action_index = {n:i for i,n in enumerate(self.action_names)}
        self.max_linear = 0.28
        self.actions = {
            'FORWARD': (0.50, 0.0),
            'LEFT':    (0.22, 0.985),
            'RIGHT':   (0.22, -0.985),
            'STOP':    (0.0,  0.0)
        }

        # Discretization (match trainer)
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

        # Enumerate states
        self.all_states = [(l,f,rf,r)
                           for l in self.L_STATES
                           for f in self.F_STATES
                           for rf in self.RF_STATES
                           for r in self.R_STATES]

        # Q-table
        self.q_table = {s: [0.0, 0.0, 0.0, 0.0] for s in self.all_states}
        # small forward bias in case table missing:
        self.q_table[self.all_states[0]][self.action_index['FORWARD']] += 0.5
        self._loaded = self.try_load_qtable(SAVE_PATH)
        self.summarize_qtable()

        # ROS I/O
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_cb, qos_profile_sensor_data)
        self.cmd_topics = [
            '/cmd_vel',
            '/model/turtlebot3_burger/cmd_vel',
            '/model/turtlebot3_burger_0/cmd_vel',
            '/turtlebot3/cmd_vel',
        ]
        self.cmd_pubs = [self.create_publisher(Twist, t, 10) for t in self.cmd_topics]
        self.cmd_pub_stamped = self.create_publisher(TwistStamped, '/cmd_vel_stamped', 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_cb, 20)

        self._last_odom = None
        self._last_pose = None
        self._ever_moved = False

        # Turn helpers (match trainer)
        self.turn_ctrl = TurnController(self.current_yaw, ang_speed=0.7, tol_deg=4.0, min_ticks=8, max_ticks=120)
        self.supervisor = JunctionSupervisor(min_consec=3)

        # Sticky small turns (avoid chatter)
        self.sticky_steps_left = 0
        self.last_action_idx   = None

        # Startup nudge
        self.STARTUP_MAX_SECS  = 8.0
        self.STARTUP_MIN_SCANS = 3
        self._startup_end_time = time.time() + self.STARTUP_MAX_SECS
        self._got_scan_count   = 0
        self._startup_timer = self.create_timer(0.05, self._startup_tick)  # 20 Hz

        # Watchdog (cmd_vel subscriber counts)
        self._watchdog_timer = self.create_timer(1.0, self._watchdog_tick)
        self._watchdog_prints = 0

        self._last_debug = 0.0
        self.get_logger().info("[RUN] Ready | Greedy with tie-breaks | startup nudge + cmd_vel watchdog enabled")

    # ---------- startup open-loop ----------
    def _startup_tick(self):
        now = time.time()
        if self._ever_moved or (self._got_scan_count >= self.STARTUP_MIN_SCANS) or (now >= self._startup_end_time):
            self._startup_timer.cancel()
            self.get_logger().info("[RUN] Startup finished; handing control to greedy policy.")
            return
        tw = Twist(); tw.linear.x = 0.30; tw.angular.z = 0.0
        self.publish_twist_all(tw)

    # ---------- watchdog ----------
    def _watchdog_tick(self):
        counts = [p.get_subscription_count() for p in self.cmd_pubs]
        moved = False
        if self._last_pose is not None and self._last_odom is not None:
            x0, y0 = self._last_pose
            x1 = self._last_odom.pose.pose.position.x
            y1 = self._last_odom.pose.pose.position.y
            if (x1-x0)*(x1-x0) + (y1-y0)*(y1-y0) > 1e-4:
                moved = True; self._ever_moved = True
        if self._last_odom is not None:
            self._last_pose = (
                self._last_odom.pose.pose.position.x,
                self._last_odom.pose.pose.position.y
            )
        if self._watchdog_prints < 10:
            pairs = [f"{t}:{c}" for t,c in zip(self.cmd_topics, counts)]
            self.get_logger().info("[WATCHDOG] cmd_vel subs -> " + " | ".join(pairs) + f" | moved={moved}")
            self._watchdog_prints += 1
        if all(c == 0 for c in counts):
            self.get_logger().warn("[WATCHDOG] No subscribers on any cmd_vel topic — check ros_gz_bridge parameter_bridge wiring.")

    # ---------- ROS callbacks ----------
    def odom_cb(self, msg: Odometry):
        self._last_odom = msg

    def current_yaw(self) -> float:
        if self._last_odom is None: return 0.0
        return quat_to_yaw(self._last_odom.pose.pose.orientation)

    def scan_cb(self, msg: LaserScan):
        self._got_scan_count += 1

        # 1) Macro turn in progress?
        tcmd = self.turn_ctrl.step()
        if tcmd is not None:
            self.publish_twist_all(tcmd); return

        # 2) Discrete state from LiDAR
        s, d = self.determine_state(msg.ranges)
        L, F, RF, R = d

        # 3) Supervisor (junctions / dead-ends)
        f, l, r = self.supervisor.update(L, F, R)
        event = self.supervisor.maybe_turn(f, l, r, self.turn_ctrl,
                                           front_block=0.45, side_block=0.40, left_open=0.90)
        if event is not None:
            stop = Twist(); stop.linear.x = 0.0; stop.angular.z = 0.0
            self.publish_twist_all(stop); return

        # 4) Greedy action with robust tie-breaks
        qvals = self.q_table.get(s, [0.0, 0.0, 0.0, 0.0])
        qarr = np.array(qvals, dtype=float)
        maxq = float(np.max(qarr)); minq = float(np.min(qarr))
        all_equal = (abs(maxq - minq) < 1e-9)

        if all_equal:
            # If front is clear, go forward; else pick side with more free space
            if s[1] in ('MEDIUM','FAR') and math.isfinite(F) and F > 0.7:
                a = self.action_index['FORWARD']
                self.get_logger().info("[RUN] Tie-break -> FORWARD (front clear)")
            else:
                a = self.choose_turn_by_space(L, R)
                self.get_logger().info("[RUN] Tie-break -> " + self.action_names[a])
        else:
            a = int(np.argmax(qarr))

        # Never STOP unless front is blocked
        if self.action_names[a] == 'STOP' and s[1] in ('MEDIUM','FAR'):
            a = self.action_index['FORWARD']

        # small stickiness for LEFT/RIGHT to avoid chattering
        if self.sticky_steps_left > 0 and self.last_action_idx is not None:
            a = self.last_action_idx
            self.sticky_steps_left -= 1
        else:
            if self.action_names[a] in ('LEFT','RIGHT'):
                self.sticky_steps_left = 6
                self.last_action_idx = a

        self.execute(self.action_names[a])

        # debug ~2 Hz
        now = time.time()
        if now - self._last_debug > 0.5:
            self._last_debug = now
            self.get_logger().info(f"[RUN] state={s} q={[round(float(x),2) for x in qvals]} -> {self.action_names[a]}")

    # ---------- Q-table ----------
    def try_load_qtable(self, path: Path) -> bool:
        if path.exists():
            try:
                data = np.load(path, allow_pickle=True).item()
                if isinstance(data, dict):
                    self.q_table = data
                    self.get_logger().info(f"[Q] Loaded table from {path}")
                    return True
                else:
                    self.get_logger().warn(f"[Q] File not a dict: {path}")
            except Exception as e:
                self.get_logger().warn(f"[Q] Failed to load table: {e}")
        else:
            self.get_logger().warn(f"[Q] Table not found at {path} — will still run with forward bias.")
        return False

    def summarize_qtable(self):
        arr = np.array(list(self.q_table.values()), dtype=float)
        if arr.size == 0:
            self.get_logger().warn("[Q] Table empty!"); return
        max_abs = np.max(np.abs(arr))
        nz = int(np.sum(np.abs(arr) > 1e-6))
        self.get_logger().info(f"[Q] Stats: max|Q|={max_abs:.3f}, nonzero_entries={nz}/{arr.size}")

    # ---------- publish/execute ----------
    def publish_twist_all(self, tw: Twist):
        for p in self.cmd_pubs:
            p.publish(tw)
        tws = TwistStamped()
        tws.header.stamp = self.get_clock().now().to_msg()
        tws.twist = tw
        self.cmd_pub_stamped.publish(tws)

    def execute(self, name: str):
        lin, ang = self.actions[name]
        tw = Twist()
        tw.linear.x  = float(max(min(lin, self.max_linear), -self.max_linear))
        tw.angular.z = float(ang)
        self.publish_twist_all(tw)

    # ---------- state helpers ----------
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
        s = (
            self.dist_to_state(L, 'left'),
            self.dist_to_state(F, 'front'),
            self.dist_to_state(RF, 'right_front'),
            self.dist_to_state(R, 'right'),
        )
        return s, (L, F, RF, R)

    # ---------- simple side choice when front blocked or tie-break ----------
    def choose_turn_by_space(self, L, R) -> int:
        # If left side is more open, turn LEFT, else RIGHT
        return self.action_index['LEFT'] if (not math.isinf(L) and L >= R) else self.action_index['RIGHT']

# ---------- Main ----------
def main(argv=None):
    rclpy.init(args=argv)
    node = QRun()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try: node.destroy_node()
        except Exception: pass
        try: rclpy.shutdown()
        except Exception: pass

if __name__ == '__main__':
    main()
