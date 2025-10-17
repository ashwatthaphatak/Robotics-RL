#!/usr/bin/env python3
import time, math
from pathlib import Path
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist, TwistStamped, Quaternion, Point
from nav_msgs.msg import Odometry
from ros_gz_interfaces.srv import SetEntityPose

SAVE_PATH = Path.home() / '.ros' / 'wf_qtable.npy'

ENTITY_NAME_CANDIDATES = [
    'turtlebot3_burger', 'turtlebot3_burger_0', 'burger', 'turtlebot3'
]

# -------- Quaternion helpers --------
def quat_to_rpy(q: Quaternion):
    sinr_cosp = 2.0 * (q.w*q.x + q.y*q.z)
    cosr_cosp = 1.0 - 2.0 * (q.x*q.x + q.y*q.y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (q.w*q.y - q.z*q.x)
    pitch = math.copysign(math.pi/2, sinp) if abs(sinp) >= 1.0 else math.asin(sinp)
    siny_cosp = 2.0 * (q.w*q.z + q.x*q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y*q.y + q.z*q.z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw

class EMA:
    def __init__(self, alpha=0.3, init=None):
        self.alpha = alpha; self.s = init
    def update(self, x):
        self.s = x if self.s is None else self.alpha*x + (1-self.alpha)*self.s
        return self.s

# -------- Macro turn controller --------
class TurnController:
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

# -------- Junction supervisor --------
class JunctionSupervisor:
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

# -------- RUN Node --------
class QRun(Node):
    def __init__(self):
        super().__init__('q_td_run')

        # Actions (match trainer’s caps)
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

        self.all_states = [(l,f,rf,r)
                           for l in self.L_STATES
                           for f in self.F_STATES
                           for rf in self.RF_STATES
                           for r in self.R_STATES]

        # Q-table
        self.q_table = {s: [0.0, 0.0, 0.0, 0.0] for s in self.all_states}
        self.try_load_qtable(SAVE_PATH); self.summarize_qtable()

        # Subscriptions
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_cb, qos_profile_sensor_data)

        # Multi-source odom bind
        self._active_odom_idx = None
        self._last_odom = None
        self._last_odom_t = 0.0
        self._odom_topics = [
            '/odom',
            '/model/turtlebot3_burger/odometry',
            '/model/turtlebot3_burger_0/odometry',
        ]
        self._odom_subs = [
            self.create_subscription(Odometry, t, self._mk_odom_cb(i), 20)
            for i, t in enumerate(self._odom_topics)
        ]

        # cmd_vel bind
        self.cmd_topics = [
            '/cmd_vel',
            '/model/turtlebot3_burger/cmd_vel',
            '/model/turtlebot3_burger_0/cmd_vel',
            '/turtlebot3/cmd_vel',
        ]
        self.cmd_pubs = [self.create_publisher(Twist, t, 10) for t in self.cmd_topics]
        self.cmd_pub_stamped = self.create_publisher(TwistStamped, '/cmd_vel_stamped', 10)
        self._active_pub_idx = None

        # Controllers / supervisors
        self.turn_ctrl = TurnController(self.current_yaw, ang_speed=0.7, tol_deg=4.0, min_ticks=8, max_ticks=120)
        self.supervisor = JunctionSupervisor(min_consec=3)

        # Reflex thresholds
        self.reflex_front_block = 0.45   # meters
        self.backoff_speed = -0.18
        self.backoff_ticks = 6           # ~0.3 s
        self._backoff_count = 0

        # Sticky small turns
        self.sticky_steps_left = 0
        self.last_action_idx   = None

        # Startup nudge
        self.STARTUP_MAX_SECS  = 8.0
        self.STARTUP_MIN_SCANS = 3
        self._startup_end_time = time.time() + self.STARTUP_MAX_SECS
        self._got_scan_count   = 0
        self._startup_timer = self.create_timer(0.05, self._startup_tick)

        # Watchdogs
        self._watchdog_timer = self.create_timer(1.0, self._watchdog_tick)
        self._watchdog_prints = 0

        # Spin recovery
        self._spin_ticks = 0
        self._spin_lin_eps = 0.02
        self._spin_ang_thresh = 0.6
        self._spin_tick_limit = 15

        # Topple / upright
        self.roll_thresh = 0.35
        self.pitch_thresh = 0.35
        self._last_upright_time = 0.0
        self._upright_cooldown_s = 3.0
        self.entity_name = None
        self.reset_cli = self.create_client(SetEntityPose, '/world/default/set_pose')

        self._last_debug = 0.0
        self.get_logger().info("[RUN] Ready | Greedy with reflex turns, spin/topple recovery, single-topic binding")

    # ---------- utilities ----------
    def now(self) -> float:
        return float(self.get_clock().now().nanoseconds) * 1e-9

    def _mk_odom_cb(self, idx):
        def _cb(msg: Odometry):
            self._last_odom = msg
            self._last_odom_t = self.now()
            if self._active_odom_idx is None:
                self._active_odom_idx = idx
                self.get_logger().info(f"[BIND] Using odometry: {self._odom_topics[idx]}")
        return _cb

    def current_yaw(self) -> float:
        if self._last_odom is None: return 0.0
        _, _, yaw = quat_to_rpy(self._last_odom.pose.pose.orientation)
        return yaw

    def _watchdog_tick(self):
        counts = [p.get_subscription_count() for p in self.cmd_pubs]
        if self._active_pub_idx is None:
            for i, c in enumerate(counts):
                if c > 0:
                    self._active_pub_idx = i
                    self.get_logger().info(f"[BIND] Using cmd_vel: {self.cmd_topics[i]}")
                    break
        if self._watchdog_prints < 8:
            pairs = [f"{t}:{c}" for t,c in zip(self.cmd_topics, counts)]
            self.get_logger().info("[WATCHDOG] cmd_vel subs -> " + " | ".join(pairs))
            self._watchdog_prints += 1
        # Odom freshness gate
        if self._last_odom is not None and (self.now() - self._last_odom_t) > 1.5:
            tw = Twist(); tw.linear.x = 0.0; tw.angular.z = 0.0
            self.publish_twist(tw)
            self.get_logger().warn("[WATCHDOG] No odom in 1.5s — stopping commands.")

    def _startup_tick(self):
        if (self._got_scan_count >= self.STARTUP_MIN_SCANS) or (time.time() >= self._startup_end_time):
            self._startup_timer.cancel()
            self.get_logger().info("[RUN] Startup finished; handing control to greedy policy.")
            return
        tw = Twist(); tw.linear.x = 0.30; tw.angular.z = 0.0
        self.publish_twist(tw)

    # ---------- topple / upright ----------
    def maybe_upright(self) -> bool:
        if self._last_odom is None: return False
        r,p,y = quat_to_rpy(self._last_odom.pose.pose.orientation)
        if abs(r) < self.roll_thresh and abs(p) < self.pitch_thresh:
            return False
        if (time.time() - self._last_upright_time) < self._upright_cooldown_s:
            return True
        if not self.reset_cli.service_is_ready():
            self.publish_twist(Twist()); self._last_upright_time = time.time()
            self.get_logger().warn("[UPRIGHT] Topple detected; service not ready — stopped.")
            return True
        if not self.entity_name:
            for name in ENTITY_NAME_CANDIDATES:
                if self._try_set_pose_probe(name):
                    self.entity_name = name
                    self.get_logger().info(f"[UPRIGHT] Using Gazebo entity: {name}")
                    break
        if not self.entity_name:
            self.publish_twist(Twist()); self._last_upright_time = time.time()
            self.get_logger().warn("[UPRIGHT] No entity — stopped.")
            return True
        px = float(self._last_odom.pose.pose.position.x)
        py = float(self._last_odom.pose.pose.position.y)
        qz = math.sin(y/2.0); qw = math.cos(y/2.0)
        req = SetEntityPose.Request()
        req.entity.name = self.entity_name
        req.pose.position = Point(x=px, y=py, z=0.05)
        req.pose.orientation = Quaternion(x=0.0, y=0.0, z=qz, w=qw)
        fut = self.reset_cli.call_async(req)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=2.0)
        self.publish_twist(Twist())
        self._last_upright_time = time.time()
        self.get_logger().warn("[UPRIGHT] Topple — uprighted in place.")
        return True

    def _try_set_pose_probe(self, name: str) -> bool:
        try:
            req = SetEntityPose.Request()
            req.entity.name = name
            req.pose.position = Point(x=0.0, y=0.0, z=0.05)
            req.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
            fut = self.reset_cli.call_async(req)
            rclpy.spin_until_future_complete(self, fut, timeout_sec=0.2)
            return fut.done() and (fut.result() is not None)
        except Exception:
            return False

    # ---------- publish ----------
    def publish_twist(self, tw: Twist):
        if self._active_pub_idx is not None:
            self.cmd_pubs[self._active_pub_idx].publish(tw)
        else:
            for p in self.cmd_pubs: p.publish(tw)
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

    # ---------- LiDAR helpers ----------
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

    # ---------- Main callback ----------
    def scan_cb(self, msg: LaserScan):
        self._got_scan_count += 1

        # Macro turn in progress?
        tcmd = self.turn_ctrl.step()
        if tcmd is not None:
            self.publish_twist(tcmd); return

        # Topple upright first
        if self.maybe_upright():
            return

        # Odom freshness gate
        if (self._last_odom is None) or ((self.now() - self._last_odom_t) > 1.5):
            self.publish_twist(Twist()); return

        # Discrete state + raw distances
        s, d = self.determine_state(msg.ranges)
        L, F, RF, R = d

        # -------- Reflex override: if front blocked, back off + 90° turn to open side --------
        if (not math.isinf(F) and F < self.reflex_front_block) or (s[1] in ('TOO_CLOSE', 'CLOSE')):
            # back-off a few ticks
            if self._backoff_count < self.backoff_ticks:
                tw = Twist(); tw.linear.x = self.backoff_speed; tw.angular.z = 0.0
                self.publish_twist(tw)
                self._backoff_count += 1
                return
            self._backoff_count = 0
            # choose side with more room; if equal prefer LEFT
            prefer_left = (math.isinf(R) and not math.isinf(L)) or (L >= R)
            delta = 90.0 if prefer_left else -90.0
            self.get_logger().warn("[REFLEX] Front blocked — backoff then 90° turn")
            self.turn_ctrl.start(delta)
            self.publish_twist(Twist())  # stop this tick
            return

        # Junction supervisor (dead-ends / open-left)
        f, l, r = self.supervisor.update(L, F, R)
        event = self.supervisor.maybe_turn(f, l, r, self.turn_ctrl,
                                           front_block=0.45, side_block=0.40, left_open=0.90)
        if event is not None:
            self.publish_twist(Twist()); return

        # Greedy from Q with tie-breaks
        qvals = self.q_table.get(s, [0.0, 0.0, 0.0, 0.0])
        qarr = np.array(qvals, dtype=float)
        maxq, minq = float(np.max(qarr)), float(np.min(qarr))
        all_equal = (abs(maxq - minq) < 1e-9)

        if all_equal:
            if s[1] in ('MEDIUM','FAR') and math.isfinite(F) and F > 0.7:
                a = self.action_index['FORWARD']
            else:
                a = self.action_index['LEFT'] if (not math.isinf(L) and L >= R) else self.action_index['RIGHT']
        else:
            a = int(np.argmax(qarr))

        # Never STOP unless genuinely needed (front very open -> prefer forward)
        if self.action_names[a] == 'STOP' and s[1] in ('MEDIUM','FAR'):
            a = self.action_index['FORWARD']

        # Sticky small turns
        if self.sticky_steps_left > 0 and self.last_action_idx is not None:
            a = self.last_action_idx
            self.sticky_steps_left -= 1
        else:
            if self.action_names[a] in ('LEFT','RIGHT'):
                self.sticky_steps_left = 6
                self.last_action_idx = a

        self.execute(self.action_names[a])

        # Spin-in-place recovery
        vx = abs(self._last_odom.twist.twist.linear.x)
        wz = self._last_odom.twist.twist.angular.z
        if vx < 0.02 and abs(wz) > 0.6:
            self._spin_ticks += 1
        else:
            self._spin_ticks = 0
        if self._spin_ticks >= self._spin_tick_limit and not self.turn_ctrl.active:
            prefer_left = L >= R
            delta = 90.0 if prefer_left else -90.0
            self.get_logger().warn("[RECOVERY] Spin-in-place -> committing 90° turn")
            self.turn_ctrl.start(delta)
            self._spin_ticks = 0
            self.publish_twist(Twist()); return

        # Debug ~2 Hz
        now = time.time()
        if now - self._last_debug > 0.5:
            self._last_debug = now
            self.get_logger().info(f"[RUN] state={s} q={[round(float(x),2) for x in qvals]} -> {self.action_names[a]}")

    # ---------- Q-table I/O ----------
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
            self.get_logger().warn(f"[Q] Table not found at {path} — running with fallback tie-breakers.")
        return False

    def summarize_qtable(self):
        arr = np.array(list(self.q_table.values()), dtype=float)
        if arr.size == 0:
            self.get_logger().warn("[Q] Table empty!"); return
        max_abs = np.max(np.abs(arr))
        nz = int(np.sum(np.abs(arr) > 1e-6))
        self.get_logger().info(f"[Q] Stats: max|Q|={max_abs:.3f}, nonzero_entries={nz}/{arr.size}")

# -------- main --------
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
