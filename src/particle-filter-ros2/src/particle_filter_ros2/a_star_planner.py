#!/usr/bin/env python3
"""
Simple A* planner + waypoint follower for TurtleBot3.

Subscribes:
- /map (nav_msgs/OccupancyGrid)
- /initialpose (geometry_msgs/PoseWithCovarianceStamped)
- /amcl_pose (geometry_msgs/PoseWithCovarianceStamped) [optional]
- /goal_pose (geometry_msgs/PoseStamped) [from RViz 2D Nav Goal]
- /odom (nav_msgs/Odometry) [fallback pose]

Publishes:
- /plan (nav_msgs/Path)
- /cmd_vel (geometry_msgs/Twist)

Usage:
- Launch map_server and this node (see launch/a_star_nav.launch.py)
- Set initial pose in RViz (2D Pose Estimate)
- Set goal in RViz (2D Nav Goal)

The node computes a path using A* on the occupancy grid and follows it
with a simple lookahead controller.
"""

import math
import heapq
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    HistoryPolicy,
    DurabilityPolicy,
)

from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Twist, TransformStamped
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from ros_gz_interfaces.msg import Entity
from ros_gz_interfaces.srv import SetEntityPose
from tf2_ros import TransformBroadcaster


def angle_normalize(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


class AStarPlanner(Node):
    def __init__(self) -> None:
        super().__init__('a_star_planner')

        # Parameters
        self.declare_parameter('occupancy_threshold', 50)
        self.declare_parameter('treat_unknown_as_obstacle', True)
        self.declare_parameter('allow_diagonal', True)
        self.declare_parameter('robot_radius', 0.12)  # meters
        self.declare_parameter('lookahead', 0.30)     # meters
        self.declare_parameter('k_lin', 0.6)
        self.declare_parameter('k_ang', 1.8)
        self.declare_parameter('max_lin_vel', 0.22)   # TB3 Burger max ~0.22 m/s
        self.declare_parameter('max_ang_vel', 2.84)   # TB3 Burger max ~2.84 rad/s
        self.declare_parameter('goal_tolerance_lin', 0.10)
        self.declare_parameter('yaw_threshold_move', 0.4)
        self.declare_parameter('sync_robot_pose', False)
        self.declare_parameter('set_pose_service', '/world/default/set_pose')
        default_entity = os.environ.get('TURTLEBOT3_MODEL', 'turtlebot3_burger')
        self.declare_parameter('entity_name', default_entity)

        self.occ_threshold = int(self.get_parameter('occupancy_threshold').value)
        self.unknown_block = bool(self.get_parameter('treat_unknown_as_obstacle').value)
        self.allow_diagonal = bool(self.get_parameter('allow_diagonal').value)
        self.robot_radius = float(self.get_parameter('robot_radius').value)

        self.lookahead = float(self.get_parameter('lookahead').value)
        self.k_lin = float(self.get_parameter('k_lin').value)
        self.k_ang = float(self.get_parameter('k_ang').value)
        self.max_lin = float(self.get_parameter('max_lin_vel').value)
        self.max_ang = float(self.get_parameter('max_ang_vel').value)
        self.goal_tol = float(self.get_parameter('goal_tolerance_lin').value)
        self.yaw_threshold_move = float(self.get_parameter('yaw_threshold_move').value)
        self.sync_robot_pose = bool(self.get_parameter('sync_robot_pose').value)
        self.set_pose_service = self.get_parameter('set_pose_service').value
        self.entity_name = self.get_parameter('entity_name').value

        self.pose_client = None
        if self.sync_robot_pose:
            self.pose_client = self.create_client(SetEntityPose, self.set_pose_service)

        self.tf_broadcaster = TransformBroadcaster(self)
        # map->odom will be computed from initial pose; start empty to avoid stale offsets
        self.map_to_odom: Optional[Tuple[float, float, float]] = None
        self.odom_pose: Optional[Tuple[float, float, float]] = None

        # Map state
        self.map: Optional[OccupancyGrid] = None
        self.width: Optional[int] = None
        self.height: Optional[int] = None
        self.resolution: Optional[float] = None
        self.origin_xy: Optional[Tuple[float, float]] = None
        self.occ: Optional[np.ndarray] = None           # raw occupancy (int16)
        self.blocked: Optional[np.ndarray] = None       # inflated obstacles (bool)
        self.inflation_offsets: Optional[List[Tuple[int, int]]] = None

        # Pose state (map frame)
        self.curr_pose: Optional[Tuple[float, float, float]] = None  # x, y, yaw
        self.have_start: bool = False

        # Goal + path state
        self.goal_xy: Optional[Tuple[float, float]] = None
        self.path_world: List[Tuple[float, float]] = []
        self.path_pub = self.create_publisher(Path, '/plan', 10)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.pending_map_pose: Optional[Tuple[float, float, float]] = None

        # Subscriptions
        map_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.create_subscription(OccupancyGrid, '/map', self.map_cb, map_qos)
        self.create_subscription(PoseWithCovarianceStamped, '/initialpose', self.initialpose_cb, 10)
        self.create_subscription(PoseWithCovarianceStamped, '/amcl_pose', self.amcl_pose_cb, 10)
        # Accept both common RViz goal topics
        self.create_subscription(PoseStamped, '/goal_pose', self.goal_cb, 10)
        self.create_subscription(PoseStamped, '/move_base_simple/goal', self.goal_cb, 10)
        self.create_subscription(Odometry, '/odom', self.odom_cb, 10)

        # Control loop
        self.timer = self.create_timer(0.05, self.control_cb)  # 20 Hz
        self.tf_timer = self.create_timer(0.05, self.publish_map_to_odom_tf)

        self.get_logger().info('A* planner ready. Set initial pose and goal in RViz.')

    # -------------------- ROS callbacks --------------------
    def map_cb(self, msg: OccupancyGrid) -> None:
        self.map = msg
        self.width = msg.info.width
        self.height = msg.info.height
        self.resolution = msg.info.resolution
        self.origin_xy = (msg.info.origin.position.x, msg.info.origin.position.y)

        data = np.array(msg.data, dtype=np.int16).reshape((self.height, self.width))
        self.occ = data

        # Build blocked grid and inflate by robot radius
        blocked = (data >= self.occ_threshold)
        if self.unknown_block:
            blocked |= (data < 0)

        # Precompute inflation offsets once per map (or when resolution changes)
        if self.inflation_offsets is None and self.resolution is not None:
            cell_radius = int(math.ceil(self.robot_radius / self.resolution))
            offsets: List[Tuple[int, int]] = []
            for dy in range(-cell_radius, cell_radius + 1):
                for dx in range(-cell_radius, cell_radius + 1):
                    if dx == 0 and dy == 0:
                        offsets.append((dy, dx))
                        continue
                    dist = math.hypot(dx, dy) * self.resolution
                    if dist <= self.robot_radius:
                        offsets.append((dy, dx))
            self.inflation_offsets = offsets

        # Inflate obstacles
        self.blocked = self._inflate_blocked(blocked)
        self.get_logger().info('Map received. Size: %dx%d, res=%.3f' % (self.width, self.height, self.resolution))

    def initialpose_cb(self, msg: PoseWithCovarianceStamped) -> None:
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        yaw = self._quat_to_yaw(q.x, q.y, q.z, q.w)
        self.curr_pose = (x, y, yaw)
        self.have_start = True
        self.get_logger().info('Initial pose set: (%.2f, %.2f, %.2f)' % (x, y, yaw))
        if self.sync_robot_pose:
            self._sync_robot_pose(msg)
        self.pending_map_pose = (x, y, yaw)
        if self.odom_pose is not None:
            self._update_map_to_odom(x, y, yaw)

    def amcl_pose_cb(self, msg: PoseWithCovarianceStamped) -> None:
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        yaw = self._quat_to_yaw(q.x, q.y, q.z, q.w)
        self.curr_pose = (x, y, yaw)

    def odom_cb(self, msg: Odometry) -> None:
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        yaw = self._quat_to_yaw(q.x, q.y, q.z, q.w)
        self.odom_pose = (x, y, yaw)

        if self.pending_map_pose is not None:
            self._update_map_to_odom(*self.pending_map_pose)

        if self.map_to_odom is not None:
            tx, ty, yaw_offset = self.map_to_odom
            cos_o = math.cos(yaw_offset)
            sin_o = math.sin(yaw_offset)
            map_x = tx + cos_o * x - sin_o * y
            map_y = ty + sin_o * x + cos_o * y
            map_yaw = angle_normalize(yaw_offset + yaw)
            self.curr_pose = (map_x, map_y, map_yaw)
        elif not self.have_start:
            self.curr_pose = (x, y, yaw)

    def goal_cb(self, msg: PoseStamped) -> None:
        self.goal_xy = (msg.pose.position.x, msg.pose.position.y)
        self.get_logger().info('Goal received: (%.2f, %.2f)' % self.goal_xy)
        self.plan_and_publish()

    # -------------------- Planning --------------------
    def plan_and_publish(self) -> None:
        if self.map is None or self.blocked is None or self.resolution is None or self.origin_xy is None:
            self.get_logger().warn('No map yet; cannot plan.')
            return
        if not self.have_start or self.curr_pose is None:
            self.get_logger().warn('No start pose yet; set /initialpose or publish /amcl_pose.')
            return
        if self.goal_xy is None:
            self.get_logger().warn('No goal set yet; click 2D Nav Goal in RViz.')
            return

        sx, sy, _ = self.curr_pose
        gx, gy = self.goal_xy

        start = self.world_to_map(sx, sy)
        goal = self.world_to_map(gx, gy)
        if not self.in_bounds(*start) or not self.in_bounds(*goal):
            self.get_logger().error('Start or goal out of map bounds.')
            return

        if self.blocked[start[1], start[0]]:
            self.get_logger().warn('Start cell is occupied/inflated; trying to plan anyway.')
        if self.blocked[goal[1], goal[0]]:
            self.get_logger().error('Goal cell is occupied/inflated. Choose a different goal.')
            return

        path_cells = self.a_star(start, goal)
        if not path_cells:
            self.get_logger().error('A* failed to find a path.')
            return

        # Convert to world and publish Path
        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = 'map'

        self.path_world = []
        for mx, my in path_cells:
            xw, yw = self.map_to_world(mx, my)
            self.path_world.append((xw, yw))
            ps = PoseStamped()
            ps.header = path_msg.header
            ps.pose.position.x = float(xw)
            ps.pose.position.y = float(yw)
            ps.pose.position.z = 0.0
            # Orientation along path not set (RViz will show as points/line)
            ps.pose.orientation.w = 1.0
            path_msg.poses.append(ps)

        self.path_pub.publish(path_msg)
        self.get_logger().info('Published path with %d waypoints.' % len(self.path_world))

    def a_star(self, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        # Neighbors: 4 or 8-connected
        if self.allow_diagonal:
            neighbors = [
                (+1, 0, 1.0), (-1, 0, 1.0), (0, +1, 1.0), (0, -1, 1.0),
                (+1, +1, math.sqrt(2)), (-1, +1, math.sqrt(2)), (+1, -1, math.sqrt(2)), (-1, -1, math.sqrt(2)),
            ]
        else:
            neighbors = [(+1, 0, 1.0), (-1, 0, 1.0), (0, +1, 1.0), (0, -1, 1.0)]

        def h(a: Tuple[int, int], b: Tuple[int, int]) -> float:
            return math.hypot(a[0] - b[0], a[1] - b[1])

        open_heap: List[Tuple[float, int, Tuple[int, int]]] = []
        g_score: Dict[Tuple[int, int], float] = {start: 0.0}
        f_start = h(start, goal)
        counter = 0
        heapq.heappush(open_heap, (f_start, counter, start))
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        closed: set = set()

        while open_heap:
            _, _, current = heapq.heappop(open_heap)
            if current in closed:
                continue
            if current == goal:
                return self._reconstruct_path(came_from, current)
            closed.add(current)

            cx, cy = current
            for dx, dy, step_cost in neighbors:
                nx, ny = cx + dx, cy + dy
                if not self.in_bounds(nx, ny):
                    continue
                # Block cells that are inflated obstacles
                if self.blocked[ny, nx]:
                    continue
                # Disallow corner cutting through diagonals
                if dx != 0 and dy != 0:
                    if self.blocked[cy, nx] or self.blocked[ny, cx]:
                        continue

                tentative_g = g_score[current] + step_cost
                neighbor = (nx, ny)
                if neighbor in closed and tentative_g >= g_score.get(neighbor, float('inf')):
                    continue
                if tentative_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + h(neighbor, goal)
                    counter += 1
                    heapq.heappush(open_heap, (f_score, counter, neighbor))

        return []

    def _reconstruct_path(
        self,
        came_from: Dict[Tuple[int, int], Tuple[int, int]],
        current: Tuple[int, int],
    ) -> List[Tuple[int, int]]:
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    # -------------------- Control --------------------
    def control_cb(self) -> None:
        if not self.path_world or self.curr_pose is None:
            return

        x, y, yaw = self.curr_pose
        # If close to final goal, stop and clear path
        gx, gy = self.path_world[-1]
        dist_to_goal = math.hypot(gx - x, gy - y)
        if dist_to_goal <= self.goal_tol:
            self._publish_cmd(0.0, 0.0)
            # Keep path published but stop following
            self.path_world = []
            self.get_logger().info('Reached goal. Stopping.')
            return

        # Choose lookahead target along path
        target_x, target_y = self._find_lookahead_target(x, y, self.lookahead)
        # Control to target
        dx = target_x - x
        dy = target_y - y
        target_dist = math.hypot(dx, dy)
        target_heading = math.atan2(dy, dx)
        yaw_err = angle_normalize(target_heading - yaw)

        # Turn-in-place if heading error large
        if abs(yaw_err) > self.yaw_threshold_move:
            v = 0.0
        else:
            v = max(0.0, min(self.k_lin * target_dist, self.max_lin))
        w = max(-self.max_ang, min(self.k_ang * yaw_err, self.max_ang))

        self._publish_cmd(v, w)

    def _find_lookahead_target(self, x: float, y: float, lookahead: float) -> Tuple[float, float]:
        # Find the closest point on the path
        if not self.path_world:
            return x, y
        # Find nearest index
        dists = [math.hypot(px - x, py - y) for (px, py) in self.path_world]
        i = int(np.argmin(dists))
        # March forward until accumulated distance exceeds lookahead
        acc = 0.0
        for j in range(i, len(self.path_world) - 1):
            x0, y0 = self.path_world[j]
            x1, y1 = self.path_world[j + 1]
            seg = math.hypot(x1 - x0, y1 - y0)
            acc += seg
            if acc >= lookahead:
                return x1, y1
        return self.path_world[-1]

    def _publish_cmd(self, v: float, w: float) -> None:
        t = Twist()
        t.linear.x = float(v)
        t.angular.z = float(w)
        self.cmd_pub.publish(t)

    # -------------------- Helpers --------------------
    def _inflate_blocked(self, blocked: np.ndarray) -> np.ndarray:
        if self.inflation_offsets is None:
            return blocked.copy()
        inflated = blocked.copy()
        ys, xs = np.where(blocked)
        for y, x in zip(ys, xs):
            for dy, dx in self.inflation_offsets:
                ny, nx = y + dy, x + dx
                if 0 <= ny < blocked.shape[0] and 0 <= nx < blocked.shape[1]:
                    inflated[ny, nx] = True
        return inflated

    def world_to_map(self, x: float, y: float) -> Tuple[int, int]:
        assert self.origin_xy is not None and self.resolution is not None
        mx = int((x - self.origin_xy[0]) / self.resolution)
        my = int((y - self.origin_xy[1]) / self.resolution)
        return mx, my

    def map_to_world(self, mx: int, my: int) -> Tuple[float, float]:
        assert self.origin_xy is not None and self.resolution is not None
        x = self.origin_xy[0] + (mx + 0.5) * self.resolution
        y = self.origin_xy[1] + (my + 0.5) * self.resolution
        return x, y

    def in_bounds(self, mx: int, my: int) -> bool:
        return (
            self.width is not None
            and self.height is not None
            and 0 <= mx < self.width
            and 0 <= my < self.height
        )

    @staticmethod
    def _quat_to_yaw(x: float, y: float, z: float, w: float) -> float:
        # Yaw from quaternion (z-axis rotation)
        # Using formula derived from quaternion to Euler (roll-pitch-yaw)
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)

    def _sync_robot_pose(self, pose_msg: PoseWithCovarianceStamped) -> None:
        if self.pose_client is None:
            return
        if not self.pose_client.service_is_ready():
            if not self.pose_client.wait_for_service(timeout_sec=0.5):
                self.get_logger().warn('Set pose service unavailable; cannot move robot.')
                return
        req = SetEntityPose.Request()
        req.entity.name = self.entity_name
        req.entity.type = Entity.MODEL
        req.pose.position.x = pose_msg.pose.pose.position.x
        req.pose.position.y = pose_msg.pose.pose.position.y
        req.pose.position.z = 0.01
        req.pose.orientation = pose_msg.pose.pose.orientation

        def done_cb(future):
            try:
                resp = future.result()
            except Exception as exc:  # pylint: disable=broad-except
                self.get_logger().error(f'Failed to sync robot pose: {exc}')
                return
            if resp.success:
                self.get_logger().info('Gazebo robot moved to RViz initial pose.')
            else:
                self.get_logger().error(f'Set pose service returned failure: {resp.status}')

        future = self.pose_client.call_async(req)
        future.add_done_callback(done_cb)

    def _update_map_to_odom(self, x_map: float, y_map: float, yaw_map: float) -> None:
        if self.odom_pose is None:
            # Assume odom is at origin if we haven't received it yet
            x_odom, y_odom, yaw_odom = 0.0, 0.0, 0.0
        else:
            x_odom, y_odom, yaw_odom = self.odom_pose
        yaw_offset = angle_normalize(yaw_map - yaw_odom)
        cos_o = math.cos(yaw_offset)
        sin_o = math.sin(yaw_offset)
        tx = x_map - (cos_o * x_odom - sin_o * y_odom)
        ty = y_map - (sin_o * x_odom + cos_o * y_odom)
        self.map_to_odom = (tx, ty, yaw_offset)
        self.pending_map_pose = None

    def publish_map_to_odom_tf(self) -> None:
        if self.map_to_odom is None:
            # Publish identity so RViz has the frame even before init
            tx = ty = yaw = 0.0
        else:
            tx, ty, yaw = self.map_to_odom
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'map'
        t.child_frame_id = 'odom'
        t.transform.translation.x = float(tx)
        t.transform.translation.y = float(ty)
        t.transform.translation.z = 0.0
        quat = self.yaw_to_quaternion(yaw)
        t.transform.rotation.x = quat[0]
        t.transform.rotation.y = quat[1]
        t.transform.rotation.z = quat[2]
        t.transform.rotation.w = quat[3]
        self.tf_broadcaster.sendTransform(t)

    @staticmethod
    def yaw_to_quaternion(yaw: float) -> Tuple[float, float, float, float]:
        half = yaw / 2.0
        return (0.0, 0.0, math.sin(half), math.cos(half))


def main(args=None) -> None:
    rclpy.init(args=args)
    node = AStarPlanner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
