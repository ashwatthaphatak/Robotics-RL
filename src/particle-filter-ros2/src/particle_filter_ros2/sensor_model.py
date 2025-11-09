#!/usr/bin/env python3
import os
import math
import cv2
import yaml
import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Pose, PoseArray, PoseWithCovarianceStamped, Quaternion
from tf2_ros import Buffer, TransformException, TransformListener
from ament_index_python.packages import get_package_share_directory

class SensorModel(Node):
    def __init__(self):
        super().__init__('sensor_model')

        # Parameters
        self.declare_parameter('map_yaml', '')
        self.declare_parameter('sigma_hit', 0.2)
        self.declare_parameter('z_hit', 0.8)
        self.declare_parameter('z_rand', 0.2)
        self.declare_parameter('z_max', 3.5)
        self.declare_parameter('beam_subsample', 20)
        self.declare_parameter('distance_field_output', '')
        self.declare_parameter('distance_field_invert', True)
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('robot_frame', 'base_link')
        self.declare_parameter('publish_corrections', True)

        self._share_dir = get_package_share_directory('particle-filter-ros2')
        default_map = os.path.join(self._share_dir, 'maps', 'map.yaml')

        p = self.get_parameter
        self.map_yaml = self._resolve_map_yaml(p('map_yaml').value, default_map)
        self.sigma_hit = p('sigma_hit').value
        self.z_hit = p('z_hit').value
        self.z_rand = p('z_rand').value
        self.z_max = p('z_max').value
        self.beam_subsample = max(1, int(p('beam_subsample').value))
        self.distance_field_output = self._resolve_output_path(
            p('distance_field_output').value,
            os.path.join(os.path.dirname(self.map_yaml), 'distance_field.png')
        )
        self.distance_field_invert = bool(p('distance_field_invert').value)
        self.map_frame = p('map_frame').value or 'map'
        self.robot_frame = p('robot_frame').value or 'base_link'
        self.publish_corrections = bool(p('publish_corrections').value)

        self.dist_field = None
        self.map_resolution = None
        self.map_origin = None
        self.map_shape = None

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=True)
        self.correction_pub = self.create_publisher(
            PoseWithCovarianceStamped, '/sensor_model/correction', 10
        )
        self.beam_debug_pub = self.create_publisher(
            PoseArray, '/sensor_model/beam_endpoints', 10
        )

        self.load_map(self.map_yaml)
        self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.get_logger().info('Sensor model node started.')

    def _resolve_map_yaml(self, value, default_map):
        candidate = os.path.expanduser(value) if value else ''
        if not candidate:
            return default_map

        if os.path.isabs(candidate) and os.path.exists(candidate):
            return candidate

        search_roots = [
            self._share_dir,
            os.path.join(self._share_dir, 'config'),
        ]
        for root in search_roots:
            maybe = os.path.normpath(os.path.join(root, candidate))
            if os.path.exists(maybe):
                return maybe

        self.get_logger().warn(
            f"map_yaml parameter '{value}' not found; falling back to default {default_map}"
        )
        return default_map

    def _resolve_output_path(self, value, default_output):
        candidate = os.path.expanduser(value) if value else ''
        if not candidate:
            return default_output

        if os.path.isabs(candidate):
            output_dir = os.path.dirname(candidate)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            return candidate

        resolved = os.path.normpath(os.path.join(self._share_dir, candidate))
        output_dir = os.path.dirname(resolved)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        return resolved

    def load_map(self, map_yaml):
        if not map_yaml or not os.path.exists(map_yaml):
            self.get_logger().error(f"Map YAML not found: {map_yaml}")
            return

        with open(map_yaml, 'r') as f:
            info = yaml.safe_load(f)

        image_path = info['image']
        if not os.path.isabs(image_path):
            image_path = os.path.join(os.path.dirname(map_yaml), image_path)

        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            self.get_logger().error(f"Failed to read map image: {image_path}")
            return

        occ_prob = img.astype(np.float32) / 255.0
        if not info.get('negate', 0):
            occ_prob = 1.0 - occ_prob

        occupied = occ_prob >= float(info.get('occupied_thresh', 0.65))
        binary = np.logical_not(occupied).astype(np.uint8) * 255
        dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        self.map_resolution = float(info['resolution'])
        self.map_origin = info['origin']
        self.map_shape = dist.shape[::-1]  # (width, height)
        self.dist_field = dist * self.map_resolution
        if np.max(dist) > 0:
            normalized = (dist/np.max(dist)*255).astype(np.uint8)
        else:
            normalized = dist.astype(np.uint8)
        if self.distance_field_invert:
            normalized = 255 - normalized
        cv2.imwrite(self.distance_field_output, normalized)
        self.get_logger().info(f"Saved likelihood field → {self.distance_field_output}")

    def scan_callback(self, scan):
        if self.dist_field is None:
            return

        source_frame = scan.header.frame_id or 'base_link'
        target_time = Time.from_msg(scan.header.stamp)
        try:
            transform = self.tf_buffer.lookup_transform(
                self.map_frame,
                source_frame,
                target_time,
                timeout=Duration(seconds=0.2)
            )
        except TransformException as exc:
            try:
                transform = self.tf_buffer.lookup_transform(
                    self.map_frame,
                    source_frame,
                    Time(),
                    timeout=Duration(seconds=0.2)
                )
                self.get_logger().warn(
                    f"TF lookup failed at stamp ({target_time.nanoseconds}ns): {exc}. "
                    "Using latest available transform instead."
                )
            except TransformException as exc_latest:
                self.get_logger().warn(f"TF lookup failed: {exc_latest}")
                return

        robot_tf = self._lookup_robot_pose(target_time)
        if robot_tf is None and source_frame == self.robot_frame:
            robot_tf = transform

        trans = transform.transform.translation
        rot = transform.transform.rotation
        yaw = self.quaternion_to_yaw(rot)
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        origin_x, origin_y = trans.x, trans.y

        total_loglik = 0.0
        count = 0
        beam_poses = []
        range_step = self.beam_subsample
        for i in range(0, len(scan.ranges), range_step):
            r = scan.ranges[i]
            if not math.isfinite(r) or r >= self.z_max or r <= scan.range_min:
                continue
            beam_angle = scan.angle_min + i * scan.angle_increment
            sx = r * math.cos(beam_angle)
            sy = r * math.sin(beam_angle)
            wx = origin_x + cos_yaw * sx - sin_yaw * sy
            wy = origin_y + sin_yaw * sx + cos_yaw * sy
            dist_to_obs = self._lookup_distance(wx, wy)
            if dist_to_obs is None:
                continue
            pz = self.prob_hit(dist_to_obs)
            total_loglik += math.log(max(pz, 1e-9))
            count += 1
            pose = Pose()
            pose.position.x = float(wx)
            pose.position.y = float(wy)
            pose.orientation = self._yaw_to_quaternion(yaw + beam_angle)
            beam_poses.append(pose)
        if count:
            self.get_logger().info(f"Avg log-likelihood: {total_loglik/count:.3f}")
        self._publish_correction(robot_tf)
        self._publish_beam_debug(scan.header.stamp, beam_poses)

    def prob_hit(self, dist):
        norm = 1.0 / (self.sigma_hit * math.sqrt(2*math.pi))
        return self.z_hit * norm * math.exp(-0.5 * (dist/self.sigma_hit)**2) + \
               self.z_rand * (1.0/self.z_max)

    @staticmethod
    def quaternion_to_yaw(q):
        return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                          1.0 - 2.0 * (q.y * q.y + q.z * q.z))

    @staticmethod
    def _yaw_to_quaternion(yaw):
        half = yaw / 2.0
        quat = Quaternion()
        quat.x = 0.0
        quat.y = 0.0
        quat.z = math.sin(half)
        quat.w = math.cos(half)
        return quat

    def _lookup_robot_pose(self, target_time):
        try:
            return self.tf_buffer.lookup_transform(
                self.map_frame,
                self.robot_frame,
                target_time,
                timeout=Duration(seconds=0.2)
            )
        except TransformException as exc:
            try:
                latest = self.tf_buffer.lookup_transform(
                    self.map_frame,
                    self.robot_frame,
                    Time(),
                    timeout=Duration(seconds=0.2)
                )
                self.get_logger().warn(
                    f"Robot pose lookup failed at stamp ({target_time.nanoseconds}ns): {exc}. "
                    "Using latest available transform instead."
                )
                return latest
            except TransformException as exc_latest:
                self.get_logger().warn(f"Robot pose lookup failed: {exc_latest}")
                return None

    def _publish_correction(self, transform):
        if not self.publish_corrections or transform is None:
            return
        msg = PoseWithCovarianceStamped()
        msg.header = transform.header
        msg.header.frame_id = self.map_frame
        msg.pose.pose.position.x = transform.transform.translation.x
        msg.pose.pose.position.y = transform.transform.translation.y
        msg.pose.pose.position.z = transform.transform.translation.z
        msg.pose.pose.orientation = transform.transform.rotation

        variance_xy = max(self.sigma_hit ** 2, 1e-6)
        variance_theta = max(self.z_hit ** 2, 1e-6)
        cov = [0.0] * 36
        cov[0] = variance_xy
        cov[7] = variance_xy
        cov[35] = variance_theta
        msg.pose.covariance = cov
        self.correction_pub.publish(msg)

    def _publish_beam_debug(self, stamp, beam_poses):
        if not beam_poses or self.beam_debug_pub.get_subscription_count() == 0:
            return
        msg = PoseArray()
        msg.header.stamp = stamp
        msg.header.frame_id = self.map_frame
        msg.poses = beam_poses
        self.beam_debug_pub.publish(msg)

    def _world_to_map(self, x, y):
        if self.map_resolution is None or self.map_origin is None or self.map_shape is None:
            return None
        origin_x, origin_y = self.map_origin[0], self.map_origin[1]
        mx = int((x - origin_x) / self.map_resolution)
        my = int((y - origin_y) / self.map_resolution)
        width, height = self.map_shape
        my_img = height - 1 - my
        if mx < 0 or mx >= width or my_img < 0 or my_img >= height:
            return None
        return mx, my_img

    def _lookup_distance(self, x, y):
        cell = self._world_to_map(x, y)
        if cell is None:
            return None
        mx, my = cell
        return float(self.dist_field[my, mx])

def main(args=None):
    rclpy.init(args=args)
    node = SensorModel()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
