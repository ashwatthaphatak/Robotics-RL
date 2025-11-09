#!/usr/bin/env python3
import math
import os
from typing import Optional

import cv2
import numpy as np
import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Pose, Pose2D, PoseArray, PoseStamped, Quaternion, TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import LaserScan
from tf2_ros import TransformBroadcaster

from .motion_model import MotionModel


class LikelihoodField:
    """Distance transform helper for likelihood-field measurement model."""

    def __init__(self, map_yaml: str):
        self.map_yaml = map_yaml
        self.distance_field = None
        self.free_points = None
        self.resolution = None
        self.origin = None
        self.height = None
        self.width = None
        self._load()

    def _load(self):
        if not os.path.exists(self.map_yaml):
            raise FileNotFoundError(f"Map YAML not found: {self.map_yaml}")

        with open(self.map_yaml, "r") as stream:
            metadata = yaml.safe_load(stream)

        image_path = metadata["image"]
        if not os.path.isabs(image_path):
            image_path = os.path.join(os.path.dirname(self.map_yaml), image_path)

        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise RuntimeError(f"Failed to read map image: {image_path}")

        negate = int(metadata.get("negate", 0))
        self.resolution = float(metadata["resolution"])
        self.origin = np.array(metadata["origin"], dtype=float)
        self.height, self.width = image.shape

        occ_prob = image.astype(np.float32) / 255.0
        if not negate:
            occ_prob = 1.0 - occ_prob

        occupied_thresh = float(metadata.get("occupied_thresh", 0.65))
        free_thresh = float(metadata.get("free_thresh", 0.196))
        occupied = occ_prob >= occupied_thresh
        free_mask = occ_prob <= free_thresh

        binary = np.logical_not(occupied).astype(np.uint8) * 255
        dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        self.distance_field = dist * self.resolution

        free_indices = np.column_stack(np.nonzero(free_mask))
        if free_indices.size == 0:
            raise RuntimeError("No free cells available for particle initialization.")
        ys = free_indices[:, 0]
        xs = free_indices[:, 1]
        world_x = self.origin[0] + (xs + 0.5) * self.resolution
        world_y = self.origin[1] + (self.height - ys - 0.5) * self.resolution
        self.free_points = np.column_stack((world_x, world_y))

    def sample_free(self, count: int, rng: np.random.Generator) -> np.ndarray:
        indices = rng.choice(len(self.free_points), size=count, replace=True)
        return self.free_points[indices]

    def world_to_index(self, x: float, y: float) -> Optional[tuple]:
        mx = int((x - self.origin[0]) / self.resolution)
        my = int((y - self.origin[1]) / self.resolution)
        if mx < 0 or mx >= self.width or my < 0 or my >= self.height:
            return None
        img_y = self.height - 1 - my
        return mx, img_y

    def lookup(self, x: float, y: float) -> Optional[float]:
        cell = self.world_to_index(x, y)
        if cell is None:
            return None
        return float(self.distance_field[cell[1], cell[0]])


class ParticleFilter(Node):
    def __init__(self):
        super().__init__("particle_filter")
        # Declare parameters
        self.declare_parameter("num_particles", 200)
        self.declare_parameter("resample_threshold", 0.5)
        self.declare_parameter("map_yaml", "")
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("publish_rate", 10.0)
        self.declare_parameter("beam_subsample", 10)
        self.declare_parameter("sigma_hit", 0.2)
        self.declare_parameter("z_hit", 0.8)
        self.declare_parameter("z_rand", 0.2)
        self.declare_parameter("z_max", 3.5)
        self.declare_parameter("alpha1", 0.02)
        self.declare_parameter("alpha2", 0.02)
        self.declare_parameter("alpha3", 0.02)
        self.declare_parameter("alpha4", 0.02)
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("odom_frame", "odom")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("publish_tf", True)
        self.declare_parameter("initialization_mode", "uniform")
        self.declare_parameter("initial_std_xy", 0.25)
        self.declare_parameter("initial_std_theta", 0.25)
        self.declare_parameter("initial_pose_x", 0.0)
        self.declare_parameter("initial_pose_y", 0.0)
        self.declare_parameter("initial_pose_theta", 0.0)
        self.declare_parameter("particle_topic", "/pf_particle_cloud")

        self.num_particles = int(self.get_parameter("num_particles").value)
        self.resample_threshold = float(self.get_parameter("resample_threshold").value)
        self.map_frame  = self.get_parameter("map_frame").value
        self.odom_frame = self.get_parameter("odom_frame").value
        self.base_frame = self.get_parameter("base_frame").value
        self.publish_tf = bool(self.get_parameter("publish_tf").value)
        self.init_mode = str(self.get_parameter("initialization_mode").value).lower()
        self.init_std_xy = float(self.get_parameter("initial_std_xy").value)
        self.init_std_theta = float(self.get_parameter("initial_std_theta").value)

        # Create TF broadcaster early so initialization routines can use it
        self.tf_broadcaster = TransformBroadcaster(self) if self.publish_tf else None

        map_yaml_param = self.get_parameter("map_yaml").value
        self.map_yaml = self._resolve_map(map_yaml_param)
        self.field = LikelihoodField(self.map_yaml)

        self.alpha = np.array([
            float(self.get_parameter("alpha1").value),
            float(self.get_parameter("alpha2").value),
            float(self.get_parameter("alpha3").value),
            float(self.get_parameter("alpha4").value),
        ])

        self.sigma_hit = float(self.get_parameter("sigma_hit").value)
        self.z_hit = float(self.get_parameter("z_hit").value)
        self.z_rand = float(self.get_parameter("z_rand").value)
        self.z_max = float(self.get_parameter("z_max").value)
        self.beam_subsample = max(1, int(self.get_parameter("beam_subsample").value))

        self.publish_rate = float(self.get_parameter("publish_rate").value)
        self.odom_topic = self.get_parameter("odom_topic").value
        self.scan_topic = self.get_parameter("scan_topic").value
        self.particle_topic = self.get_parameter("particle_topic").value

        self.manual_pose = Pose2D()
        self.manual_pose.x = float(self.get_parameter("initial_pose_x").value)
        self.manual_pose.y = float(self.get_parameter("initial_pose_y").value)
        self.manual_pose.theta = MotionModel.normalize_angle(
            float(self.get_parameter("initial_pose_theta").value)
        )

        self.rng = np.random.default_rng()
        self.particles = np.zeros((self.num_particles, 3), dtype=np.float64)
        self.weights = np.ones(self.num_particles, dtype=np.float64) / self.num_particles
        self.initialized = False
        self.current_estimate = np.zeros(3, dtype=np.float64)

        if self.init_mode == "uniform":
            self._initialize_uniform()
        elif self.init_mode == "manual":
            self._initialize_manual()
        elif self.init_mode == "odom":
            self.get_logger().info("Waiting for first odometry message to initialize particle cloud.")
        else:
            self.get_logger().warn(
                f"Unknown initialization_mode '{self.init_mode}', falling back to uniform sampling."
            )
            self._initialize_uniform()

        self.prev_odom: Optional[Pose2D] = None
        self.latest_odom: Optional[Pose2D] = None
        self.last_scan: Optional[LaserScan] = None
        self.last_pose: Optional[PoseStamped] = None

        qos = QoSProfile(
            depth=10,
            history=QoSHistoryPolicy.KEEP_LAST,
            reliability=QoSReliabilityPolicy.RELIABLE,
        )

        self.create_subscription(Odometry, self.odom_topic, self.odom_callback, qos)
        self.create_subscription(LaserScan, self.scan_topic, self.scan_callback, qos)
        self.particle_pub = self.create_publisher(PoseArray, self.particle_topic, 10)
        self.pose_pub = self.create_publisher(PoseStamped, "/pose_estimate", 10)

        self.create_timer(1.0 / self.publish_rate, self.publish_outputs)
        self.get_logger().info("Particle filter node initialized.")

    def _resolve_map(self, candidate: str) -> str:
        if candidate:
            expanded = os.path.expanduser(candidate)
            if os.path.isabs(expanded):
                return expanded
            share = get_package_share_directory("particle-filter-ros2")
            resolved = os.path.join(share, expanded)
            if os.path.exists(resolved):
                return resolved
            if os.path.exists(expanded):
                return expanded
            raise FileNotFoundError(f"Unable to resolve map_yaml '{candidate}'")
        share = get_package_share_directory("particle-filter-ros2")
        return os.path.join(share, "maps", "map.yaml")

    @staticmethod
    def _odom_to_pose2d(msg: Odometry) -> Pose2D:
        pose = Pose2D()
        pose.x = msg.pose.pose.position.x
        pose.y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        pose.theta = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                                1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        return pose

    def odom_callback(self, msg: Odometry):
        current = self._odom_to_pose2d(msg)
        if self.prev_odom is None:
            self.prev_odom = current
            self.latest_odom = current
            if not self.initialized and self.init_mode == "odom":
                self._initialize_from_pose(current, current)
            return

        dr1, dtrans, dr2 = MotionModel.compute_motion(self.prev_odom, current)
        self.prev_odom = current
        self.latest_odom = current
        if not self.initialized:
            return
        if dtrans == 0.0 and dr1 == 0.0 and dr2 == 0.0:
            return
        self._predict(dr1, dtrans, dr2)

    def scan_callback(self, scan: LaserScan):
        if self.prev_odom is None or not self.initialized:
            return
        log_weights = np.array([self._measurement_log_likelihood(particle, scan)
                                for particle in self.particles])
        prior = np.log(np.clip(self.weights, 1e-12, None))
        combined = prior + log_weights
        max_log = np.max(combined)
        stable = np.exp(combined - max_log)
        total = np.sum(stable)
        if total <= 0.0 or not np.isfinite(total):
            self.weights[:] = 1.0 / self.num_particles
        else:
            self.weights = stable / total

        self.last_scan = scan
        self._maybe_resample()
        self._update_pose_estimate(scan.header.stamp)

    def _predict(self, dr1: float, dtrans: float, dr2: float):
        a1, a2, a3, a4 = self.alpha
        sigma1 = math.sqrt(a1 * dr1 * dr1 + a2 * dtrans * dtrans)
        sigma2 = math.sqrt(a3 * dtrans * dtrans + a4 * (dr1 * dr1 + dr2 * dr2))
        sigma3 = math.sqrt(a1 * dr2 * dr2 + a2 * dtrans * dtrans)

        dr1_hat = dr1 + self.rng.normal(0.0, sigma1, size=self.num_particles)
        dt_hat = dtrans + self.rng.normal(0.0, sigma2, size=self.num_particles)
        dr2_hat = dr2 + self.rng.normal(0.0, sigma3, size=self.num_particles)

        headings = self.particles[:, 2] + dr1_hat
        self.particles[:, 0] += dt_hat * np.cos(headings)
        self.particles[:, 1] += dt_hat * np.sin(headings)
        self.particles[:, 2] = np.array([
            MotionModel.normalize_angle(theta)
            for theta in (self.particles[:, 2] + dr1_hat + dr2_hat)
        ])

    def _measurement_log_likelihood(self, particle: np.ndarray, scan: LaserScan) -> float:
        total = 0.0
        valid = 0
        px, py, pt = particle
        for i in range(0, len(scan.ranges), self.beam_subsample):
            rng = scan.ranges[i]
            if not math.isfinite(rng) or rng <= scan.range_min or rng >= self.z_max:
                continue
            beam = scan.angle_min + i * scan.angle_increment
            end_x = px + rng * math.cos(pt + beam)
            end_y = py + rng * math.sin(pt + beam)
            dist = self.field.lookup(end_x, end_y)
            if dist is None:
                continue
            prob = self._prob_hit(dist)
            total += math.log(max(prob, 1e-9))
            valid += 1
        if valid == 0:
            return math.log(1e-9)
        return total

    def _prob_hit(self, dist: float) -> float:
        norm = 1.0 / (self.sigma_hit * math.sqrt(2.0 * math.pi))
        return self.z_hit * norm * math.exp(-0.5 * (dist / self.sigma_hit) ** 2) + self.z_rand * (1.0 / self.z_max)

    def _maybe_resample(self):
        neff = 1.0 / np.sum(self.weights ** 2)
        if (neff / self.num_particles) < self.resample_threshold:
            self._low_variance_resample()
            self.weights[:] = 1.0 / self.num_particles

    def _low_variance_resample(self):
        cumulative = np.cumsum(self.weights)
        step = 1.0 / self.num_particles
        r = self.rng.uniform(0.0, step)
        indices = np.zeros(self.num_particles, dtype=int)
        i = 0
        for m in range(self.num_particles):
            u = r + m * step
            while u > cumulative[i]:
                i += 1
                if i >= self.num_particles:
                    i = self.num_particles - 1
                    break
            indices[m] = i
        self.particles = self.particles[indices].copy()

    def _update_pose_estimate(self, stamp):
        mean_xy = np.average(self.particles[:, :2], weights=self.weights, axis=0)
        sin_sum = np.sum(self.weights * np.sin(self.particles[:, 2]))
        cos_sum = np.sum(self.weights * np.cos(self.particles[:, 2]))
        yaw = math.atan2(sin_sum, cos_sum)

        pose = PoseStamped()
        pose.header.stamp = stamp
        pose.header.frame_id = self.map_frame
        pose.pose.position.x = float(mean_xy[0])
        pose.pose.position.y = float(mean_xy[1])
        pose.pose.position.z = 0.0
        pose.pose.orientation = self._yaw_to_quaternion(yaw)

        self.pose_pub.publish(pose)
        self.last_pose = pose
        self.current_estimate[:] = [mean_xy[0], mean_xy[1], yaw]

        if self.tf_broadcaster and self.latest_odom is not None:
            tx, ty, theta = self._compute_map_to_odom_transform(mean_xy, yaw, self.latest_odom)
            self._broadcast_transform(stamp, tx, ty, theta)

    def publish_outputs(self):
        if not self.initialized:
            return
        cloud = PoseArray()
        cloud.header.frame_id = self.map_frame
        if self.last_scan:
            cloud.header.stamp = self.last_scan.header.stamp
        poses = []
        for particle in self.particles:
            pose = Pose()
            pose.position.x = float(particle[0])
            pose.position.y = float(particle[1])
            pose.orientation = self._yaw_to_quaternion(particle[2])
            poses.append(pose)
        cloud.poses = poses
        self.particle_pub.publish(cloud)

        if self.last_pose:
            self.pose_pub.publish(self.last_pose)

    def _initialize_uniform(self):
        positions = self.field.sample_free(self.num_particles, self.rng)
        headings = self.rng.uniform(-math.pi, math.pi, size=(self.num_particles, 1))
        self.particles = np.hstack((positions, headings))
        self.weights.fill(1.0 / self.num_particles)
        self.initialized = True
        self.current_estimate[:] = [
            np.mean(self.particles[:, 0]),
            np.mean(self.particles[:, 1]),
            MotionModel.normalize_angle(np.mean(self.particles[:, 2])),
        ]

    def _initialize_from_pose(self, map_pose: Pose2D, odom_reference: Optional[Pose2D] = None):
        noise_xy = self.rng.normal(0.0, self.init_std_xy, size=(self.num_particles, 2))
        noise_theta = self.rng.normal(0.0, self.init_std_theta, size=(self.num_particles, 1))
        self.particles[:, 0] = map_pose.x + noise_xy[:, 0]
        self.particles[:, 1] = map_pose.y + noise_xy[:, 1]
        headings = map_pose.theta + noise_theta[:, 0]
        self.particles[:, 2] = np.array([MotionModel.normalize_angle(h) for h in headings])
        self.weights.fill(1.0 / self.num_particles)
        self.initialized = True
        self.current_estimate[:] = [
            np.mean(self.particles[:, 0]),
            np.mean(self.particles[:, 1]),
            MotionModel.normalize_angle(np.mean(self.particles[:, 2])),
        ]
        stamp = self.get_clock().now().to_msg()
        self.last_pose = PoseStamped()
        self.last_pose.header.stamp = stamp
        self.last_pose.header.frame_id = self.map_frame
        self.last_pose.pose.position.x = float(self.current_estimate[0])
        self.last_pose.pose.position.y = float(self.current_estimate[1])
        self.last_pose.pose.position.z = 0.0
        self.last_pose.pose.orientation = self._yaw_to_quaternion(self.current_estimate[2])
        if self.tf_broadcaster:
            tx, ty, theta = self._compute_map_to_odom_transform(
                self.current_estimate[:2], self.current_estimate[2], odom_reference
            )
            self._broadcast_transform(stamp, tx, ty, theta)
        self.get_logger().info(
            f"Initialized particles around map pose (x={map_pose.x:.2f}, y={map_pose.y:.2f}, θ={map_pose.theta:.2f})."
        )

    def _initialize_manual(self):
        manual_pose = Pose2D()
        manual_pose.x = self.manual_pose.x
        manual_pose.y = self.manual_pose.y
        manual_pose.theta = self.manual_pose.theta
        odom_reference = Pose2D()
        odom_reference.x = 0.0
        odom_reference.y = 0.0
        odom_reference.theta = 0.0
        self._initialize_from_pose(manual_pose, odom_reference)

    @staticmethod
    def _yaw_to_quaternion(yaw: float) -> Quaternion:
        half = yaw / 2.0
        q = Quaternion()
        q.x = 0.0
        q.y = 0.0
        q.z = math.sin(half)
        q.w = math.cos(half)
        return q

    def _broadcast_transform(self, stamp, tx, ty, theta):
        if not self.tf_broadcaster:
            return
        t = TransformStamped()
        t.header.stamp       = stamp
        t.header.frame_id    = self.map_frame
        t.child_frame_id     = self.odom_frame
        t.transform.translation.x = float(tx)
        t.transform.translation.y = float(ty)
        t.transform.translation.z = 0.0
        t.transform.rotation = self._yaw_to_quaternion(theta)
        self.tf_broadcaster.sendTransform(t)

    def _compute_map_to_odom_transform(self, mean_xy, yaw, odom_pose):
        if odom_pose is None:
            return 0.0, 0.0, 0.0
        dx     = mean_xy[0] - odom_pose.x
        dy     = mean_xy[1] - odom_pose.y
        theta  = MotionModel.normalize_angle(yaw - odom_pose.theta)
        return dx, dy, theta


def main(args=None):
    rclpy.init(args=args)
    node = ParticleFilter()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
