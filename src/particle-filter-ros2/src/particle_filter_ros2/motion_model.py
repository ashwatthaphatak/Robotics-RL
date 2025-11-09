#!/usr/bin/env python3
import math
import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose2D, PoseStamped, PoseWithCovarianceStamped, Quaternion

class MotionModel(Node):
    def __init__(self):
        super().__init__('motion_model')

        # Parameters α1–α4 control motion noise
        self.declare_parameter('alpha1', 0.02)
        self.declare_parameter('alpha2', 0.02)
        self.declare_parameter('alpha3', 0.02)
        self.declare_parameter('alpha4', 0.02)
        self.declare_parameter('initial_pose_x', 0.0)
        self.declare_parameter('initial_pose_y', 0.0)
        self.declare_parameter('initial_pose_theta', 0.0)
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('correction_gain', 1.0)

        self.alpha = [self.get_parameter(f'alpha{i+1}').value for i in range(4)]
        self.map_frame = self.get_parameter('map_frame').value or 'map'
        self.correction_gain = float(self.get_parameter('correction_gain').value)

        self.state = Pose2D()
        self.state.x = float(self.get_parameter('initial_pose_x').value)
        self.state.y = float(self.get_parameter('initial_pose_y').value)
        self.state.theta = self.normalize_angle(
            float(self.get_parameter('initial_pose_theta').value)
        )

        self.pose_pub = self.create_publisher(PoseStamped, '/motion_model/pose', 10)
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.create_subscription(
            PoseWithCovarianceStamped,
            '/initialpose',
            self.initial_pose_callback,
            10,
        )
        self.create_subscription(
            PoseWithCovarianceStamped,
            '/sensor_model/correction',
            self.sensor_correction_callback,
            10,
        )

        self.prev_odom = None
        self.get_logger().info(
            f"Motion model node started with initial state "
            f"(x={self.state.x:.3f}, y={self.state.y:.3f}, θ={self.state.theta:.3f})."
        )

    def odom_callback(self, msg: Odometry):
        # Convert quaternion → yaw
        q = msg.pose.pose.orientation
        yaw = math.atan2(2*(q.w*q.z + q.x*q.y),
                         1 - 2*(q.y*q.y + q.z*q.z))
        curr = Pose2D(
            x=msg.pose.pose.position.x,
            y=msg.pose.pose.position.y,
            theta=yaw
        )

        if self.prev_odom is None:
            self.prev_odom = curr
            if self.state is None:
                self.state = Pose2D(x=curr.x, y=curr.y, theta=curr.theta)
                self.get_logger().info(
                    "Motion model state seeded from first odom message "
                    f"(x={self.state.x:.3f}, y={self.state.y:.3f}, θ={self.state.theta:.3f})."
                )
            return

        drot1, dtrans, drot2 = self.compute_motion(self.prev_odom, curr)
        self.prev_odom = curr
        self.state = self.sample_motion_model_odometry(self.state,
                                                       drot1, dtrans, drot2)
        self.get_logger().info(
            f"Δrot1={drot1:.3f}, Δtrans={dtrans:.3f}, Δrot2={drot2:.3f} → "
            f"State=({self.state.x:.3f},{self.state.y:.3f},{self.state.theta:.3f})"
        )
        self._publish_state(msg.header.stamp)

    def initial_pose_callback(self, msg: PoseWithCovarianceStamped):
        pose2d = self._pose2d_from_pose(msg.pose.pose)
        self.state = pose2d
        self.get_logger().info(
            f"Initial pose update received: "
            f"(x={pose2d.x:.3f}, y={pose2d.y:.3f}, θ={pose2d.theta:.3f})."
        )
        self._publish_state(msg.header.stamp)

    def sensor_correction_callback(self, msg: PoseWithCovarianceStamped):
        if self.state is None:
            self.initial_pose_callback(msg)
            return
        measurement = self._pose2d_from_pose(msg.pose.pose)
        gain = max(0.0, min(1.0, self.correction_gain))
        self.state.x += gain * (measurement.x - self.state.x)
        self.state.y += gain * (measurement.y - self.state.y)
        heading_error = self.normalize_angle(measurement.theta - self.state.theta)
        self.state.theta = self.normalize_angle(self.state.theta + gain * heading_error)
        self.get_logger().info(
            f"Sensor correction applied (gain={gain:.2f}): "
            f"(x={self.state.x:.3f}, y={self.state.y:.3f}, θ={self.state.theta:.3f})."
        )
        self._publish_state(msg.header.stamp)

    @staticmethod
    def compute_motion(prev, curr):
        dx, dy = curr.x - prev.x, curr.y - prev.y
        dtrans = math.hypot(dx, dy)
        if dtrans < 1e-6:
            drot1 = 0.0
            drot2 = MotionModel.normalize_angle(curr.theta - prev.theta)
        else:
            drot1 = MotionModel.normalize_angle(math.atan2(dy, dx) - prev.theta)
            drot2 = MotionModel.normalize_angle(curr.theta - prev.theta - drot1)
        return drot1, dtrans, drot2

    def sample_motion_model_odometry(self, prev_pose, dr1, dt, dr2):
        a1, a2, a3, a4 = self.alpha
        sigma1 = math.sqrt(a1 * dr1 * dr1 + a2 * dt * dt)
        sigma2 = math.sqrt(a3 * dt * dt + a4 * (dr1 * dr1 + dr2 * dr2))
        sigma3 = math.sqrt(a1 * dr2 * dr2 + a2 * dt * dt)

        dr1_hat = dr1 - np.random.normal(0.0, sigma1)
        dt_hat = dt - np.random.normal(0.0, sigma2)
        dr2_hat = dr2 - np.random.normal(0.0, sigma3)

        heading = prev_pose.theta + dr1_hat
        x_new = prev_pose.x + dt_hat * math.cos(heading)
        y_new = prev_pose.y + dt_hat * math.sin(heading)
        theta_new = MotionModel.normalize_angle(prev_pose.theta + dr1_hat + dr2_hat)
        return Pose2D(x=x_new, y=y_new, theta=theta_new)

    def _publish_state(self, stamp):
        if self.state is None:
            return
        if stamp is None or (not getattr(stamp, 'sec', 0) and not getattr(stamp, 'nanosec', 0)):
            stamp = self.get_clock().now().to_msg()
        pose_msg = PoseStamped()
        pose_msg.header.frame_id = self.map_frame
        pose_msg.header.stamp = stamp
        pose_msg.pose.position.x = float(self.state.x)
        pose_msg.pose.position.y = float(self.state.y)
        pose_msg.pose.position.z = 0.0
        pose_msg.pose.orientation = self._yaw_to_quaternion(self.state.theta)
        self.pose_pub.publish(pose_msg)

    def _pose2d_from_pose(self, pose):
        yaw = self._quaternion_to_yaw(pose.orientation)
        pose2d = Pose2D()
        pose2d.x = pose.position.x
        pose2d.y = pose.position.y
        pose2d.theta = self.normalize_angle(yaw)
        return pose2d

    @staticmethod
    def normalize_angle(angle):
        return math.atan2(math.sin(angle), math.cos(angle))

    @staticmethod
    def _quaternion_to_yaw(q):
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

def main(args=None):
    rclpy.init(args=args)
    node = MotionModel()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
