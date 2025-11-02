#!/usr/bin/env python3
import math
import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose2D

class MotionModel(Node):
    def __init__(self):
        super().__init__('motion_model')

        # Parameters α1–α4 control motion noise
        self.declare_parameter('alpha1', 0.02)
        self.declare_parameter('alpha2', 0.02)
        self.declare_parameter('alpha3', 0.02)
        self.declare_parameter('alpha4', 0.02)
        self.alpha = [self.get_parameter(f'alpha{i+1}').value for i in range(4)]

        # Subscribe to odometry
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.prev_odom = None
        self.get_logger().info('Motion model node started.')

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
            return

        drot1, dtrans, drot2 = self.compute_motion(self.prev_odom, curr)
        sample = self.sample_motion_model_odometry(self.prev_odom,
                                                   drot1, dtrans, drot2)
        self.get_logger().info(
            f"Δrot1={drot1:.3f}, Δtrans={dtrans:.3f}, Δrot2={drot2:.3f} → "
            f"Sample=({sample.x:.3f},{sample.y:.3f},{sample.theta:.3f})"
        )
        self.prev_odom = curr

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

    @staticmethod
    def normalize_angle(angle):
        return math.atan2(math.sin(angle), math.cos(angle))

def main(args=None):
    rclpy.init(args=args)
    node = MotionModel()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
