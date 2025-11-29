#!/usr/bin/env python3
"""Utility node that moves the TurtleBot3 inside Gazebo after spawn."""

import math
import os

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Pose
from ros_gz_interfaces.msg import Entity
from ros_gz_interfaces.srv import SetEntityPose


def yaw_to_quat(yaw: float):
    half = yaw / 2.0
    return (0.0, 0.0, math.sin(half), math.cos(half))


class PoseSetter(Node):
    def __init__(self) -> None:
        super().__init__('set_gz_pose')

        default_entity = os.environ.get('TURTLEBOT3_MODEL', 'turtlebot3_burger')
        self.declare_parameter('entity_name', default_entity)
        self.declare_parameter('x', 0.0)
        self.declare_parameter('y', 0.0)
        self.declare_parameter('z', 0.01)
        self.declare_parameter('yaw', 0.0)

        pose = Pose()
        pose.position.x = float(self.get_parameter('x').value)
        pose.position.y = float(self.get_parameter('y').value)
        pose.position.z = float(self.get_parameter('z').value)
        quat = yaw_to_quat(float(self.get_parameter('yaw').value))
        pose.orientation.x = quat[0]
        pose.orientation.y = quat[1]
        pose.orientation.z = quat[2]
        pose.orientation.w = quat[3]

        self.entity_name = self.get_parameter('entity_name').value
        self.pose = pose

        self.cli = self.create_client(SetEntityPose, '/world/default/set_pose')
        self.timer = self.create_timer(0.5, self._tick)
        self.request_in_flight = False
        self.attempts = 0
        self.max_attempts = 30

    def _tick(self):
        if self.request_in_flight:
            return
        if self.attempts >= self.max_attempts:
            self.get_logger().error('Max attempts reached without moving robot; giving up.')
            self.destroy_node()
            return
        if not self.cli.wait_for_service(timeout_sec=0.5):
            self.get_logger().warn('Waiting for /world/default/set_pose service...')
            return

        req = SetEntityPose.Request()
        req.entity.name = self.entity_name
        req.entity.type = Entity.MODEL
        req.pose = self.pose

        self.get_logger().info(
            f"Requesting pose ({self.pose.position.x:.2f}, {self.pose.position.y:.2f})"
        )
        future = self.cli.call_async(req)
        future.add_done_callback(self._done)
        self.request_in_flight = True
        self.attempts += 1

    def _done(self, future):
        try:
            resp = future.result()
        except Exception as exc:  # pylint: disable=broad-except
            self.get_logger().error(f'Failed to set pose: {exc}')
        else:
            if resp.success:
                self.get_logger().info(
                    f"Spawn pose set for {self.entity_name} at "
                    f"({self.pose.position.x:.2f}, {self.pose.position.y:.2f})"
                )
                self.destroy_node()
                return

            self.get_logger().warn(
                f'Set pose failed (attempt {self.attempts}): {resp.status}. Will retry.'
            )
        finally:
            self.request_in_flight = False


def main(args=None):
    rclpy.init(args=args)
    node = PoseSetter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
