#!/usr/bin/env python3
"""
Diagnostic tool to check particle weighting.
Run this while the particle filter is running to see why particles are being weighted incorrectly.
"""
import sys
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
import numpy as np

class DiagnosticListener(Node):
    def __init__(self):
        super().__init__('diagnostic_listener')
        
        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )
        
        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )
        
        self.first_scan = True
        self.last_odom = None
        
    def odom_callback(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        
        from scipy.spatial.transform import Rotation
        quat = msg.pose.pose.orientation
        r = Rotation.from_quat([quat.x, quat.y, quat.z, quat.w])
        theta = r.as_euler('xyz')[2]
        
        self.last_odom = (x, y, theta)
    
    def scan_callback(self, msg):
        if self.first_scan:
            self.first_scan = False
            self.get_logger().info("="*70)
            self.get_logger().info("LASER SCAN FRAME CHECK")
            self.get_logger().info("="*70)
            self.get_logger().info(f"Frame ID: {msg.header.frame_id}")
            self.get_logger().info(f"Angle min: {msg.angle_min:.4f} rad ({np.degrees(msg.angle_min):.1f}°)")
            self.get_logger().info(f"Angle max: {msg.angle_max:.4f} rad ({np.degrees(msg.angle_max):.1f}°)")
            self.get_logger().info(f"Angle increment: {msg.angle_increment:.4f} rad ({np.degrees(msg.angle_increment):.1f}°)")
            self.get_logger().info(f"Range min: {msg.range_min:.3f} m")
            self.get_logger().info(f"Range max: {msg.range_max:.3f} m")
            self.get_logger().info(f"Num ranges: {len(msg.ranges)}")
            
            # Check first and last few ranges
            valid_ranges = [r for r in msg.ranges if not np.isinf(r) and r > msg.range_min]
            if valid_ranges:
                self.get_logger().info(f"Valid ranges: {len(valid_ranges)}")
                self.get_logger().info(f"Range values (first 5): {msg.ranges[:5]}")
            
            self.get_logger().info("="*70)
            self.get_logger().info("EXPECTED BEHAVIOR:")
            self.get_logger().info("- If angle_min < 0: LEFT beam is negative angle (left side)")
            self.get_logger().info("- If angle_min > 0: Angle increases from LEFT to RIGHT")
            self.get_logger().info("- In TurtleBot3: Typically angle_min ≈ -2.35 (270°) to angle_max ≈ 2.35 (90°)")
            self.get_logger().info("="*70)

def main():
    rclpy.init()
    listener = DiagnosticListener()
    
    print("\nListening for first scan to print diagnostics...")
    rclpy.spin(listener)

if __name__ == '__main__':
    main()
