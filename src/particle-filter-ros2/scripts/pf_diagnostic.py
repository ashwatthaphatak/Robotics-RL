#!/usr/bin/env python3
"""
Comprehensive particle filter diagnostic.
Logs detailed information about particle updates.
"""
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
import numpy as np
from scipy.spatial.transform import Rotation

class PFDiagnostic(Node):
    def __init__(self):
        super().__init__('pf_diagnostic')
        
        self.scan_count = 0
        self.odom_count = 0
        self.last_robot_pose = None
        
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
        
        self.pf_cloud_sub = self.create_subscription(
            geometry_msgs.msg.PoseArray,
            '/particlecloud',
            self.pf_cloud_callback,
            10
        )
        
        self.get_logger().info("PF Diagnostic node started")
    
    def odom_callback(self, msg):
        self.odom_count += 1
        
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        
        quat = msg.pose.pose.orientation
        r = Rotation.from_quat([quat.x, quat.y, quat.z, quat.w])
        theta = r.as_euler('xyz')[2]
        
        self.last_robot_pose = (x, y, theta)
        
        if self.odom_count % 5 == 0:
            self.get_logger().info(f"Robot pose: ({x:.2f}, {y:.2f}, {np.degrees(theta):.1f}°)")
    
    def scan_callback(self, msg):
        self.scan_count += 1
        
        if self.scan_count % 5 == 0:
            valid_ranges = [r for r in msg.ranges if not np.isinf(r) and r > msg.range_min and r < msg.range_max]
            self.get_logger().info(
                f"Scan {self.scan_count}: {len(valid_ranges)}/{len(msg.ranges)} valid beams, "
                f"range: {min(valid_ranges):.2f}-{max(valid_ranges):.2f}m"
            )
    
    def pf_cloud_callback(self, msg):
        if len(msg.poses) == 0:
            return
        
        # Calculate particle cloud stats
        xs = [p.position.x for p in msg.poses]
        ys = [p.position.y for p in msg.poses]
        
        mean_x = np.mean(xs)
        mean_y = np.mean(ys)
        std_x = np.std(xs)
        std_y = np.std(ys)
        
        if self.scan_count % 5 == 0:
            self.get_logger().info(
                f"Particles: mean=({mean_x:.2f}, {mean_y:.2f}), "
                f"std=({std_x:.2f}, {std_y:.2f}), "
                f"count={len(msg.poses)}"
            )
            
            if self.last_robot_pose:
                dist_to_robot = np.sqrt(
                    (mean_x - self.last_robot_pose[0])**2 + 
                    (mean_y - self.last_robot_pose[1])**2
                )
                self.get_logger().info(
                    f"Distance from particles to robot: {dist_to_robot:.2f}m " +
                    ("(converged!)" if dist_to_robot < 0.5 else "(not converged)")
                )

def main():
    rclpy.init()
    node = PFDiagnostic()
    
    print("\n" + "="*70)
    print("Particle Filter Diagnostic")
    print("="*70)
    print("Monitoring:")
    print("  - Robot odometry")
    print("  - Laser scans")
    print("  - Particle cloud convergence")
    print("="*70 + "\n")
    
    rclpy.spin(node)

if __name__ == '__main__':
    import geometry_msgs.msg
    main()
