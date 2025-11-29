#!/usr/bin/env python3
"""
Diagnostic script to identify heading reference misalignment between particles and ego vehicle.
Run this while the particle filter is running.

Usage: ros2 run particle_filter_ros2 diagnose_heading
"""
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
import numpy as np
from scipy.spatial.transform import Rotation
from collections import deque

class HeadingDiagnostic(Node):
    def __init__(self):
        super().__init__('heading_diagnostic')
        
        self.particle_sub = self.create_subscription(
            PoseArray,
            '/particlecloud',
            self.particle_callback,
            10
        )
        
        self.pose_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            '/amcl_pose',
            self.pose_callback,
            10
        )
        
        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )
        
        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )
        
        # Store recent data
        self.particle_headings = deque(maxlen=5)
        self.estimated_heading = None
        self.odom_heading = None
        self.scan_info = None
        self.callback_count = 0
        
    def quat_to_yaw(self, quat):
        """Convert quaternion to yaw angle."""
        r = Rotation.from_quat([quat.x, quat.y, quat.z, quat.w])
        return r.as_euler('xyz')[2]
    
    def normalize_angle(self, angle):
        """Normalize angle to [-pi, pi]."""
        return np.arctan2(np.sin(angle), np.cos(angle))
    
    def particle_callback(self, msg):
        """Extract particle headings."""
        headings = []
        for pose in msg.poses:
            yaw = self.quat_to_yaw(pose.orientation)
            headings.append(yaw)
        
        if headings:
            self.particle_headings.append(headings)
    
    def pose_callback(self, msg):
        """Extract estimated pose heading."""
        self.estimated_heading = self.quat_to_yaw(msg.pose.pose.orientation)
    
    def odom_callback(self, msg):
        """Extract odometry heading."""
        self.odom_heading = self.quat_to_yaw(msg.pose.pose.orientation)
    
    def scan_callback(self, msg):
        """Store scan information."""
        self.scan_info = {
            'frame_id': msg.header.frame_id,
            'angle_min': msg.angle_min,
            'angle_max': msg.angle_max,
            'angle_increment': msg.angle_increment,
            'range_min': msg.range_min,
            'range_max': msg.range_max,
            'num_ranges': len(msg.ranges)
        }
        
        # Print diagnostic every N scans
        self.callback_count += 1
        if self.callback_count % 10 == 0:
            self.print_diagnostics()
    
    def print_diagnostics(self):
        """Print heading diagnostic information."""
        print("\n" + "="*80)
        print(f"HEADING DIAGNOSTIC (Callback #{self.callback_count})")
        print("="*80)
        
        # Print odometry heading
        if self.odom_heading is not None:
            print(f"\nOdometry Heading (ego vehicle):")
            print(f"  Yaw: {self.odom_heading:.4f} rad ({np.degrees(self.odom_heading):.2f}°)")
        
        # Print particle headings
        if self.particle_headings:
            recent_headings = self.particle_headings[-1]
            mean_heading = np.mean(recent_headings)
            std_heading = np.std(recent_headings)
            
            print(f"\nParticle Headings:")
            print(f"  Mean: {mean_heading:.4f} rad ({np.degrees(mean_heading):.2f}°)")
            print(f"  Std Dev: {std_heading:.4f} rad ({np.degrees(std_heading):.2f}°)")
            print(f"  Min: {np.min(recent_headings):.4f} rad ({np.degrees(np.min(recent_headings)):.2f}°)")
            print(f"  Max: {np.max(recent_headings):.4f} rad ({np.degrees(np.max(recent_headings)):.2f}°)")
        
        # Print estimated pose heading
        if self.estimated_heading is not None:
            print(f"\nEstimated Pose Heading (from particle cloud):")
            print(f"  Yaw: {self.estimated_heading:.4f} rad ({np.degrees(self.estimated_heading):.2f}°)")
        
        # Compare headings
        if self.odom_heading is not None and self.particle_headings:
            recent_headings = self.particle_headings[-1]
            mean_particle_heading = np.mean(recent_headings)
            
            # Calculate differences
            diff = self.normalize_angle(self.odom_heading - mean_particle_heading)
            print(f"\nHeading Alignment Analysis:")
            print(f"  Difference (Odom - Particle Mean): {diff:.4f} rad ({np.degrees(diff):.2f}°)")
            
            # Check if there's a consistent offset
            if abs(diff) > 0.1:  # More than ~5.7 degrees
                print(f"  ⚠️  WARNING: Significant heading offset detected!")
                print(f"  Particles offset by ~{np.degrees(diff):.1f}° from ego vehicle")
                
                # Try to identify common offsets
                if abs(diff - np.pi) < 0.1:
                    print(f"  ➜ Possible issue: 180° rotation (heading directions opposite)")
                elif abs(diff - np.pi/2) < 0.1:
                    print(f"  ➜ Possible issue: 90° CCW rotation")
                elif abs(diff + np.pi/2) < 0.1:
                    print(f"  ➜ Possible issue: 90° CW rotation")
            else:
                print(f"  ✓ Headings appear aligned")
        
        # Print scan information
        if self.scan_info:
            print(f"\nLaser Scan Configuration:")
            print(f"  Frame ID: {self.scan_info['frame_id']}")
            print(f"  Angle range: {self.scan_info['angle_min']:.4f} to {self.scan_info['angle_max']:.4f} rad")
            print(f"    ({np.degrees(self.scan_info['angle_min']):.1f}° to {np.degrees(self.scan_info['angle_max']):.1f}°)")
            print(f"  Angle increment: {self.scan_info['angle_increment']:.4f} rad ({np.degrees(self.scan_info['angle_increment']):.2f}°)")
            print(f"  Range: {self.scan_info['range_min']:.3f} to {self.scan_info['range_max']:.3f} m")
            print(f"  Num ranges: {self.scan_info['num_ranges']}")
        
        print("="*80 + "\n")

def main():
    rclpy.init()
    diagnostic = HeadingDiagnostic()
    
    print("\n" + "="*80)
    print("HEADING REFERENCE DIAGNOSTIC NODE")
    print("="*80)
    print("Listening for particles, odometry, pose estimates, and laser scans...")
    print("Diagnostic output will appear every 10 scans (roughly every 5-10 seconds)")
    print("="*80 + "\n")
    
    rclpy.spin(diagnostic)

if __name__ == '__main__':
    main()
