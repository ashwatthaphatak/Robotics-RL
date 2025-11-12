#!/usr/bin/env python3
"""
Particle Filter for Robot Localization
Implements Monte Carlo Localization (MCL) algorithm
"""
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseArray, Pose, PoseWithCovarianceStamped
import numpy as np
import os
import sys

from ament_index_python.packages import get_package_share_directory
from particle_filter_ros2.motion_model import MotionModel
from particle_filter_ros2.sensor_model import SensorModel


class ParticleFilter(Node):
    """
    Particle Filter Node for robot localization.
    Implements the MCL algorithm with motion and sensor models.
    """
    
    def __init__(self):
        super().__init__('particle_filter')
        
        self.get_logger().info('=== PARTICLE FILTER STARTING ===')
        
        # Parameters
        self.declare_parameter('num_particles', 500)
        self.declare_parameter('alpha1', 0.1)
        self.declare_parameter('alpha2', 0.1)
        self.declare_parameter('alpha3', 0.1)
        self.declare_parameter('alpha4', 0.1)
        self.declare_parameter('z_hit', 0.95)
        self.declare_parameter('z_random', 0.05)
        self.declare_parameter('sigma_hit', 0.2)
        
        self.num_particles = self.get_parameter('num_particles').value
        
        # Initialize particles uniformly over the map
        self.particles = None
        self.weights = None
        self.initialize_particles()
        
        # Initialize motion model
        self.motion_model = MotionModel(
            alpha1=self.get_parameter('alpha1').value,
            alpha2=self.get_parameter('alpha2').value,
            alpha3=self.get_parameter('alpha3').value,
            alpha4=self.get_parameter('alpha4').value
        )
        
        self.get_logger().info('Motion model initialized')
        
        # Initialize sensor model
        try:
            pkg_dir = get_package_share_directory('particle-filter-ros2')
            maps_dir = os.path.join(pkg_dir, 'maps')
            
            self.get_logger().info(f'Loading maps from: {maps_dir}')
            
            self.sensor_model = SensorModel(
                likelihood_field_path=os.path.join(maps_dir, 'likelihood_field.npy'),
                metadata_path=os.path.join(maps_dir, 'likelihood_field_metadata.yaml'),
                distance_map_path=os.path.join(maps_dir, 'likelihood_field_distance.npy'),
                z_hit=self.get_parameter('z_hit').value,
                z_random=self.get_parameter('z_random').value,
                sigma_hit=self.get_parameter('sigma_hit').value
            )
            
            self.get_logger().info('Sensor model initialized')
            
        except Exception as e:
            self.get_logger().error(f'Failed to initialize sensor model: {e}')
            raise
        
        # Subscribers
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
        
        self.initial_pose_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            '/initialpose',
            self.initial_pose_callback,
            10
        )
        
        # Publishers
        self.particle_pub = self.create_publisher(
            PoseArray,
            '/particlecloud',
            10
        )
        
        self.pose_pub = self.create_publisher(
            PoseWithCovarianceStamped,
            '/amcl_pose',
            10
        )
        
        # Timer for visualization (5 Hz for stability)
        self.viz_timer = self.create_timer(0.2, self.publish_particles)
        
        # State
        self.last_odom = None
        self.first_scan_received = False
        self.particles_updated = True  # Publish initial particles
        
        self.get_logger().info(f'=== PARTICLE FILTER INITIALIZED with {self.num_particles} particles ===')
    
    def initialize_particles(self):
        """Initialize particles uniformly over the map."""
        x_min, x_max = -5.0, 5.0
        y_min, y_max = -5.0, 5.0
        
        self.particles = np.zeros((self.num_particles, 3))
        self.particles[:, 0] = np.random.uniform(x_min, x_max, self.num_particles)
        self.particles[:, 1] = np.random.uniform(y_min, y_max, self.num_particles)
        self.particles[:, 2] = np.random.uniform(-np.pi, np.pi, self.num_particles)
        
        self.weights = np.ones(self.num_particles) / self.num_particles
        
        self.get_logger().info('Particles initialized uniformly')
    
    def initial_pose_callback(self, msg):
        """Handle initial pose estimate from RViz."""
        self.get_logger().info('Received initial pose estimate')
        
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        
        quat = msg.pose.pose.orientation
        from scipy.spatial.transform import Rotation
        r = Rotation.from_quat([quat.x, quat.y, quat.z, quat.w])
        theta = r.as_euler('xyz')[2]
        
        self.particles[:, 0] = np.random.normal(x, 0.5, self.num_particles)
        self.particles[:, 1] = np.random.normal(y, 0.5, self.num_particles)
        self.particles[:, 2] = np.random.normal(theta, 0.3, self.num_particles)
        
        self.particles[:, 2] = np.arctan2(np.sin(self.particles[:, 2]), 
                                          np.cos(self.particles[:, 2]))
        
        self.weights = np.ones(self.num_particles) / self.num_particles
        self.motion_model.prev_odom = None
        
        self.get_logger().info(f'Particles reinitialized around ({x:.2f}, {y:.2f}, {theta:.2f})')
        self.particles_updated = True
    
    def odom_callback(self, msg):
        """Process odometry and update particles using motion model."""
        odom_change = self.motion_model.compute_odometry_change(msg)
        
        if odom_change is None:
            return
        
        delta_rot1, delta_trans, delta_rot2 = odom_change
        
        self.particles = self.motion_model.sample_motion_model(
            self.particles, delta_rot1, delta_trans, delta_rot2
        )
        
        # DON'T publish here - let timer handle it
        self.particles_updated = True
    
    def scan_callback(self, msg):
        """Process laser scan and update weights using sensor model."""
        if not self.first_scan_received:
            self.first_scan_received = True
            self.get_logger().info('First scan received')
        
        self.weights = self.sensor_model.compute_weights(self.particles, msg)
        self.resample()
        self.publish_estimated_pose()
        # DON'T publish here either
        self.particles_updated = True
    
    def resample(self):
        """Resample particles using low variance resampling."""
        new_particles = np.zeros_like(self.particles)
        c = np.cumsum(self.weights)
        r = np.random.uniform(0, 1.0 / self.num_particles)
        
        i = 0
        for m in range(self.num_particles):
            u = r + m * (1.0 / self.num_particles)
            while u > c[i]:
                i += 1
            new_particles[m] = self.particles[i]
        
        self.particles = new_particles
        self.weights = np.ones(self.num_particles) / self.num_particles
    
    def publish_particles(self):
        """Publish particle cloud for visualization - called by timer at 5 Hz."""
        if not self.particles_updated:
            return
            
        try:
            msg = PoseArray()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = 'map'

            # Publish ALL particles - no subsampling
            for i in range(len(self.particles)):
                pose = Pose()
                pose.position.x = float(self.particles[i, 0])
                pose.position.y = float(self.particles[i, 1])
                pose.position.z = 0.0

                # Convert yaw to quaternion
                theta = float(self.particles[i, 2])
                pose.orientation.x = 0.0
                pose.orientation.y = 0.0
                pose.orientation.z = float(np.sin(theta / 2.0))
                pose.orientation.w = float(np.cos(theta / 2.0))

                msg.poses.append(pose)

            self.particle_pub.publish(msg)
            self.particles_updated = False

        except Exception as e:
            self.get_logger().error(f'Error publishing particles: {e}')

    def publish_estimated_pose(self):
        """Publish estimated robot pose."""
        x_mean = np.sum(self.weights * self.particles[:, 0])
        y_mean = np.sum(self.weights * self.particles[:, 1])
        
        sin_sum = np.sum(self.weights * np.sin(self.particles[:, 2]))
        cos_sum = np.sum(self.weights * np.cos(self.particles[:, 2]))
        theta_mean = np.arctan2(sin_sum, cos_sum)
        
        msg = PoseWithCovarianceStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        
        msg.pose.pose.position.x = float(x_mean)
        msg.pose.pose.position.y = float(y_mean)
        msg.pose.pose.position.z = 0.0
        
        from scipy.spatial.transform import Rotation
        r = Rotation.from_euler('xyz', [0, 0, theta_mean])
        quat = r.as_quat()
        msg.pose.pose.orientation.x = float(quat[0])
        msg.pose.pose.orientation.y = float(quat[1])
        msg.pose.pose.orientation.z = float(quat[2])
        msg.pose.pose.orientation.w = float(quat[3])
        
        msg.pose.covariance[0] = 0.1
        msg.pose.covariance[7] = 0.1
        msg.pose.covariance[35] = 0.1
        
        self.pose_pub.publish(msg)


def main(args=None):
    print("=== STARTING PARTICLE FILTER NODE ===")
    
    try:
        rclpy.init(args=args)
        particle_filter = ParticleFilter()
        
        print("=== PARTICLE FILTER READY ===")
        rclpy.spin(particle_filter)
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()