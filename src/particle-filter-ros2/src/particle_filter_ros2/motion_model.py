#!/usr/bin/env python3
import numpy as np
from scipy.spatial.transform import Rotation

def sample_normal_distribution(b):
    """
    Sample from zero-mean normal distribution with variance b^2/6.
    This approximates using sum of 12 uniform random samples.
    """
    return (b / np.sqrt(6)) * np.sum(np.random.uniform(-1, 1, 12))

class MotionModel:
    """
    Odometry-based motion model using sample_motion_model algorithm.
    Based on Probabilistic Robotics (Thrun et al.)
    """
    
    def __init__(self, alpha1=0.1, alpha2=0.1, alpha3=0.1, alpha4=0.1):
        """
        Initialize motion model with noise parameters.
        
        Args:
            alpha1: Rotation noise from rotation
            alpha2: Rotation noise from translation
            alpha3: Translation noise from translation
            alpha4: Translation noise from rotation
        """
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.alpha3 = alpha3
        self.alpha4 = alpha4
        
        self.prev_odom = None
    
    def quaternion_to_yaw(self, quat):
        """Convert quaternion to yaw angle."""
        r = Rotation.from_quat([quat.x, quat.y, quat.z, quat.w])
        return r.as_euler('xyz')[2]
    
    def compute_odometry_change(self, odom_msg):
        """
        Compute odometry change (δ_rot1, δ_trans, δ_rot2) from consecutive odometry readings.
        This is the "u" in the algorithm.
        
        Args:
            odom_msg: Current odometry message
            
        Returns:
            tuple: (δ_rot1, δ_trans, δ_rot2) or None if first call
        """
        # Extract current pose
        x_curr = odom_msg.pose.pose.position.x
        y_curr = odom_msg.pose.pose.position.y
        theta_curr = self.quaternion_to_yaw(odom_msg.pose.pose.orientation)
        
        # First call - store and return None
        if self.prev_odom is None:
            self.prev_odom = (x_curr, y_curr, theta_curr)
            return None
        
        # Extract previous pose (x, y, θ) from prev odometry
        x_prev, y_prev, theta_prev = self.prev_odom
        
        # Compute deltas according to algorithm
        delta_x = x_curr - x_prev
        delta_y = y_curr - y_prev
        
        # δ_trans = sqrt((x' - x)^2 + (y' - y)^2)
        delta_trans = np.sqrt(delta_x**2 + delta_y**2)
        
        # δ_rot1 = atan2(y' - y, x' - x) - θ
        delta_rot1 = np.arctan2(delta_y, delta_x) - theta_prev
        
        # δ_rot2 = θ' - θ - δ_rot1
        delta_rot2 = theta_curr - theta_prev - delta_rot1
        
        # Normalize angles to [-pi, pi]
        delta_rot1 = np.arctan2(np.sin(delta_rot1), np.cos(delta_rot1))
        delta_rot2 = np.arctan2(np.sin(delta_rot2), np.cos(delta_rot2))
        
        # Update previous odometry
        self.prev_odom = (x_curr, y_curr, theta_curr)
        
        return delta_rot1, delta_trans, delta_rot2
    
    def sample_motion_model_single(self, x, y, theta, delta_rot1, delta_trans, delta_rot2):
        """
        Algorithm sample_motion_model(u, x) for a single particle.
        
        Args:
            x, y, theta: Current particle pose
            delta_rot1, delta_rot2, delta_trans: Odometry reading u
            
        Returns:
            x', y', θ': Updated particle pose
        """
        # Line 1: δ̂_rot1 = δ_rot1 + sample(α1|δ_rot1| + α2·δ_trans)
        delta_rot1_hat = delta_rot1 + sample_normal_distribution(
            self.alpha1 * abs(delta_rot1) + self.alpha2 * delta_trans
        )
        
        # Line 2: δ̂_trans = δ_trans + sample(α3·δ_trans + α4(|δ_rot1| + |δ_rot2|))
        delta_trans_hat = delta_trans + sample_normal_distribution(
            self.alpha3 * delta_trans + self.alpha4 * (abs(delta_rot1) + abs(delta_rot2))
        )
        
        # Line 3: δ̂_rot2 = δ_rot2 + sample(α1|δ_rot2| + α2·δ_trans)
        delta_rot2_hat = delta_rot2 + sample_normal_distribution(
            self.alpha1 * abs(delta_rot2) + self.alpha2 * delta_trans
        )
        
        # Line 4: x' = x + δ̂_trans · cos(θ + δ̂_rot1)
        x_new = x + delta_trans_hat * np.cos(theta + delta_rot1_hat)
        
        # Line 5: y' = y + δ̂_trans · sin(θ + δ̂_rot1)
        y_new = y + delta_trans_hat * np.sin(theta + delta_rot1_hat)
        
        # Line 6: θ' = θ + δ̂_rot1 + δ̂_rot2
        theta_new = theta + delta_rot1_hat + delta_rot2_hat
        
        # Normalize angle to [-pi, pi]
        theta_new = np.arctan2(np.sin(theta_new), np.cos(theta_new))
        
        # Line 7: Return (x', y', θ')
        return x_new, y_new, theta_new
    
    def sample_motion_model(self, particles, delta_rot1, delta_trans, delta_rot2):
        """
        Apply motion model to all particles.
        
        Args:
            particles: Nx3 array of particle poses [x, y, theta]
            delta_rot1, delta_trans, delta_rot2: Odometry reading
            
        Returns:
            Updated Nx3 array of particles
        """
        num_particles = particles.shape[0]
        new_particles = np.zeros_like(particles)
        
        for i in range(num_particles):
            x, y, theta = particles[i]
            x_new, y_new, theta_new = self.sample_motion_model_single(
                x, y, theta, delta_rot1, delta_trans, delta_rot2
            )
            new_particles[i] = [x_new, y_new, theta_new]
        
        return new_particles