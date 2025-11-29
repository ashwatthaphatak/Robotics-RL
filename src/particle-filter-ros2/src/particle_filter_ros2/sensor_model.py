#!/usr/bin/env python3
import numpy as np
import yaml

class SensorModel:
    """
    Likelihood field range finder model.
    Algorithm: likelihood_field_range_finder_model(z_t, x_t, m)
    """
    
    def __init__(self, likelihood_field_path, metadata_path, distance_map_path,
                 z_hit=0.95, z_random=0.05, sigma_hit=0.2, laser_offset_angle=0.0):
        """
        Initialize sensor model.
        
        Args:
            likelihood_field_path: Path to likelihood field .npy file
            metadata_path: Path to metadata YAML file
            distance_map_path: Path to distance map .npy file
            z_hit: Weight for hit probability (z_hit in algorithm)
            z_random: Weight for random measurement (z_random in algorithm)
            sigma_hit: Standard deviation for measurement noise (σ_hit)
            laser_offset_angle: Mounting offset angle of laser relative to base_link (radians)
                               Default 0.0 assumes laser points forward same as base_link
        """
        # Load likelihood field (not actually used in simplified version)
        # self.likelihood_field = np.load(likelihood_field_path)
        
        # Load distance map (this is the "dist" lookup table)
        self.distance_map = np.load(distance_map_path)
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = yaml.safe_load(f)
        
        self.resolution = metadata['resolution']
        self.origin = np.array(metadata['origin'][:2])  # [x, y]
        self.height, self.width = self.distance_map.shape
        
        # Sensor model parameters
        self.z_hit = z_hit
        self.z_random = z_random
        self.sigma_hit = sigma_hit
        self.laser_offset_angle = laser_offset_angle
        
        # Sensor specifications (will be updated from LaserScan)
        self.z_max = None  # Maximum range
        
    def world_to_map(self, x, y):
        """
        Convert world coordinates to map pixel coordinates.
        
        Args:
            x, y: World coordinates (meters)
            
        Returns:
            mx, my: Map coordinates (pixels)
        """
        mx = int((x - self.origin[0]) / self.resolution)
        my = int((y - self.origin[1]) / self.resolution)
        return mx, my
    
    def prob(self, dist, sigma):
        """
        Compute probability density for Gaussian distribution.
        prob(dist, σ_hit) = exp(-dist^2 / (2 * σ^2)) / (sqrt(2π) * σ)
        
        For simplicity, we can use unnormalized Gaussian:
        exp(-dist^2 / (2 * σ^2))
        """
        return np.exp(-dist**2 / (2 * sigma**2))
    
    def likelihood_field_range_finder_model(self, z_t, x_t, laser_scan):
        """
        Algorithm likelihood_field_range_finder_model(z_t, x_t, m).
        
        Args:
            z_t: Laser scan (ranges array)
            x_t: Particle pose [x, y, θ]
            laser_scan: Full LaserScan message for metadata
            
        Returns:
            q: Likelihood/weight for this particle
        """
        # Extract particle pose
        x, y, theta = x_t
        
        # Update z_max from laser scan
        self.z_max = laser_scan.range_max
        
        # Line 1: q = 1
        q = 1.0
        
        # Line 2: for all k do (iterate through laser beams)
        # Use ALL beams for maximum heading discrimination
        step = 1  # Use every beam - no subsampling
        
        beam_count = 0
        for k in range(0, len(z_t), step):
            z_k = z_t[k]
            
            # Line 3: if z_k^t ≠ z_max (skip max range readings)
            if np.isinf(z_k) or z_k >= laser_scan.range_max:
                continue
            
            # Also skip invalid readings
            if z_k < laser_scan.range_min:
                continue
            
            beam_count += 1
            
            # Beam angle in world frame
            theta_k = laser_scan.angle_min + k * laser_scan.angle_increment
            
            # Account for laser mounting offset relative to base_link
            theta_k_world = theta_k + self.laser_offset_angle
            
            # Assuming sensor is at robot center (x_k,sens = 0, y_k,sens = 0)
            # Line 4-5: Compute endpoint location
            # x_z_k = x + x_k,sens·cos(θ) - y_k,sens·sin(θ) + z_k·cos(θ + θ_k,sens)
            # y_z_k = y + y_k,sens·cos(θ) + x_k,sens·sin(θ) + z_k·sin(θ + θ_k,sens)
            x_z_k = x + z_k * np.cos(theta + theta_k_world)
            y_z_k = y + z_k * np.sin(theta + theta_k_world)
            
            # Line 6: Check lookup table (distance matrix)
            # Convert to map coordinates
            mx, my = self.world_to_map(x_z_k, y_z_k)
            
            # Check if within map bounds
            if 0 <= mx < self.width and 0 <= my < self.height:
                # Get minimum distance from distance map
                dist = self.distance_map[my, mx]
            else:
                # Outside map - assign large distance (penalizes wrong heading)
                dist = self.z_max
            
            # Line 7: Compute probability density
            # q = q · (z_hit · prob(dist, σ_hit) + z_random / z_max)
            p_hit = self.prob(dist, self.sigma_hit)
            p_random = 1.0 / self.z_max
            p = self.z_hit * p_hit + self.z_random * p_random
            
            # Line 8: Multiply probabilities
            q *= p
        
        # If we didn't process any beams, return low weight
        if beam_count == 0:
            q = 1e-10
        
        # Line 9: return q
        return q
    
    def compute_weights(self, particles, laser_scan):
        """
        Compute weights for all particles using sensor model.
        
        Args:
            particles: Nx3 array of particle poses
            laser_scan: LaserScan message
            
        Returns:
            weights: N-length array of particle weights
        """
        num_particles = particles.shape[0]
        weights = np.zeros(num_particles)
        
        z_t = np.array(laser_scan.ranges)
        
        for i in range(num_particles):
            weights[i] = self.likelihood_field_range_finder_model(
                z_t, particles[i], laser_scan
            )
        
        # Normalize weights
        weights += 1e-300  # Avoid division by zero
        weights /= np.sum(weights)
        
        return weights