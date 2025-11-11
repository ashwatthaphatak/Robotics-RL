#!/usr/bin/env python3
import numpy as np
import yaml

class SensorModel:
    """
    Likelihood field range finder model.
    Algorithm: likelihood_field_range_finder_model(z_t, x_t, m)
    """
    
    def __init__(self, likelihood_field_path, metadata_path, distance_map_path,
                 z_hit=0.95, z_random=0.05, sigma_hit=0.2):
        """
        Initialize sensor model.
        
        Args:
            likelihood_field_path: Path to likelihood field .npy file
            metadata_path: Path to metadata YAML file
            distance_map_path: Path to distance map .npy file
            z_hit: Weight for hit probability (z_hit in algorithm)
            z_random: Weight for random measurement (z_random in algorithm)
            sigma_hit: Standard deviation for measurement noise (σ_hit)
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
    
    def likelihood_field_range_finder_model(self, z_t, x_t, laser_scan, debug=False):
        """
        Algorithm likelihood_field_range_finder_model(z_t, x_t, m).
        
        Args:
            z_t: Laser scan (ranges array)
            x_t: Particle pose [x, y, θ]
            laser_scan: Full LaserScan message for metadata
            debug: If True, print debug info for this particle
            
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
        # Downsample for efficiency (every 10th beam)
        step = 10
        
        beam_count = 0
        valid_count = 0
        
        for k in range(0, len(z_t), step):
            z_k = z_t[k]
            
            # Line 3: if z_k^t ≠ z_max (skip max range readings)
            if np.isinf(z_k) or z_k >= laser_scan.range_max:
                continue
            
            # Also skip invalid readings
            if z_k < laser_scan.range_min:
                continue
            
            valid_count += 1
            beam_count += 1
            
            # Beam angle in sensor/robot frame
            theta_k = laser_scan.angle_min + k * laser_scan.angle_increment
            
            # Transform to world frame: 
            # The beam angle must be rotated by the particle's theta
            # Then we add the range to get the endpoint in world coordinates
            beam_angle_world = theta + theta_k
            
            # Compute endpoint location in world frame
            x_z_k = x + z_k * np.cos(beam_angle_world)
            y_z_k = y + z_k * np.sin(beam_angle_world)
            
            # Convert to map coordinates
            mx, my = self.world_to_map(x_z_k, y_z_k)
            
            # Check if within map bounds
            if 0 <= mx < self.width and 0 <= my < self.height:
                # Get minimum distance from distance map
                # NOTE: distance_map is indexed as [y, x] (row, col)
                dist = self.distance_map[my, mx]
            else:
                # Outside map - assign large distance (penalize out-of-bounds hits)
                dist = self.z_max
            
            # Compute probability density
            # q = q · (z_hit · prob(dist, σ_hit) + z_random / z_max)
            p_hit = self.prob(dist, self.sigma_hit)
            p_random = 1.0 / self.z_max
            p = self.z_hit * p_hit + self.z_random * p_random
            
            # Multiply probabilities
            q *= p
            
            if debug and valid_count <= 3:
                import sys
                print(f"  Beam {k}: range={z_k:.3f}m, angle_sensor={np.degrees(theta_k):.1f}°, "
                      f"angle_world={np.degrees(beam_angle_world):.1f}°, "
                      f"endpoint=({x_z_k:.2f}, {y_z_k:.2f}), "
                      f"map_px=({mx}, {my}), dist={dist:.4f}, p={p:.6f}", 
                      file=sys.stderr)
        
        if debug:
            import sys
            print(f"Particle ({x:.2f}, {y:.2f}, {np.degrees(theta):.1f}°): "
                  f"valid_beams={valid_count}, final_weight={q:.10f}", 
                  file=sys.stderr)
        
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