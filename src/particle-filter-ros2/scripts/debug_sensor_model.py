#!/usr/bin/env python3
"""
Debug script to verify particle filter weighting is correct.
Tests sensor model with synthetic data.
"""
import sys
sys.path.insert(0, '/home/aaphatak/ros2_ws/src/particle-filter-ros2/src')

import numpy as np
from particle_filter_ros2.sensor_model import SensorModel
from sensor_msgs.msg import LaserScan
import os

# Initialize sensor model
maps_dir = '/home/aaphatak/ros2_ws/src/particle-filter-ros2/maps'

sensor_model = SensorModel(
    likelihood_field_path=os.path.join(maps_dir, 'likelihood_field.npy'),
    metadata_path=os.path.join(maps_dir, 'map_metadata.yaml'),
    distance_map_path=os.path.join(maps_dir, 'distance_field.npy'),
    z_hit=0.8,
    z_random=0.2,
    sigma_hit=0.2
)

print("="*70)
print("SENSOR MODEL DEBUG TEST")
print("="*70)

# Load distance field to check values
distance_field = np.load(os.path.join(maps_dir, 'distance_field.npy'))
print(f"\nDistance field stats:")
print(f"  Min: {distance_field.min():.4f}")
print(f"  Max: {distance_field.max():.4f}")
print(f"  Mean: {distance_field.mean():.4f}")

# Test world_to_map conversion
print(f"\nSensor model metadata:")
print(f"  Origin: {sensor_model.origin}")
print(f"  Resolution: {sensor_model.resolution}")
print(f"  Map size: {sensor_model.width} x {sensor_model.height}")

# Test coordinate transforms
test_points = [
    (0, 0, "World origin"),
    (5, 5, "World (5,5)"),
    (-5.711, -5.038, "Map origin"),
]

print(f"\nCoordinate transform test:")
for wx, wy, label in test_points:
    mx, my = sensor_model.world_to_map(wx, wy)
    in_bounds = (0 <= mx < sensor_model.width and 0 <= my < sensor_model.height)
    if in_bounds:
        dist_val = distance_field[my, mx]
        print(f"  {label:20s} -> world({wx:6.2f}, {wy:6.2f}) -> map({mx:3d}, {my:3d}) -> dist={dist_val:.4f} (in bounds: ✓)")
    else:
        print(f"  {label:20s} -> world({wx:6.2f}, {wy:6.2f}) -> map({mx:3d}, {my:3d}) (OUT OF BOUNDS ✗)")

# Create synthetic laser scan
print(f"\nCreating synthetic laser scan...")
num_beams = 360
laser_scan = LaserScan()
laser_scan.angle_min = -np.pi
laser_scan.angle_max = np.pi
laser_scan.angle_increment = 2 * np.pi / num_beams
laser_scan.range_min = 0.1
laser_scan.range_max = 3.5

# Create simple laser scan: all beams same distance
laser_scan.ranges = [2.0] * num_beams
laser_scan.header.frame_id = 'base_link'

print(f"  Beams: {len(laser_scan.ranges)}")
print(f"  All ranges: {laser_scan.ranges[0]}m")
print(f"  Angle range: {np.degrees(laser_scan.angle_min):.1f}° to {np.degrees(laser_scan.angle_max):.1f}°")

# Test particles at different locations
test_particles = [
    np.array([-2.0, -0.5, 0.0]),      # Near known position
    np.array([0.0, 0.0, 0.0]),        # World origin
    np.array([5.0, 5.0, 0.0]),        # Far away
    np.array([-5.7, -5.0, 0.0]),      # Near map corner
]

print(f"\nTesting particles with synthetic laser scan:")
print(f"{'Particle Position':<25s} {'X':<8s} {'Y':<8s} {'Weight':<12s} {'Log Weight':<12s}")
print("-"*70)

weights = []
for particle in test_particles:
    w = sensor_model.likelihood_field_range_finder_model(
        np.array(laser_scan.ranges), particle, laser_scan, debug=False
    )
    weights.append(w)
    log_w = np.log(w) if w > 0 else -np.inf
    print(f"({particle[0]:6.2f}, {particle[1]:6.2f}){'':<5s} {particle[0]:8.2f} {particle[1]:8.2f} {w:.2e}       {log_w:.2f}")

# Normalize and check if reasonable
weights = np.array(weights)
weights_norm = weights / weights.sum()

print(f"\nNormalized weights:")
for i, (particle, w) in enumerate(zip(test_particles, weights_norm)):
    print(f"  Particle {i} at ({particle[0]:6.2f}, {particle[1]:6.2f}): {w:.4f}")

best_idx = np.argmax(weights)
worst_idx = np.argmin(weights)

print(f"\n{'='*70}")
if weights[best_idx] > weights[worst_idx] * 100:
    print("✓ Weight discrimination is good!")
    print(f"  Best particle:  {weights[best_idx]:.2e}")
    print(f"  Worst particle: {weights[worst_idx]:.2e}")
    print(f"  Ratio: {weights[best_idx]/weights[worst_idx]:.1f}x")
else:
    print("✗ Weight discrimination is POOR!")
    print(f"  Best particle:  {weights[best_idx]:.2e}")
    print(f"  Worst particle: {weights[worst_idx]:.2e}")
    print(f"  Ratio: {weights[best_idx]/weights[worst_idx]:.1f}x (should be >> 1)")
    print("\nDEBUG: Running with debug=True...")
    
    # Re-run best particle with debug
    print(f"\nBest particle at ({test_particles[best_idx][0]:.2f}, {test_particles[best_idx][1]:.2f}):")
    _ = sensor_model.likelihood_field_range_finder_model(
        np.array(laser_scan.ranges), test_particles[best_idx], laser_scan, debug=True
    )
    
    print(f"\nWorst particle at ({test_particles[worst_idx][0]:.2f}, {test_particles[worst_idx][1]:.2f}):")
    _ = sensor_model.likelihood_field_range_finder_model(
        np.array(laser_scan.ranges), test_particles[worst_idx], laser_scan, debug=True
    )

print("="*70)
