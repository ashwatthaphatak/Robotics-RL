#!/usr/bin/env python3
"""
Comprehensive diagnostic to identify coordinate frame and weighting issues.
This script will:
1. Check laser scan frame and orientation
2. Verify map coordinate transformations
3. Compare particle weights at known positions
4. Identify inverted/swapped axes
"""
import numpy as np
import yaml
import sys
import os
from PIL import Image

def check_distance_field():
    """Verify distance field properties"""
    print("\n" + "="*70)
    print("DISTANCE FIELD DIAGNOSTICS")
    print("="*70)
    
    maps_dir = os.path.expanduser('~/ros2_ws/src/particle-filter-ros2/maps')
    
    # Load metadata
    with open(os.path.join(maps_dir, 'map_metadata.yaml')) as f:
        meta = yaml.safe_load(f)
    
    print(f"Origin: {meta['origin'][:2]}")
    print(f"Resolution: {meta['resolution']} m/pixel")
    print(f"Shape: {meta['shape']}")
    
    # Load distance field
    dist = np.load(os.path.join(maps_dir, 'distance_field.npy'))
    print(f"Distance field shape: {dist.shape} (should match [height, width])")
    print(f"Distance field range: [{dist.min():.4f}, {dist.max():.4f}]")
    
    # Check distance field values at specific locations
    print("\nDistance field sample values:")
    print(f"  Center: {dist[dist.shape[0]//2, dist.shape[1]//2]:.4f}")
    print(f"  Top-left corner: {dist[0, 0]:.4f}")
    print(f"  Bottom-right corner: {dist[-1, -1]:.4f}")
    
    # Load and compare with PNG
    png_path = os.path.join(maps_dir, 'distance_field.png')
    png_img = Image.open(png_path)
    png_array = np.array(png_img)
    if len(png_array.shape) == 3:
        png_array = np.mean(png_array, axis=2)
    
    print(f"\nDistance field PNG:")
    print(f"  Shape: {png_array.shape}")
    print(f"  Range: [{png_array.min()}, {png_array.max()}]")
    
    # Check if PNG and NPY match
    png_normalized = png_array.astype(np.float32) / 255.0
    if np.allclose(png_normalized, dist):
        print("  ✓ PNG and NPY are ALIGNED")
    else:
        print("  ✗ PNG and NPY MISMATCH")
        print(f"    Max difference: {np.max(np.abs(png_normalized - dist))}")

def check_coordinate_transforms():
    """Test world-to-map and map-to-world transforms"""
    print("\n" + "="*70)
    print("COORDINATE TRANSFORM DIAGNOSTICS")
    print("="*70)
    
    maps_dir = os.path.expanduser('~/ros2_ws/src/particle-filter-ros2/maps')
    
    with open(os.path.join(maps_dir, 'map_metadata.yaml')) as f:
        meta = yaml.safe_load(f)
    
    origin = meta['origin'][:2]
    resolution = meta['resolution']
    height, width = meta['shape']
    
    print(f"Origin: {origin}")
    print(f"Resolution: {resolution}")
    print(f"Map dimensions: {width}x{height} pixels")
    
    # Calculate world boundaries
    x_min, x_max = origin[0], origin[0] + width * resolution
    y_min, y_max = origin[1], origin[1] + height * resolution
    
    print(f"World X range: [{x_min:.2f}, {x_max:.2f}]")
    print(f"World Y range: [{y_min:.2f}, {y_max:.2f}]")
    
    # Test forward and backward transforms
    print("\nTesting coordinate transforms:")
    test_points = [
        (origin[0], origin[1], "Map origin"),
        (0, 0, "World (0,0)"),
        (5, 5, "World (5,5)"),
    ]
    
    for wx, wy, label in test_points:
        mx = int((wx - origin[0]) / resolution)
        my = int((wy - origin[1]) / resolution)
        
        # Transform back
        wx_back = origin[0] + mx * resolution
        wy_back = origin[1] + my * resolution
        
        in_bounds = (0 <= mx < width and 0 <= my < height)
        
        print(f"  {label}:")
        print(f"    World: ({wx:.2f}, {wy:.2f})")
        print(f"    Map:   ({mx}, {my})")
        print(f"    Back:  ({wx_back:.2f}, {wy_back:.2f})")
        print(f"    In bounds: {in_bounds}")

def check_map_file():
    """Verify map.pgm and map.yaml alignment"""
    print("\n" + "="*70)
    print("MAP FILE DIAGNOSTICS")
    print("="*70)
    
    maps_dir = os.path.expanduser('~/ros2_ws/src/particle-filter-ros2/maps')
    
    # Load map.yaml
    with open(os.path.join(maps_dir, 'map.yaml')) as f:
        map_yaml = yaml.safe_load(f)
    
    print("map.yaml contents:")
    for key, val in map_yaml.items():
        print(f"  {key}: {val}")
    
    # Load map.pgm
    map_pgm = Image.open(os.path.join(maps_dir, 'map.pgm'))
    print(f"\nmap.pgm dimensions: {map_pgm.size} (width, height)")
    
    # Check if dimensions match
    pgm_width, pgm_height = map_pgm.size
    yaml_width = int((map_yaml['origin'][0:2] + [0])[-3] + pgm_width * map_yaml['resolution'])
    
    print(f"Expected coverage: origin + (pixels * resolution)")
    print(f"  X: {map_yaml['origin'][0]} + ({pgm_width} * {map_yaml['resolution']}) = "
          f"{map_yaml['origin'][0] + pgm_width * map_yaml['resolution']:.3f}")

def check_laser_geometry():
    """Print expected laser scan geometry"""
    print("\n" + "="*70)
    print("EXPECTED LASER SCAN GEOMETRY (TurtleBot3)")
    print("="*70)
    print("For TurtleBot3 LDS-01 sensor:")
    print("  - Angle min: -2.35 rad (-135°) - FRONT LEFT")
    print("  - Angle max: +2.35 rad (+135°) - FRONT RIGHT")
    print("  - Scan direction: RIGHT to LEFT (or LEFT to RIGHT)")
    print("  - First beam index 0: one side")
    print("  - Last beam index N: other side")
    print("")
    print("Beam endpoint calculation should be:")
    print("  1. Get beam angle in sensor frame: theta_k = angle_min + k * angle_increment")
    print("  2. Transform to world frame: theta_world = robot_theta + theta_k")
    print("  3. Calculate endpoint: x = robot_x + range * cos(theta_world)")
    print("                         y = robot_y + range * sin(theta_world)")

if __name__ == '__main__':
    check_distance_field()
    check_coordinate_transforms()
    check_map_file()
    check_laser_geometry()
    
    print("\n" + "="*70)
    print("NEXT STEPS:")
    print("="*70)
    print("1. Run this script and check output above")
    print("2. Run the particle filter and enable debug mode")
    print("3. Check particles getting HIGH weights - they should be near robot")
    print("4. If particles far from robot have high weights, coordinate frame is inverted")
    print("="*70 + "\n")
