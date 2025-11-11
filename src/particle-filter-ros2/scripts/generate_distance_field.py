#!/usr/bin/env python3
"""
Generate distance field from map image.
This creates a distance field that can be used for particle filter localization.
"""
import numpy as np
from PIL import Image
import yaml
import os
from scipy.ndimage import distance_transform_edt

def generate_distance_field(map_yaml_path, distance_field_png_path, output_dir):
    """
    Generate distance field from map image.
    
    Args:
        map_yaml_path: Path to map.yaml file
        distance_field_png_path: Path to distance_field.png file
        output_dir: Directory to save output files
    """
    # Load map metadata
    with open(map_yaml_path, 'r') as f:
        map_config = yaml.safe_load(f)
    
    print(f"Map config: {map_config}")
    
    # Load distance field PNG
    distance_img = Image.open(distance_field_png_path)
    distance_array = np.array(distance_img)
    
    # If image is RGB, convert to grayscale
    if len(distance_array.shape) == 3:
        distance_array = np.mean(distance_array, axis=2)
    
    print(f"Distance field shape: {distance_array.shape}")
    print(f"Distance field dtype: {distance_array.dtype}")
    print(f"Distance field min/max (raw): {distance_array.min()}/{distance_array.max()}")
    
    # Normalize to [0, 1] if needed
    if distance_array.dtype == np.uint8:
        distance_array = distance_array.astype(np.float32) / 255.0
    
    # IMPORTANT: In distance_field.png:
    # - Black (0) = FREE SPACE
    # - White (255/1.0) = OBSTACLES
    #
    # We want to create a distance-to-obstacle map where:
    # - Pixels far from obstacles get value close to 1.0
    # - Pixels near/at obstacles get value close to 0.0
    #
    # Strategy:
    # 1. Create binary map: free space = 1, obstacles = 0
    # 2. Use distance transform to get distance from each pixel to nearest obstacle
    # 3. Normalize distances to [0, 1]
    
    print("\nComputing distance transform...")
    
    # Create binary free-space map (black=free=1, white=obstacles=0)
    free_space_map = 1.0 - distance_array  # Invert so free space = 1
    
    # Distance transform: returns distance to nearest False value
    # This gives us distance from each pixel to nearest obstacle
    distances = distance_transform_edt(free_space_map > 0.5)
    
    print(f"Raw distance transform min/max: {distances.min()}/{distances.max()}")
    
    # Normalize distances to [0, 1]
    # Where max distance (farthest from obstacles) maps to 1.0
    if distances.max() > 0:
        distance_field = distances / distances.max()
    else:
        distance_field = distances
    
    distance_field = distance_field.astype(np.float32)
    
    print(f"Normalized distance field min/max: {distance_field.min():.4f}/{distance_field.max():.4f}")
    print(f"Mean distance: {distance_field.mean():.4f}")
    print(f"Median distance: {np.median(distance_field):.4f}")
    
    # Count how many pixels have different distance ranges
    for threshold in [0.1, 0.3, 0.5, 0.7, 0.9]:
        count = np.sum(distance_field >= threshold)
        print(f"Pixels with distance >= {threshold}: {count} ({100*count/distance_field.size:.1f}%)")
    
    # Create metadata for the distance field
    metadata = {
        'resolution': map_config['resolution'],
        'origin': map_config['origin'],
        'shape': list(distance_array.shape),
        'max_distance': 2.0,
        'sigma': 0.2
    }
    
    # Save distance field as NPY
    distance_npy_path = os.path.join(output_dir, 'distance_field.npy')
    np.save(distance_npy_path, distance_field)
    print(f"\nSaved distance field to: {distance_npy_path}")
    
    # Save metadata as YAML
    metadata_yaml_path = os.path.join(output_dir, 'map_metadata.yaml')
    with open(metadata_yaml_path, 'w') as f:
        yaml.dump(metadata, f)
    print(f"Saved metadata to: {metadata_yaml_path}")
    
    # Also save visualization of distance field
    distance_vis = (distance_field * 255).astype(np.uint8)
    distance_vis_path = os.path.join(output_dir, 'distance_field_visualization.png')
    Image.fromarray(distance_vis).save(distance_vis_path)
    print(f"Saved visualization to: {distance_vis_path}")
    
    return distance_npy_path, metadata_yaml_path

if __name__ == '__main__':
    import sys
    
    maps_dir = os.path.join(os.path.dirname(__file__), '..', 'maps')
    
    map_yaml = os.path.join(maps_dir, 'map.yaml')
    distance_png = os.path.join(maps_dir, 'distance_field.png')
    
    if not os.path.exists(map_yaml):
        print(f"Error: {map_yaml} not found")
        sys.exit(1)
    
    if not os.path.exists(distance_png):
        print(f"Error: {distance_png} not found")
        sys.exit(1)
    
    print(f"Generating distance field from:")
    print(f"  Map YAML: {map_yaml}")
    print(f"  Distance PNG: {distance_png}")
    print(f"  Output dir: {maps_dir}")
    
    generate_distance_field(map_yaml, distance_png, maps_dir)
