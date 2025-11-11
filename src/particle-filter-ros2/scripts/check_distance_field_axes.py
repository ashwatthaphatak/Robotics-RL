#!/usr/bin/env python3
"""
Check if map axes are inverted by comparing expected vs actual distance field values.
"""
import numpy as np
import yaml
import os
from PIL import Image
import matplotlib.pyplot as plt

maps_dir = os.path.expanduser('~/ros2_ws/src/particle-filter-ros2/maps')

# Load metadata
with open(os.path.join(maps_dir, 'map_metadata.yaml')) as f:
    meta = yaml.safe_load(f)

# Load distance field
dist = np.load(os.path.join(maps_dir, 'distance_field.npy'))

# Load PNG for visualization
png_img = Image.open(os.path.join(maps_dir, 'distance_field.png'))
png_array = np.array(png_img)
if len(png_array.shape) == 3:
    png_array = np.mean(png_array, axis=2)

print("="*70)
print("DISTANCE FIELD AXIS CHECK")
print("="*70)

origin = meta['origin'][:2]
resolution = meta['resolution']
height, width = meta['shape']

print(f"Metadata shape: height={height}, width={width}")
print(f"Distance field array shape: {dist.shape}")
print(f"PNG array shape: {png_array.shape}")

# Check if distance field is mostly zero or non-zero
zero_count = np.sum(dist == 0)
nonzero_count = np.sum(dist > 0)
total_count = dist.size

print(f"\nDistance field values:")
print(f"  Zero pixels: {zero_count} ({100*zero_count/total_count:.1f}%)")
print(f"  Non-zero pixels: {nonzero_count} ({100*nonzero_count/total_count:.1f}%)")
print(f"  Mean: {dist.mean():.4f}")
print(f"  Std: {dist.std():.4f}")

# Check corners to detect if axes are swapped
print("\nCorner analysis:")
corners = {
    "Top-left [0,0]": dist[0, 0],
    "Top-right [0,-1]": dist[0, -1],
    "Bottom-left [-1,0]": dist[-1, 0],
    "Bottom-right [-1,-1]": dist[-1, -1],
    "Center": dist[height//2, width//2]
}

for name, val in corners.items():
    print(f"  {name}: {val:.4f}")

# Create a simple test: most distance fields should have high values (free space)
# and low values (obstacles)
print("\nStatistics:")
print(f"  Percentile 25%: {np.percentile(dist, 25):.4f}")
print(f"  Percentile 50%: {np.percentile(dist, 50):.4f}")
print(f"  Percentile 75%: {np.percentile(dist, 75):.4f}")
print(f"  Percentile 90%: {np.percentile(dist, 90):.4f}")

# Check if distance field looks reasonable
# Most distances should NOT be 0 (that would indicate obstacle/wall everywhere)
if zero_count > total_count * 0.5:
    print("\n⚠️  WARNING: More than 50% of distance field is 0!")
    print("   This might indicate:")
    print("   - Distance field PNG is inverted (black=free, white=obstacle)")
    print("   - Distance field is not actually a distance map")

print("\n" + "="*70)
print("To debug further:")
print("1. Plot distance_field.npy with matplotlib to visualize")
print("2. Check if white areas are FREE SPACE (should have high distances)")
print("3. Check if black areas are OBSTACLES (should have distance=0)")
print("="*70)
