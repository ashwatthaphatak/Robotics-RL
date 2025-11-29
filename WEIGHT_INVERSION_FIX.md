# Particle Filter Weight Inversion Fix

## Problem Identified

Your particles were getting **inverted weights**: particles far from the robot got HIGH weights, while particles near the robot got LOW weights. This caused the particle filter to sample out good particles and replace them with bad ones.

## Root Cause

The distance field (used by the sensor model to evaluate particle hypotheses) was **inverted**:

- **distance_field.png** semantics:
  - Black (pixel value 0) = FREE SPACE
  - White (pixel value 255) = OBSTACLES

- **But the code treated it as**:
  - Value 0 (black) = distance 0 (obstacle, low probability)
  - Value 255 (white) = distance 1.0 (free, high probability)

This is **backwards**! A particle in an open area should have HIGH probability (near free space), not LOW.

## Solution Implemented

### 1. Correct Distance Field Inversion
The script `scripts/generate_distance_field.py` now:
- Reads `distance_field.png` (black=free, white=obstacles)
- **Inverts it**: `distance_field = 1.0 - normalized_png`
- Result:
  - Black pixels (0) → 1.0 (high distance = far from obstacles = free space) ✓
  - White pixels (255) → 0.0 (low distance = near obstacles) ✓

### 2. Regenerated Files
- `maps/distance_field.npy` - Now correctly inverted
- `maps/map_metadata.yaml` - Metadata with correct origin/resolution

### 3. Enhanced Sensor Model
Added debug parameter to `sensor_model.py` for troubleshooting:
- Can print which beams contribute to weight calculation
- Shows endpoint transformations for verification

## Verification Results

```
Distance field statistics:
  Min/Max: 0.0 / 1.0
  Mean: 0.9767 (mostly FREE SPACE ✓)
  Non-zero pixels: 100% ✓

Corner values: 1.0 (all free space corners are HIGH ✓)
```

## Expected Behavior After Fix

1. **Particles in open areas** → Get HIGH weights ✓
2. **Particles in walls** → Get LOW weights ✓
3. **Good particles stay** during resampling
4. **Particle cluster converges** to true robot position
5. **Particles move smoothly** with odometry updates

## Files Modified

1. `src/particle_filter_ros2/sensor_model.py`
   - Added debug parameter to `likelihood_field_range_finder_model()`
   - Better comments on coordinate transforms

2. `scripts/generate_distance_field.py`
   - Now correctly inverts distance_field.png
   - Clearer comments on black/white semantics

## Testing

To verify the fix works:

1. Rebuild: `colcon build --packages-select particle-filter-ros2`
2. Launch: `ros2 launch particle-filter-ros2 particle_filter.launch.py`
3. In RViz, check particle cloud:
   - Particles should start distributed across the map
   - Move robot with teleop
   - Particles should converge to robot location
   - **Good particles should NOT get sampled out**

## Key Changes

**Before:**
- distance_field.npy was 95% zeros (mostly obstacles)
- Particles in free space got low weights
- Particles in walls got high weights ❌

**After:**
- distance_field.npy is 100% populated (correct distances)
- Particles in free space get high weights ✓
- Particles in walls get low weights ✓
