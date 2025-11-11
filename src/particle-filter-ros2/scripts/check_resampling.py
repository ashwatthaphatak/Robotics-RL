#!/usr/bin/env python3
"""
Check if adaptive resampling threshold is working correctly.
"""
import numpy as np

def check_resampling():
    print("="*70)
    print("RESAMPLING THRESHOLD CHECK")
    print("="*70)
    
    num_particles = 500
    threshold = 0.5
    
    # Scenario 1: All particles equal weight (after initialization)
    weights_uniform = np.ones(num_particles) / num_particles
    N_eff_uniform = 1.0 / np.sum(weights_uniform**2)
    ratio_uniform = N_eff_uniform / num_particles
    
    print(f"\nScenario 1: All particles equally weighted")
    print(f"  N_eff: {N_eff_uniform:.1f}/{num_particles}")
    print(f"  Ratio: {ratio_uniform:.4f}")
    print(f"  Threshold: {threshold}")
    print(f"  Should resample? {ratio_uniform < threshold} {'✓' if ratio_uniform < threshold else '✗'}")
    
    # Scenario 2: One particle dominates
    weights_dominated = np.zeros(num_particles)
    weights_dominated[0] = 0.99
    weights_dominated[1:] = 0.01 / (num_particles - 1)
    weights_dominated /= weights_dominated.sum()
    N_eff_dominated = 1.0 / np.sum(weights_dominated**2)
    ratio_dominated = N_eff_dominated / num_particles
    
    print(f"\nScenario 2: One particle dominates (99% weight)")
    print(f"  N_eff: {N_eff_dominated:.1f}/{num_particles}")
    print(f"  Ratio: {ratio_dominated:.4f}")
    print(f"  Threshold: {threshold}")
    print(f"  Should resample? {ratio_dominated < threshold} {'✓' if ratio_dominated < threshold else '✗'}")
    
    # Scenario 3: Weights converging (sensor update)
    weights_converged = np.zeros(num_particles)
    # 50 particles have weight 0.01, rest have near zero
    weights_converged[:50] = 0.01
    weights_converged[50:] = 0.001 / (num_particles - 50)
    weights_converged /= weights_converged.sum()
    N_eff_converged = 1.0 / np.sum(weights_converged**2)
    ratio_converged = N_eff_converged / num_particles
    
    print(f"\nScenario 3: Weights somewhat converged")
    print(f"  N_eff: {N_eff_converged:.1f}/{num_particles}")
    print(f"  Ratio: {ratio_converged:.4f}")
    print(f"  Threshold: {threshold}")
    print(f"  Should resample? {ratio_converged < threshold} {'✓' if ratio_converged < threshold else '✗'}")
    
    print("\n" + "="*70)
    print("INTERPRETATION:")
    print("="*70)
    print("Scenario 1 (after init): Don't resample if N_eff ≥ 50% of particles")
    print("Scenario 2 (bad convergence): Resample (one particle too dominant)")
    print("Scenario 3 (good convergence): Resample if N_eff < 50% particles")
    print("\nIf threshold is 0.5:")
    print("  - Resample when N_eff < 0.5 * num_particles")
    print("  - Don't resample when N_eff >= 0.5 * num_particles")

check_resampling()
