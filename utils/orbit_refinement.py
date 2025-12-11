"""
Orbit refinement using Doppler measurements.

This module refines satellite orbits from TLEs using observed Doppler shifts,
reducing position errors from km-level to 10s-100s of meters.
"""

import numpy as np
from typing import List, Tuple, Dict
import logging

logger = logging.getLogger(__name__)

# Speed of light
C = 299792458.0


def refine_satellite_position(sat_pos_tle: np.ndarray, sat_vel_tle: np.ndarray,
                               doppler_measurements: List[Dict],
                               user_pos: np.ndarray, carrier_freq: float,
                               max_iterations: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Refine satellite position/velocity using Doppler measurements via least squares.
    
    This corrects for TLE propagation errors by adjusting the satellite state
    to minimize residuals between predicted and measured Doppler.
    
    Args:
        sat_pos_tle: Initial satellite position from TLE [x, y, z] (meters)
        sat_vel_tle: Initial satellite velocity from TLE [vx, vy, vz] (m/s)
        doppler_measurements: List of Doppler observations with keys:
            - 'doppler': measured Doppler (Hz)
            - 'time': measurement time offset (seconds)
            - 'user_pos': user position at measurement time
        user_pos: User position ECEF (meters)
        carrier_freq: Carrier frequency (Hz)
        max_iterations: Maximum iterations for least squares
        
    Returns:
        (refined_pos, refined_vel): Refined satellite position and velocity
    """
    if len(doppler_measurements) < 2:
        logger.debug("Not enough measurements for orbit refinement")
        return sat_pos_tle, sat_vel_tle
    
    # Initial state: [px, py, pz, vx, vy, vz]
    state = np.concatenate([sat_pos_tle, sat_vel_tle])
    
    for iteration in range(max_iterations):
        # Build Jacobian matrix and residual vector
        H = []  # Jacobian
        residuals = []
        
        for meas in doppler_measurements[:10]:  # Limit to 10 measurements for speed
            # Predicted Doppler from current state
            sat_pos = state[0:3]
            sat_vel = state[3:6]
            
            # Use per-measurement user position if available (handles moving receiver)
            user_pos_i = meas.get('user_pos', user_pos)
            
            # Line of sight
            los = sat_pos - user_pos_i
            los_norm = np.linalg.norm(los)
            if los_norm < 1e-6:
                continue
            los_unit = los / los_norm
            
            # Predicted Doppler
            doppler_pred = -(carrier_freq / C) * np.dot(los_unit, sat_vel)
            
            # Residual
            residuals.append(meas['doppler'] - doppler_pred)
            
            # Jacobian: ∂doppler/∂state
            # ∂doppler/∂pos: depends on LOS direction change
            d_dop_d_pos = (carrier_freq / C) * sat_vel / los_norm
            
            # ∂doppler/∂vel: direct dependency
            d_dop_d_vel = -(carrier_freq / C) * los_unit
            
            H.append(np.concatenate([d_dop_d_pos, d_dop_d_vel]))
        
        if len(residuals) < 2:
            break
        
        H = np.array(H)
        residuals = np.array(residuals)
        
        # Weighted least squares (weight by inverse of residual magnitude)
        weights = 1.0 / (np.abs(residuals) + 1.0)  # Avoid division by zero
        W = np.diag(weights)
        
        # Solve: delta_state = (H^T W H)^-1 H^T W residuals
        try:
            HTW = H.T @ W
            delta_state = np.linalg.solve(HTW @ H, HTW @ residuals)
        except np.linalg.LinAlgError:
            logger.warning("Orbit refinement: singular matrix, stopping")
            break
        
        # Update state
        state += delta_state
        
        # Check convergence
        if np.linalg.norm(delta_state[0:3]) < 1.0:  # Position change < 1 meter
            logger.debug(f"Orbit refinement converged in {iteration+1} iterations")
            break
    
    refined_pos = state[0:3]
    refined_vel = state[3:6]
    
    # Log correction magnitude
    pos_correction = np.linalg.norm(refined_pos - sat_pos_tle)
    if pos_correction > 10.0:  # Only log if significant correction
        logger.info(f"Orbit refined: position correction = {pos_correction:.1f} m")
    
    return refined_pos, refined_vel


def compute_orbit_quality(doppler_measurements: List[Dict], 
                          sat_pos: np.ndarray, sat_vel: np.ndarray,
                          user_pos: np.ndarray, carrier_freq: float) -> float:
    """
    Compute orbit quality metric based on Doppler residuals.
    
    Args:
        doppler_measurements: List of Doppler observations
        sat_pos: Satellite position (meters)
        sat_vel: Satellite velocity (m/s)
        user_pos: User position (meters)
        carrier_freq: Carrier frequency (Hz)
        
    Returns:
        RMS Doppler residual (Hz) - lower is better
    """
    if not doppler_measurements:
        return float('inf')
    
    residuals = []
    for meas in doppler_measurements:
        # Use per-measurement user position if available
        user_pos_i = meas.get('user_pos', user_pos)
        
        # Line of sight
        los = sat_pos - user_pos_i
        los_norm = np.linalg.norm(los)
        if los_norm < 1e-6:
            continue
        los_unit = los / los_norm
        
        # Predicted Doppler
        doppler_pred = -(carrier_freq / C) * np.dot(los_unit, sat_vel)
        
        # Residual
        residuals.append(meas['doppler'] - doppler_pred)
    
    if not residuals:
        return float('inf')
    
    return np.sqrt(np.mean(np.array(residuals)**2))
