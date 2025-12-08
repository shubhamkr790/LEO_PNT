"""
Measurement models for EKF updates.

Defines how measurements relate to states and their associated uncertainties.
Includes adaptive noise covariance matrices with elevation and multipath factors.
"""

import numpy as np
from typing import Dict


def gnss_measurement_model(fix_quality: Dict) -> np.ndarray:
    """
    Compute GNSS measurement noise covariance.
    
    Args:
        fix_quality: Dict with hdop, num_sats, etc.
    
    Returns:
        R matrix [3x3] for position measurement (meters²)
    """
    hdop = fix_quality.get('hdop', 2.0)
    num_sats = fix_quality.get('num_sats', 6)
    
    # Base position error (meters)
    base_error = 2.0
    
    # Scale by DOP
    pos_std = base_error * hdop
    
    # Reduce uncertainty with more satellites
    if num_sats >= 10:
        pos_std *= 0.8
    elif num_sats < 5:
        pos_std *= 1.5
    
    # Covariance matrix (assume independent errors)
    R = np.eye(3) * (pos_std ** 2)
    
    return R


def doppler_measurement_model(snr_db: float, integration_time: float = 0.1,
                               elevation_deg: float = 30.0,
                               multipath_indicator: float = 1.0) -> float:
    """
    Compute adaptive Doppler measurement noise variance.
    
    Args:
        snr_db: Signal-to-noise ratio in dB
        integration_time: Integration time in seconds
        elevation_deg: Satellite elevation angle (degrees)
        multipath_indicator: Multipath risk factor [1.0 = none, 2-5 = high]
    
    Returns:
        R variance for Doppler measurement (Hz²)
    """
    # Doppler measurement accuracy depends on SNR and integration time
    # Theoretical bound: σ_f ≈ 1 / (2π * T * sqrt(SNR))
    
    snr_linear = 10 ** (snr_db / 10)
    
    if snr_linear > 0 and integration_time > 0:
        doppler_std = 1.0 / (2 * np.pi * integration_time * np.sqrt(snr_linear))
    else:
        doppler_std = 100.0  # Default high uncertainty
    
    # Add practical noise floor
    doppler_std = max(doppler_std, 5.0)  # At least 5 Hz std
    
    # Elevation-based scaling (low elevation = higher multipath + atmospheric uncertainty)
    if elevation_deg < 10.0:
        elevation_factor = 3.0  # Very high uncertainty
    elif elevation_deg < 20.0:
        elevation_factor = 2.0  # High uncertainty
    elif elevation_deg < 30.0:
        elevation_factor = 1.5  # Moderate uncertainty
    else:
        elevation_factor = 1.0  # Nominal
    
    # Apply elevation scaling
    doppler_std *= elevation_factor
    
    # Apply multipath scaling
    doppler_std *= multipath_indicator
    
    R = doppler_std ** 2
    
    return R


def imu_measurement_model(quality: str = 'medium') -> np.ndarray:
    """
    Compute IMU measurement noise covariance.
    
    Args:
        quality: 'low', 'medium', or 'high'
    
    Returns:
        R matrix [3x3] for acceleration measurement (m/s²)²
    """
    # Accelerometer noise density (m/s²/√Hz)
    noise_density = {
        'low': 0.01,     # Consumer grade (MPU-9250)
        'medium': 0.001,  # Industrial grade
        'high': 0.0001   # Tactical grade
    }
    
    density = noise_density.get(quality, 0.01)
    
    # Assume 100 Hz sampling
    accel_std = density * np.sqrt(100)
    
    R = np.eye(3) * (accel_std ** 2)
    
    return R


def barometer_measurement_model(altitude: float) -> float:
    """
    Compute barometer measurement noise variance.
    
    Args:
        altitude: Current altitude (meters)
    
    Returns:
        R variance for altitude measurement (m²)
    """
    # Barometer accuracy degrades with altitude
    base_std = 1.0  # 1 meter at sea level
    
    # Add altitude-dependent error (pressure decreases with altitude)
    altitude_factor = 1.0 + altitude / 10000.0  # 10% per 1000m
    
    alt_std = base_std * altitude_factor
    
    R = alt_std ** 2
    
    return R


def compute_measurement_weight_matrix(measurements: list, cognitive_weights: list) -> np.ndarray:
    """
    Compute weighted measurement covariance for multiple LEO satellites.
    
    Args:
        measurements: List of measurement dicts
        cognitive_weights: List of cognitive weights [0, 1]
    
    Returns:
        Weighted R matrix
    """
    # Weight measurements by cognitive trust scores
    # Lower weight = higher uncertainty
    
    weighted_R = []
    
    for meas, weight in zip(measurements, cognitive_weights):
        snr = meas.get('snr', 0.0)
        R_base = doppler_measurement_model(snr)
        
        # Increase uncertainty for low-weight measurements
        if weight > 0:
            R_weighted = R_base / weight
        else:
            R_weighted = R_base * 1e6  # Effectively reject
        
        weighted_R.append(R_weighted)
    
    return np.array(weighted_R)
