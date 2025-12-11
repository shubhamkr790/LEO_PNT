"""
Atmospheric corrections for LEO satellite signals.

Compensates for ionospheric and tropospheric effects on Doppler measurements.
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)


def apply_ionospheric_correction(doppler_hz: float, elevation_deg: float, 
                                   frequency_hz: float) -> float:
    """
    Apply ionospheric correction to Doppler measurement using simplified Klobuchar model.
    
    Ionospheric delay is frequency-dependent and elevation-dependent.
    For L-band (~1621 MHz), typical delay is 5-50 cm, affecting Doppler by ~0.5-5 Hz.
    
    Args:
        doppler_hz: Measured Doppler shift (Hz)
        elevation_deg: Satellite elevation angle (degrees)
        frequency_hz: Carrier frequency (Hz)
        
    Returns:
        Corrected Doppler (Hz)
    """
    # Simplified ionospheric correction
    # Full Klobuchar model requires GPS almanac parameters, so we use a statistical model
    
    if elevation_deg < 5.0:
        # Very low elevation - ionospheric effect is large and uncertain
        correction_hz = 0.0  # Don't correct, measurement unreliable anyway
    else:
        # Elevation mapping function (increases delay at low elevations)
        elevation_rad = np.radians(elevation_deg)
        mapping = 1.0 / np.sin(elevation_rad)
        
        # Typical ionospheric delay at zenith (L-band): ~10 cm = ~0.5 Hz Doppler effect
        # Scale by frequency squared (ionosphere is dispersive)
        base_correction_hz = 0.5 * (1.5e9 / frequency_hz)**2
        
        # Apply elevation mapping
        correction_hz = base_correction_hz * mapping
        
        # Ionospheric delay slows down the signal, reducing apparent Doppler
        # Correction is additive
        correction_hz = min(correction_hz, 5.0)  # Cap at 5 Hz
    
    corrected_doppler = doppler_hz + correction_hz
    
    logger.debug(f"Iono correction: {correction_hz:.2f} Hz (elev={elevation_deg:.1f}°)")
    
    return corrected_doppler


def apply_tropospheric_correction(doppler_hz: float, elevation_deg: float,
                                    altitude_m: float = 0.0,
                                    latitude_deg: float = 18.5,
                                    pressure_hpa: float = 1013.25) -> float:
    """
    Apply tropospheric correction using Saastamoinen model.
    
    Tropospheric delay is frequency-independent and depends on meteorological conditions.
    For LEO satellites, typical delay is 2-20 cm, affecting Doppler by ~0.2-2 Hz.
    
    Args:
        doppler_hz: Measured Doppler shift (Hz)
        elevation_deg: Satellite elevation angle (degrees)
        altitude_m: User altitude above sea level (meters)
        latitude_deg: User latitude (degrees, default 18.5 for Pune)
        pressure_hpa: Atmospheric pressure (hPa/mbar)
        
    Returns:
        Corrected Doppler (Hz)
    """
    if elevation_deg < 5.0:
        # Very low elevation - tropospheric effect is large and uncertain
        correction_hz = 0.0
    else:
        # Elevation mapping function (Niell mapping function, simplified)
        elevation_rad = np.radians(elevation_deg)
        sin_elev = np.sin(elevation_rad)
        
        # Simplified Niell mapping
        a = 0.0012
        b = 0.0003
        mapping = (1.0 + a / (1.0 + b)) / (sin_elev + a / (sin_elev + b))
        
        # Zenith tropospheric delay (Saastamoinen model)
        # Pressure correction for altitude
        pressure_corrected = pressure_hpa * np.exp(-altitude_m / 8000.0)
        
        # Zenith delay in meters (using actual user latitude)
        zenith_delay_m = (0.002277 * pressure_corrected) / (1.0 - 0.00266 * np.cos(2 * np.radians(latitude_deg)))
        
        # Apply mapping function
        slant_delay_m = zenith_delay_m * mapping
        
        # Convert delay to Doppler effect (derivative of delay)
        # Troposphere delay rate for LEO is ~0.1-1 cm/s = 0.1-1 Hz
        correction_hz = slant_delay_m * 0.01  # Rough conversion
        correction_hz = min(correction_hz, 2.0)  # Cap at 2 Hz
    
    corrected_doppler = doppler_hz + correction_hz
    
    logger.debug(f"Tropo correction: {correction_hz:.2f} Hz (elev={elevation_deg:.1f}°)")
    
    return corrected_doppler


def compute_total_atmospheric_correction(doppler_hz: float, elevation_deg: float,
                                          frequency_hz: float, altitude_m: float = 0.0,
                                          latitude_deg: float = 18.5) -> float:
    """
    Apply both ionospheric and tropospheric corrections.
    
    Args:
        doppler_hz: Measured Doppler shift (Hz)
        elevation_deg: Satellite elevation angle (degrees)
        frequency_hz: Carrier frequency (Hz)
        altitude_m: User altitude (meters)
        latitude_deg: User latitude (degrees, default 18.5 for Pune)
        
    Returns:
        Fully corrected Doppler (Hz)
    """
    # Apply ionospheric correction first
    doppler_iono = apply_ionospheric_correction(doppler_hz, elevation_deg, frequency_hz)
    
    # Then tropospheric
    doppler_corrected = apply_tropospheric_correction(doppler_iono, elevation_deg, altitude_m, latitude_deg)
    
    total_correction = doppler_corrected - doppler_hz
    
    logger.debug(f"Total atmos correction: {total_correction:.2f} Hz")
    
    return doppler_corrected


def compute_elevation_angle(user_pos_ecef: np.ndarray, sat_pos_ecef: np.ndarray) -> float:
    """
    Compute satellite elevation angle from user's perspective.
    
    Args:
        user_pos_ecef: User position in ECEF [x, y, z] (meters)
        sat_pos_ecef: Satellite position in ECEF [x, y, z] (meters)
        
    Returns:
        Elevation angle in degrees (0° = horizon, 90° = zenith)
    """
    # Vector from user to satellite
    los_vector = sat_pos_ecef - user_pos_ecef
    los_norm = los_vector / np.linalg.norm(los_vector)
    
    # Local vertical (user position normalized)
    local_vertical = user_pos_ecef / np.linalg.norm(user_pos_ecef)
    
    # Elevation = 90° - angle between LOS and local horizontal
    # cos(elevation) = dot(LOS, local_vertical)
    cos_elev = np.dot(los_norm, local_vertical)
    elevation_rad = np.arcsin(cos_elev)  # arcsin(cos(theta)) gives elevation
    
    elevation_deg = np.degrees(elevation_rad)
    
    return elevation_deg
