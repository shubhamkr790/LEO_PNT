"""
Coordinate transformation functions.
Supports: Geodetic (LLA) ↔ ECEF, NED ↔ ECEF
"""

import numpy as np
from typing import Tuple

# WGS84 ellipsoid parameters
WGS84_A = 6378137.0  # Semi-major axis (m)
WGS84_F = 1.0 / 298.257223563  # Flattening
WGS84_E2 = 2 * WGS84_F - WGS84_F ** 2  # Eccentricity squared


def geodetic_to_ecef(lat: float, lon: float, alt: float) -> np.ndarray:
    """
    Convert geodetic coordinates (latitude, longitude, altitude) to ECEF.
    
    Args:
        lat: Latitude in degrees
        lon: Longitude in degrees
        alt: Altitude above ellipsoid in meters
    
    Returns:
        np.array([x, y, z]) in ECEF coordinates (meters)
    """
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    
    # Radius of curvature in prime vertical
    N = WGS84_A / np.sqrt(1 - WGS84_E2 * np.sin(lat_rad) ** 2)
    
    x = (N + alt) * np.cos(lat_rad) * np.cos(lon_rad)
    y = (N + alt) * np.cos(lat_rad) * np.sin(lon_rad)
    z = (N * (1 - WGS84_E2) + alt) * np.sin(lat_rad)
    
    return np.array([x, y, z])


def ecef_to_geodetic(x: float, y: float, z: float) -> Tuple[float, float, float]:
    """
    Convert ECEF coordinates to geodetic (latitude, longitude, altitude).
    Uses iterative method (Bowring's formula).
    
    Args:
        x, y, z: ECEF coordinates in meters
    
    Returns:
        (lat, lon, alt) in degrees and meters
    """
    # Longitude
    lon = np.arctan2(y, x)
    
    # Distance from Z-axis
    p = np.sqrt(x**2 + y**2)
    
    # Initial latitude estimate
    lat = np.arctan2(z, p * (1 - WGS84_E2))
    
    # Iterative refinement
    for _ in range(5):
        N = WGS84_A / np.sqrt(1 - WGS84_E2 * np.sin(lat) ** 2)
        alt = p / np.cos(lat) - N
        lat = np.arctan2(z, p * (1 - WGS84_E2 * N / (N + alt)))
    
    # Final altitude
    N = WGS84_A / np.sqrt(1 - WGS84_E2 * np.sin(lat) ** 2)
    alt = p / np.cos(lat) - N
    
    return np.degrees(lat), np.degrees(lon), alt


def velocity_ned_to_ecef(lat: float, lon: float, vel_ned: np.ndarray) -> np.ndarray:
    """
    Transform velocity from NED (North-East-Down) to ECEF frame.
    
    Args:
        lat: Latitude in degrees
        lon: Longitude in degrees
        vel_ned: Velocity in NED frame [vn, ve, vd] (m/s)
    
    Returns:
        Velocity in ECEF frame [vx, vy, vz] (m/s)
    """
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    
    # Rotation matrix from NED to ECEF
    R = np.array([
        [-np.sin(lat_rad) * np.cos(lon_rad), -np.sin(lon_rad), -np.cos(lat_rad) * np.cos(lon_rad)],
        [-np.sin(lat_rad) * np.sin(lon_rad),  np.cos(lon_rad), -np.cos(lat_rad) * np.sin(lon_rad)],
        [ np.cos(lat_rad),                    0,                -np.sin(lat_rad)]
    ])
    
    vel_ecef = R @ vel_ned
    return vel_ecef


def velocity_ecef_to_ned(lat: float, lon: float, vel_ecef: np.ndarray) -> np.ndarray:
    """
    Transform velocity from ECEF to NED (North-East-Down) frame.
    
    Args:
        lat: Latitude in degrees
        lon: Longitude in degrees
        vel_ecef: Velocity in ECEF frame [vx, vy, vz] (m/s)
    
    Returns:
        Velocity in NED frame [vn, ve, vd] (m/s)
    """
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    
    # Rotation matrix from ECEF to NED (transpose of NED-to-ECEF)
    R = np.array([
        [-np.sin(lat_rad) * np.cos(lon_rad), -np.sin(lat_rad) * np.sin(lon_rad),  np.cos(lat_rad)],
        [-np.sin(lon_rad),                    np.cos(lon_rad),                    0              ],
        [-np.cos(lat_rad) * np.cos(lon_rad), -np.cos(lat_rad) * np.sin(lon_rad), -np.sin(lat_rad)]
    ])
    
    vel_ned = R @ vel_ecef
    return vel_ned


def compute_los_vector(user_ecef: np.ndarray, sat_ecef: np.ndarray) -> np.ndarray:
    """
    Compute Line-of-Sight (LOS) unit vector from user to satellite.
    
    Args:
        user_ecef: User position in ECEF [x, y, z]
        sat_ecef: Satellite position in ECEF [x, y, z]
    
    Returns:
        Unit LOS vector [x, y, z]
    """
    los = sat_ecef - user_ecef
    los_norm = np.linalg.norm(los)
    
    if los_norm == 0:
        return np.zeros(3)
    
    return los / los_norm


def compute_range(user_ecef: np.ndarray, sat_ecef: np.ndarray) -> float:
    """
    Compute range (distance) from user to satellite.
    
    Args:
        user_ecef: User position in ECEF [x, y, z]
        sat_ecef: Satellite position in ECEF [x, y, z]
    
    Returns:
        Range in meters
    """
    return np.linalg.norm(sat_ecef - user_ecef)


def compute_elevation_azimuth(user_ecef: np.ndarray, sat_ecef: np.ndarray) -> Tuple[float, float]:
    """
    Compute elevation and azimuth angles from user to satellite.
    
    Args:
        user_ecef: User position in ECEF [x, y, z]
        sat_ecef: Satellite position in ECEF [x, y, z]
    
    Returns:
        (elevation, azimuth) in degrees
    """
    # Convert user position to geodetic
    lat, lon, _ = ecef_to_geodetic(*user_ecef)
    
    # LOS vector in ECEF
    los_ecef = sat_ecef - user_ecef
    
    # Transform to NED frame
    los_ned = velocity_ecef_to_ned(lat, lon, los_ecef)
    
    # Compute elevation and azimuth
    down = los_ned[2]
    north = los_ned[0]
    east = los_ned[1]
    
    range_horizontal = np.sqrt(north**2 + east**2)
    elevation = np.degrees(np.arctan2(-down, range_horizontal))
    azimuth = np.degrees(np.arctan2(east, north)) % 360
    
    return elevation, azimuth
