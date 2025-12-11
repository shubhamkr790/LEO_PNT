"""
TLE (Two-Line Element) parsing and satellite position/velocity computation.
Uses Skyfield library for orbital propagation.
"""

import numpy as np
from skyfield.api import load, EarthSatellite, wgs84
from skyfield.toposlib import GeographicPosition
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

# Speed of light (m/s)
C = 299792458.0


class TLEManager:
    """Manage TLE data and satellite orbit propagation."""
    
    def __init__(self, tle_file: str):
        """
        Initialize TLE manager with TLE file.
        
        Args:
            tle_file: Path to TLE file
        """
        self.tle_file = Path(tle_file)
        self.satellites: Dict[str, EarthSatellite] = {}
        self.ts = load.timescale()
        
        self._load_tles()
        
    def _load_tles(self):
        """Load TLE data from file."""
        if not self.tle_file.exists():
            logger.error(f"TLE file not found: {self.tle_file}")
            raise FileNotFoundError(f"TLE file not found: {self.tle_file}")
        
        with open(self.tle_file, 'r') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
        
        # Parse TLEs (expect groups of 3 lines: name, line1, line2)
        i = 0
        while i < len(lines) - 2:
            name = lines[i]
            line1 = lines[i + 1]
            line2 = lines[i + 2]
            
            if line1.startswith('1 ') and line2.startswith('2 '):
                try:
                    sat = EarthSatellite(line1, line2, name, self.ts)
                    self.satellites[name] = sat
                    logger.debug(f"Loaded TLE: {name}")
                except Exception as e:
                    logger.warning(f"Failed to parse TLE for {name}: {e}")
            
            i += 3
        
        logger.info(f"Loaded {len(self.satellites)} satellites from TLE file")
    
    def get_satellite(self, name: str) -> Optional[EarthSatellite]:
        """Get satellite by name."""
        return self.satellites.get(name)
    
    def list_satellites(self) -> List[str]:
        """Get list of available satellite names."""
        return list(self.satellites.keys())
    
    def compute_position_velocity(self, sat_name: str, timestamp: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute satellite position and velocity in ECEF/ITRS at given time.
        
        Args:
            sat_name: Satellite name
            timestamp: Unix timestamp
        
        Returns:
            (position_ecef, velocity_ecef)
            position_ecef: [x, y, z] in ECEF/ITRS (meters)
            velocity_ecef: [vx, vy, vz] in ECEF/ITRS (m/s)
        """
        sat = self.satellites.get(sat_name)
        if not sat:
            raise ValueError(f"Satellite not found: {sat_name}")
        
        # Convert timestamp to Skyfield time
        # ts.ut1() expects (year, month, day, hour, minute, second)
        dt = datetime.utcfromtimestamp(timestamp)
        t = self.ts.ut1(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second + dt.microsecond / 1e6)
        
        # Compute geocentric position (in GCRS/inertial frame)
        geocentric = sat.at(t)
        
        # Convert from GCRS (inertial) to ITRS (Earth-fixed/ECEF)
        # Use Skyfield's frame_xyz with ITRS frame
        from skyfield.framelib import itrs
        pos_itrs_km = geocentric.frame_xyz(itrs).km
        position_ecef = pos_itrs_km * 1000.0  # Convert km to meters
        
        # Velocity: Transform using finite differences in ITRS frame
        dt_step = 0.1  # seconds
        dt_plus = datetime.utcfromtimestamp(timestamp + dt_step)
        t_plus = self.ts.ut1(dt_plus.year, dt_plus.month, dt_plus.day, dt_plus.hour, dt_plus.minute, dt_plus.second + dt_plus.microsecond / 1e6)
        geocentric_plus = sat.at(t_plus)
        pos_plus_itrs_km = geocentric_plus.frame_xyz(itrs).km
        
        # Velocity in ITRS/ECEF (accounts for Earth rotation)
        velocity_ecef = (pos_plus_itrs_km - pos_itrs_km) * 1000.0 / dt_step  # m/s
        
        return position_ecef, velocity_ecef
    
    def compute_subpoint(self, sat_name: str, timestamp: float) -> Tuple[float, float, float]:
        """
        Compute satellite subpoint (ground track) position.
        
        Args:
            sat_name: Satellite name
            timestamp: Unix timestamp
        
        Returns:
            (latitude, longitude, altitude) in degrees and meters
        """
        sat = self.satellites.get(sat_name)
        if not sat:
            raise ValueError(f"Satellite not found: {sat_name}")
        
        dt = datetime.utcfromtimestamp(timestamp)
        t = self.ts.ut1(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second + dt.microsecond / 1e6)
        geocentric = sat.at(t)
        
        subpoint = wgs84.subpoint(geocentric)
        
        return subpoint.latitude.degrees, subpoint.longitude.degrees, subpoint.elevation.m


def compute_satellite_position(tle_mgr: TLEManager, sat_name: str, timestamp: float) -> Dict:
    """
    Compute comprehensive satellite position data.
    
    Args:
        tle_mgr: TLE manager instance
        sat_name: Satellite name
        timestamp: Unix timestamp
    
    Returns:
        Dict with position_ecef, velocity_ecef, lat, lon, alt
    """
    position_ecef, velocity_ecef = tle_mgr.compute_position_velocity(sat_name, timestamp)
    lat, lon, alt = tle_mgr.compute_subpoint(sat_name, timestamp)
    
    return {
        'name': sat_name,
        'timestamp': timestamp,
        'position_ecef': position_ecef,
        'velocity_ecef': velocity_ecef,
        'lat': lat,
        'lon': lon,
        'alt': alt
    }


def compute_doppler_prediction(sat_position_ecef: np.ndarray, sat_velocity_ecef: np.ndarray,
                               user_position_ecef: np.ndarray, carrier_freq: float) -> float:
    """
    Compute predicted Doppler shift based on satellite and user geometry.
    
    Doppler formula:
        f_doppler = -f_carrier * (LOS · v_relative) / c
    
    Args:
        sat_position_ecef: Satellite position [x, y, z] in meters
        sat_velocity_ecef: Satellite velocity [vx, vy, vz] in m/s
        user_position_ecef: User position [x, y, z] in meters
        carrier_freq: Carrier frequency in Hz
    
    Returns:
        Predicted Doppler shift in Hz
    """
    from .coordinates import compute_los_vector
    
    # Line-of-sight vector (user to satellite)
    los = compute_los_vector(user_position_ecef, sat_position_ecef)
    
    # Relative velocity (satellite - user, assuming user stationary)
    v_relative = sat_velocity_ecef
    
    # Radial velocity component (along LOS)
    v_radial = np.dot(los, v_relative)
    
    # Doppler shift
    doppler_shift = -(carrier_freq / C) * v_radial
    
    return doppler_shift


def compute_visible_satellites(tle_mgr: TLEManager, user_position_ecef: np.ndarray,
                               timestamp: float, min_elevation: float = 1.0) -> List[Dict]:
    """
    Compute list of visible satellites above elevation mask.
    
    Args:
        tle_mgr: TLE manager
        user_position_ecef: User position in ECEF
        timestamp: Unix timestamp
        min_elevation: Minimum elevation angle in degrees
    
    Returns:
        List of visible satellite info dicts
    """
    from .coordinates import compute_elevation_azimuth
    
    visible = []
    
    for sat_name in tle_mgr.list_satellites():
        try:
            sat_data = compute_satellite_position(tle_mgr, sat_name, timestamp)
            sat_pos = sat_data['position_ecef']
            
            elevation, azimuth = compute_elevation_azimuth(user_position_ecef, sat_pos)
            
            if elevation >= min_elevation:
                sat_data['elevation'] = elevation
                sat_data['azimuth'] = azimuth
                visible.append(sat_data)
                
        except Exception as e:
            logger.warning(f"Failed to compute visibility for {sat_name}: {e}")
    
    return visible
