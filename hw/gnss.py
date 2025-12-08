"""
GNSS receiver driver for L89HA module.
Parses NMEA sentences for position, velocity, and timing.
PPS integration handled by system chrony.
"""

import serial
import pynmea2
import time
import logging
from typing import Optional, Dict
import numpy as np

logger = logging.getLogger(__name__)


class GNSSReceiver:
    """GNSS receiver for position and timing reference."""
    
    def __init__(self, port: str = '/dev/ttyUSB0', baudrate: int = 9600, timeout: float = 1.0):
        """
        Initialize GNSS receiver.
        
        Args:
            port: Serial port (e.g., '/dev/ttyUSB0', 'COM3')
            baudrate: Baud rate (typically 9600 or 115200)
            timeout: Read timeout in seconds
        """
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.serial: Optional[serial.Serial] = None
        
        # Latest fix data
        self.last_fix: Optional[Dict] = None
        self.last_update_time: Optional[float] = None
        
    def open(self):
        """Open serial connection to GNSS module."""
        try:
            self.serial = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=self.timeout
            )
            logger.info(f"GNSS opened on {self.port} @ {self.baudrate} baud")
            
        except Exception as e:
            logger.error(f"Failed to open GNSS: {e}")
            raise
            
    def close(self):
        """Close serial connection."""
        if self.serial:
            self.serial.close()
            self.serial = None
            logger.info("GNSS closed")
            
    def read_sentence(self) -> Optional[str]:
        """Read one NMEA sentence from serial port."""
        if not self.serial:
            raise RuntimeError("GNSS not opened")
        
        try:
            line = self.serial.readline().decode('ascii', errors='ignore').strip()
            return line if line.startswith('$') else None
        except Exception as e:
            logger.warning(f"NMEA read error: {e}")
            return None
            
    def parse_and_update(self) -> Optional[Dict]:
        """
        Parse NMEA sentence and update position/velocity.
        
        Returns:
            Dict with position/velocity data if valid fix, else None
        """
        sentence = self.read_sentence()
        if not sentence:
            return None
        
        try:
            msg = pynmea2.parse(sentence)
            
            # GGA: Position fix
            if isinstance(msg, pynmea2.types.talker.GGA):
                if msg.gps_qual > 0:  # Valid fix
                    self.last_fix = {
                        'type': 'GGA',
                        'lat': msg.latitude,
                        'lon': msg.longitude,
                        'alt': msg.altitude,
                        'num_sats': msg.num_sats,
                        'hdop': msg.horizontal_dil,
                        'timestamp': time.time()
                    }
                    self.last_update_time = time.time()
                    return self.last_fix
                    
            # RMC: Position + velocity
            elif isinstance(msg, pynmea2.types.talker.RMC):
                if msg.status == 'A':  # Active/valid
                    self.last_fix = {
                        'type': 'RMC',
                        'lat': msg.latitude,
                        'lon': msg.longitude,
                        'speed': msg.spd_over_grnd * 0.51444 if msg.spd_over_grnd else 0,  # knots to m/s
                        'course': msg.true_course if msg.true_course else 0,
                        'timestamp': time.time()
                    }
                    self.last_update_time = time.time()
                    return self.last_fix
                    
            # VTG: Velocity (course and speed)
            elif isinstance(msg, pynmea2.types.talker.VTG):
                if self.last_fix and msg.spd_over_grnd_kmph:
                    self.last_fix['speed_kmh'] = msg.spd_over_grnd_kmph
                    self.last_fix['speed'] = msg.spd_over_grnd_kmph / 3.6  # to m/s
                    
        except pynmea2.ParseError as e:
            logger.debug(f"NMEA parse error: {e}")
            
        return None
        
    def get_position_ecef(self) -> Optional[np.ndarray]:
        """
        Get current position in ECEF coordinates.
        
        Returns:
            np.array([x, y, z]) in meters, or None if no fix
        """
        if not self.last_fix or 'lat' not in self.last_fix:
            return None
        
        from ..utils.coordinates import geodetic_to_ecef
        
        lat = self.last_fix['lat']
        lon = self.last_fix['lon']
        alt = self.last_fix.get('alt', 0)
        
        return geodetic_to_ecef(lat, lon, alt)
        
    def get_velocity_ecef(self) -> Optional[np.ndarray]:
        """
        Get current velocity in ECEF coordinates.
        
        Returns:
            np.array([vx, vy, vz]) in m/s, or None if no velocity
        """
        if not self.last_fix or 'speed' not in self.last_fix:
            return None
        
        from ..utils.coordinates import velocity_ned_to_ecef
        
        lat = self.last_fix['lat']
        lon = self.last_fix['lon']
        speed = self.last_fix['speed']
        course = self.last_fix.get('course', 0)
        
        # Convert to NED velocity
        vn = speed * np.cos(np.radians(course))
        ve = speed * np.sin(np.radians(course))
        vd = 0  # No vertical velocity from GNSS
        
        return velocity_ned_to_ecef(lat, lon, np.array([vn, ve, vd]))
        
    def is_fix_valid(self, max_age: float = 2.0) -> bool:
        """Check if fix is recent and valid."""
        if not self.last_fix or not self.last_update_time:
            return False
        return (time.time() - self.last_update_time) < max_age
        
    def __enter__(self):
        self.open()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
