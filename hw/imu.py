"""
IMU reader for MPU-9250 or BNO055.
Provides acceleration, gyroscope, and magnetometer data.
"""

import numpy as np
import time
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

try:
    import smbus2 as smbus
    HAS_SMBUS = True
except ImportError:
    HAS_SMBUS = False
    logger.warning("smbus2 not available - IMU will not work")


class IMUReader:
    """IMU reader for acceleration and gyroscope data."""
    
    # MPU-9250 I2C addresses and registers
    MPU9250_ADDR = 0x68
    ACCEL_XOUT_H = 0x3B
    GYRO_XOUT_H = 0x43
    PWR_MGMT_1 = 0x6B
    
    def __init__(self, bus_number: int = 1, device_addr: int = 0x68, 
                 imu_type: str = 'mpu9250'):
        """
        Initialize IMU reader.
        
        Args:
            bus_number: I2C bus number (typically 1 on Raspberry Pi)
            device_addr: I2C device address (0x68 for MPU-9250, 0x28 for BNO055)
            imu_type: 'mpu9250' or 'bno055'
        """
        self.bus_number = bus_number
        self.device_addr = device_addr
        self.imu_type = imu_type.lower()
        self.bus: Optional[smbus.SMBus] = None
        
        # Calibration offsets (to be populated from config)
        self.accel_bias = np.zeros(3)
        self.gyro_bias = np.zeros(3)
        
        # Scale factors
        self.accel_scale = 16384.0  # LSB/g for ±2g range
        self.gyro_scale = 131.0     # LSB/°/s for ±250°/s range
        
    def open(self):
        """Open I2C bus and initialize IMU."""
        if not HAS_SMBUS:
            logger.error("smbus2 not installed - cannot open IMU")
            raise RuntimeError("smbus2 library required for IMU")
        
        try:
            self.bus = smbus.SMBus(self.bus_number)
            
            if self.imu_type == 'mpu9250':
                # Wake up MPU-9250
                self.bus.write_byte_data(self.device_addr, self.PWR_MGMT_1, 0)
                time.sleep(0.1)
                
            logger.info(f"IMU opened: {self.imu_type} at 0x{self.device_addr:02X}")
            
        except Exception as e:
            logger.error(f"Failed to open IMU: {e}")
            raise
            
    def close(self):
        """Close I2C bus."""
        if self.bus:
            self.bus.close()
            self.bus = None
            logger.info("IMU closed")
            
    def read_raw_accel(self) -> np.ndarray:
        """Read raw accelerometer data (3-axis)."""
        if not self.bus:
            raise RuntimeError("IMU not opened")
        
        data = self.bus.read_i2c_block_data(self.device_addr, self.ACCEL_XOUT_H, 6)
        
        # Combine high and low bytes (signed 16-bit)
        ax = np.int16((data[0] << 8) | data[1])
        ay = np.int16((data[2] << 8) | data[3])
        az = np.int16((data[4] << 8) | data[5])
        
        return np.array([ax, ay, az], dtype=np.float64)
        
    def read_raw_gyro(self) -> np.ndarray:
        """Read raw gyroscope data (3-axis)."""
        if not self.bus:
            raise RuntimeError("IMU not opened")
        
        data = self.bus.read_i2c_block_data(self.device_addr, self.GYRO_XOUT_H, 6)
        
        gx = np.int16((data[0] << 8) | data[1])
        gy = np.int16((data[2] << 8) | data[3])
        gz = np.int16((data[4] << 8) | data[5])
        
        return np.array([gx, gy, gz], dtype=np.float64)
        
    def read_accel(self) -> np.ndarray:
        """
        Read calibrated acceleration in m/s².
        
        Returns:
            np.array([ax, ay, az]) in m/s²
        """
        raw = self.read_raw_accel()
        accel_g = (raw - self.accel_bias) / self.accel_scale
        return accel_g * 9.81  # Convert to m/s²
        
    def read_gyro(self) -> np.ndarray:
        """
        Read calibrated angular velocity in rad/s.
        
        Returns:
            np.array([gx, gy, gz]) in rad/s
        """
        raw = self.read_raw_gyro()
        gyro_dps = (raw - self.gyro_bias) / self.gyro_scale
        return np.radians(gyro_dps)  # Convert to rad/s
        
    def read_imu_data(self) -> Dict:
        """
        Read complete IMU data.
        
        Returns:
            Dict with 'accel', 'gyro', 'timestamp'
        """
        timestamp = time.time()
        
        try:
            accel = self.read_accel()
            gyro = self.read_gyro()
            
            return {
                'accel': accel,
                'gyro': gyro,
                'timestamp': timestamp
            }
            
        except Exception as e:
            logger.error(f"IMU read failed: {e}")
            return {
                'accel': np.zeros(3),
                'gyro': np.zeros(3),
                'timestamp': timestamp
            }
            
    def calibrate(self, num_samples: int = 100):
        """
        Calibrate IMU by computing bias offsets.
        Device must be stationary during calibration.
        
        Args:
            num_samples: Number of samples for averaging
        """
        logger.info(f"Calibrating IMU ({num_samples} samples)...")
        
        accel_sum = np.zeros(3)
        gyro_sum = np.zeros(3)
        
        for _ in range(num_samples):
            accel_sum += self.read_raw_accel()
            gyro_sum += self.read_raw_gyro()
            time.sleep(0.01)
            
        self.accel_bias = accel_sum / num_samples
        self.gyro_bias = gyro_sum / num_samples
        
        # Subtract gravity from Z-axis bias (assuming device is level)
        self.accel_bias[2] -= self.accel_scale  # 1g in LSB
        
        logger.info(f"IMU calibrated: accel_bias={self.accel_bias}, gyro_bias={self.gyro_bias}")
        
    def __enter__(self):
        self.open()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
