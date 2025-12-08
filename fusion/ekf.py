"""
Extended Kalman Filter for LEO-PNT navigation.

State vector (7 states):
    [px, py, pz, vx, vy, vz, baro_bias]
    
    px, py, pz: Position in ECEF (meters)
    vx, vy, vz: Velocity in ECEF (m/s)
    baro_bias: Barometer altitude bias (meters)
    
Note: Clock bias/drift removed as they are unobservable with Doppler-only measurements.
      Clock errors are implicitly absorbed in position/velocity states when GNSS is available.
"""

import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class ExtendedKalmanFilter:
    """Extended Kalman Filter for multi-sensor fusion."""
    
    def __init__(self, initial_state: Optional[np.ndarray] = None,
                 initial_covariance: Optional[np.ndarray] = None):
        """
        Initialize EKF.
        
        Args:
            initial_state: Initial state vector [7]
            initial_covariance: Initial covariance matrix [7x7]
        """
        # State dimension
        self.n_states = 7
        
        # State vector: [px, py, pz, vx, vy, vz, baro_bias]
        if initial_state is not None:
            self.x = np.array(initial_state, dtype=np.float64)
        else:
            self.x = np.zeros(self.n_states)
        
        # State covariance matrix
        if initial_covariance is not None:
            self.P = np.array(initial_covariance, dtype=np.float64)
        else:
            # Initialize with large uncertainty
            self.P = np.eye(self.n_states) * 1000.0
            self.P[6, 6] = 50.0   # Barometer bias (meters)
        
        # Process noise covariance
        self.Q = self._build_process_noise()
        
        # Last update time
        self.last_time = None
        
        logger.info("EKF initialized")
    
    def _build_process_noise(self) -> np.ndarray:
        """Build process noise covariance matrix."""
        Q = np.zeros((self.n_states, self.n_states))
        
        # Position process noise (very small, driven by velocity)
        Q[0:3, 0:3] = np.eye(3) * 0.01
        
        # Velocity process noise (acceleration uncertainty)
        # Increased from 1.0 to 10.0 for realistic handheld/pedestrian motion
        Q[3:6, 3:6] = np.eye(3) * 10.0  # ~10 m/s² uncertainty (handheld device)
        
        # Barometer bias process noise
        Q[6, 6] = 0.1  # meters
        
        return Q
    
    def predict(self, dt: float):
        """
        Prediction step: propagate state forward in time.
        
        Args:
            dt: Time step in seconds
        """
        # State transition matrix (linear approximation)
        F = self._build_state_transition_matrix(dt)
        
        # Predict state: x = F * x
        self.x = F @ self.x
        
        # Predict covariance: P = F * P * F^T + Q
        self.P = F @ self.P @ F.T + self.Q * dt
        
        logger.debug(f"EKF predict: dt={dt:.3f}s")
    
    def _build_state_transition_matrix(self, dt: float) -> np.ndarray:
        """
        Build state transition matrix F.
        
        State propagation:
            position = position + velocity * dt
            velocity = velocity (constant velocity model)
            baro_bias = baro_bias (constant)
        """
        F = np.eye(self.n_states)
        
        # Position depends on velocity
        F[0, 3] = dt  # px += vx * dt
        F[1, 4] = dt  # py += vy * dt
        F[2, 5] = dt  # pz += vz * dt
        
        # Baro bias is constant (no dynamics)
        
        return F
    
    def update_gnss(self, z_pos: np.ndarray, R_pos: np.ndarray):
        """
        Update with GNSS position measurement.
        
        Args:
            z_pos: Measured position [x, y, z] in ECEF (meters)
            R_pos: Measurement covariance [3x3]
        """
        # Measurement model: H maps state to measurement
        H = np.zeros((3, self.n_states))
        H[0:3, 0:3] = np.eye(3)  # Measure position directly
        
        # Innovation
        z_pred = H @ self.x
        y = z_pos - z_pred
        
        # Innovation covariance
        S = H @ self.P @ H.T + R_pos
        
        # Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # Update state
        self.x = self.x + K @ y
        
        # Update covariance
        I_KH = np.eye(self.n_states) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R_pos @ K.T  # Joseph form
        
        logger.debug(f"EKF update (GNSS position): innovation norm={np.linalg.norm(y):.2f}m")
    
    def update_doppler(self, z_doppler: float, sat_los: np.ndarray, sat_vel: np.ndarray,
                       carrier_freq: float, R_doppler: float):
        """
        Update with LEO Doppler measurement.
        
        Args:
            z_doppler: Measured Doppler shift (Hz)
            sat_los: Line-of-sight unit vector from user to satellite [3]
            sat_vel: Satellite velocity in ECEF [vx, vy, vz] (m/s)
            carrier_freq: Carrier frequency (Hz)
            R_doppler: Measurement variance (Hz²)
        """
        from utils.tle import C  # Speed of light
        
        # Predicted Doppler based on current state
        # Doppler = -(f_carrier / c) * (LOS · (v_sat - v_user))
        v_user = self.x[3:6]
        v_relative = sat_vel - v_user
        doppler_pred = -(carrier_freq / C) * np.dot(sat_los, v_relative)
        
        # Innovation
        y = z_doppler - doppler_pred
        
        # Measurement Jacobian H (linearization)
        H = np.zeros(self.n_states)
        # Doppler depends on user velocity: ∂doppler/∂v_user = (f_carrier/c) * LOS
        H[3:6] = (carrier_freq / C) * sat_los
        H = H.reshape(1, -1)
        
        # Innovation covariance
        S = H @ self.P @ H.T + R_doppler
        
        # Kalman gain
        K = self.P @ H.T / S
        
        # Update state
        self.x = self.x + K.flatten() * y
        
        # Update covariance
        I_KH = np.eye(self.n_states) - np.outer(K, H)
        self.P = I_KH @ self.P
        
        logger.debug(f"EKF update (Doppler): innovation={y:.2f}Hz")
    
    def update_imu(self, z_accel: np.ndarray, dt: float, R_accel: np.ndarray):
        """
        Update with IMU acceleration measurement.
        
        Args:
            z_accel: Measured acceleration [ax, ay, az] in m/s²
            dt: Time since last IMU update
            R_accel: Measurement covariance [3x3]
        """
        # Use acceleration to update velocity
        # Predicted velocity change: dv = a * dt
        dv_pred = z_accel * dt
        
        # Measurement: velocity change
        v_prev = self.x[3:6] - dv_pred  # Reverse propagate
        
        # For simplicity, use IMU to constrain velocity updates
        # This is a simplified approach; full INS integration would be more complex
        
        # Measurement model: measure velocity change
        H = np.zeros((3, self.n_states))
        H[0:3, 3:6] = np.eye(3)
        
        # Innovation (using velocity)
        y = dv_pred - np.zeros(3)  # Expected velocity change vs. zero
        
        # This is simplified - in practice, you'd do full INS mechanization
        logger.debug("EKF update (IMU) - simplified integration")
    
    def update_barometer(self, z_alt: float, R_alt: float):
        """
        Update with barometer altitude measurement.
        
        Args:
            z_alt: Measured altitude (meters above ellipsoid)
            R_alt: Measurement variance (m²)
        """
        from utils.coordinates import ecef_to_geodetic
        
        # Convert current ECEF position to geodetic
        lat, lon, alt_pred = ecef_to_geodetic(*self.x[0:3])
        
        # Apply barometer bias
        alt_pred_biased = alt_pred + self.x[6]  # baro_bias is at index 6
        
        # Innovation
        y = z_alt - alt_pred_biased
        
        # Measurement Jacobian (approximate - altitude depends on pz mainly)
        H = np.zeros(self.n_states)
        # Altitude ≈ pz component (simplified)
        H[2] = 1.0  # ∂alt/∂pz ≈ 1 (simplified)
        H[6] = 1.0  # Altitude also depends on baro bias
        H = H.reshape(1, -1)
        
        # Innovation covariance
        S = H @ self.P @ H.T + R_alt
        
        # Kalman gain
        K = self.P @ H.T / S
        
        # Update state
        self.x = self.x + K.flatten() * y
        
        # Update covariance
        I_KH = np.eye(self.n_states) - np.outer(K, H)
        self.P = I_KH @ self.P
        
        logger.debug(f"EKF update (Barometer): innovation={y:.2f}m")
    
    def get_position_ecef(self) -> np.ndarray:
        """Get current position estimate in ECEF."""
        return self.x[0:3].copy()
    
    def get_velocity_ecef(self) -> np.ndarray:
        """Get current velocity estimate in ECEF."""
        return self.x[3:6].copy()
    
    def get_position_uncertainty(self) -> float:
        """Get position uncertainty (3D RMS)."""
        pos_cov = self.P[0:3, 0:3]
        return np.sqrt(np.trace(pos_cov))
    
    def get_velocity_uncertainty(self) -> float:
        """Get velocity uncertainty (3D RMS)."""
        vel_cov = self.P[3:6, 3:6]
        return np.sqrt(np.trace(vel_cov))
    
    def get_position_geodetic(self) -> Tuple[float, float, float]:
        """Get current position in geodetic coordinates."""
        from utils.coordinates import ecef_to_geodetic
        return ecef_to_geodetic(*self.x[0:3])
    
    def get_state(self) -> np.ndarray:
        """Get full state vector."""
        return self.x.copy()
    
    def get_covariance(self) -> np.ndarray:
        """Get state covariance matrix."""
        return self.P.copy()
