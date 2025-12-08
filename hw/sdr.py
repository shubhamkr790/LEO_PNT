"""
RTL-SDR v4 driver for wideband IQ sampling.
Supports 700-4000 MHz via Netboon panel antenna + NooElec LANA LNA.
"""

import numpy as np
from rtlsdr import RtlSdr
import time
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class RTLSDRCapture:
    """RTL-SDR IQ capture with configurable center frequency and sample rate."""
    
    def __init__(self, device_index: int = 0, sample_rate: float = 2.4e6,
                 center_freq: float = 1621e6, gain = 'auto', ppm: int = 0):
        """
        Initialize RTL-SDR device.
        
        Args:
            device_index: RTL-SDR device index (0 for first device)
            sample_rate: Sample rate in Hz (default 2.4 MHz)
            center_freq: Center frequency in Hz (default 1621 MHz for Iridium)
            gain: Gain setting ('auto' or dB value)
            ppm: Frequency correction in PPM
        """
        self.device_index = device_index
        self.sample_rate = sample_rate
        self.center_freq = center_freq
        self.gain = gain
        self.ppm = ppm
        self.sdr: Optional[RtlSdr] = None
        
    def open(self):
        """Open and configure the SDR device."""
        try:
            self.sdr = RtlSdr(device_index=self.device_index)
            self.sdr.sample_rate = self.sample_rate
            self.sdr.center_freq = self.center_freq
            # ⚠ RTL-SDR Blog V4 on Windows: freq_correction can cause LIBUSB_ERROR_INVALID_PARAM
            # Disable it for now. PPM error at 1621 MHz (1-3 ppm) = ~1.6-4.8 kHz bias.
            # This constant bias doesn't affect Doppler detection significantly.
            #self.sdr.freq_correction = self.ppm
            
            # Set gain: Iridium signals are weak, prefer manual high gain
            if self.gain == 'auto':
                # Override 'auto' with fixed 45 dB for weak LEO signals
                self.sdr.gain = 45
                logger.info("Gain set to 45 dB (auto overridden for weak Iridium signals)")
            else:
                self.sdr.gain = float(self.gain)
                logger.info(f"Gain set to {self.sdr.gain} dB (manual)")

            
            logger.info(f"RTL-SDR opened: {self.sample_rate/1e6:.2f} MHz @ {self.center_freq/1e6:.2f} MHz")
            logger.info(f"Gain: {self.sdr.gain}, PPM: {self.ppm}")
            
        except Exception as e:
            logger.error(f"Failed to open RTL-SDR: {e}")
            raise
            
    def close(self):
        """Close the SDR device."""
        if self.sdr:
            self.sdr.close()
            self.sdr = None
            logger.info("RTL-SDR closed")
            
    def capture_iq(self, duration: float, timestamp: Optional[float] = None) -> Tuple[np.ndarray, float]:
        """
        Capture IQ samples for specified duration.
        
        Args:
            duration: Capture duration in seconds
            timestamp: Optional PPS-aligned timestamp (from GNSS)
        
        Returns:
            (iq_samples, capture_timestamp)
            iq_samples: Complex64 array of IQ samples
            capture_timestamp: Unix timestamp of capture start
        """
        if not self.sdr:
            raise RuntimeError("SDR not opened. Call open() first.")
        
        num_samples = int(self.sample_rate * duration)
        
        # Use provided PPS timestamp or current time
        capture_time = timestamp if timestamp else time.time()
        
        try:
            # Read IQ samples (complex64)
            iq_samples = self.sdr.read_samples(num_samples)
            logger.debug(f"Captured {len(iq_samples)} IQ samples ({duration:.3f}s)")
            
            return iq_samples, capture_time
            
        except Exception as e:
            logger.error(f"IQ capture failed: {e}")
            raise
    
    def retune(self, center_freq: float):
        """Retune to different center frequency."""
        if not self.sdr:
            raise RuntimeError("SDR not opened")
        
        self.center_freq = center_freq
        self.sdr.center_freq = center_freq
        logger.info(f"Retuned to {center_freq/1e6:.2f} MHz")
        
    def __enter__(self):
        self.open()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        
    @staticmethod
    def list_devices() -> list:
        """List available RTL-SDR devices."""
        from rtlsdr import RtlSdr
        try:
            # Try to open devices
            devices = []
            for i in range(10):  # Check first 10 indices
                try:
                    sdr = RtlSdr(device_index=i)
                    devices.append(i)
                    sdr.close()
                except:
                    break
            return devices
        except:
            return []
