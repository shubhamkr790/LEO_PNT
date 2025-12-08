"""
Spectrogram and PSD computation for signal analysis.
"""

import numpy as np
from scipy import signal
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


def compute_stft(iq_samples: np.ndarray, sample_rate: float, 
                 nperseg: int = 1024, noverlap: int = 512,
                 window: str = 'hann') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Short-Time Fourier Transform (STFT) of IQ samples.
    
    Args:
        iq_samples: Complex IQ samples
        sample_rate: Sample rate in Hz
        nperseg: Segment length for STFT
        noverlap: Overlap between segments
        window: Window function ('hann', 'hamming', etc.)
    
    Returns:
        (frequencies, times, Sxx)
        frequencies: Frequency bins in Hz
        times: Time bins in seconds
        Sxx: Complex STFT values [freq x time]
    """
    f, t, Zxx = signal.stft(
        iq_samples,
        fs=sample_rate,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        return_onesided=False,  # Keep both positive and negative frequencies
        boundary=None,
        padded=False
    )
    
    # Shift zero frequency to center
    f = np.fft.fftshift(f)
    Zxx = np.fft.fftshift(Zxx, axes=0)
    
    logger.debug(f"STFT computed: {Zxx.shape[0]} freq bins x {Zxx.shape[1]} time bins")
    
    return f, t, Zxx


def compute_psd(iq_samples: np.ndarray, sample_rate: float,
                nperseg: int = 1024, noverlap: int = 512) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Power Spectral Density (PSD) using Welch's method.
    
    Args:
        iq_samples: Complex IQ samples
        sample_rate: Sample rate in Hz
        nperseg: Segment length
        noverlap: Overlap between segments
    
    Returns:
        (frequencies, psd_db)
        frequencies: Frequency bins in Hz
        psd_db: PSD in dB
    """
    f, Pxx = signal.welch(
        iq_samples,
        fs=sample_rate,
        nperseg=nperseg,
        noverlap=noverlap,
        return_onesided=False,
        scaling='density'
    )
    
    # Shift to center frequency
    f = np.fft.fftshift(f)
    Pxx = np.fft.fftshift(Pxx)
    
    # Convert to dB
    psd_db = 10 * np.log10(Pxx + 1e-20)  # Add epsilon to avoid log(0)
    
    return f, psd_db


def compute_spectrogram_image(iq_samples: np.ndarray, sample_rate: float,
                              nperseg: int = 1024, noverlap: int = 512) -> np.ndarray:
    """
    Compute spectrogram magnitude for ML input (CNN).
    
    Args:
        iq_samples: Complex IQ samples
        sample_rate: Sample rate in Hz
        nperseg: Segment length
        noverlap: Overlap
    
    Returns:
        Spectrogram magnitude in dB [freq x time]
    """
    _, _, Zxx = compute_stft(iq_samples, sample_rate, nperseg, noverlap)
    
    # Compute magnitude and convert to dB
    spec_mag = np.abs(Zxx)
    spec_db = 20 * np.log10(spec_mag + 1e-20)
    
    return spec_db


def normalize_spectrogram(spec_db: np.ndarray, vmin: float = -80, vmax: float = 0) -> np.ndarray:
    """
    Normalize spectrogram to [0, 1] range for ML.
    
    Args:
        spec_db: Spectrogram in dB
        vmin: Minimum dB value
        vmax: Maximum dB value
    
    Returns:
        Normalized spectrogram [0, 1]
    """
    spec_norm = (spec_db - vmin) / (vmax - vmin)
    return np.clip(spec_norm, 0, 1)
