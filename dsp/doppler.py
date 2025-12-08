"""
Doppler shift extraction from IQ signals.
"""

import numpy as np
from scipy import signal
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def extract_doppler_shift(iq_samples: np.ndarray, sample_rate: float,
                          center_freq: float, expected_doppler: Optional[float] = None,
                          search_range: float = 10e3) -> Tuple[float, float, float]:
    """
    Extract Doppler shift from IQ samples using FFT peak search.
    
    Args:
        iq_samples: Complex IQ samples
        sample_rate: Sample rate in Hz
        center_freq: SDR center frequency in Hz
        expected_doppler: Expected Doppler shift in Hz (for narrower search)
        search_range: Doppler search range in Hz (±)
    
    Returns:
        (doppler_shift_hz, snr_db, peak_power_db)
        doppler_shift_hz: Detected Doppler shift relative to center_freq
        snr_db: Signal-to-noise ratio
        peak_power_db: Peak signal power in dB
    """
    # Compute FFT
    fft_size = len(iq_samples)
    fft_result = np.fft.fft(iq_samples)
    fft_mag = np.abs(fft_result)
    fft_power = fft_mag ** 2
    
    # Frequency bins
    freqs = np.fft.fftfreq(fft_size, 1.0 / sample_rate)
    
    # Search region
    if expected_doppler is not None:
        # Narrow search around expected Doppler
        search_mask = np.abs(freqs - expected_doppler) < search_range
    else:
        # Full search
        search_mask = np.abs(freqs) < search_range
    
    # Find peak within search region
    search_power = fft_power[search_mask]
    search_freqs = freqs[search_mask]
    
    if len(search_power) == 0:
        logger.warning("No samples in Doppler search range")
        return 0.0, -100.0, -100.0
    
    peak_idx = np.argmax(search_power)
    doppler_shift = search_freqs[peak_idx]
    peak_power = search_power[peak_idx]
    
    # Estimate SNR (peak vs median noise)
    noise_floor = np.median(search_power)
    snr = 10 * np.log10((peak_power / noise_floor) if noise_floor > 0 else 1)
    peak_power_db = 10 * np.log10(peak_power + 1e-20)
    
    logger.debug(f"Doppler: {doppler_shift:.2f} Hz, SNR: {snr:.2f} dB, Power: {peak_power_db:.2f} dB")
    
    return doppler_shift, snr, peak_power_db


def sliding_doppler_extraction(iq_samples: np.ndarray, sample_rate: float,
                                center_freq: float, window_size: float = 0.1,
                                overlap: float = 0.5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract Doppler shifts over sliding windows for time-series analysis.
    
    Args:
        iq_samples: Complex IQ samples
        sample_rate: Sample rate in Hz
        center_freq: SDR center frequency
        window_size: Window size in seconds
        overlap: Overlap fraction (0-1)
    
    Returns:
        (times, doppler_shifts, snrs)
        times: Time stamps for each window
        doppler_shifts: Doppler shift in Hz for each window
        snrs: SNR in dB for each window
    """
    samples_per_window = int(window_size * sample_rate)
    hop_size = int(samples_per_window * (1 - overlap))
    
    num_windows = (len(iq_samples) - samples_per_window) // hop_size + 1
    
    times = np.zeros(num_windows)
    doppler_shifts = np.zeros(num_windows)
    snrs = np.zeros(num_windows)
    
    for i in range(num_windows):
        start_idx = i * hop_size
        end_idx = start_idx + samples_per_window
        
        if end_idx > len(iq_samples):
            break
        
        window_samples = iq_samples[start_idx:end_idx]
        window_time = start_idx / sample_rate
        
        doppler, snr, _ = extract_doppler_shift(window_samples, sample_rate, center_freq)
        
        times[i] = window_time
        doppler_shifts[i] = doppler
        snrs[i] = snr
    
    return times[:i+1], doppler_shifts[:i+1], snrs[:i+1]


def estimate_snr(iq_samples: np.ndarray, sample_rate: float,
                 signal_bw: float = 20e3) -> float:
    """
    Estimate SNR from IQ samples.
    
    Args:
        iq_samples: Complex IQ samples
        sample_rate: Sample rate in Hz
        signal_bw: Expected signal bandwidth in Hz
    
    Returns:
        SNR in dB
    """
    # Compute PSD
    freqs, psd = signal.welch(
        iq_samples,
        fs=sample_rate,
        nperseg=1024,
        return_onesided=False
    )
    
    # Find signal region (center)
    center_mask = np.abs(freqs) < signal_bw / 2
    noise_mask = np.abs(freqs) > signal_bw
    
    signal_power = np.mean(psd[center_mask])
    noise_power = np.median(psd[noise_mask])
    
    snr_db = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else 0
    
    return snr_db


def refine_doppler_parabolic(fft_mag: np.ndarray, peak_idx: int) -> float:
    """
    Refine Doppler estimate using parabolic interpolation.
    
    Args:
        fft_mag: FFT magnitude array
        peak_idx: Index of peak
    
    Returns:
        Refined peak offset (fractional bins)
    """
    if peak_idx <= 0 or peak_idx >= len(fft_mag) - 1:
        return 0.0
    
    # Parabolic interpolation
    alpha = fft_mag[peak_idx - 1]
    beta = fft_mag[peak_idx]
    gamma = fft_mag[peak_idx + 1]
    
    if beta > alpha and beta > gamma:
        offset = 0.5 * (alpha - gamma) / (alpha - 2*beta + gamma)
        return offset
    
    return 0.0


def detect_bursts_in_spectrogram(iq_samples: np.ndarray, sample_rate: float,
                                   threshold_db: float = 10.0,
                                   nperseg: int = 256) -> list:
    """
    Detect signal bursts in IQ data using STFT spectrogram analysis.
    
    This is designed for TDMA/FDMA signals like Iridium where bursts appear
    as time-frequency tiles above the noise floor.
    
    Args:
        iq_samples: Complex IQ samples
        sample_rate: Sample rate in Hz
        threshold_db: Detection threshold above median noise floor (dB)
        nperseg: STFT window size (smaller = better time resolution)
    
    Returns:
        List of burst dicts with keys:
            - time_start, time_end: Time bounds (seconds)
            - freq_center: Center frequency offset (Hz)
            - freq_span: Frequency span (Hz)
            - peak_power_db: Peak power in burst (dB)
            - snr_db: SNR estimate
    """
    from .spectrogram import compute_stft
    
    # Compute STFT with high time resolution
    freqs, times, Zxx = compute_stft(iq_samples, sample_rate, 
                                      nperseg=nperseg, 
                                      noverlap=nperseg // 2)
    
    # Power spectrogram in dB
    power = np.abs(Zxx) ** 2
    power_db = 10 * np.log10(power + 1e-20)
    
    # Estimate noise floor (median across all time-freq bins)
    noise_floor_db = np.median(power_db)
    
    # Threshold mask
    threshold_abs = noise_floor_db + threshold_db
    burst_mask = power_db > threshold_abs
    
    # Count how many time-freq bins exceed threshold
    num_burst_bins = np.sum(burst_mask)
    
    if num_burst_bins == 0:
        logger.debug("No bursts detected above threshold")
        return []
    
    # Simple burst extraction: find connected regions
    # For now, just find the strongest burst
    bursts = []
    
    # Find peak in spectrogram
    peak_idx = np.unravel_index(np.argmax(power_db), power_db.shape)
    peak_freq_idx, peak_time_idx = peak_idx
    
    peak_power = power_db[peak_freq_idx, peak_time_idx]
    peak_freq = freqs[peak_freq_idx]
    peak_time = times[peak_time_idx]
    
    # Estimate burst extent (rough - find surrounding bins above threshold)
    time_indices = np.where(burst_mask[peak_freq_idx, :])[0]
    freq_indices = np.where(burst_mask[:, peak_time_idx])[0]
    
    if len(time_indices) > 0 and len(freq_indices) > 0:
        time_start = times[time_indices[0]] if time_indices[0] > 0 else times[0]
        time_end = times[time_indices[-1]] if time_indices[-1] < len(times)-1 else times[-1]
        
        freq_min = freqs[freq_indices[0]]
        freq_max = freqs[freq_indices[-1]]
        freq_center = (freq_min + freq_max) / 2
        freq_span = freq_max - freq_min
        
        snr_db = peak_power - noise_floor_db
        
        bursts.append({
            'time_start': time_start,
            'time_end': time_end,
            'freq_center': freq_center,
            'freq_span': freq_span,
            'peak_power_db': peak_power,
            'snr_db': snr_db
        })
        
        logger.debug(f"Detected burst: t=[{time_start:.3f}, {time_end:.3f}]s, "
                     f"f={freq_center:.1f} Hz, SNR={snr_db:.1f} dB")
    
    return bursts
