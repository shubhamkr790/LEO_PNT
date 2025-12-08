"""
PPS timing alignment and synchronization.
Assumes chrony handles PPS hardware synchronization.
"""

import time
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def get_pps_timestamp() -> float:
    """
    Get current PPS-aligned timestamp.
    
    In real implementation, this would interface with chrony or NTP.
    For now, returns system time.
    
    Returns:
        Unix timestamp
    """
    # In production, read from /var/lib/chrony or use ntpq
    return time.time()


def align_to_pps(target_time: Optional[float] = None, tolerance: float = 0.01) -> float:
    """
    Wait for next PPS pulse and return timestamp.
    
    Args:
        target_time: Optional target time to wait for
        tolerance: Timing tolerance in seconds
    
    Returns:
        PPS timestamp
    """
    current_time = time.time()
    
    if target_time is None:
        # Wait for next second boundary
        next_pps = int(current_time) + 1
    else:
        next_pps = target_time
    
    sleep_time = next_pps - current_time
    
    if sleep_time > 0:
        time.sleep(sleep_time)
    
    actual_time = time.time()
    
    if abs(actual_time - next_pps) > tolerance:
        logger.warning(f"PPS alignment error: {abs(actual_time - next_pps):.6f}s")
    
    return actual_time


def calculate_time_offset(gps_time: float, system_time: float) -> float:
    """
    Calculate offset between GPS time and system time.
    
    Args:
        gps_time: GPS timestamp
        system_time: System timestamp
    
    Returns:
        Time offset in seconds
    """
    return gps_time - system_time


def sync_check(last_sync: float, max_drift: float = 1.0) -> bool:
    """
    Check if time synchronization is still valid.
    
    Args:
        last_sync: Last synchronization timestamp
        max_drift: Maximum acceptable drift in seconds
    
    Returns:
        True if sync is valid
    """
    drift = abs(time.time() - last_sync)
    return drift < max_drift
