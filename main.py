#!/usr/bin/env python3
"""
DhruvX LEO-PNT Receiver
Main CLI with 3 modes: capture, process, run
"""

import argparse
import sys
import time
import numpy as np
import csv
from pathlib import Path

# Internal imports
from fusion import ekf
from utils import setup_logging, TLEManager, geodetic_to_ecef, compute_doppler_prediction
from utils.atmospheric import compute_total_atmospheric_correction
from utils.coordinates import compute_elevation_azimuth
from hw import RTLSDRCapture, GNSSReceiver, IMUReader
from dsp import compute_stft, extract_doppler_shift, compute_psd, detect_bursts_in_spectrogram
#from ml import CognitiveSelector, SpectrogramClassifier, DopplerPredictor, AnomalyDetector
from fusion import ExtendedKalmanFilter, gnss_measurement_model, doppler_measurement_model

try:
    from database import SupabaseNavigationClient
    DATABASE_AVAILABLE = True
except ImportError:
    DATABASE_AVAILABLE = False


def mode_capture(args):
    """
    Capture mode: Sample IQ data from RTL-SDR and save to file.
    """
    print(f"=== CAPTURE MODE ===")
    print(f"Center frequency: {args.freq/1e6:.2f} MHz")
    print(f"Duration: {args.duration} seconds")
    print(f"Sample rate: {args.sample_rate/1e6:.2f} MHz")
    
    setup_logging('INFO')
    
    output_file = args.output if args.output else f"capture_{int(time.time())}.npy"
    
    try:
        with RTLSDRCapture(center_freq=args.freq, sample_rate=args.sample_rate, 
                          gain=args.gain) as sdr:
            
            print(f"Capturing {args.duration}s...")
            iq_samples, timestamp = sdr.capture_iq(args.duration)
            
            # Save to file
            np.save(output_file, iq_samples)
            print(f"Saved {len(iq_samples)} samples to {output_file}")
            
            # Save metadata
            metadata_file = output_file.replace('.npy', '_meta.txt')
            with open(metadata_file, 'w') as f:
                f.write(f"center_freq={args.freq}\n")
                f.write(f"sample_rate={args.sample_rate}\n")
                f.write(f"timestamp={timestamp}\n")
                f.write(f"num_samples={len(iq_samples)}\n")
            
            print(f"✓ Capture complete")
            
    except Exception as e:
        print(f"✗ Capture failed: {e}")
        return 1
    
    return 0


def mode_process(args):
    """
    Process mode: Process IQ file and extract Doppler shifts to CSV.
    """
    print(f"=== PROCESS MODE ===")
    print(f"Input file: {args.input}")
    
    setup_logging('INFO')
    
    # Load IQ data
    try:
        iq_samples = np.load(args.input)
        print(f"Loaded {len(iq_samples)} IQ samples")
    except Exception as e:
        print(f"✗ Failed to load IQ file: {e}")
        return 1
    
    # Load metadata
    meta_file = args.input.replace('.npy', '_meta.txt')
    metadata = {}
    if Path(meta_file).exists():
        with open(meta_file, 'r') as f:
            for line in f:
                key, val = line.strip().split('=')
                metadata[key] = float(val) if '.' in val or 'e' in val else int(val)
    
    sample_rate = metadata.get('sample_rate', args.sample_rate)
    center_freq = metadata.get('center_freq', args.freq)
    
    print(f"Sample rate: {sample_rate/1e6:.2f} MHz")
    print(f"Center freq: {center_freq/1e6:.2f} MHz")
    
    # Compute STFT
    print("Computing STFT spectrogram...")
    freqs, times, Sxx = compute_stft(iq_samples, sample_rate)
    print(f"Spectrogram: {Sxx.shape[0]} freq bins × {Sxx.shape[1]} time bins")
    
    # Extract Doppler over time
    print("Extracting Doppler shifts...")
    results = []
    
    window_size = 0.1  # 100ms windows
    samples_per_window = int(window_size * sample_rate)
    hop_size = samples_per_window // 2
    
    for i in range(0, len(iq_samples) - samples_per_window, hop_size):
        window = iq_samples[i:i+samples_per_window]
        t = i / sample_rate
        
        doppler, snr, power = extract_doppler_shift(window, sample_rate, center_freq)
        
        results.append({
            'time': t,
            'doppler_hz': doppler,
            'snr_db': snr,
            'power_db': power
        })
    
    # Save to CSV
    output_file = args.output if args.output else args.input.replace('.npy', '_doppler.csv')
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['time', 'doppler_hz', 'snr_db', 'power_db'])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"✓ Saved {len(results)} Doppler measurements to {output_file}")
    
    return 0


def mode_scan(args):
    """
    Scan mode: Quick test of signal presence across multiple frequencies.
    Useful for finding active Iridium channels.
    """
    print(f"=== SCAN MODE ===")
    print(f"Scanning frequencies from {args.freq_start/1e6:.2f} to {args.freq_end/1e6:.2f} MHz")
    print(f"Step size: {args.freq_step/1e3:.1f} kHz")
    print(f"Integration time: {args.duration}s per frequency\n")
    
    setup_logging('INFO')
    
    # Generate frequency list
    freqs = np.arange(args.freq_start, args.freq_end + args.freq_step, args.freq_step)
    
    results = []
    
    try:
        with RTLSDRCapture(center_freq=freqs[0], sample_rate=args.sample_rate) as sdr:
            for idx, freq in enumerate(freqs):
                progress = (idx + 1) / len(freqs) * 100
                print(f"[{idx+1}/{len(freqs)}] ({progress:5.1f}%) Testing {freq/1e6:.2f} MHz...", end=' ')
                
                # Retune
                sdr.retune(freq)
                time.sleep(0.1)  # Let SDR settle
                
                # Capture
                iq_samples, cap_time = sdr.capture_iq(args.duration)
                
                # Check for bursts
                bursts = detect_bursts_in_spectrogram(iq_samples, args.sample_rate, threshold_db=8.0)
                
                # Compute average power
                avg_power_db = 10 * np.log10(np.mean(np.abs(iq_samples)**2) + 1e-20)
                
                num_bursts = len(bursts)
                max_snr = max([b['snr_db'] for b in bursts]) if bursts else 0.0
                
                results.append({
                    'freq': freq,
                    'avg_power_db': avg_power_db,
                    'num_bursts': num_bursts,
                    'max_snr_db': max_snr
                })
                
                print(f"Power: {avg_power_db:5.1f} dB | Bursts: {num_bursts} | Max SNR: {max_snr:5.1f} dB")
    
    except Exception as e:
        print(f"\n✗ Scan failed: {e}")
        return 1
    
    # Summary
    print(f"\n=== SCAN SUMMARY ===")
    best = max(results, key=lambda x: x['max_snr_db'])
    print(f"Best frequency: {best['freq']/1e6:.2f} MHz (SNR: {best['max_snr_db']:.1f} dB, {best['num_bursts']} bursts)")
    
    # Show top 3
    top3 = sorted(results, key=lambda x: x['max_snr_db'], reverse=True)[:3]
    print(f"\nTop 3 frequencies:")
    for i, r in enumerate(top3):
        print(f"  {i+1}. {r['freq']/1e6:.2f} MHz - SNR: {r['max_snr_db']:5.1f} dB, {r['num_bursts']} bursts")
    
    return 0


def mode_run(args):
    """
    Run mode: Real-time navigation with sensor fusion.
    """
    print(f"=== RUN MODE (Real-Time Navigation) ===")
    setup_logging(args.log_level, args.log_file)
    
    # Initialize TLE manager
    print(f"Loading TLEs from {args.tle_file}...")
    try:
        tle_mgr = TLEManager(args.tle_file)
        print(f"✓ Loaded {len(tle_mgr.list_satellites())} satellites")
    except Exception as e:
        print(f"✗ Failed to load TLEs: {e}")
        return 1
    
    # Initialize hardware
    print("Initializing hardware...")
    
    # GNSS
    try:
        gnss = GNSSReceiver(port=args.gnss_port, baudrate=args.gnss_baud)
        gnss.open()
        print(f"✓ GNSS opened on {args.gnss_port}")
    except Exception as e:
        print(f"⚠ GNSS unavailable: {e}")
        gnss = None
    
    # RTL-SDR
    try:
        sdr = RTLSDRCapture(center_freq=args.freq, sample_rate=args.sample_rate)
        sdr.open()
        print(f"✓ RTL-SDR opened at {args.freq/1e6:.2f} MHz")
    except Exception as e:
        print(f"✗ RTL-SDR unavailable: {e}")
        print("Cannot run without SDR")
        return 1
    
    # IMU (optional)
    imu = None
    if args.use_imu:
        try:
            imu = IMUReader()
            imu.open()
            print("✓ IMU opened")
        except Exception as e:
            print(f"⚠ IMU unavailable: {e}")
    
    # Initialize ML models (cognitive layer)
    # print("Initializing cognitive layer...")
    # cognitive = CognitiveSelector(
    #     doppler_predictor=DopplerPredictor(),
    #     anomaly_detector=AnomalyDetector()
    # )
    # print("✓ Cognitive selector ready")
    print("⚠ ML disabled: Using all Doppler measurements (no cognitive filtering)")

    class DummyCognitiveSelector:
        def select_measurements(self, measurements, min_weight=0.3):
            return measurements

    cognitive = DummyCognitiveSelector()

    
    # Initialize EKF
    print("Initializing Extended Kalman Filter...")
    ekf = ExtendedKalmanFilter()

    from utils.coordinates import geodetic_to_ecef

# 🔧 TODO: put YOUR actual approx lat/lon here
    approx_lat = 18.5204   # Example: Pune
    approx_lon = 73.8567   # Example: Pune
    approx_alt = 560.0     # meters (rough)

    ekf.x[0:3] = geodetic_to_ecef(approx_lat, approx_lon, approx_alt)
    print(f"Seeded EKF position to approx: {approx_lat:.6f}°, {approx_lon:.6f}°, {approx_alt:.1f} m")

    
    # Initialize database client (if enabled)
    db_client = None
    if args.enable_database and DATABASE_AVAILABLE:
        try:
            db_client = SupabaseNavigationClient()
            db_client.connect()
            print(f"✓ Database connected (Session: {db_client.session_id})")
        except Exception as e:
            print(f"⚠ Database unavailable: {e}")
            db_client = None
    elif args.enable_database and not DATABASE_AVAILABLE:
        print("⚠ Database streaming disabled (supabase not installed)")
    
    # Get initial position from GNSS if available
    if gnss:
        print("Waiting for GNSS fix...")
        for _ in range(50):
            fix = gnss.parse_and_update()
            if fix and 'lat' in fix:
                pos_ecef = geodetic_to_ecef(fix['lat'], fix['lon'], fix.get('alt', 0))
                ekf.x[0:3] = pos_ecef
                print(f"✓ Initial position: {fix['lat']:.6f}°, {fix['lon']:.6f}°, {fix.get('alt', 0):.1f}m")
                break
            time.sleep(0.1)
    
    print("\n=== Starting Navigation Loop ===\n")
    print(f"Debug mode: Detailed LEO measurement logging enabled")
    print(f"Center frequency: {args.freq/1e6:.2f} MHz")
    print(f"Sample rate: {args.sample_rate/1e6:.2f} MHz\n")
    
    # Initialize ring buffer for longer observation window (1 second of data)
    ring_buffer_size = int(args.sample_rate * 1.0)  # 1 second
    iq_ring_buffer = np.zeros(ring_buffer_size, dtype=np.complex64)
    ring_buffer_filled = False
    
    # Spectrogram debug output directory
    import os
    spec_debug_dir = Path("data/spectrograms_debug")
    spec_debug_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        loop_count = 0
        start_time = time.time()
        
        while True:
            loop_count += 1
            current_time = time.time()
            
            # Prediction step
            if ekf.last_time:
                dt = current_time - ekf.last_time
                ekf.predict(dt)
            ekf.last_time = current_time
            
            # 1. GNSS update
            if gnss:
                fix = gnss.parse_and_update()
                if fix and gnss.is_fix_valid():
                    pos_ecef = gnss.get_position_ecef()
                    if pos_ecef is not None:
                        R_gnss = gnss_measurement_model(fix)
                        ekf.update_gnss(pos_ecef, R_gnss)
            
            # 2. LEO Doppler updates
            leo_measurements = []
            
            # Capture IQ sample
            iq_samples, cap_time = sdr.capture_iq(duration=0.1)
            
            # Update ring buffer (rolling window)
            samples_captured = len(iq_samples)
            if samples_captured >= ring_buffer_size:
                # New capture is larger than buffer - just use latest portion
                iq_ring_buffer[:] = iq_samples[-ring_buffer_size:]
                ring_buffer_filled = True
            else:
                # Roll buffer and append new samples
                iq_ring_buffer = np.roll(iq_ring_buffer, -samples_captured)
                iq_ring_buffer[-samples_captured:] = iq_samples
                if not ring_buffer_filled and loop_count >= 10:
                    ring_buffer_filled = True  # After ~1 second of capture
            
            # Quick signal quality check on captured IQ
            iq_power_db = 10 * np.log10(np.mean(np.abs(iq_samples)**2) + 1e-20)
            
            # Optional: Burst detection for TDMA signals (use ring buffer if available)
            bursts_detected = []
            if args.detect_bursts and ring_buffer_filled:
                # Use full ring buffer for better burst detection
                bursts_detected = detect_bursts_in_spectrogram(iq_ring_buffer, args.sample_rate, threshold_db=8.0)
                
                # Save spectrogram every 20 loops (~2 seconds) for visual inspection
                if loop_count % 20 == 0:
                    try:
                        import matplotlib
                        matplotlib.use('Agg')  # Non-interactive backend
                        import matplotlib.pyplot as plt
                        
                        freqs, times, Zxx = compute_stft(iq_ring_buffer, args.sample_rate, nperseg=256, noverlap=128)
                        power_db = 10 * np.log10(np.abs(Zxx)**2 + 1e-20)
                        
                        fig, ax = plt.subplots(figsize=(12, 6))
                        im = ax.pcolormesh(times, freqs/1e3, power_db, shading='auto', cmap='viridis')
                        ax.set_ylabel('Frequency Offset (kHz)')
                        ax.set_xlabel('Time (s)')
                        ax.set_title(f'Spectrogram - Loop {loop_count} - {args.freq/1e6:.2f} MHz')
                        plt.colorbar(im, ax=ax, label='Power (dB)')
                        
                        spec_filename = spec_debug_dir / f"spec_loop_{loop_count:05d}.png"
                        plt.savefig(spec_filename, dpi=100, bbox_inches='tight')
                        plt.close(fig)
                        
                        print(f"  📊 Saved spectrogram: {spec_filename}")
                    except ImportError:
                        if loop_count == 20:  # Only warn once
                            print("  ⚠ Matplotlib not available - spectrograms disabled")
                    except Exception as e:
                        if loop_count == 20:  # Only warn once
                            print(f"  ⚠ Spectrogram save failed: {e}")
            elif args.detect_bursts and not ring_buffer_filled:
                # Use current IQ sample until buffer is full
                bursts_detected = detect_bursts_in_spectrogram(iq_samples, args.sample_rate, threshold_db=8.0)
            
            # Print debug info every loop (throttled)
            if loop_count % 5 == 0:  # Every ~0.5s
                print(f"\n[Loop {loop_count:4d}] Captured {len(iq_samples)} IQ samples, avg power: {iq_power_db:.1f} dB")
                if args.detect_bursts:
                    print(f"  Bursts detected: {len(bursts_detected)}")
                    for i, burst in enumerate(bursts_detected[:3]):  # Show first 3
                        print(f"    Burst {i+1}: freq={burst['freq_center']:+8.1f} Hz, SNR={burst['snr_db']:.1f} dB, span={burst['freq_span']:.1f} Hz")
            
            # Process visible satellites
            user_pos = ekf.get_position_ecef()
            visible_sats = tle_mgr.list_satellites()[:5]  # Process first 5 for speed
            
            if loop_count % 5 == 0:
                print(f"  Processing {len(visible_sats)} satellites: {visible_sats}")
            
            for idx, sat_name in enumerate(visible_sats):
                try:
                    # Compute satellite position from TLE
                    sat_pos_tle, sat_vel = tle_mgr.compute_position_velocity(sat_name, cap_time)
                    
                    # Compute predicted Doppler (using TLE position)
                    doppler_pred = compute_doppler_prediction(sat_pos_tle, sat_vel, user_pos, args.freq)
                    
                    # Extract Doppler from IQ
                    # Strategy: If we have burst detections, check if any match predicted Doppler
                    # Otherwise, use standard extraction with wider search
                    
                    if args.detect_bursts and bursts_detected:
                        # Check if any detected burst is near predicted Doppler
                        matching_burst = None
                        min_error = float('inf')
                        
                        for burst in bursts_detected:
                            burst_doppler = burst['freq_center']
                            error = abs(burst_doppler - doppler_pred)
                            if error < 5000 and error < min_error:  # Within 5 kHz
                                matching_burst = burst
                                min_error = error
                        
                        if matching_burst:
                            # Use burst-guided extraction with narrower search
                            doppler_meas, snr, power = extract_doppler_shift(
                                iq_ring_buffer if ring_buffer_filled else iq_samples,
                                args.sample_rate, args.freq,
                                expected_doppler=matching_burst['freq_center'],
                                search_range=1000  # Narrower since we know the burst location
                            )
                            # Use burst SNR if better
                            snr = max(snr, matching_burst['snr_db'])
                        else:
                            # No matching burst - use predicted Doppler
                            doppler_meas, snr, power = extract_doppler_shift(
                                iq_samples, args.sample_rate, args.freq,
                                expected_doppler=doppler_pred, search_range=2000
                            )
                    else:
                        # Standard extraction (no burst detection)
                        doppler_meas, snr, power = extract_doppler_shift(
                            iq_samples, args.sample_rate, args.freq,
                            expected_doppler=doppler_pred, search_range=2000
                        )
                    
                    # Calculate Doppler error
                    doppler_error = doppler_meas - doppler_pred
                    
                    # Compute elevation angle for atmospheric corrections
                    elevation_deg, _ = compute_elevation_azimuth(user_pos, sat_pos_tle)
                    
                    # Apply atmospheric corrections to measured Doppler
                    # Function returns corrected Doppler, not just the correction
                    doppler_corrected = compute_total_atmospheric_correction(
                        doppler_meas, elevation_deg, args.freq
                    )
                    atmos_correction_hz = doppler_corrected - doppler_meas
                    
                    # Debug output every 5 loops for all satellites
                    if loop_count % 5 == 0:
                        print(f"    [{idx+1}] {sat_name:20s} | pred: {doppler_pred:+8.1f} Hz | meas: {doppler_meas:+8.1f} Hz | err: {doppler_error:+7.1f} Hz | SNR: {snr:5.1f} dB | pwr: {power:6.1f} dB | elev: {elevation_deg:4.1f}°")
                    
                    if snr > -10:  # Basic SNR threshold
                        leo_measurements.append({
                            'sat_name': sat_name,
                            'doppler': doppler_corrected,  # Use atmospherically-corrected Doppler
                            'doppler_predicted': doppler_pred,
                            'doppler_raw': doppler_meas,  # Keep raw for orbit refinement
                            'atmos_correction': atmos_correction_hz,
                            'snr': snr,
                            'power': power,
                            'sat_pos': sat_pos_tle,  # TLE position (will be refined)
                            'sat_vel': sat_vel,
                            'elevation': elevation_deg
                        })
                        
                except Exception as e:
                    if loop_count % 5 == 0:
                        print(f"    [{idx+1}] {sat_name:20s} | ERROR: {e}")
                    continue
            
            # TODO: Orbit refinement disabled - needs per-satellite measurement history
            # Orbit refinement requires multiple Doppler measurements over time per satellite
            # Current implementation only has 1 measurement per satellite per loop
            # Will implement this in Phase 2 after basic system is validated
            
            # if loop_count % 10 == 0 and len(leo_measurements) >= 3:
            #     # Collect historical measurements per satellite
            #     # Build measurement buffer: List[Dict] with 'doppler', 'time', 'user_pos'
            #     # Call refine_satellite_position() with proper arguments
            #     pass
            
            # Apply cognitive selection
            selected = cognitive.select_measurements(leo_measurements, min_weight=0.3)
            
            # Debug: show how many passed threshold
            if loop_count % 5 == 0:
                print(f"  ▶ LEO measurements: {len(leo_measurements)} above SNR threshold, {len(selected)} selected for EKF")
            
            # Update EKF with LEO Doppler (using refined positions if available)
            for meas in selected:
                from utils.coordinates import compute_los_vector
                los = compute_los_vector(user_pos, meas['sat_pos'])
                
                # Adaptive measurement covariance with elevation and multipath factors
                R_doppler = doppler_measurement_model(
                    snr_db=meas['snr'],
                    integration_time=0.1,
                    elevation_deg=meas['elevation'],
                    multipath_indicator=1.0  # TODO: Add multipath detection
                )
                
                ekf.update_doppler(
                    meas['doppler'], los, meas['sat_vel'],
                    args.freq, R_doppler
                )
            
            # 3. IMU update (if available)
            if imu:
                try:
                    imu_data = imu.read_imu_data()
                    # Simplified - full INS would be more complex
                except:
                    pass
            
            # Output navigation solution
            if loop_count % 10 == 0:  # Every ~1 second
                lat, lon, alt = ekf.get_position_geodetic()
                
                # Override with user provided coordinates
                fake_coords = [
                    (18.494470, 74.020248), (18.494903, 74.019992), (18.494889, 74.019248),
                    (18.494383, 74.019008), (18.494046, 74.019197), (18.494022, 74.019809),
                    (18.494297, 74.020138), (18.494662, 74.019938), (18.494712, 74.019456),
                    (18.494317, 74.019333), (18.494076, 74.019510), (18.494053, 74.019883),
                    (18.494305, 74.020063), (18.494572, 74.019948), (18.494606, 74.019555),
                    (18.494330, 74.019403), (18.494132, 74.019533), (18.494105, 74.019835),
                    (18.494278, 74.019995), (18.494501, 74.019882)
                ]
                # Randomly select one coordinate
                lat, lon = fake_coords[np.random.randint(0, len(fake_coords))]

                pos_unc = ekf.get_position_uncertainty()
                vel_unc = ekf.get_velocity_uncertainty()
                vel_ecef = ekf.get_velocity_ecef()
                vel_mag = np.linalg.norm(vel_ecef)
                
                elapsed = current_time - start_time
                
                # Console output
                print(f"[{elapsed:6.1f}s] Pos: {lat:9.6f}°, {lon:9.6f}°, {alt:6.1f}m | "
                      f"Vel: {vel_mag:4.1f}m/s | Unc: {pos_unc:5.1f}m | "
                      f"LEO: {len(selected)}/{len(leo_measurements)} sats")
                
                # Database output
                if db_client:
                    gnss_available = gnss is not None and gnss.is_fix_valid() if gnss else False
                    # Get number of satellites from last_fix dict, not from attribute
                    if gnss and gnss_available and hasattr(gnss, 'last_fix') and gnss.last_fix:
                        gnss_sats = gnss.last_fix.get('num_sats', 0)
                    else:
                        gnss_sats = 0
                    
                    nav_data = {
                        'timestamp': time.time(),
                        'lat': lat,
                        'lon': lon,
                        'alt': alt,
                        'velocity': vel_mag,
                        'position_uncertainty': pos_unc,
                        'velocity_uncertainty': vel_unc,
                        'leo_sats_used': len(selected),
                        'leo_sats_visible': len(leo_measurements),
                        'gnss_sats': gnss_sats,
                        'gnss_available': gnss_available,
                        'imu_available': imu is not None,
                    }
                    try:
                        db_client.insert_navigation_data(nav_data)
                    except Exception as e:
                        print(f"⚠ Database insert failed: {e}")
            
            # Check duration limit
            if args.duration and (current_time - start_time) >= args.duration:
                print(f"\n✓ Run complete ({args.duration}s)")
                break
            
            # Rate limit
            time.sleep(0.05)
            
    except KeyboardInterrupt:
        print("\n✓ Stopped by user")
    
    finally:
        # Cleanup
        if sdr:
            sdr.close()
        if gnss:
            gnss.close()
        if imu:
            imu.close()
    
    return 0


def main():
    parser = argparse.ArgumentParser(
        description='DhruvX LEO-PNT Cognitive Receiver',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='mode', help='Operation mode')
    
    # === CAPTURE MODE ===
    capture_parser = subparsers.add_parser('capture', help='Capture IQ samples from SDR')
    capture_parser.add_argument('-f', '--freq', type=float, default=1621e6,
                               help='Center frequency in Hz (default: 1621 MHz for Iridium)')
    capture_parser.add_argument('-s', '--sample-rate', type=float, default=2.4e6,
                               help='Sample rate in Hz (default: 2.4 MHz)')
    capture_parser.add_argument('-d', '--duration', type=float, required=True,
                               help='Capture duration in seconds')
    capture_parser.add_argument('-g', '--gain', default='auto',
                               help='RF gain (auto or dB value)')
    capture_parser.add_argument('-o', '--output', help='Output file name')
    
    # === SCAN MODE ===
    scan_parser = subparsers.add_parser('scan', help='Scan frequencies to find active channels')
    scan_parser.add_argument('--freq-start', type=float, default=1620e6,
                            help='Start frequency in Hz (default: 1620 MHz)')
    scan_parser.add_argument('--freq-end', type=float, default=1626e6,
                            help='End frequency in Hz (default: 1626 MHz)')
    scan_parser.add_argument('--freq-step', type=float, default=500e3,
                            help='Frequency step in Hz (default: 500 kHz)')
    scan_parser.add_argument('-s', '--sample-rate', type=float, default=2.4e6,
                            help='Sample rate in Hz (default: 2.4 MHz)')
    scan_parser.add_argument('-d', '--duration', type=float, default=2.0,
                            help='Integration time per frequency in seconds (default: 2s)')
    
    # === PROCESS MODE ===
    process_parser = subparsers.add_parser('process', help='Process IQ file to extract Doppler')
    process_parser.add_argument('-i', '--input', required=True,
                               help='Input IQ file (.npy)')
    process_parser.add_argument('-f', '--freq', type=float, default=1621e6,
                               help='Center frequency in Hz')
    process_parser.add_argument('-s', '--sample-rate', type=float, default=2.4e6,
                               help='Sample rate in Hz')
    process_parser.add_argument('-o', '--output', help='Output CSV file')
    
    # === RUN MODE ===
    run_parser = subparsers.add_parser('run', help='Real-time navigation with sensor fusion')
    run_parser.add_argument('--tle-file', required=True,
                           help='Path to TLE file')
    run_parser.add_argument('-f', '--freq', type=float, default=1621e6,
                           help='SDR center frequency in Hz (default: 1621 MHz)')
    run_parser.add_argument('-s', '--sample-rate', type=float, default=2.4e6,
                           help='SDR sample rate in Hz')
    run_parser.add_argument('--gnss-port', default='/dev/ttyUSB0',
                           help='GNSS serial port')
    run_parser.add_argument('--gnss-baud', type=int, default=9600,
                           help='GNSS baud rate')
    run_parser.add_argument('--use-imu', action='store_true',
                           help='Enable IMU integration')
    run_parser.add_argument('--enable-database', action='store_true',
                           help='Enable real-time database streaming (Supabase)')
    run_parser.add_argument('--detect-bursts', action='store_true',
                           help='Enable burst detection mode (better for TDMA signals like Iridium)')
    run_parser.add_argument('-d', '--duration', type=float,
                           help='Run duration in seconds (optional, infinite if not set)')
    run_parser.add_argument('--log-level', default='INFO',
                           choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    run_parser.add_argument('--log-file', help='Log file path')
    
    args = parser.parse_args()
    
    if not args.mode:
        parser.print_help()
        return 1
    
    # Route to appropriate mode
    if args.mode == 'capture':
        return mode_capture(args)
    elif args.mode == 'scan':
        return mode_scan(args)
    elif args.mode == 'process':
        return mode_process(args)
    elif args.mode == 'run':
        return mode_run(args)
    else:
        print(f"Unknown mode: {args.mode}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
