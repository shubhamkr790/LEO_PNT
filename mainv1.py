import argparse
import sys
import time
import numpy as np
import csv
import json
from pathlib import Path
from datetime import datetime

from fusion import ekf
from utils import setup_logging, TLEManager, geodetic_to_ecef, compute_doppler_prediction, compute_visible_satellites
from utils.atmospheric import compute_total_atmospheric_correction
from utils.coordinates import compute_elevation_azimuth
from hw import RTLSDRCapture, GNSSReceiver, IMUReader
from dsp import compute_stft, extract_doppler_shift, compute_psd, detect_bursts_in_spectrogram
from fusion import ExtendedKalmanFilter, gnss_measurement_model, doppler_measurement_model

try:
    from database import SupabaseNavigationClient
    DATABASE_AVAILABLE = True
except ImportError:
    DATABASE_AVAILABLE = False


def mode_capture(args):
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
            
            np.save(output_file, iq_samples)
            print(f"Saved {len(iq_samples)} samples to {output_file}")
            
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
    print(f"=== PROCESS MODE ===")
    print(f"Input file: {args.input}")
    
    setup_logging('INFO')
    
    try:
        iq_samples = np.load(args.input)
        print(f"Loaded {len(iq_samples)} IQ samples")
    except Exception as e:
        print(f"✗ Failed to load IQ file: {e}")
        return 1
    
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
    
    print("Computing STFT spectrogram...")
    freqs, times, Sxx = compute_stft(iq_samples, sample_rate)
    print(f"Spectrogram: {Sxx.shape[0]} freq bins × {Sxx.shape[1]} time bins")
    
    print("Extracting Doppler shifts...")
    results = []
    
    window_size = 0.1
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
    
    output_file = args.output if args.output else args.input.replace('.npy', '_doppler.csv')
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['time', 'doppler_hz', 'snr_db', 'power_db'])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"✓ Saved {len(results)} Doppler measurements to {output_file}")
    
    return 0


def mode_scan(args):
    print(f"=== SCAN MODE ===")
    print(f"Scanning frequencies from {args.freq_start/1e6:.2f} to {args.freq_end/1e6:.2f} MHz")
    print(f"Step size: {args.freq_step/1e3:.1f} kHz")
    print(f"Integration time: {args.duration}s per frequency\n")
    
    setup_logging('INFO')
    
    # Create output directory for burst captures
    scan_dir = Path("data/scan_captures")
    scan_dir.mkdir(parents=True, exist_ok=True)
    print(f"Burst captures will be saved to: {scan_dir}\n")
    
    freqs = np.arange(args.freq_start, args.freq_end + args.freq_step, args.freq_step)
    
    results = []
    burst_captures = []
    
    try:
        with RTLSDRCapture(center_freq=freqs[0], sample_rate=args.sample_rate) as sdr:
            for idx, freq in enumerate(freqs):
                progress = (idx + 1) / len(freqs) * 100
                print(f"[{idx+1}/{len(freqs)}] ({progress:5.1f}%) Testing {freq/1e6:.2f} MHz...", end=' ')
                
                sdr.retune(freq)
                time.sleep(0.1)
                
                iq_samples, cap_time = sdr.capture_iq(args.duration)
                
                bursts = detect_bursts_in_spectrogram(
                    iq_samples, 
                    args.sample_rate, 
                    threshold_db=8.0,
                    nperseg=512,
                    min_duration_ms=20.0,
                    min_snr_db=6.0,
                    persistence_required=1,
                    remove_dc=True
                )
                
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
                
                # Save IQ data if bursts detected
                if num_bursts > 0:
                    timestamp_str = datetime.utcfromtimestamp(cap_time).strftime('%Y%m%d_%H%M%S')
                    freq_mhz_str = f"{freq/1e6:.2f}".replace('.', '_')
                    base_filename = f"burst_{freq_mhz_str}MHz_{timestamp_str}"
                    
                    # Save as .npy (NumPy binary format)
                    npy_file = scan_dir / f"{base_filename}.npy"
                    np.save(npy_file, iq_samples)
                    
                    # Save as .iq (raw binary format - interleaved I/Q float32)
                    iq_file = scan_dir / f"{base_filename}.iq"
                    # Convert complex64 to interleaved float32 (I, Q, I, Q, ...)
                    iq_interleaved = np.empty(len(iq_samples) * 2, dtype=np.float32)
                    iq_interleaved[0::2] = iq_samples.real.astype(np.float32)
                    iq_interleaved[1::2] = iq_samples.imag.astype(np.float32)
                    iq_interleaved.tofile(iq_file)
                    
                    # Save metadata as JSON
                    json_file = scan_dir / f"{base_filename}.json"
                    metadata = {
                        'timestamp_utc': datetime.utcfromtimestamp(cap_time).isoformat() + 'Z',
                        'timestamp_unix': float(cap_time),
                        'center_frequency_hz': float(freq),
                        'center_frequency_mhz': float(freq / 1e6),
                        'sample_rate_hz': float(args.sample_rate),
                        'sample_rate_mhz': float(args.sample_rate / 1e6),
                        'num_samples': int(len(iq_samples)),
                        'duration_seconds': float(args.duration),
                        'avg_power_db': float(avg_power_db),
                        'num_bursts_detected': int(num_bursts),
                        'max_burst_snr_db': float(max_snr),
                        'bursts': [
                            {
                                'time_start_s': float(b['time_start']),
                                'time_end_s': float(b['time_end']),
                                'duration_ms': float(b.get('duration_ms', 0)),
                                'freq_center_hz': float(b['freq_center']),
                                'freq_span_hz': float(b['freq_span']),
                                'snr_db': float(b['snr_db']),
                                'peak_power_db': float(b['peak_power_db'])
                            }
                            for b in bursts
                        ],
                        'files': {
                            'npy': str(npy_file.name),
                            'iq': str(iq_file.name),
                            'json': str(json_file.name)
                        }
                    }
                    
                    with open(json_file, 'w') as f:
                        json.dump(metadata, f, indent=2)
                    
                    burst_captures.append({
                        'freq': freq,
                        'num_bursts': num_bursts,
                        'files': base_filename
                    })
                    
                    print(f"  💾 Saved: {base_filename}.{{npy,iq,json}}")
    
    except Exception as e:
        print(f"\n✗ Scan failed: {e}")
        return 1
    
    print(f"\n=== SCAN SUMMARY ===")
    best = max(results, key=lambda x: x['max_snr_db'])
    print(f"Best frequency: {best['freq']/1e6:.2f} MHz (SNR: {best['max_snr_db']:.1f} dB, {best['num_bursts']} bursts)")
    
    top3 = sorted(results, key=lambda x: x['max_snr_db'], reverse=True)[:3]
    print(f"\nTop 3 frequencies:")
    for i, r in enumerate(top3):
        print(f"  {i+1}. {r['freq']/1e6:.2f} MHz - SNR: {r['max_snr_db']:5.1f} dB, {r['num_bursts']} bursts")
    
    # Summary of saved captures
    if burst_captures:
        print(f"\n=== BURST CAPTURES ===")
        print(f"Saved {len(burst_captures)} frequency captures with bursts:")
        for cap in burst_captures:
            print(f"  • {cap['freq']/1e6:.2f} MHz - {cap['num_bursts']} burst(s) - {cap['files']}.{{npy,iq,json}}")
        print(f"\nAll files saved to: {scan_dir.absolute()}")
    else:
        print(f"\n⚠ No bursts detected - no captures saved")
    
    if args.auto_run and best['num_bursts'] > 0:
        print(f"\n{'='*60}")
        print(f"🚀 AUTO-RUN MODE ENABLED")
        print(f"{'='*60}")
        print(f"Starting navigation at best frequency: {best['freq']/1e6:.2f} MHz")
        print(f"Duration: {args.run_duration}s")
        print(f"TLE file: {args.tle_file}")
        print(f"{'='*60}\n")
        
        time.sleep(2)
        
        class RunArgs:
            def __init__(self):
                self.tle_file = args.tle_file
                self.freq = best['freq']
                self.sample_rate = args.sample_rate
                self.gnss_port = '/dev/ttyUSB0'
                self.gnss_baud = 9600
                self.use_imu = args.use_imu
                self.enable_database = False
                self.detect_bursts = True
                self.duration = args.run_duration
                self.log_level = 'INFO'
                self.log_file = None
        
        run_args = RunArgs()
        return mode_run(run_args)
    
    return 0


def mode_run(args):
    print(f"=== RUN MODE (Real-Time Navigation) ===")
    setup_logging(args.log_level, args.log_file)
    
    print(f"Loading TLEs from {args.tle_file}...")
    try:
        tle_mgr = TLEManager(args.tle_file)
        print(f"✓ Loaded {len(tle_mgr.list_satellites())} satellites")
    except Exception as e:
        print(f"✗ Failed to load TLEs: {e}")
        return 1
    
    print("Initializing hardware...")
    
    try:
        gnss = GNSSReceiver(port=args.gnss_port, baudrate=args.gnss_baud)
        gnss.open()
        print(f"✓ GNSS opened on {args.gnss_port}")
    except Exception as e:
        print(f"⚠ GNSS unavailable: {e}")
        gnss = None
    
    try:
        sdr = RTLSDRCapture(center_freq=args.freq, sample_rate=args.sample_rate)
        sdr.open()
        print(f"✓ RTL-SDR opened at {args.freq/1e6:.2f} MHz")
    except Exception as e:
        print(f"✗ RTL-SDR unavailable: {e}")
        print("Cannot run without SDR")
        return 1
    
    imu = None
    if args.use_imu:
        try:
            imu = IMUReader()
            imu.open()
            print("✓ IMU opened")
        except Exception as e:
            print(f"⚠ IMU unavailable: {e}")
    
    print("⚠ ML disabled: Using all Doppler measurements (no cognitive filtering)")

    class DummyCognitiveSelector:
        def select_measurements(self, measurements, min_weight=0.3):
            return measurements

    cognitive = DummyCognitiveSelector()

    
    print("Initializing Extended Kalman Filter...")
    ekf = ExtendedKalmanFilter()

    from utils.coordinates import geodetic_to_ecef

    # ⚠️ UPDATE THESE COORDINATES FOR YOUR TEST LOCATION ⚠️
    approx_lat = 18.4946     # Your latitude (degrees)
    approx_lon = 73.0207     # Your longitude (degrees)
    known_alt_m = 534.0      # Your FIXED ALTITUDE in meters (MUST BE ACCURATE!)
    altitude_sigma_m = 5.0   # Altitude uncertainty (±5m) - keeps altitude constrained

    # Seed EKF with known position
    ekf.x[0:3] = geodetic_to_ecef(approx_lat, approx_lon, known_alt_m)
    print(f"Seeded EKF position to: {approx_lat:.6f}°, {approx_lon:.6f}°, {known_alt_m:.1f}m")
    
    # Apply altitude constraint for 2-3 satellite positioning
    ekf.constrain_altitude(known_alt_m, altitude_sigma_m)
    print(f"✓ Altitude constrained to {known_alt_m:.1f}m ±{altitude_sigma_m:.1f}m (enables 2D positioning with 2-3 satellites)")
    print(f"  With altitude fixed, you can get accurate lat/lon from just 2 satellites!")

    
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
    
    ring_buffer_size = int(args.sample_rate * 1.0)
    iq_ring_buffer = np.zeros(ring_buffer_size, dtype=np.complex64)
    ring_buffer_filled = False
    
    import os
    spec_debug_dir = Path("data/spectrograms_debug")
    spec_debug_dir.mkdir(parents=True, exist_ok=True)
    
    
    try:
        loop_count = 0
        start_time = time.time()
        
        while True:
            loop_count += 1
            current_time = time.time()
            
            if ekf.last_time:
                dt = current_time - ekf.last_time
                ekf.predict(dt)
            ekf.last_time = current_time
            
            if gnss:
                fix = gnss.parse_and_update()
                if fix and gnss.is_fix_valid():
                    pos_ecef = gnss.get_position_ecef()
                    if pos_ecef is not None:
                        R_gnss = gnss_measurement_model(fix)
                        ekf.update_gnss(pos_ecef, R_gnss)
            
            leo_measurements = []
            
            iq_samples, cap_time = sdr.capture_iq(duration=0.1)
            
            samples_captured = len(iq_samples)
            if samples_captured >= ring_buffer_size:
                iq_ring_buffer[:] = iq_samples[-ring_buffer_size:]
                ring_buffer_filled = True
            else:
                iq_ring_buffer = np.roll(iq_ring_buffer, -samples_captured)
                iq_ring_buffer[-samples_captured:] = iq_samples
                if not ring_buffer_filled and loop_count >= 10:
                    ring_buffer_filled = True
            
            iq_power_db = 10 * np.log10(np.mean(np.abs(iq_samples)**2) + 1e-20)
            
            # Get user position and compute actually VISIBLE satellites BEFORE burst detection
            user_pos = ekf.get_position_ecef()
            
            # Get satellites actually above horizon (min 1° elevation)
            visible_sat_data = compute_visible_satellites(tle_mgr, user_pos, cap_time, min_elevation=-10.0)
            
            # Sort by elevation (highest first) and take top 8
            visible_sat_data = sorted(visible_sat_data, key=lambda x: x['elevation'], reverse=True)[:8]
            visible_sats = [sat['name'] for sat in visible_sat_data]
            
            # Diagnostics on first loop
            if loop_count == 1 and visible_sat_data:
                print(f"\n✓ Found {len(visible_sat_data)} visible satellites above 1° elevation:")
                for sat in visible_sat_data:
                    print(f"  • {sat['name']:20s} - Elevation: {sat['elevation']:5.1f}°, Azimuth: {sat['azimuth']:6.1f}°")
                print()
            
            # Fallback: if no satellites above horizon, use first 5 from TLE file
            if not visible_sats:
                visible_sats = tle_mgr.list_satellites()[:5]
                if loop_count == 1:
                    print(f"\n⚠ WARNING: No satellites above 1° elevation!")
                    print(f"  Your position: {ekf.get_position_geodetic()}")
                    print(f"  This usually means: outdated TLE, wrong position, or wrong system time")
                    print(f"  Falling back to first 5 satellites from TLE file (may be below horizon)\n")
                elif loop_count % 20 == 0:
                    print("  ⚠ Still no satellites above horizon")
            
            bursts_detected = []
            if args.detect_bursts and ring_buffer_filled:
                bursts_detected = detect_bursts_in_spectrogram(
                    iq_ring_buffer, 
                    args.sample_rate, 
                    threshold_db=8.0,
                    nperseg=512,
                    min_duration_ms=15.0,
                    min_freq_span_hz=1000.0,
                    min_snr_db=5.0,
                    persistence_required=1,
                    remove_dc=True
                )
                
                if loop_count % 20 == 0:
                    try:
                        import matplotlib
                        matplotlib.use('Agg')
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
                        if loop_count == 20:
                            print("  ⚠ Matplotlib not available - spectrograms disabled")
                    except Exception as e:
                        if loop_count == 20:
                            print(f"  ⚠ Spectrogram save failed: {e}")
            elif args.detect_bursts and not ring_buffer_filled:
                bursts_detected = detect_bursts_in_spectrogram(
                    iq_samples, 
                    args.sample_rate, 
                    threshold_db=8.0,
                    nperseg=512,
                    min_duration_ms=15.0,
                    min_snr_db=5.0,
                    persistence_required=1,
                    remove_dc=True
                )
            
            if loop_count % 5 == 0:
                print(f"\n[Loop {loop_count:4d}] Captured {len(iq_samples)} IQ samples, avg power: {iq_power_db:.1f} dB")
                if args.detect_bursts:
                    print(f"  Bursts detected: {len(bursts_detected)+1}")
                    for i, burst in enumerate(bursts_detected[:3]):
                        doppler_info = f", pred={burst.get('matched_doppler', 0):+8.1f} Hz, err={burst.get('doppler_error', 0):+6.1f} Hz" if 'matched_doppler' in burst else ""
                        print(f"    Burst {i+1}: freq={burst['freq_center']:+8.1f} Hz, SNR={burst['snr_db']:.1f} dB, "
                              f"span={burst['freq_span']:.1f} Hz, dur={burst.get('duration_ms', 0):.1f}ms{doppler_info}")
            
            # user_pos and visible_sats already defined above (before burst detection)
            
            if loop_count % 5 == 0:
                print(f"  Processing {len(visible_sats)} satellites: {visible_sats}")
            
            for idx, sat_name in enumerate(visible_sats):
                try:
                    sat_pos_tle, sat_vel = tle_mgr.compute_position_velocity(sat_name, cap_time)
                    
                    doppler_pred = compute_doppler_prediction(sat_pos_tle, sat_vel, user_pos, args.freq)
                    
                    if args.detect_bursts and bursts_detected:
                        matching_burst = None
                        min_error = float('inf')
                        
                        for burst in bursts_detected:
                            burst_doppler = burst['freq_center']
                            error = abs(burst_doppler - doppler_pred)
                            if error < 5000 and error < min_error:
                                matching_burst = burst
                                min_error = error
                        
                        if matching_burst:
                            doppler_meas, snr, power = extract_doppler_shift(
                                iq_ring_buffer if ring_buffer_filled else iq_samples,
                                args.sample_rate, args.freq,
                                expected_doppler=matching_burst['freq_center'],
                                search_range=1000
                            )
                            snr = max(snr, matching_burst['snr_db'])
                        else:
                            doppler_meas, snr, power = extract_doppler_shift(
                                iq_samples, args.sample_rate, args.freq,
                                expected_doppler=doppler_pred, search_range=2000
                            )
                    else:
                        doppler_meas, snr, power = extract_doppler_shift(
                            iq_samples, args.sample_rate, args.freq,
                            expected_doppler=doppler_pred, search_range=2000
                        )
                    
                    doppler_error = doppler_meas - doppler_pred
                    
                    elevation_deg, _ = compute_elevation_azimuth(user_pos, sat_pos_tle)
                    
                    doppler_corrected = compute_total_atmospheric_correction(
                        doppler_meas, elevation_deg, args.freq
                    )
                    atmos_correction_hz = doppler_corrected - doppler_meas
                    
                    if loop_count % 5 == 0:
                        print(f"    [{idx+1}] {sat_name:20s} | pred: {doppler_pred:+8.1f} Hz | meas: {doppler_meas:+8.1f} Hz | err: {doppler_error:+7.1f} Hz | SNR: {snr:5.1f} dB | pwr: {power:6.1f} dB | elev: {elevation_deg:4.1f}°")
                    
                    if snr > -10:
                        leo_measurements.append({
                            'sat_name': sat_name,
                            'doppler': doppler_corrected,
                            'doppler_predicted': doppler_pred,
                            'doppler_raw': doppler_meas,
                            'atmos_correction': atmos_correction_hz,
                            'snr': snr,
                            'power': power,
                            'sat_pos': sat_pos_tle,
                            'sat_vel': sat_vel,
                            'elevation': elevation_deg
                        })
                        
                except Exception as e:
                    if loop_count % 5 == 0:
                        print(f"    [{idx+1}] {sat_name:20s} | ERROR: {e}")
                    continue
            
            selected = cognitive.select_measurements(leo_measurements, min_weight=0.3)
            
            if loop_count % 5 == 0:
                print(f"  ▶ LEO measurements: {len(leo_measurements)} above SNR threshold, {len(selected)} selected for EKF")
            
            for meas in selected:
                from utils.coordinates import compute_los_vector
                los = compute_los_vector(user_pos, meas['sat_pos'])
                
                R_doppler = doppler_measurement_model(
                    snr_db=meas['snr'],
                    integration_time=0.1,
                    elevation_deg=meas['elevation'],
                    multipath_indicator=1.0
                )
                
                ekf.update_doppler(
                    meas['doppler'], los, meas['sat_vel'],
                    args.freq, R_doppler
                )
            
            # Re-apply altitude constraint every loop to keep altitude fixed
            ekf.update_altitude_pseudomeasurement(known_alt_m, altitude_sigma_m)
            
            if imu:
                try:
                    imu_data = imu.read_imu_data()
                    if imu_data['timestamp'] > 0:
                        accel = imu_data['accel']
                        R_imu = np.eye(3) * 0.1
                        ekf.update_imu(accel, dt=0.1, R_accel=R_imu)
                        
                        if loop_count % 50 == 0:
                            print(f"  📡 IMU: accel=[{accel[0]:6.2f}, {accel[1]:6.2f}, {accel[2]:6.2f}] m/s²")
                except Exception as e:
                    if loop_count % 100 == 0:
                        print(f"  ⚠ IMU read error: {e}")
            
            if loop_count % 5 == 0:
                lat, lon, alt = ekf.get_position_geodetic()
                pos_unc = ekf.get_position_uncertainty()
                vel_unc = ekf.get_velocity_uncertainty()
                vel_ecef = ekf.get_velocity_ecef()
                vel_mag = np.linalg.norm(vel_ecef)
                
                elapsed = current_time - start_time
                
                print(f"[{elapsed:6.1f}s] Pos: {lat:9.6f}°, {lon:9.6f}°, {alt:6.1f}m | "
                      f"Vel: {vel_mag:4.1f}m/s | Unc: {pos_unc:5.1f}m | "
                      f"LEO: {len(selected)}/{len(leo_measurements)} sats")
                
                if db_client:
                    gnss_available = gnss is not None and gnss.is_fix_valid() if gnss else False
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
            
            if args.duration and (current_time - start_time) >= args.duration:
                print(f"\n✓ Run complete ({args.duration}s)")
                break
            
            time.sleep(0.05)
            
    except KeyboardInterrupt:
        print("\n✓ Stopped by user")
    
    finally:
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
    scan_parser.add_argument('--auto-run', action='store_true',
                            help='Automatically start run mode with best frequency after scan')
    scan_parser.add_argument('--tle-file', default='tles/iridium.txt',
                            help='TLE file for auto-run mode (default: tles/iridium.txt)')
    scan_parser.add_argument('--run-duration', type=float, default=60,
                            help='Run mode duration in seconds for auto-run (default: 60s)')
    scan_parser.add_argument('--use-imu', action='store_true',
                            help='Enable IMU in auto-run mode')
    
    process_parser = subparsers.add_parser('process', help='Process IQ file to extract Doppler')
    process_parser.add_argument('-i', '--input', required=True,
                               help='Input IQ file (.npy)')
    process_parser.add_argument('-f', '--freq', type=float, default=1621e6,
                               help='Center frequency in Hz')
    process_parser.add_argument('-s', '--sample-rate', type=float, default=2.4e6,
                               help='Sample rate in Hz')
    process_parser.add_argument('-o', '--output', help='Output CSV file')
    
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
