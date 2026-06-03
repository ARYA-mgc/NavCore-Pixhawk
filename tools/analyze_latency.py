#!/usr/bin/env python3
"""
analyze_latency.py
Analyzes MAVLink logs (.tlog) to statistically measure the temporal latency 
between RAW_IMU integration timestamps and GPS_RAW_INT / VISION_POSITION_ESTIMATE reception.
"""
import sys
import os
import argparse
import numpy as np

try:
    from pymavlink import mavutil
except ImportError:
    print("Error: pymavlink is required to run this script (pip install pymavlink)")
    sys.exit(1)

def analyze_latency(log_path: str):
    if not os.path.exists(log_path):
        print(f"Error: Log file {log_path} not found.")
        sys.exit(1)

    print(f"Analyzing MAVLink log: {log_path}")
    
    # pymavlink setup
    mlog = mavutil.mavlink_connection(log_path)
    
    gps_latencies = []
    vio_latencies = []
    
    last_imu_time_usec = 0
    
    # Stream messages
    while True:
        msg = mlog.recv_match(type=['RAW_IMU', 'HIGHRES_IMU', 'GPS_RAW_INT', 'VISION_POSITION_ESTIMATE'])
        if msg is None:
            break
            
        msg_type = msg.get_type()
        
        if msg_type in ['RAW_IMU', 'HIGHRES_IMU']:
            last_imu_time_usec = msg.time_usec
            
        elif msg_type == 'GPS_RAW_INT':
            # GPS messages typically have a time_usec field (system time when message was generated)
            if hasattr(msg, 'time_usec') and last_imu_time_usec > 0:
                # This is a rough estimation of when the GPS was integrated into the system vs IMU
                delay_us = msg.time_usec - last_imu_time_usec
                # Filter out negative delays (e.g. if GPS timestamp is generated before IMU) or giant delays
                if 0 < delay_us < 2000000:  
                    gps_latencies.append(delay_us / 1e6)
                    
        elif msg_type == 'VISION_POSITION_ESTIMATE':
            if hasattr(msg, 'usec') and last_imu_time_usec > 0:
                delay_us = msg.usec - last_imu_time_usec
                if 0 < delay_us < 2000000:
                    vio_latencies.append(delay_us / 1e6)

    print("\n--- Latency Analysis Results ---")
    
    if gps_latencies:
        gps_lat = np.array(gps_latencies)
        print(f"GPS Latency (N={len(gps_lat)}):")
        print(f"  Mean: {np.mean(gps_lat)*1000:.1f} ms")
        print(f"  Min:  {np.min(gps_lat)*1000:.1f} ms")
        print(f"  Max:  {np.max(gps_lat)*1000:.1f} ms")
        print(f"  Std:  {np.std(gps_lat)*1000:.1f} ms")
        print("  --- Percentiles ---")
        print(f"  50th: {np.percentile(gps_lat, 50)*1000:.1f} ms")
        print(f"  90th: {np.percentile(gps_lat, 90)*1000:.1f} ms")
        print(f"  95th: {np.percentile(gps_lat, 95)*1000:.1f} ms")
        print(f"  99th: {np.percentile(gps_lat, 99)*1000:.1f} ms")
    else:
        print("No GPS latency data found. (Check if GPS_RAW_INT and RAW_IMU exist in log)")

    if vio_latencies:
        vio_lat = np.array(vio_latencies)
        print(f"\nVIO Latency (N={len(vio_lat)}):")
        print(f"  Mean: {np.mean(vio_lat)*1000:.1f} ms")
        print(f"  Min:  {np.min(vio_lat)*1000:.1f} ms")
        print(f"  Max:  {np.max(vio_lat)*1000:.1f} ms")
        print(f"  Std:  {np.std(vio_lat)*1000:.1f} ms")
        print("  --- Percentiles ---")
        print(f"  50th: {np.percentile(vio_lat, 50)*1000:.1f} ms")
        print(f"  90th: {np.percentile(vio_lat, 90)*1000:.1f} ms")
        print(f"  95th: {np.percentile(vio_lat, 95)*1000:.1f} ms")
        print(f"  99th: {np.percentile(vio_lat, 99)*1000:.1f} ms")
    else:
        print("No VIO latency data found. (Check if VISION_POSITION_ESTIMATE exists)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze sensor latency from MAVLink logs.")
    parser.add_argument("log_file", help="Path to .tlog or .bin file")
    args = parser.parse_args()
    analyze_latency(args.log_file)
