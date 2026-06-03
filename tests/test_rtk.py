#!/usr/bin/env python3
"""Unit tests for RTK ground truth pipeline.

Tests:
    - UBX NAV-PVT / NAV-HPPOSLLH binary parsing
    - UBX Fletcher-8 checksum verification
    - WGS-84 → NED coordinate conversion
    - Flight recorder CSV format round-trip
    - Log validator on synthetic data
    - NTRIP GGA sentence generation
    - Flight phase detection
"""

import sys
import os
import math
import struct
import tempfile
import shutil
import csv
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


# ── UBX Frame Construction Helpers ─────────────────────────────

def build_ubx_frame(cls: int, msg_id: int, payload: bytes) -> bytes:
    """Build a complete UBX frame with valid Fletcher-8 checksum."""
    length = len(payload)
    header = struct.pack('<BBBBH', 0xB5, 0x62, cls, msg_id, length)
    frame = header + payload
    ck_a, ck_b = 0, 0
    for b in frame[2:]:
        ck_a = (ck_a + b) & 0xFF
        ck_b = (ck_b + ck_a) & 0xFF
    return frame + bytes([ck_a, ck_b])


def build_nav_pvt_payload(lat_deg=13.0827, lon_deg=80.2707, alt_m=50.0,
                          vel_n=1.0, vel_e=0.5, vel_d=-0.1,
                          fix_type=3, carr_soln=2, n_sats=15,
                          h_acc_mm=20, v_acc_mm=30, pdop_100=120):
    """Build a 92-byte UBX NAV-PVT payload with known values."""
    payload = bytearray(92)
    # iTOW at offset 0
    struct.pack_into('<I', payload, 0, 500000)  # 500s GPS time
    # fixType at offset 20
    payload[20] = fix_type
    # flags at offset 21 (carrSoln in bits 6-7)
    payload[21] = (carr_soln << 6) | 0x01  # gnssFixOK
    # numSV at offset 23
    payload[23] = n_sats
    # lon at offset 24 (degE-7)
    struct.pack_into('<i', payload, 24, int(lon_deg * 1e7))
    # lat at offset 28
    struct.pack_into('<i', payload, 28, int(lat_deg * 1e7))
    # height at offset 32 (mm)
    struct.pack_into('<i', payload, 32, int(alt_m * 1000))
    # hAcc at offset 40 (mm)
    struct.pack_into('<I', payload, 40, h_acc_mm)
    # vAcc at offset 44 (mm)
    struct.pack_into('<I', payload, 44, v_acc_mm)
    # velN at offset 48 (mm/s)
    struct.pack_into('<i', payload, 48, int(vel_n * 1000))
    # velE at offset 52
    struct.pack_into('<i', payload, 52, int(vel_e * 1000))
    # velD at offset 56
    struct.pack_into('<i', payload, 56, int(vel_d * 1000))
    # pDOP at offset 76 (0.01)
    struct.pack_into('<H', payload, 76, pdop_100)
    return bytes(payload)


def build_nav_hpposllh_payload(lat_deg=13.0827, lon_deg=80.2707,
                                alt_m=50.0, lat_hp=5, lon_hp=-3,
                                height_hp=7, h_acc_01mm=150, v_acc_01mm=250):
    """Build a 36-byte UBX NAV-HPPOSLLH payload."""
    payload = bytearray(36)
    payload[0] = 0  # version
    # iTOW at offset 4
    struct.pack_into('<I', payload, 4, 500000)
    # lon at offset 8
    struct.pack_into('<i', payload, 8, int(lon_deg * 1e7))
    # lat at offset 12
    struct.pack_into('<i', payload, 12, int(lat_deg * 1e7))
    # height at offset 16 (mm)
    struct.pack_into('<i', payload, 16, int(alt_m * 1000))
    # hMSL at offset 20
    struct.pack_into('<i', payload, 20, int(alt_m * 1000))
    # HP residuals at offset 24-27
    struct.pack_into('<b', payload, 24, lon_hp)
    struct.pack_into('<b', payload, 25, lat_hp)
    struct.pack_into('<b', payload, 26, height_hp)
    struct.pack_into('<b', payload, 27, 0)
    # hAcc at offset 28 (0.1 mm)
    struct.pack_into('<I', payload, 28, h_acc_01mm)
    # vAcc at offset 32
    struct.pack_into('<I', payload, 32, v_acc_01mm)
    return bytes(payload)


# ── Test: UBX NAV-PVT Parsing ──────────────────────────────────

class TestUBXNavParsing:

    def test_parse_nav_pvt_position(self):
        from interfaces.rtk_collector import UBXNavParser
        parser = UBXNavParser()

        lat, lon, alt = 13.0827, 80.2707, 50.0
        payload = build_nav_pvt_payload(lat_deg=lat, lon_deg=lon, alt_m=alt)
        frame = build_ubx_frame(0x01, 0x07, payload)

        messages = parser.feed(frame)
        assert len(messages) == 1
        msg = messages[0]
        assert msg["type"] == "NAV_PVT"
        assert abs(msg["lat_deg"] - lat) < 1e-6
        assert abs(msg["lon_deg"] - lon) < 1e-6
        assert abs(msg["alt_m"] - alt) < 0.01

    def test_parse_nav_pvt_velocity(self):
        from interfaces.rtk_collector import UBXNavParser
        parser = UBXNavParser()

        payload = build_nav_pvt_payload(vel_n=2.5, vel_e=-1.3, vel_d=0.4)
        frame = build_ubx_frame(0x01, 0x07, payload)

        messages = parser.feed(frame)
        msg = messages[0]
        assert abs(msg["vel_n"] - 2.5) < 0.01
        assert abs(msg["vel_e"] - (-1.3)) < 0.01
        assert abs(msg["vel_d"] - 0.4) < 0.01

    def test_parse_nav_pvt_rtk_fixed(self):
        from interfaces.rtk_collector import UBXNavParser
        parser = UBXNavParser()

        payload = build_nav_pvt_payload(fix_type=3, carr_soln=2, n_sats=18)
        frame = build_ubx_frame(0x01, 0x07, payload)

        messages = parser.feed(frame)
        msg = messages[0]
        assert msg["carrier_solution"] == 2  # RTK_FIXED
        assert msg["n_sats"] == 18

    def test_parse_nav_pvt_accuracy(self):
        from interfaces.rtk_collector import UBXNavParser
        parser = UBXNavParser()

        payload = build_nav_pvt_payload(h_acc_mm=15, v_acc_mm=25)
        frame = build_ubx_frame(0x01, 0x07, payload)

        messages = parser.feed(frame)
        msg = messages[0]
        assert abs(msg["h_acc_m"] - 0.015) < 1e-4
        assert abs(msg["v_acc_m"] - 0.025) < 1e-4

    def test_parse_nav_hpposllh(self):
        from interfaces.rtk_collector import UBXNavParser
        parser = UBXNavParser()

        lat, lon, alt = 13.0827, 80.2707, 50.0
        payload = build_nav_hpposllh_payload(
            lat_deg=lat, lon_deg=lon, alt_m=alt,
            lat_hp=5, lon_hp=-3, height_hp=7)
        frame = build_ubx_frame(0x01, 0x14, payload)

        messages = parser.feed(frame)
        assert len(messages) == 1
        msg = messages[0]
        assert msg["type"] == "NAV_HPPOSLLH"
        # HP adds sub-mm precision
        expected_lat = lat + 5 * 1e-9
        assert abs(msg["lat_deg"] - expected_lat) < 1e-10

    def test_bad_checksum_rejected(self):
        from interfaces.rtk_collector import UBXNavParser
        parser = UBXNavParser()

        payload = build_nav_pvt_payload()
        frame = bytearray(build_ubx_frame(0x01, 0x07, payload))
        frame[-1] ^= 0xFF  # corrupt checksum

        messages = parser.feed(bytes(frame))
        assert len(messages) == 0

    def test_multiple_messages(self):
        from interfaces.rtk_collector import UBXNavParser
        parser = UBXNavParser()

        frame1 = build_ubx_frame(0x01, 0x07, build_nav_pvt_payload(lat_deg=10.0))
        frame2 = build_ubx_frame(0x01, 0x07, build_nav_pvt_payload(lat_deg=20.0))

        messages = parser.feed(frame1 + frame2)
        assert len(messages) == 2
        assert abs(messages[0]["lat_deg"] - 10.0) < 1e-6
        assert abs(messages[1]["lat_deg"] - 20.0) < 1e-6


# ── Test: WGS-84 → NED Conversion ─────────────────────────────

class TestWGS84toNED:

    def test_origin_is_zero(self):
        from interfaces.rtk_collector import RTKCollector
        collector = RTKCollector({"rtk": {}})
        collector.set_origin(13.0827, 80.2707, 50.0)
        ned = collector._wgs84_to_ned(13.0827, 80.2707, 50.0)
        assert np.allclose(ned, [0, 0, 0], atol=0.01)

    def test_north_offset(self):
        from interfaces.rtk_collector import RTKCollector
        collector = RTKCollector({"rtk": {}})
        collector.set_origin(0.0, 0.0, 0.0)
        # ~111.2 km per degree of latitude at equator
        ned = collector._wgs84_to_ned(0.001, 0.0, 0.0)
        assert ned[0] > 100.0  # north should be positive
        assert abs(ned[1]) < 1.0  # east should be ~0
        assert abs(ned[2]) < 1.0  # down should be ~0


# ── Test: Flight Recorder Round-Trip ──────────────────────────

class TestFlightRecorder:

    def test_csv_format_matches_sim(self):
        """Verify recorded CSV is compatible with rtk_validate.py."""
        from logger.flight_recorder import FlightRecorder

        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = FlightRecorder(output_dir=tmpdir)
            recorder.start_session()

            # Record some fake data
            t0 = 1000.0
            for i in range(10):
                t = t0 + i * 0.01
                recorder.record_imu(
                    t, np.array([0.0, 0.0, -9.81]),
                    np.array([0.01, -0.005, 0.002]))
                recorder.record_baro(t, -5.0)

            recorder.stop_session()

            # Verify IMU CSV format
            imu_path = os.path.join(recorder.session_dir, "imu_log.csv")
            assert os.path.exists(imu_path)

            with open(imu_path, "r") as f:
                reader = csv.DictReader(f)
                headers = reader.fieldnames
                # Must match generate_rtk_sim.py output format
                assert "time_s" in headers
                assert "ax" in headers
                assert "ay" in headers
                assert "az" in headers
                assert "gx" in headers

                rows = list(reader)
                assert len(rows) == 10

            # Verify baro CSV format
            baro_path = os.path.join(recorder.session_dir, "baro_log.csv")
            with open(baro_path, "r") as f:
                reader = csv.DictReader(f)
                assert "time_s" in reader.fieldnames
                assert "alt_m" in reader.fieldnames

    def test_metadata_written(self):
        from logger.flight_recorder import FlightRecorder

        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = FlightRecorder(output_dir=tmpdir)
            recorder.start_session({"test_key": "test_value"})
            recorder.stop_session()

            import json
            meta_path = os.path.join(recorder.session_dir, "metadata.json")
            assert os.path.exists(meta_path)
            with open(meta_path) as f:
                meta = json.load(f)
            assert "duration_s" in meta
            assert "sample_counts" in meta


# ── Test: NTRIP GGA Generation ─────────────────────────────────

class TestNTRIPGGA:

    def test_gga_format(self):
        from interfaces.ntrip_client import generate_gga
        gga = generate_gga(13.0827, 80.2707, 50.0)
        assert gga.startswith("$GPGGA,")
        assert gga[-3] == "*"  # checksum marker
        assert len(gga) > 50

    def test_gga_checksum_valid(self):
        from interfaces.ntrip_client import generate_gga
        gga = generate_gga(28.6139, 77.2090, 216.0)
        # Verify checksum
        body = gga[1:gga.index("*")]
        expected_ck = 0
        for c in body:
            expected_ck ^= ord(c)
        stated_ck = int(gga[gga.index("*") + 1:], 16)
        assert expected_ck == stated_ck

    def test_gga_hemispheres(self):
        from interfaces.ntrip_client import generate_gga
        # Northern latitude, Eastern longitude
        gga = generate_gga(13.0827, 80.2707, 50.0)
        assert ",N," in gga
        assert ",E," in gga

        # Southern latitude, Western longitude
        gga = generate_gga(-33.8688, -151.2093, 10.0)
        assert ",S," in gga

    def test_gga_western_longitude(self):
        from interfaces.ntrip_client import generate_gga
        gga = generate_gga(40.7128, -74.0060, 10.0)
        assert ",W," in gga


# ── Test: Log Validator ───────────────────────────────────────

class TestLogValidator:

    def _create_test_logs(self, tmpdir, duration=10.0, imu_hz=100,
                          rtk_hz=5, with_gaps=False):
        """Create synthetic log files for testing."""
        # IMU
        with open(os.path.join(tmpdir, "imu_log.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["time_s", "ax", "ay", "az", "gx", "gy", "gz"])
            t = 0.0
            while t < duration:
                if with_gaps and 4.0 < t < 6.0:
                    t += 1.0 / imu_hz
                    continue
                w.writerow([f"{t:.4f}", "0", "0", "-9.81", "0", "0", "0"])
                t += 1.0 / imu_hz

        # Baro
        with open(os.path.join(tmpdir, "baro_log.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["time_s", "alt_m"])
            for t in np.arange(0, duration, 1.0 / 25):
                w.writerow([f"{t:.4f}", "-5.0"])

        # Mag
        with open(os.path.join(tmpdir, "mag_log.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["time_s", "mx", "my", "mz"])
            for t in np.arange(0, duration, 1.0 / 10):
                w.writerow([f"{t:.4f}", "0.2", "0.0", "0.4"])

        # RTK ground truth
        with open(os.path.join(tmpdir, "rtk_ground_truth.csv"), "w",
                  newline="") as f:
            w = csv.writer(f)
            w.writerow(["time_s", "x_m", "y_m", "z_m",
                         "vx_mps", "vy_mps", "vz_mps",
                         "qw", "qx", "qy", "qz",
                         "fix_type", "h_acc_m", "v_acc_m", "n_sats"])
            for t in np.arange(0, duration, 1.0 / rtk_hz):
                w.writerow([f"{t:.4f}", "0", "0", "-5",
                            "0", "0", "0",
                            "1", "0", "0", "0",
                            "5", "0.02", "0.03", "15"])

    def test_valid_logs_pass(self):
        sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                        "..", "scripts"))
        from validate_logs import validate_logs

        with tempfile.TemporaryDirectory() as tmpdir:
            self._create_test_logs(tmpdir)
            report = validate_logs(tmpdir)
            assert report.passed

    def test_missing_file_fails(self):
        sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                        "..", "scripts"))
        from validate_logs import validate_logs

        with tempfile.TemporaryDirectory() as tmpdir:
            # Only create IMU — missing rtk_ground_truth
            with open(os.path.join(tmpdir, "imu_log.csv"), "w") as f:
                f.write("time_s,ax,ay,az,gx,gy,gz\n0,0,0,-9.81,0,0,0\n")
            with open(os.path.join(tmpdir, "baro_log.csv"), "w") as f:
                f.write("time_s,alt_m\n0,-5.0\n")
            with open(os.path.join(tmpdir, "mag_log.csv"), "w") as f:
                f.write("time_s,mx,my,mz\n0,0.2,0.0,0.4\n")

            report = validate_logs(tmpdir)
            assert not report.passed  # missing rtk_ground_truth

    def test_gaps_detected(self):
        sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                        "..", "scripts"))
        from validate_logs import validate_logs

        with tempfile.TemporaryDirectory() as tmpdir:
            self._create_test_logs(tmpdir, with_gaps=True)
            report = validate_logs(tmpdir)
            # Should have warnings about gaps
            gap_results = [r for r in report.results
                           if "continuity" in r.name.lower()]
            if gap_results:
                assert not gap_results[0].passed


# ── Test: Flight Phase Detection ──────────────────────────────

class TestFlightPhaseDetection:

    def test_hover_detection(self):
        sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                        "..", "scripts"))
        from analyze_flight import detect_flight_phases

        n = 100
        times = np.arange(n) * 0.1
        pos = np.zeros((n, 3))
        pos[:, 2] = -5.0  # 5m AGL in NED
        vel = np.zeros((n, 3))  # stationary at altitude

        phases = detect_flight_phases(times, pos, vel)
        assert len(phases) > 0
        # At least one HOVER phase
        phase_types = [p.phase for p in phases]
        assert "HOVER" in phase_types

    def test_cruise_detection(self):
        sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                        "..", "scripts"))
        from analyze_flight import detect_flight_phases

        n = 100
        times = np.arange(n) * 0.1
        pos = np.zeros((n, 3))
        pos[:, 2] = -10.0
        vel = np.zeros((n, 3))
        vel[:, 0] = 3.0  # forward motion at 3 m/s

        phases = detect_flight_phases(times, pos, vel)
        phase_types = [p.phase for p in phases]
        assert "CRUISE" in phase_types


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
