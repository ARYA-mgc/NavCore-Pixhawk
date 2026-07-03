# NavCore-Pixhawk: State Vector & Frame Reference

This document serves as the single source of truth for the mathematics and coordinate frames used within the `NavCore-Pixhawk` INS system.

## Coordinate Frames

### World Frame (NED)
The primary navigational frame is **North-East-Down (NED)**.
*   **X-axis:** Points to True North
*   **Y-axis:** Points to East
*   **Z-axis:** Points Down (parallel to gravity)
*   **Origin:** The position where the filter initializes, or the first valid GPS lock.

### Body Frame (FRD)
The vehicle's local frame is **Forward-Right-Down (FRD)**.
*   **X-axis:** Points forward through the nose of the vehicle.
*   **Y-axis:** Points to the right wing.
*   **Z-axis:** Points down (towards the ground when level).

### IMU/Sensor Frames
By default, all raw IMU data from MAVLink (`RAW_IMU`, `SCALED_IMU`) is expected to be translated by ArduPilot into the **FRD Body Frame**. The filter assumes sensors are aligned to FRD.

## State Vector (21 States)

The core `ESKF` tracks a 21-element state vector (`x`), structured as follows:

| Index | Name | Description | Units | Frame |
| :--- | :--- | :--- | :--- | :--- |
| `0..2` | `POS` | Position (North, East, Down) | meters (m) | NED |
| `3..5` | `VEL` | Velocity (North, East, Down) | meters/second (m/s) | NED |
| `6..9` | `QUAT` | Attitude Quaternion $(q_w, q_x, q_y, q_z)$ | unitless | Body → NED |
| `10..12`| `ABIAS` | Accelerometer Bias | $m/s^2$ | Body |
| `13..15`| `GBIAS` | Gyroscope Bias | $rad/s$ | Body |
| `16` | `BARO_BIAS` | Barometer Altitude Bias | meters (m) | - |
| `17` | `CLK_BIAS` | GNSS Receiver Clock Bias | meters (m) | - |
| `18` | `CLK_DRIFT` | GNSS Receiver Clock Drift | meters/second (m/s) | - |
| `19..20`| `WIND` | Wind Velocity (North, East) | meters/second (m/s) | NED |

### Error State ($\delta x$)
The Kalman filter update step computes an error state ($\delta x$) of dimension 20. The attitude quaternion error is represented as a 3-DOF rotation vector (Euler angles).

| Index | Name | Description |
| :--- | :--- | :--- |
| `0..2` | `E_POS` | Position Error |
| `3..5` | `E_VEL` | Velocity Error |
| `6..8` | `E_ATT` | Attitude Error (Roll, Pitch, Yaw) |
| `9..11` | `E_ABIAS`| Accel Bias Error |
| `12..14`| `E_GBIAS`| Gyro Bias Error |
| `15` | `E_BARO` | Baro Bias Error |
| `16` | `E_CLK_B`| Clock Bias Error |
| `17` | `E_CLK_D`| Clock Drift Error |
| `18..19`| `E_WIND` | Wind Velocity Error |

## Sensor Integrations & Conventions

*   **GPS:** Position uses WGS-84 coordinates converted to local NED Cartesian coordinates via `pymap3d`.
*   **Barometer:** Provides altitude (-Z). The filter state Z is positive-down, so barometric altitude is negated to match NED.
*   **Magnetometer:** Calibrated magnetometer data (body frame) is used to observe yaw.
*   **Lidar Rangefinder:** Measures distance along the Body-Z axis. The filter projects this to the NED-Z axis using the current pitch and roll estimates to prevent bank-angle-induced altitude jumps.
*   **Optical Flow:** Measures angular rates and velocities in the Body XY plane. Internally mapped to NED velocities using the current attitude and rangefinder distance.

## Covariance Matrix (P)

The covariance matrix `P` is a $20 \times 20$ matrix corresponding to the uncertainties of the error states.
In the Square Root form of the ESKF, `P` is factorized and maintained as an upper triangular matrix `U` and a lower triangular matrix `S` where $P = S S^T$.

## System Math 

*   **Earth Gravity:** $g = 9.80665 \ m/s^2$
*   **Attitude Updates:** The quaternion is updated via RK4 integration of the body rates.
*   **Measurement Updates:** The filter uses Joseph-form (or sequential Potter's) updates to preserve covariance symmetry and positive-definiteness.
