# NavCore-Pixhawk Architecture: State Vector & Frames

This document serves as the single source of truth for the Square-Root Error-State Kalman Filter (SR-ESKF) state layout, units, and coordinate frames.

## 1. Coordinate Frames

### Global Frame (NED)
- **Definition:** North-East-Down (Local Tangent Plane).
- **Origin:** Set automatically upon acquiring the first valid 3D GPS fix.
- **Axes:**
  - `X`: True North (meters)
  - `Y`: True East (meters)
  - `Z`: Down (towards Earth's center, meters) - *Note: Altitude above ground is therefore a negative Z value.*

### Body Frame (FRD)
- **Definition:** Forward-Right-Down.
- **Axes:**
  - `X`: Forward (nose of the aircraft)
  - `Y`: Right (starboard wing)
  - `Z`: Down (belly of the aircraft)

---

## 2. Nominal State Vector (21 States)

The nominal state $\mathbf{x}$ represents the full filter estimate.

| Index | Symbol | Description | Units | Frame |
|---|---|---|---|---|
| `0:2` | $\mathbf{p}$ | Position (North, East, Down) | $m$ | NED |
| `3:5` | $\mathbf{v}$ | Velocity (North, East, Down) | $m/s$ | NED |
| `6:9` | $\mathbf{q}$ | Attitude Quaternion $(q_w, q_x, q_y, q_z)$ | unitless | Body $\rightarrow$ NED |
| `10:12` | $\mathbf{b}_a$ | Accelerometer Bias | $m/s^2$ | Body |
| `13:15` | $\mathbf{b}_g$ | Gyroscope Bias | $rad/s$ | Body |
| `16` | $b_{baro}$ | Barometer Altitude Bias | $m$ | NED (Z) |
| `17` | $b_{clk}$ | GPS Receiver Clock Bias | $m$ | - |
| `18` | $d_{clk}$ | GPS Receiver Clock Drift | $m/s$ | - |
| `19:20` | $\mathbf{w}$ | Wind Velocity (North, East) | $m/s$ | NED |

---

## 3. Error State Vector (20 States)

The error state $\delta\mathbf{x}$ tracks the small perturbations estimated by the Kalman Filter before they are injected into the nominal state.

| Index | Symbol | Description | Units | Frame |
|---|---|---|---|---|
| `0:2` | $\delta\mathbf{p}$ | Position Error | $m$ | NED |
| `3:5` | $\delta\mathbf{v}$ | Velocity Error | $m/s$ | NED |
| `6:8` | $\delta\mathbf{\theta}$ | Attitude Error (Roll, Pitch, Yaw) | $rad$ | Body |
| `9:11` | $\delta\mathbf{b}_a$ | Accelerometer Bias Error | $m/s^2$ | Body |
| `12:14` | $\delta\mathbf{b}_g$ | Gyroscope Bias Error | $rad/s$ | Body |
| `15` | $\delta b_{baro}$ | Barometer Bias Error | $m$ | NED (Z) |
| `16` | $\delta b_{clk}$ | Clock Bias Error | $m$ | - |
| `17` | $\delta d_{clk}$ | Clock Drift Error | $m/s$ | - |
| `18:19` | $\delta\mathbf{w}$ | Wind Velocity Error | $m/s$ | NED |

### Attitude Error Injection Mapping
The 3-element attitude error $\delta\mathbf{\theta}$ is injected into the 4-element quaternion $\mathbf{q}$ using the small-angle approximation:
$\delta\mathbf{q} = \begin{bmatrix} 1 \\ \delta\mathbf{\theta}/2 \end{bmatrix}$
$\mathbf{q}^+ = \mathbf{q}^- \otimes \delta\mathbf{q}$
