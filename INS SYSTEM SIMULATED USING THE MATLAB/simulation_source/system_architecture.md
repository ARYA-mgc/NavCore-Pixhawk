# System Architecture — UAV INS (GPS-Denied)

## Overview

The system implements a **loosely-coupled INS** architecture where the EKF state estimator runs at full IMU rate, and aiding sensor corrections are applied asynchronously when new measurements arrive.

## Coordinate Frames

| Frame | Convention | Notes |
|-------|-----------|-------|
| NED | North-East-Down | Inertial reference frame |
| Body | X=fwd, Y=right, Z=down | Fixed to UAV |
| Navigation | Aligned with NED at init | Rotated by Euler angles |

## Rotation Convention

ZYX (aerospace): Yaw → Pitch → Roll applied in that order.

```
R_body_ned = Rz(ψ) · Ry(θ) · Rx(φ)
```

## State Propagation

### Mechanisation Equations (body → NED)
```
ṗ = v
v̇ = R_bn · f_b + g_ned
η̇ = T(η)⁻¹ · ω_b
```

Where:
- `f_b` = specific force in body frame (accelerometer measurement minus gravity in body)  
- `g_ned = [0, 0, 9.81]ᵀ` m/s² (NED, z-down)  
- `T(η)` = Euler angle rate transformation matrix  
- `ω_b` = angular rate in body frame (gyroscope measurement)

## EKF Linearisation

The Jacobian **F** is computed analytically at each step:

```
F = ∂f/∂x |_{x̂}

F(1:3, 4:6) = I₃ · dt          ← dp/dv
F(4:6, 7:9) = ∂(R_bn·f_b)/∂η  ← dv/dEuler (via skew-symmetric)
F(7:9, 7:9) = ∂(T⁻¹ω)/∂η     ← dη/dη (approximated)
```

## Aiding Sensor Models

### Barometer
```
z_baro = -p_z + ν_baro,   ν_baro ~ N(0, σ²_baro)
H_baro = [0,0,1, 0,0,0, 0,0,0]
```
(Negative because NED z is positive downward, altitude is positive upward)

### Magnetometer (yaw only)
```
z_mag = ψ + ν_mag,   ν_mag ~ N(0, σ²_mag)
H_mag = [0,0,0, 0,0,0, 0,0,1]
```

## Limitations

1. **No accelerometer bias estimation** in state vector — bias is assumed constant and absorbed into process noise Q.
2. **Euler singularity at pitch = ±90°** — for aggressive aerobatics, replace with quaternion representation.
3. **Flat-Earth assumption** — valid for local flights < 10 km radius.
4. **Magnetometer subject to hard/soft-iron distortion** — calibration required in practice.

## Future Improvements

- [ ] Quaternion-based attitude representation (avoids gimbal lock)
- [ ] Accelerometer and gyroscope bias states in EKF (15-state INS)
- [ ] UKF / particle filter comparison
- [ ] Vision-aided INS (optical flow integration)
- [ ] Hardware-in-the-loop testing with real IMU data
