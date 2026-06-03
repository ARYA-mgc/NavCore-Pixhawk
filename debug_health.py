import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
import numpy as np
from core.eskf import ESKF
from utils.noise import IMUNoiseParams

noise = IMUNoiseParams()
eskf = ESKF(noise)
eskf.x[6:10] = eskf._euler_to_quat(0, 0, 0)
eskf._initialized = True

accel = np.array([0.0, 0.0, -9.80665])
rng = np.random.default_rng(42)
for i in range(300):
    eskf.predict(accel, np.zeros(3), 0.01)
    if i % 10 == 0:
        eskf.update_baro(rng.normal(0, 0.3))
    if i % 2 == 0:
        eskf.update_mag(rng.normal(0, 0.02))

print(f"Health: {eskf.health}")
print(f"z_cov: {eskf.P[2, 2]}")
