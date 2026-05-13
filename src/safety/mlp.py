#!/usr/bin/env python3
# The psychic brain. Predicts crashes before the ESKF even knows it's sick.
# Uses a lightweight IsolationForest to flag anomalies in vibration and covariance.

import numpy as np
import logging
from sklearn.ensemble import IsolationForest

log = logging.getLogger("ml_predictive")

class MLAnomalyDetector:
    # fast, embedded-friendly anomaly detection
    def __init__(self, contamination=0.01):
        # 50 trees is enough for a Pi to handle without crying
        self.clf = IsolationForest(
            n_estimators=50,
            contamination=contamination,
            max_samples=256,
            random_state=42
        )
        self.is_trained = False
        self.training_buffer = []
        self.TRAIN_SAMPLES = 500   # About 5-10 seconds of nominal flight to baseline
        
    def check_health(self, accel_var: float, gyro_var: float, p_trace: float) -> bool:
        # Feed the beast. Returns True if we're about to crash.
        features = np.array([[accel_var, gyro_var, p_trace]])
        
        if not self.is_trained:
            self.training_buffer.append(features[0])
            if len(self.training_buffer) >= self.TRAIN_SAMPLES:
                log.info("ML Predictor: Bootstrapping baseline...")
                self.clf.fit(np.array(self.training_buffer))
                self.is_trained = True
                log.info("ML Predictor: Online and watching.")
            return False  # assume good while learning
            
        # 1 = Normal, -1 = Anomaly (Impending doom)
        prediction = self.clf.predict(features)[0]
        if prediction == -1:
            log.critical(f"ML PREDICTION: Structural/Sensor anomaly! [a={accel_var:.2f}, g={gyro_var:.2f}, P={p_trace:.1f}]")
            return True
            
        return False
