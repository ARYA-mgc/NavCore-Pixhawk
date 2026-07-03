#!/usr/bin/env python3
# Machine learning anomaly detection.
# Flags unusual sensor/filter behaviour so the fault manager can react.

import numpy as np
import logging
from sklearn.ensemble import IsolationForest

log = logging.getLogger("ml_predictive")


class MLAnomalyDetector:
    # Lightweight anomaly detector.
    def __init__(self, contamination=0.01):
        # Initialize IsolationForest.
        self.clf = IsolationForest(
            n_estimators=50,
            contamination=contamination,
            max_samples=256,
            random_state=42,
        )
        self.is_trained = False
        self.training_buffer = []
        self.TRAIN_SAMPLES = 500  # About 5-10 seconds of nominal flight to baseline

    def check_health(
        self,
        accel_var: float,
        gyro_var: float,
        p_trace: float,
        vel_var: float = 0.0,
        gps_sats: int = 10,
        gps_hdop: float = 1.0,
        gps_vdop: float = 1.0,
    ) -> bool:
        # Evaluate current state for anomalies (including GPS spoofing detection features).
        features = np.array(
            [
                [
                    accel_var,
                    gyro_var,
                    p_trace,
                    vel_var,
                    float(gps_sats),
                    gps_hdop,
                    gps_vdop,
                ]
            ]
        )

        if not self.is_trained:
            self.training_buffer.append(features[0])
            if len(self.training_buffer) >= self.TRAIN_SAMPLES:
                log.info("ML Predictor: Bootstrapping baseline...")
                self.clf.fit(np.array(self.training_buffer))
                self.is_trained = True
                log.info("ML Predictor: Online and watching.")
            return False  # Return nominal during baseline phase.

        # 1 = Normal, -1 = Anomaly (Anomaly detected)
        prediction = self.clf.predict(features)[0]
        if prediction == -1:
            log.critical(
                f"ML PREDICTION: Structural/Sensor anomaly! [a={accel_var:.2f}, g={gyro_var:.2f}, P={p_trace:.1f}, "
                f"v_var={vel_var:.2f}, sats={gps_sats}, hdop={gps_hdop:.1f}, vdop={gps_vdop:.1f}]"
            )
            return True

        return False
