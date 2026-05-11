# Writing JSON logs for the fancy dashboards.

import os
import json
import logging
import numpy as np
from datetime import datetime

log = logging.getLogger("structured_logger")


class NumpyEncoder(json.JSONEncoder):
    """Custom encoder for numpy data types."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        if hasattr(obj, 'name'):  # For Enums
            return obj.name
        return super(NumpyEncoder, self).default(obj)


class StructuredLogger:
    def __init__(self, log_dir: str = "logs"):
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filepath = os.path.join(log_dir, f"ins_structured_{timestamp}.jsonl")
        
        try:
            self._file = open(self.filepath, "w")
            log.info(f"Structured logging started: {self.filepath}")
        except Exception as e:
            log.error(f"Failed to open structured log: {e}")
            self._file = None

    def log_state(self, t: float, state: dict, covariance: np.ndarray, 
                  health_status: str, safety_action: str, timing_ms: float):
        """
        Log the full estimator state, covariance metrics, and health.
        """
        if self._file is None:
            return
            
        record = {
            "t": t,
            "type": "STATE",
            "health": health_status,
            "safety": safety_action,
            "dt_ms": timing_ms,
            "state": {
                "pos": state["pos"],
                "vel": state["vel"],
                "euler": state["euler"],
                "quat": state["quat"],
                "bias_a": state["accel_bias"],
                "bias_g": state["gyro_bias"]
            },
            "cov": {
                "trace": float(np.trace(covariance)),
                "diag": np.diag(covariance),
                "cond": float(np.linalg.cond(covariance)) if covariance.size > 0 else 0.0
            }
        }
        self._write_record(record)

    def log_innovation(self, t: float, sensor: str, nis: float, 
                       y: np.ndarray, S: np.ndarray, rejected: bool):
        """
        Log measurement innovations for offline tuning (Allan variance/RMSE).
        """
        if self._file is None:
            return
            
        record = {
            "t": t,
            "type": "INNOVATION",
            "sensor": sensor,
            "nis": float(nis),
            "rejected": rejected,
            "y": y,
            "S_diag": np.diag(S)
        }
        self._write_record(record)

    def _write_record(self, record: dict):
        if self._file is not None:
            try:
                json_str = json.dumps(record, cls=NumpyEncoder)
                self._file.write(json_str + "\n")
            except Exception as e:
                log.debug(f"Failed to write structured log record: {e}")

    def close(self):
        if self._file is not None:
            self._file.flush()
            self._file.close()
            log.info(f"Structured log closed: {self.filepath}")
