import os
import sys

# Add the project root to Python's import path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from Inference.mlops_logger import MLOpsLogger


LOG_PATH = os.path.join(
    PROJECT_ROOT,
    "Data",
    "mlops_logs.csv"
)


logger = MLOpsLogger(LOG_PATH)


test_status = {
    "lighting": "NORMAL",
    "brightness": 142.43,
    "contrast": 53.47,
    "blur_score": 419.35,
    "status": "MONITORING",
    "event_active": False,
    "confidence": 0.71,
    "movement": 6.2,
    "vertical": 1.5,
    "direction": "STABLE",
    "ratio": 1.48,
    "ratio_change": -0.03,
    "recovery": "0/5",
    "candidate": False,
    "score": 0,
    "event_timer": 0.0,
}


logger.log(test_status)


print("MLOps logger test successful.")
print(f"Log file: {LOG_PATH}")