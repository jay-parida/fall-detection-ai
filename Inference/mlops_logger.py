import csv
import os
import threading
from datetime import datetime


class MLOpsLogger:
    def __init__(self, log_path):
        self.log_path = log_path
        self.lock = threading.Lock()

        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

        self.fieldnames = [
            "timestamp",
            "lighting",
            "brightness",
            "contrast",
            "blur_score",
            "status",
            "event_active",
            "confidence",
            "movement",
            "vertical",
            "direction",
            "ratio",
            "ratio_change",
            "recovery",
            "candidate",
            "score",
            "event_timer",
        ]

        self._create_file()

    def _create_file(self):
        if not os.path.exists(self.log_path):
            with open(
                self.log_path,
                "w",
                newline="",
                encoding="utf-8"
            ) as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=self.fieldnames
                )
                writer.writeheader()

    def log(self, status):
        row = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "lighting": status.get("lighting", "UNKNOWN"),
            "brightness": status.get("brightness", 0.0),
            "contrast": status.get("contrast", 0.0),
            "blur_score": status.get("blur_score", 0.0),
            "status": status.get("status", "UNKNOWN"),
            "event_active": status.get("event_active", False),
            "confidence": status.get("confidence", 0.0),
            "movement": status.get("movement", 0.0),
            "vertical": status.get("vertical", 0.0),
            "direction": status.get("direction", "UNKNOWN"),
            "ratio": status.get("ratio", 0.0),
            "ratio_change": status.get("ratio_change", 0.0),
            "recovery": status.get("recovery", "0/5"),
            "candidate": status.get("candidate", False),
            "score": status.get("score", 0),
            "event_timer": status.get("event_timer", 0.0),
        }

        with self.lock:
            with open(
                self.log_path,
                "a",
                newline="",
                encoding="utf-8"
            ) as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=self.fieldnames
                )
                writer.writerow(row)