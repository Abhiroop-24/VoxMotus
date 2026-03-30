"""Angle-feature training data collection for deterministic model training."""

from __future__ import annotations

import csv
import os
import time
from datetime import datetime
from typing import Dict, List

from joint_mapping import ANGLE_KEYS


class TrainingDataCollector:
    """Buffered CSV collector for labeled pose/angle training samples."""

    BASE_FIELDS = [
        "timestamp",
        "state",
        "activity",
        "camera",
        "pose_quality",
        "knee_angle",
        "elbow_angle",
        "torso_tilt",
        "body_line",
        "body_horizontal",
        "shoulder_hip_diff",
        "error_codes",
    ]

    def __init__(self, enabled: bool, output_dir: str = "outputs/datasets", flush_every: int = 250) -> None:
        self.enabled = bool(enabled)
        self.output_dir = output_dir
        self.flush_every = max(20, int(flush_every))
        self._rows: List[Dict] = []
        self._total_samples = 0
        self._file_path = ""
        self._header_written = False

        if self.enabled:
            os.makedirs(self.output_dir, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self._file_path = os.path.join(self.output_dir, f"training_angles_{stamp}.csv")

    @property
    def file_path(self) -> str:
        return self._file_path

    @property
    def sample_count(self) -> int:
        return self._total_samples

    @property
    def buffered_count(self) -> int:
        return len(self._rows)

    def add_sample(
        self,
        now_ts: float,
        state: str,
        activity: str,
        camera: str,
        pose_quality: float,
        metrics: Dict,
        joint_angles: Dict,
        error_codes: List[str],
    ) -> None:
        if not self.enabled:
            return

        row = {
            "timestamp": round(float(now_ts), 4),
            "state": str(state or ""),
            "activity": str(activity or ""),
            "camera": str(camera or ""),
            "pose_quality": round(float(pose_quality or 0.0), 5),
            "knee_angle": self._safe_float(metrics.get("knee_angle")),
            "elbow_angle": self._safe_float(metrics.get("elbow_angle")),
            "torso_tilt": self._safe_float(metrics.get("torso_tilt")),
            "body_line": self._safe_float(metrics.get("body_line")),
            "body_horizontal": int(bool(metrics.get("body_horizontal", False))),
            "shoulder_hip_diff": self._safe_float(metrics.get("shoulder_hip_diff")),
            "error_codes": "|".join(sorted(set([str(x) for x in error_codes]))),
        }

        for key in ANGLE_KEYS:
            row[key] = self._safe_float(joint_angles.get(key))

        self._rows.append(row)
        self._total_samples += 1
        if len(self._rows) >= self.flush_every:
            self.flush()

    def flush(self) -> None:
        if not self.enabled or not self._rows:
            return

        fieldnames = self.BASE_FIELDS + ANGLE_KEYS
        mode = "a" if self._header_written else "w"
        with open(self._file_path, mode, encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not self._header_written:
                writer.writeheader()
                self._header_written = True
            writer.writerows(self._rows)

        self._rows.clear()

    def close(self) -> None:
        self.flush()

    @staticmethod
    def _safe_float(value) -> float:
        try:
            return round(float(value), 5)
        except Exception:
            return float("nan")
