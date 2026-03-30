"""Exercise rep counting and progression logic."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import time

import numpy as np


DIFFICULTY_PRESETS = {
    "easy": {
        "squat_target": 8,
        "situp_target": 8,
        "jumping_jack_target": 12,
        "pushup_target": 6,
        "strictness": 0.9,
        "speed_tolerance": 1.3,
    },
    "medium": {
        "squat_target": 12,
        "situp_target": 12,
        "jumping_jack_target": 20,
        "pushup_target": 10,
        "strictness": 1.0,
        "speed_tolerance": 1.0,
    },
    "tough": {
        "squat_target": 16,
        "situp_target": 16,
        "jumping_jack_target": 28,
        "pushup_target": 14,
        "strictness": 1.1,
        "speed_tolerance": 0.8,
    },
}


@dataclass
class RepCounter:
    name: str
    down_threshold: float
    up_threshold: float
    min_down_time: float = 0.35
    min_rep_time: float = 1.0
    rep_count: int = 0
    stage: str = "up"
    last_stage_ts: float = 0.0
    last_rep_ts: float = 0.0

    def update(self, value: Optional[float]) -> Dict:
        now = time.time()
        event = None

        if value is None:
            return {"rep_count": self.rep_count, "stage": self.stage, "event": event}

        val = float(value)

        # Range guard prevents stage chatter when up/down thresholds get too close.
        threshold_gap = max(6.0, float(self.up_threshold - self.down_threshold))
        soft_reset_level = self.down_threshold + 0.45 * threshold_gap

        if self.stage == "up" and val < self.down_threshold:
            self.stage = "down"
            self.last_stage_ts = now

        elif self.stage == "down":
            down_elapsed = now - self.last_stage_ts
            if val < self.down_threshold:
                if down_elapsed >= self.min_down_time:
                    self.stage = "hold"
                    self.last_stage_ts = now
            elif val > self.up_threshold:
                if down_elapsed >= self.min_down_time and (now - self.last_rep_ts) >= self.min_rep_time:
                    self.rep_count += 1
                    self.last_rep_ts = now
                    event = "rep_completed"
                self.stage = "up"
                self.last_stage_ts = now

        elif self.stage == "hold":
            if val > self.up_threshold and (now - self.last_rep_ts) >= self.min_rep_time:
                self.stage = "up"
                self.rep_count += 1
                self.last_rep_ts = now
                event = "rep_completed"
            elif val >= soft_reset_level:
                # If user returns upward without crossing strict up threshold, reset stage safely.
                self.stage = "up"
                self.last_stage_ts = now

        return {"rep_count": self.rep_count, "stage": self.stage, "event": event}


@dataclass
class MotionCycleCounter:
    """Counts full closed->open->closed cycles from a scalar signal."""

    name: str
    closed_threshold: float
    open_threshold: float
    open_is_high: bool = True
    min_open_time: float = 0.18
    min_rep_time: float = 0.7
    rep_count: int = 0
    stage: str = "closed"
    last_stage_ts: float = 0.0
    last_rep_ts: float = 0.0

    def _is_open(self, value: float) -> bool:
        if self.open_is_high:
            return value >= self.open_threshold
        return value <= self.open_threshold

    def _is_closed(self, value: float) -> bool:
        if self.open_is_high:
            return value <= self.closed_threshold
        return value >= self.closed_threshold

    def update(self, value: Optional[float]) -> Dict:
        now = time.time()
        event = None

        if value is None:
            return {"rep_count": self.rep_count, "stage": self.stage, "event": event}

        val = float(value)
        is_open = self._is_open(val)
        is_closed = self._is_closed(val)

        if self.stage == "closed" and is_open:
            self.stage = "open"
            self.last_stage_ts = now

        elif self.stage == "open" and is_closed:
            held_open = (now - self.last_stage_ts) >= self.min_open_time
            enough_rep_gap = (now - self.last_rep_ts) >= self.min_rep_time
            if held_open and enough_rep_gap:
                self.rep_count += 1
                self.last_rep_ts = now
                event = "rep_completed"
            self.stage = "closed"
            self.last_stage_ts = now

        return {"rep_count": self.rep_count, "stage": self.stage, "event": event}


class ExerciseEngine:
    def __init__(self, difficulty: str = "medium") -> None:
        self.set_difficulty(difficulty)
        self.squat_counter = RepCounter("squat", down_threshold=110, up_threshold=155)
        self.pushup_counter = RepCounter("pushup", down_threshold=95, up_threshold=145, min_down_time=0.25)
        self.situp_counter = MotionCycleCounter(
            name="situp",
              closed_threshold=14.0,
              open_threshold=40.0,
            open_is_high=True,
              min_open_time=0.18,
              min_rep_time=0.72,
        )
        self.jumping_jack_counter = MotionCycleCounter(
            name="jumping_jack",
              closed_threshold=0.38,
              open_threshold=0.58,
            open_is_high=True,
              min_open_time=0.1,
              min_rep_time=0.42,
        )
        self.squat_calibrated = False
        self._squat_standing_samples = []
        self._squat_depth_samples = []

    def set_difficulty(self, difficulty: str) -> None:
        self.difficulty = difficulty if difficulty in DIFFICULTY_PRESETS else "medium"
        self.settings = DIFFICULTY_PRESETS[self.difficulty]

    def update(self, activity: str, metrics: Dict) -> Dict:
        if activity == "squat":
            result = self.squat_counter.update(metrics.get("knee_angle"))
            target = self.settings["squat_target"]
            return {"exercise": "squat", "target": target, **result}

        if activity == "pushup":
            # Elbow angle is a robust down/up proxy for push-up repetition.
            result = self.pushup_counter.update(metrics.get("elbow_angle"))
            target = self.settings["pushup_target"]
            return {"exercise": "pushup", "target": target, **result}

        if activity == "situp":
            result = self.situp_counter.update(metrics.get("situp_signal"))
            target = self.settings["situp_target"]
            return {"exercise": "situp", "target": target, **result}

        if activity == "jumping_jack":
            result = self.jumping_jack_counter.update(metrics.get("jumping_jack_open_score"))
            target = self.settings["jumping_jack_target"]
            return {"exercise": "jumping_jack", "target": target, **result}

        return {
            "exercise": "none",
            "rep_count": 0,
            "target": 0,
            "stage": "none",
            "event": None,
        }

    def target_for(self, exercise_name: str) -> int:
        if exercise_name == "squat":
            return self.settings["squat_target"]
        if exercise_name == "situp":
            return self.settings["situp_target"]
        if exercise_name == "jumping_jack":
            return self.settings["jumping_jack_target"]
        if exercise_name == "pushup":
            return self.settings["pushup_target"]
        return 0

    def calibrate_squat(self, activity: str, knee_angle: Optional[float]) -> None:
        """Online calibration for squat thresholds based on live user motion."""
        if knee_angle is None:
            return

        value = float(np.clip(knee_angle, 60.0, 190.0))

        if activity in ("standing", "transition", "idle") and len(self._squat_standing_samples) < 80:
            self._squat_standing_samples.append(value)

        if activity in ("squat", "transition") and len(self._squat_depth_samples) < 80:
            self._squat_depth_samples.append(value)

        if len(self._squat_standing_samples) < 15 or len(self._squat_depth_samples) < 15:
            return

        standing_ref = float(np.percentile(self._squat_standing_samples, 75))
        depth_ref = float(np.percentile(self._squat_depth_samples, 20))
        motion_range = standing_ref - depth_ref
        if motion_range < 18.0:
            return

        down = float(np.clip(depth_ref + 8.0, 90.0, 135.0))
        up = float(np.clip(standing_ref - 8.0, down + 22.0, 175.0))
        self.squat_counter.down_threshold = down
        self.squat_counter.up_threshold = up
        self.squat_calibrated = True

    def calibration_status(self) -> Dict:
        return {
            "squat_calibrated": self.squat_calibrated,
            "squat_down_threshold": round(self.squat_counter.down_threshold, 2),
            "squat_up_threshold": round(self.squat_counter.up_threshold, 2),
            "standing_samples": len(self._squat_standing_samples),
            "depth_samples": len(self._squat_depth_samples),
        }

    def reset(self) -> None:
        self.squat_counter.rep_count = 0
        self.squat_counter.stage = "up"
        self.squat_counter.last_stage_ts = 0.0
        self.squat_counter.last_rep_ts = 0.0
        self.pushup_counter.rep_count = 0
        self.pushup_counter.stage = "up"
        self.pushup_counter.last_stage_ts = 0.0
        self.pushup_counter.last_rep_ts = 0.0
        self.situp_counter.rep_count = 0
        self.situp_counter.stage = "closed"
        self.situp_counter.last_stage_ts = 0.0
        self.situp_counter.last_rep_ts = 0.0
        self.jumping_jack_counter.rep_count = 0
        self.jumping_jack_counter.stage = "closed"
        self.jumping_jack_counter.last_stage_ts = 0.0
        self.jumping_jack_counter.last_rep_ts = 0.0
        self.squat_calibrated = False
        self._squat_standing_samples.clear()
        self._squat_depth_samples.clear()
