"""Deterministic coaching messages driven by a structured feedback dataset."""

from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Optional


def _default_dataset() -> Dict:
    return {
        "state": {
            "WAIT_START_GESTURE": [
                "Welcome to ANTARDRISHTI. Show one palm to start, and two palms to stop."
            ],
            "WAIT_DIFFICULTY": [
                "Please choose your difficulty: one finger for easy, two for medium, three for tough."
            ],
            "READY": ["Difficulty confirmed. Hold steady, we will begin now."],
            "WORKOUT_SQUATS": ["Starting squats. Keep your chest up and move with control."],
            "WORKOUT_SITUPS": [
                "Starting sit-ups. Lie down, bend your knees, lift your upper body, then lower slowly."
            ],
            "WORKOUT_JUMPING_JACKS": [
                "Starting jumping jacks. Jump feet apart while raising both arms, then return to start."
            ],
            "BREAK": ["Great effort. Take a 10 second break and breathe."],
            "WORKOUT_PUSHUPS": ["Starting push-ups. Keep your body in one line."],
            "PAUSED": ["Paused. Hold position and breathe."],
            "RESUMED": ["Resuming. Stay controlled."],
            "END_COMPLETE": ["Session complete. Well done."],
            "END_STOPPED": ["Session stopped safely."],
        },
        "safety": {
            "obstacle": ["Stop. There is something ahead."],
            "camera_lost": ["I am adjusting my view. Hold on."],
        },
        "error": {
            "squat": {
                "back_tilt": {
                    "low": ["Keep your chest a little higher."],
                    "medium": ["Keep your back straight and chest up."],
                    "high": ["Stop and reset posture. Keep your back upright before continuing."],
                },
                "not_enough_depth": {
                    "low": ["Go a little deeper."],
                    "medium": ["Try bending a bit more for each squat."],
                    "high": ["Pause and reset. Reach a deeper squat before standing up."],
                },
            },
            "pushup": {
                "body_not_straight": {
                    "low": ["Keep your body straighter."],
                    "medium": ["Tighten your core and keep a straight line."],
                    "high": ["Pause and reset. Align shoulders, hips, and ankles in one line."],
                },
                "arms_not_bending_enough": {
                    "low": ["Bend your elbows a little more."],
                    "medium": ["Go lower with controlled elbow bend."],
                    "high": ["Pause and reset. Use full elbow bend before pushing up."],
                },
            },
        },
    }


class FeedbackLibrary:
    """Loads structured feedback and selects messages in deterministic round-robin order."""

    def __init__(self, dataset_path: str = "feedback_dataset.json") -> None:
        self.dataset_path = dataset_path
        self.data = self._load_dataset(dataset_path)
        self._cursor: Dict[str, int] = {}

    def _load_dataset(self, path: str) -> Dict:
        if not os.path.exists(path):
            return _default_dataset()

        try:
            with open(path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                return loaded
        except Exception:
            pass
        return _default_dataset()

    def _pick(self, namespace: str, choices: List[str], avoid: str = "") -> str:
        if not choices:
            return "Keep going."

        idx = self._cursor.get(namespace, 0) % len(choices)
        chosen = choices[idx]
        self._cursor[namespace] = idx + 1

        if avoid and chosen == avoid and len(choices) > 1:
            idx = self._cursor[namespace] % len(choices)
            chosen = choices[idx]
            self._cursor[namespace] = idx + 1
        return chosen

    def state_message(self, state_key: str, avoid: str = "") -> str:
        bucket = self.data.get("state", {})
        choices = bucket.get(state_key, [])
        return self._pick(f"state:{state_key}", choices, avoid=avoid)

    def safety_message(self, safety_key: str, avoid: str = "") -> str:
        bucket = self.data.get("safety", {})
        choices = bucket.get(safety_key, [])
        return self._pick(f"safety:{safety_key}", choices, avoid=avoid)

    def error_message(self, exercise: str, error_code: str, severity: str, avoid: str = "") -> str:
        sev = (severity or "medium").strip().lower()
        if sev not in ("low", "medium", "high"):
            sev = "medium"

        error_root = self.data.get("error", {})
        exercise_bucket = error_root.get(exercise, {}) if isinstance(error_root, dict) else {}
        code_bucket = exercise_bucket.get(error_code, {}) if isinstance(exercise_bucket, dict) else {}

        choices = []
        if isinstance(code_bucket, dict):
            choices = code_bucket.get(sev, [])
            if not choices:
                choices = code_bucket.get("medium", [])
        if not choices:
            choices = ["Adjust your form and continue with control."]

        return self._pick(f"error:{exercise}:{error_code}:{sev}", choices, avoid=avoid)


class CoachDecisionEngine:
    """Deterministic message policy with cooldown and anti-repeat controls."""

    def __init__(self, cooldown_s: float = 2.5, dataset_path: str = "feedback_dataset.json") -> None:
        self.cooldown_s = float(max(2.0, min(3.0, cooldown_s)))
        self.last_speak_ts = 0.0
        self.last_phrase: Optional[str] = None
        self.last_key: Optional[str] = None
        self.error_spoken_ts: Dict[str, float] = {}
        self.feedback = FeedbackLibrary(dataset_path=dataset_path)

    def _eligible(self, now_ts: float, key: str, force: bool = False, min_interval_s: float = 0.0) -> bool:
        if force:
            return True
        gate = max(self.cooldown_s, float(min_interval_s))
        if (now_ts - self.last_speak_ts) < gate:
            return False
        if self.last_key == key and (now_ts - self.last_speak_ts) < 5.0:
            return False
        return True

    def _remember(self, now_ts: float, key: str, phrase: str) -> str:
        self.last_speak_ts = now_ts
        self.last_key = key
        self.last_phrase = phrase
        return phrase

    def state_message(self, now_ts: float, state_key: str, force: bool = False) -> Optional[str]:
        key = f"state:{state_key}"
        if not self._eligible(now_ts, key, force=force):
            return None
        phrase = self.feedback.state_message(state_key, avoid=self.last_phrase or "")
        return self._remember(now_ts, key, phrase)

    def safety_message(self, now_ts: float, safety_key: str, force: bool = True) -> Optional[str]:
        key = f"safety:{safety_key}"
        if not self._eligible(now_ts, key, force=force):
            return None
        phrase = self.feedback.safety_message(safety_key, avoid=self.last_phrase or "")
        return self._remember(now_ts, key, phrase)

    def error_message(
        self,
        now_ts: float,
        exercise: str,
        error_code: str,
        severity: str,
        min_error_interval_s: float = 2.8,
    ) -> Optional[str]:
        scoped_key = f"error:{exercise}:{error_code}:{severity}"
        last = self.error_spoken_ts.get(scoped_key, 0.0)
        if (now_ts - last) < float(min_error_interval_s):
            return None
        if not self._eligible(now_ts, scoped_key, force=False):
            return None

        phrase = self.feedback.error_message(
            exercise=exercise,
            error_code=error_code,
            severity=severity,
            avoid=self.last_phrase or "",
        )
        self.error_spoken_ts[scoped_key] = now_ts
        return self._remember(now_ts, scoped_key, phrase)

    def end_message(self, now_ts: float, stopped_by_user: bool) -> Optional[str]:
        key = "END_STOPPED" if stopped_by_user else "END_COMPLETE"
        return self.state_message(now_ts=now_ts, state_key=key, force=True)

    def reset_memory(self) -> None:
        self.last_speak_ts = 0.0
        self.last_phrase = None
        self.last_key = None
        self.error_spoken_ts.clear()
