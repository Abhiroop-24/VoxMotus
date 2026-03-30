"""Reference comparison and performance scoring for PS4 objective alignment."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional

import numpy as np


REFERENCE_CURVES = {
    "squat": np.concatenate([np.linspace(170, 95, 12), np.linspace(95, 170, 12)]),
    "pushup": np.concatenate([np.linspace(165, 75, 10), np.linspace(75, 165, 10)]),
}


@dataclass
class ScoringState:
    rep_scores: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))
    error_tally: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    frame_quality: Deque[float] = field(default_factory=lambda: deque(maxlen=160))
    motion_deltas: Deque[float] = field(default_factory=lambda: deque(maxlen=160))
    latest_rep_score: float = 0.0


class ReferenceComparator:
    """Compares movement against light reference templates and outputs score 0-100."""

    def __init__(self) -> None:
        self.state = ScoringState()
        self.samples: Dict[str, List[float]] = {"squat": [], "pushup": []}
        self.last_signal = {"squat": None, "pushup": None}

    def reset(self) -> None:
        self.state = ScoringState()
        self.samples = {"squat": [], "pushup": []}
        self.last_signal = {"squat": None, "pushup": None}

    def _resample(self, signal: List[float], target_len: int) -> np.ndarray:
        if len(signal) < 2:
            return np.zeros(target_len, dtype=np.float32)
        x_old = np.linspace(0.0, 1.0, num=len(signal))
        x_new = np.linspace(0.0, 1.0, num=target_len)
        return np.interp(x_new, x_old, np.array(signal, dtype=np.float32))

    def _rep_score(self, exercise: str, signal: List[float], error_count: int) -> float:
        if len(signal) < 6:
            return 55.0

        reference = REFERENCE_CURVES[exercise]
        aligned = self._resample(signal, len(reference))
        mae = float(np.mean(np.abs(aligned - reference)))

        smoothness = float(np.mean(np.abs(np.diff(aligned))))
        smooth_bonus = max(0.0, 12.0 - smoothness * 3.0)

        base = 100.0 - mae * 0.9
        score = base + smooth_bonus - (error_count * 4.5)
        return float(np.clip(score, 0.0, 100.0))

    def _instant_quality(self, activity: str, metrics: Dict, errors: List[str]) -> float:
        if not metrics:
            return 0.0

        quality = 85.0
        if activity == "squat":
            quality -= max(0.0, abs(metrics.get("knee_angle", 170.0) - 100.0) - 28.0) * 0.9
            quality -= max(0.0, metrics.get("torso_tilt", 0.0) - 20.0) * 1.4
        elif activity == "pushup":
            quality -= max(0.0, abs(metrics.get("elbow_angle", 160.0) - 85.0) - 24.0) * 0.9
            quality -= max(0.0, 165.0 - metrics.get("body_line", 165.0)) * 1.0
        quality -= len(errors) * 5.5
        return float(np.clip(quality, 0.0, 100.0))

    def update(
        self,
        exercise: str,
        activity: str,
        metrics: Dict,
        rep_event: Optional[str],
        errors: List[str],
        rep_count: int,
        target: int,
    ) -> Dict:
        for err in errors:
            self.state.error_tally[err] += 1

        signal_value = None
        if exercise == "squat":
            signal_value = metrics.get("knee_angle")
        elif exercise == "pushup":
            signal_value = metrics.get("elbow_angle")

        if signal_value is not None and exercise in self.samples:
            self.samples[exercise].append(float(signal_value))

            prev = self.last_signal.get(exercise)
            if prev is not None:
                self.state.motion_deltas.append(abs(float(signal_value) - float(prev)))
            self.last_signal[exercise] = float(signal_value)

        instant_quality = self._instant_quality(activity, metrics, errors)
        self.state.frame_quality.append(instant_quality)

        if rep_event == "rep_completed" and exercise in self.samples:
            rep_errors = len(errors)
            rep_score = self._rep_score(exercise, self.samples[exercise], rep_errors)
            self.state.rep_scores[exercise].append(rep_score)
            self.state.latest_rep_score = rep_score
            self.samples[exercise].clear()

        all_rep_scores = []
        for values in self.state.rep_scores.values():
            all_rep_scores.extend(values)

        quality_score = float(np.mean(all_rep_scores)) if all_rep_scores else float(np.mean(self.state.frame_quality) if self.state.frame_quality else 0.0)

        if len(self.state.motion_deltas) > 5:
            motion_std = float(np.std(np.array(self.state.motion_deltas, dtype=np.float32)))
            coordination_score = float(np.clip(100.0 - (motion_std * 3.2), 0.0, 100.0))
        else:
            coordination_score = 65.0

        performance_score = 0.0
        if target > 0:
            performance_score = float(np.clip((rep_count / target) * 100.0, 0.0, 100.0))

        total_errors = sum(self.state.error_tally.values())
        penalty = min(18.0, total_errors * 0.45)

        skill_score = (
            (quality_score * 0.52)
            + (coordination_score * 0.22)
            + (performance_score * 0.26)
            - penalty
        )
        skill_score = float(np.clip(skill_score, 0.0, 100.0))

        return {
            "skill_score": round(skill_score, 2),
            "quality_score": round(quality_score, 2),
            "coordination_score": round(coordination_score, 2),
            "performance_score": round(performance_score, 2),
            "latest_rep_score": round(self.state.latest_rep_score, 2),
            "reference_gap": round(max(0.0, 100.0 - quality_score), 2),
            "rep_scores": {k: [round(v, 2) for v in vals[-5:]] for k, vals in self.state.rep_scores.items()},
            "error_tally": dict(self.state.error_tally),
        }
