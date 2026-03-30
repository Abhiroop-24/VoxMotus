"""Short-term coaching memory to adapt feedback style."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Optional


@dataclass
class CoachMemory:
    last_error: Optional[str] = None
    last_feedback: Optional[str] = None
    error_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    recent_rep_events: Deque[int] = field(default_factory=lambda: deque(maxlen=10))

    def register_error(self, err: str) -> None:
        self.last_error = err
        self.error_counts[err] += 1

    def register_feedback(self, message: str) -> None:
        self.last_feedback = message

    def register_rep_event(self, completed: bool) -> None:
        self.recent_rep_events.append(1 if completed else 0)

    def repeat_count(self, err: str) -> int:
        return self.error_counts.get(err, 0)

    def improvement_trend(self) -> str:
        if len(self.recent_rep_events) < 5:
            return "stable"
        score = sum(self.recent_rep_events) / len(self.recent_rep_events)
        if score > 0.7:
            return "improving"
        if score < 0.35:
            return "struggling"
        return "stable"
