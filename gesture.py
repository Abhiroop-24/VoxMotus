"""Bi-manual gesture detection with robust role mapping and temporal confirmation."""

from __future__ import annotations

import math
import time
from collections import deque
from typing import Deque, Dict, Optional, Tuple

import cv2
import mediapipe as mp


class GestureDetector:
    """Maps hand landmarks to deterministic start/stop and difficulty gestures."""

    def __init__(self) -> None:
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )
        self.drawer = mp.solutions.drawing_utils
        self.wrist_history: Dict[str, Deque[Tuple[float, float]]] = {
            "Left": deque(maxlen=22),
            "Right": deque(maxlen=22),
        }
        self._candidate: Dict[str, Optional[str]] = {"control": None, "selection": None}
        self._candidate_count: Dict[str, int] = {"control": 0, "selection": 0}
        self._candidate_since: Dict[str, float] = {"control": 0.0, "selection": 0.0}
        self._lock_until: Dict[str, float] = {"control": 0.0, "selection": 0.0}

        self.min_hold_s = 0.4
        self.min_stable_frames = 5
        self.lock_duration = 1.25
        # Reject duplicate/overlapping detections so stop requires clearly two hands.
        self.min_palm_pair_dx = 0.09
        self.min_palm_pair_distance = 0.14
        self.min_handedness_score = 0.5
        self.min_hand_box_area = 0.0038

    @staticmethod
    def _hand_stats(lm) -> Tuple[float, float, float, float, float]:
        xs = [float(point.x) for point in lm]
        ys = [float(point.y) for point in lm]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        span_x = max_x - min_x
        span_y = max_y - min_y
        area = span_x * span_y
        cx = (min_x + max_x) * 0.5
        cy = (min_y + max_y) * 0.5
        return cx, cy, span_x, span_y, area

    @staticmethod
    def _thumb_open(lm, handedness: str) -> bool:
        thumb_tip = lm[4]
        thumb_ip = lm[3]
        horiz_margin = 0.012
        if handedness == "Right":
            horiz_open = thumb_tip.x < (thumb_ip.x - horiz_margin)
        else:
            horiz_open = thumb_tip.x > (thumb_ip.x + horiz_margin)
        vertical_open = thumb_tip.y < (thumb_ip.y - 0.02)
        return horiz_open or vertical_open

    @staticmethod
    def _finger_up(lm, tip_idx: int, pip_idx: int) -> bool:
        return lm[tip_idx].y < (lm[pip_idx].y - 0.015)

    def _finger_states(self, lm, handedness: str) -> Tuple[int, int, int, int, int]:
        thumb = 1 if self._thumb_open(lm, handedness) else 0
        index = 1 if self._finger_up(lm, 8, 6) else 0
        middle = 1 if self._finger_up(lm, 12, 10) else 0
        ring = 1 if self._finger_up(lm, 16, 14) else 0
        pinky = 1 if self._finger_up(lm, 20, 18) else 0
        return (thumb, index, middle, ring, pinky)

    @staticmethod
    def _is_open_palm(fingers: Tuple[int, int, int, int, int]) -> bool:
        _, index, middle, ring, pinky = fingers
        return index == 1 and middle == 1 and ring == 1 and pinky == 1

    @staticmethod
    def _map_selection(fingers: Tuple[int, int, int, int, int]) -> Optional[str]:
        _, index, middle, ring, pinky = fingers
        # Thumb is ignored for difficulty gestures; only count visible finger numbers.
        if (index, middle, ring, pinky) == (1, 0, 0, 0):
            return "difficulty_easy"
        if (index, middle, ring, pinky) == (1, 1, 0, 0):
            return "difficulty_medium"
        if (index, middle, ring, pinky) == (1, 1, 1, 0):
            return "difficulty_tough"
        return None

    def _stable_role_gesture(self, role: str, gesture: Optional[str], now: float) -> Optional[str]:
        if now < self._lock_until[role]:
            return None

        if not gesture:
            self._candidate[role] = None
            self._candidate_count[role] = 0
            self._candidate_since[role] = 0.0
            return None

        if self._candidate[role] != gesture:
            self._candidate[role] = gesture
            self._candidate_count[role] = 1
            self._candidate_since[role] = now
            return None

        self._candidate_count[role] += 1
        held = now - self._candidate_since[role]
        if self._candidate_count[role] >= self.min_stable_frames and held >= self.min_hold_s:
            self._lock_until[role] = now + self.lock_duration
            self._candidate[role] = None
            self._candidate_count[role] = 0
            self._candidate_since[role] = 0.0
            return gesture
        return None

    def detect(self, frame) -> Dict[str, Optional[str]]:
        now = time.time()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        control = {"gesture": None, "hand": None}
        selection = {"gesture": None, "hand": None}
        raw = []

        detections = []
        if results.multi_hand_landmarks and results.multi_handedness:
            for landmarks, handed in zip(results.multi_hand_landmarks, results.multi_handedness):
                label = handed.classification[0].label
                handed_score = float(handed.classification[0].score)
                lm = landmarks.landmark
                cx, cy, span_x, span_y, hand_area = self._hand_stats(lm)
                if handed_score < self.min_handedness_score or hand_area < self.min_hand_box_area:
                    continue
                self.wrist_history[label].append((now, cx))

                fingers = self._finger_states(lm, label)
                open_palm = self._is_open_palm(fingers)
                selection_candidate = self._map_selection(fingers)
                raw.append(
                    {
                        "hand": label,
                        "cx": round(cx, 3),
                        "cy": round(cy, 3),
                        "handedness_score": round(handed_score, 3),
                        "hand_area": round(hand_area, 4),
                        "hand_span": [round(span_x, 3), round(span_y, 3)],
                        "fingers": list(fingers),
                        "open_palm": open_palm,
                        "selection_candidate": selection_candidate,
                    }
                )
                detections.append(
                    {
                        "label": label,
                        "cx": cx,
                        "cy": cy,
                        "score": handed_score,
                        "area": hand_area,
                        "open_palm": open_palm,
                        "selection": selection_candidate,
                    }
                )

                self.drawer.draw_landmarks(frame, landmarks, mp.solutions.hands.HAND_CONNECTIONS)

        role_control = None
        role_selection = None

        open_palms = [item for item in detections if item.get("open_palm")]
        palm_labels = {str(item.get("label", "")) for item in open_palms if item.get("label")}
        palm_x = sorted(float(item.get("cx", 0.5)) for item in open_palms)
        palm_span = (palm_x[-1] - palm_x[0]) if len(palm_x) >= 2 else 0.0
        palm_pair_distance = 0.0
        if len(open_palms) >= 2:
            for i in range(len(open_palms) - 1):
                ax = float(open_palms[i].get("cx", 0.5))
                ay = float(open_palms[i].get("cy", 0.5))
                for j in range(i + 1, len(open_palms)):
                    bx = float(open_palms[j].get("cx", 0.5))
                    by = float(open_palms[j].get("cy", 0.5))
                    pair_distance = math.hypot(ax - bx, ay - by)
                    if pair_distance > palm_pair_distance:
                        palm_pair_distance = pair_distance

        if (
            len(open_palms) >= 2
            and len(palm_labels) >= 2
            and palm_span >= self.min_palm_pair_dx
            and palm_pair_distance >= self.min_palm_pair_distance
        ):
            role_control = "stop"
            control["hand"] = "both"
        elif len(open_palms) == 1:
            role_control = "start"
            control["hand"] = str(open_palms[0].get("label", "single"))

        # Difficulty gesture can be shown by one hand using one/two/three fingers.
        if detections:
            ordered = sorted(detections, key=lambda item: float(item.get("cx", 0.5)))
            selection_candidates = [item for item in ordered if item.get("selection")]
            if len(selection_candidates) == 1:
                selected = selection_candidates[0]
                role_selection = selected.get("selection")
                selection["hand"] = str(selected.get("label", "unknown"))

        stable_control = self._stable_role_gesture("control", role_control, now)
        stable_selection = self._stable_role_gesture("selection", role_selection, now)

        if stable_control:
            control["gesture"] = stable_control
        if stable_selection:
            selection["gesture"] = stable_selection

        return {
            "control": control,
            "selection": selection,
            "raw": raw,
        }

    def close(self) -> None:
        self.hands.close()
