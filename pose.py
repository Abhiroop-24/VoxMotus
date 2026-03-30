"""Pose extraction and geometry helpers using MediaPipe Pose."""

from __future__ import annotations

import math
from typing import Dict, Optional

import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose


POSE_KEYPOINTS = {
    "nose": mp_pose.PoseLandmark.NOSE,
    "left_shoulder": mp_pose.PoseLandmark.LEFT_SHOULDER,
    "right_shoulder": mp_pose.PoseLandmark.RIGHT_SHOULDER,
    "left_elbow": mp_pose.PoseLandmark.LEFT_ELBOW,
    "right_elbow": mp_pose.PoseLandmark.RIGHT_ELBOW,
    "left_wrist": mp_pose.PoseLandmark.LEFT_WRIST,
    "right_wrist": mp_pose.PoseLandmark.RIGHT_WRIST,
    "left_hip": mp_pose.PoseLandmark.LEFT_HIP,
    "right_hip": mp_pose.PoseLandmark.RIGHT_HIP,
    "left_knee": mp_pose.PoseLandmark.LEFT_KNEE,
    "right_knee": mp_pose.PoseLandmark.RIGHT_KNEE,
    "left_ankle": mp_pose.PoseLandmark.LEFT_ANKLE,
    "right_ankle": mp_pose.PoseLandmark.RIGHT_ANKLE,
}


class PoseEstimator:
    def __init__(
        self,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.55,
        min_tracking_confidence: float = 0.55,
        preprocess_mode: str = "none",
        smooth_alpha: float = 0.35,
        min_visibility: float = 0.45,
    ) -> None:
        self.preprocess_mode = preprocess_mode
        self.smooth_alpha = float(np.clip(smooth_alpha, 0.01, 1.0))
        self.min_visibility = min_visibility
        self._last_landmarks: Dict[str, tuple] = {}
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        self.pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.drawer = mp.solutions.drawing_utils

    def _preprocess(self, frame):
        if frame is None:
            return frame

        processed = frame
        if self.preprocess_mode == "pi":
            h, w = processed.shape[:2]
            if w < 960:
                scale = 960.0 / float(max(1, w))
                processed = cv2.resize(
                    processed,
                    (960, int(h * scale)),
                    interpolation=cv2.INTER_CUBIC,
                )

            ycrcb = cv2.cvtColor(processed, cv2.COLOR_BGR2YCrCb)
            y, cr, cb = cv2.split(ycrcb)
            y = self._clahe.apply(y)
            processed = cv2.cvtColor(cv2.merge((y, cr, cb)), cv2.COLOR_YCrCb2BGR)
            processed = cv2.bilateralFilter(processed, 5, 35, 35)

        elif self.preprocess_mode == "laptop":
            processed = cv2.GaussianBlur(processed, (3, 3), 0)

        return processed

    def _quality_score(self, visibilities) -> float:
        if not visibilities:
            return 0.0

        mean_vis = float(np.mean(visibilities))
        strong_ratio = float(np.mean(np.array(visibilities) >= self.min_visibility))
        quality = (mean_vis * 0.65) + (strong_ratio * 0.35)
        return float(np.clip(quality, 0.0, 1.0))

    def _smooth_point(self, key: str, current: tuple) -> tuple:
        prev = self._last_landmarks.get(key)
        if prev is None:
            return current

        alpha = self.smooth_alpha
        return (
            (prev[0] * (1.0 - alpha)) + (current[0] * alpha),
            (prev[1] * (1.0 - alpha)) + (current[1] * alpha),
            (prev[2] * (1.0 - alpha)) + (current[2] * alpha),
            current[3],
        )

    def extract(self, frame) -> Dict:
        processed = self._preprocess(frame)
        rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.pose.process(rgb)

        data: Dict[str, Optional[tuple]] = {
            "raw": results,
            "present": False,
            "quality": 0.0,
            "visibility_mean": 0.0,
            "visibility_strong_ratio": 0.0,
            "preprocess_mode": self.preprocess_mode,
        }
        if not results.pose_landmarks:
            for key in POSE_KEYPOINTS:
                data[key] = None
            return data

        data["present"] = True
        landmarks = results.pose_landmarks.landmark
        visibilities = []
        smoothed_cache = {}
        for key, enum_value in POSE_KEYPOINTS.items():
            point = landmarks[enum_value.value]
            visibilities.append(point.visibility)
            current = (point.x, point.y, point.z, point.visibility)
            smoothed = self._smooth_point(key, current)
            smoothed_cache[key] = smoothed
            data[key] = smoothed

        self._last_landmarks = smoothed_cache
        data["quality"] = self._quality_score(visibilities)
        data["visibility_mean"] = float(np.mean(visibilities))
        data["visibility_strong_ratio"] = float(np.mean(np.array(visibilities) >= self.min_visibility))
        return data

    def draw(self, frame, results) -> None:
        if results and results.pose_landmarks:
            self.drawer.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
            )

    def close(self) -> None:
        self.pose.close()


def calculate_angle(a, b, c) -> float:
    """Returns the angle ABC in degrees."""
    ax, ay = a[:2]
    bx, by = b[:2]
    cx, cy = c[:2]

    radians = math.atan2(cy - by, cx - bx) - math.atan2(ay - by, ax - bx)
    angle = abs(math.degrees(radians))
    if angle > 180.0:
        angle = 360.0 - angle
    return angle


def midpoint(p1, p2):
    return ((p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0)
