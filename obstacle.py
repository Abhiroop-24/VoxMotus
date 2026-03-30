"""Obstacle detection manager with YOLOv8n primary mode and motion fallback."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import math
import threading

import cv2


@dataclass
class ObstacleMeta:
    mode: str
    present: bool
    alert: bool
    count: int
    max_area_ratio: float
    summary: str
    distance_m: float
    boxes: List[Dict]


class MotionObstacleDetector:
    """Fallback motion-based obstacle detector."""

    def __init__(self, min_contour_area: int = 7000, alert_cooldown_s: float = 6.0) -> None:
        self.prev_gray = None
        self.min_contour_area = min_contour_area
        self.alert_cooldown_s = alert_cooldown_s
        self.last_alert_time = 0.0
        self.prev_present = False

    def detect(self, frame, now_ts: float) -> ObstacleMeta:
        if frame is None:
            self.prev_gray = None
            self.prev_present = False
            return ObstacleMeta("motion", False, False, 0, 0.0, "no frame", 0.0, [])

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (11, 11), 0)

        if self.prev_gray is None:
            self.prev_gray = gray
            return ObstacleMeta("motion", False, False, 0, 0.0, "warming up", 0.0, [])

        diff = cv2.absdiff(self.prev_gray, gray)
        _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        thresh = cv2.dilate(thresh, None, iterations=2)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_area = 0.0
        count = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.min_contour_area:
                count += 1
            if area > max_area:
                max_area = area

        self.prev_gray = gray
        h, w = frame.shape[:2]
        frame_area = float(max(1, h * w))
        max_area_ratio = max_area / frame_area
        present = count > 0
        alert = False
        if present and ((not self.prev_present) or ((now_ts - self.last_alert_time) >= self.alert_cooldown_s)):
            alert = True
            self.last_alert_time = now_ts

        self.prev_present = present
        return ObstacleMeta("motion", present, alert, count, max_area_ratio, "motion", 0.0, [])


class YOLOObstacleDetector:
    """YOLOv8n detector for Pi stream obstacle alerts."""

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        conf_threshold: float = 0.35,
        min_area_ratio: float = 0.06,
        alert_cooldown_s: float = 6.0,
        person_only: bool = True,
        ignore_person: bool = False,
        device: str = "auto",
    ) -> None:
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.min_area_ratio = min_area_ratio
        self.alert_cooldown_s = alert_cooldown_s
        self.person_only = person_only
        self.ignore_person = ignore_person
        self.device = device

        self.last_alert_time = 0.0
        self.prev_present = False
        self._model = None
        self._names = {}
        self._available = False
        self._load_attempted = False
        self._load_lock = threading.Lock()
        self._init_error = ""

    def _ensure_model_loaded(self) -> None:
        if self._available or self._load_attempted:
            return
        with self._load_lock:
            if self._available or self._load_attempted:
                return
            self._load_attempted = True
            self._load_model()

    def _load_model(self) -> None:
        try:
            from ultralytics import YOLO

            self._model = YOLO(self.model_path)
            self._names = getattr(self._model, "names", {}) or {}
            self._available = True
        except Exception as exc:
            self._available = False
            self._init_error = str(exc)

    def _label(self, cls_idx: int) -> str:
        return str(self._names.get(int(cls_idx), int(cls_idx)))

    def _estimate_distance_m(self, area_ratio: float) -> float:
        if area_ratio <= 0.0:
            return 99.9
        # Calibrate so a box covering ~6% of the frame is ~1.5m away.
        k = 0.37
        dist = k / math.sqrt(max(area_ratio, 1e-6))
        return float(round(min(5.0, max(0.3, dist)), 2))

    def detect(self, frame, now_ts: float) -> ObstacleMeta:
        if frame is None:
            self.prev_present = False
            return ObstacleMeta("yolo", False, False, 0, 0.0, "no frame", 0.0, [])

        self._ensure_model_loaded()

        if not self._available:
            return ObstacleMeta("yolo", False, False, 0, 0.0, f"unavailable: {self._init_error}", 0.0, [])

        try:
            kwargs = {
                "source": frame,
                "conf": self.conf_threshold,
                "verbose": False,
                "imgsz": 640,
            }
            if self.device != "auto":
                kwargs["device"] = self.device

            results = self._model.predict(**kwargs)
            result = results[0]
            boxes = result.boxes

            h, w = frame.shape[:2]
            frame_area = float(max(1, h * w))
            min_cx = int(w * 0.12)
            max_cx = int(w * 0.88)

            kept = []
            max_area_ratio = 0.0
            nearest_dist = 0.0
            for box in boxes:
                cls_idx = int(box.cls.item())
                conf = float(box.conf.item())
                if conf < self.conf_threshold:
                    continue
                if self.person_only and cls_idx != 0:
                    continue
                if self.ignore_person and cls_idx == 0:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x1i, y1i, x2i, y2i = int(x1), int(y1), int(x2), int(y2)
                bw = max(1, x2i - x1i)
                bh = max(1, y2i - y1i)
                area_ratio = (bw * bh) / frame_area
                cx = int((x1i + x2i) / 2)

                if area_ratio < self.min_area_ratio:
                    continue
                if not (min_cx <= cx <= max_cx):
                    continue

                max_area_ratio = max(max_area_ratio, area_ratio)
                nearest_dist = self._estimate_distance_m(max_area_ratio)
                kept.append(
                    {
                        "label": self._label(cls_idx),
                        "conf": round(conf, 3),
                        "area_ratio": round(area_ratio, 4),
                        "distance_m": self._estimate_distance_m(area_ratio),
                        "xyxy": [x1i, y1i, x2i, y2i],
                    }
                )

            present = len(kept) > 0
            alert = False
            if present and ((not self.prev_present) or ((now_ts - self.last_alert_time) >= self.alert_cooldown_s)):
                alert = True
                self.last_alert_time = now_ts
            self.prev_present = present

            summary = "none"
            if kept:
                summary = ", ".join([f"{item['label']}:{item['conf']}" for item in kept[:2]])

            return ObstacleMeta("yolo", present, alert, len(kept), max_area_ratio, summary, nearest_dist, kept)
        except Exception as exc:
            return ObstacleMeta("yolo", False, False, 0, 0.0, f"infer error: {exc}", 0.0, [])


class ObstacleManager:
    """Unified obstacle detection API with YOLO and motion modes."""

    def __init__(
        self,
        mode: str = "yolo",
        min_contour_area: int = 7000,
        yolo_model_path: str = "yolov8n.pt",
        conf_threshold: float = 0.35,
        min_area_ratio: float = 0.06,
        alert_cooldown_s: float = 6.0,
        person_only: bool = True,
        ignore_person: bool = False,
        device: str = "auto",
    ) -> None:
        self.mode = (mode or "yolo").strip().lower()
        self.motion = MotionObstacleDetector(min_contour_area=min_contour_area, alert_cooldown_s=alert_cooldown_s)
        self.yolo = YOLOObstacleDetector(
            model_path=yolo_model_path,
            conf_threshold=conf_threshold,
            min_area_ratio=min_area_ratio,
            alert_cooldown_s=alert_cooldown_s,
            person_only=person_only,
            ignore_person=ignore_person,
            device=device,
        )

    def detect(self, frame, now_ts: float, enabled: bool = True) -> Tuple[bool, bool, Dict]:
        if not enabled:
            meta = ObstacleMeta(self.mode, False, False, 0, 0.0, "disabled", 0.0, [])
            return False, False, meta.__dict__

        if self.mode == "motion":
            meta = self.motion.detect(frame, now_ts)
            return meta.present, meta.alert, meta.__dict__

        meta = self.yolo.detect(frame, now_ts)
        # No fallback: Pi-only YOLO, return as-is so UI shows yolo status/errors.
        return meta.present, meta.alert, meta.__dict__
