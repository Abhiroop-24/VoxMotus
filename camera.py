"""Camera management for laptop webcam and Raspberry Pi UDP stream."""

from __future__ import annotations

import os
import threading
import time
from typing import Optional, Tuple

os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "quiet")

import cv2
import numpy as np


class CameraStream:
    """Threaded camera reader that always keeps the latest frame."""

    def __init__(
        self,
        source: int | str,
        name: str,
        backend: int = cv2.CAP_ANY,
        reconnect_delay_s: float = 0.2,
    ) -> None:
        self.source = source
        self.name = name
        self.backend = backend
        self.reconnect_delay_s = reconnect_delay_s
        self.cap: Optional[cv2.VideoCapture] = None
        self.frame: Optional[np.ndarray] = None
        self.ok = False
        self._running = False
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None

    def _open_capture(self) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(self.source, self.backend)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def start(self) -> None:
        if self._running:
            return
        self.cap = self._open_capture()
        self._running = True
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def _reader(self) -> None:
        while self._running:
            if self.cap is None or not self.cap.isOpened():
                self.ok = False
                time.sleep(self.reconnect_delay_s)
                if self.cap is not None:
                    self.cap.release()
                self.cap = self._open_capture()
                continue

            ok, frame = self.cap.read()
            self.ok = ok
            if ok:
                with self._lock:
                    self.frame = frame
            else:
                time.sleep(0.01)

    def get_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            if self.frame is None:
                return None
            return self.frame.copy()

    def is_alive(self) -> bool:
        return bool(self.ok)

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self.cap is not None:
            self.cap.release()


class DualCameraManager:
    """Keeps both cameras alive and lets main loop choose a primary source."""

    def __init__(
        self,
        laptop_source: int | str = 0,
        pi_udp_url: str = "udp://@:8080",
    ) -> None:
        self.laptop = CameraStream(laptop_source, "laptop")
        self.pi = CameraStream(pi_udp_url, "pi", backend=cv2.CAP_FFMPEG)

    def start(self) -> None:
        self.laptop.start()
        self.pi.start()

    def get_frames(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        return self.laptop.get_frame(), self.pi.get_frame()

    def choose_primary(self, activity: str, laptop_frame, pi_frame):
        if activity == "pushup" and pi_frame is not None:
            return "pi", pi_frame
        if laptop_frame is not None:
            return "laptop", laptop_frame
        if pi_frame is not None:
            return "pi", pi_frame
        return "none", None

    def status(self) -> dict:
        return {
            "laptop_alive": self.laptop.is_alive(),
            "pi_alive": self.pi.is_alive(),
        }

    def stop(self) -> None:
        self.laptop.stop()
        self.pi.stop()
