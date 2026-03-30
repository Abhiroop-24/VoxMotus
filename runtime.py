"""Long-running ANTARDRISHTI runtime managed by Flask."""

from __future__ import annotations

import os
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from activity import ActivityDetector, classify_posture_errors
from ai_assist import LocalAIAssistant
from camera import DualCameraManager
from coach import CoachDecisionEngine
from exercise import ExerciseEngine
from gesture import GestureDetector
from obstacle import ObstacleManager
from pi_control import PiCameraRemote, PiConfig
from pose import PoseEstimator
from scoring import ReferenceComparator
from joint_mapping import extract_joint_angles
from training_data import TrainingDataCollector
from ui import write_state
from voice import VoiceEngine


@dataclass
class RuntimeConfig:
    webcam: int = 0
    pi_url: str = "udp://@:8080"
    state_file: str = "state.json"
    break_seconds: int = 10
    obstacle_area: int = 7000
    voice_cooldown: float = 2.4
    mute: bool = False
    tts_backend: str = "auto"
    voice_persona: str = "coach"
    pyttsx3_voice_hint: str = ""
    edge_voice: str = "en-US-AvaNeural"
    edge_rate: str = "+0%"
    edge_pitch: str = "+0Hz"
    piper_model: str = ""
    piper_bin: str = "piper"
    piper_speaker: Optional[int] = None
    output_dir: str = "outputs"
    record_annotated: bool = True
    output_fps: int = 20
    auto_start_pi: bool = False
    pi_host: str = ""
    pi_username: str = ""
    pi_password: str = ""
    pi_port: int = 22
    pi_start_cmd: str = "cd ~ && ./start_cam.sh > /tmp/drishti_stream.log 2>&1 &"
    obstacle_enabled: bool = True
    obstacle_mode: str = "yolo"
    obstacle_yolo_model: str = "yolov8n.pt"
    obstacle_yolo_conf: float = 0.35
    obstacle_yolo_min_area: float = 0.06
    obstacle_alert_cooldown: float = 6.0
    obstacle_person_only: bool = True
    obstacle_ignore_person: bool = False
    obstacle_device: str = "cuda"
    pose_model_laptop: int = 1
    pose_model_pi: int = 2
    pose_min_detection_conf: float = 0.55
    pose_min_tracking_conf: float = 0.55
    pose_smooth_alpha: float = 0.35
    fusion_min_quality: float = 0.18
    feedback_dataset_path: str = "feedback_dataset.json"
    rephrase_enabled: bool = False
    rephrase_provider: str = "openai"
    rephrase_model: str = ""
    rephrase_api_key: str = ""
    rephrase_api_base: str = ""
    ai_weights_path: str = "models/activity_weights.json"
    ai_confidence_threshold: float = 0.82
    collect_training_data: bool = False
    training_data_dir: str = "outputs/datasets"
    training_flush_every: int = 250


class SessionController:
    """Tracks interaction state and pause/resume transitions."""

    STATES = (
        "INIT",
        "WAIT_USER",
        "WAIT_START_GESTURE",
        "WAIT_DIFFICULTY",
        "READY",
        "WORKOUT_SQUATS",
        "WORKOUT_SITUPS",
        "WORKOUT_JUMPING_JACKS",
        "BREAK",
        "WORKOUT_PUSHUPS",
        "PAUSED",
        "END",
    )

    VALID_TRANSITIONS = {
        "INIT": {"WAIT_USER"},
        "WAIT_USER": {"WAIT_START_GESTURE", "END"},
        "WAIT_START_GESTURE": {"WAIT_DIFFICULTY", "END"},
        "WAIT_DIFFICULTY": {"READY", "PAUSED", "END"},
        "READY": {"WORKOUT_SQUATS", "WORKOUT_SITUPS", "WORKOUT_JUMPING_JACKS", "WORKOUT_PUSHUPS", "PAUSED", "END"},
        "WORKOUT_SQUATS": {"BREAK", "PAUSED", "END"},
        "WORKOUT_SITUPS": {"BREAK", "PAUSED", "END"},
        "WORKOUT_JUMPING_JACKS": {"BREAK", "PAUSED", "END"},
        "BREAK": {"WORKOUT_SQUATS", "WORKOUT_SITUPS", "WORKOUT_JUMPING_JACKS", "WORKOUT_PUSHUPS", "PAUSED", "END"},
        "WORKOUT_PUSHUPS": {"PAUSED", "END"},
        "PAUSED": {"WAIT_DIFFICULTY", "READY", "WORKOUT_SQUATS", "WORKOUT_SITUPS", "WORKOUT_JUMPING_JACKS", "BREAK", "WORKOUT_PUSHUPS", "END"},
        "END": {"WAIT_START_GESTURE", "WAIT_DIFFICULTY"},
    }

    def __init__(self) -> None:
        self.state = "INIT"
        self.prev_state = "INIT"
        self.changed_ts = time.time()

    def transition(self, new_state: str, force: bool = False) -> bool:
        if new_state == self.state:
            return False
        if new_state not in self.STATES:
            return False
        if not force and new_state not in self.VALID_TRANSITIONS.get(self.state, set()):
            return False
        self.prev_state = self.state
        self.state = new_state
        self.changed_ts = time.time()
        return True

    def pause(self) -> bool:
        if self.state == "PAUSED":
            return False
        if self.state not in (
            "WAIT_DIFFICULTY",
            "READY",
            "WORKOUT_SQUATS",
            "WORKOUT_SITUPS",
            "WORKOUT_JUMPING_JACKS",
            "BREAK",
            "WORKOUT_PUSHUPS",
        ):
            return False
        self.prev_state = self.state
        self.state = "PAUSED"
        self.changed_ts = time.time()
        return True

    def resume(self) -> bool:
        if self.state != "PAUSED":
            return False
        self.state = self.prev_state if self.prev_state not in ("PAUSED", "INIT") else "READY"
        self.changed_ts = time.time()
        return True

    def since_change(self) -> float:
        return time.time() - self.changed_ts


class GestureGuard:
    """Applies gesture debounce and optional confirmation windows."""

    def __init__(self, confirm_window: float = 1.0, lock_window: float = 1.3) -> None:
        self.confirm_window = float(confirm_window)
        self.lock_window = float(lock_window)
        self._last_seen: Dict[str, Dict[str, float]] = {}

    def check(
        self,
        gesture: str,
        now_ts: float,
        require_confirm: bool = False,
        lock_override_s: Optional[float] = None,
    ) -> bool:
        if not gesture:
            return False
        lock_for = self.lock_window if lock_override_s is None else float(max(0.2, lock_override_s))
        meta = self._last_seen.get(gesture)
        if meta and now_ts < meta.get("locked_until", 0.0):
            return False

        if require_confirm:
            if meta and (now_ts - meta.get("ts", 0.0)) <= self.confirm_window:
                self._last_seen[gesture] = {
                    "ts": now_ts,
                    "locked_until": now_ts + lock_for,
                }
                return True

            # First sight should not lock when confirmation is required.
            self._last_seen[gesture] = {
                "ts": now_ts,
                "locked_until": 0.0,
            }
            return False

        self._last_seen[gesture] = {
            "ts": now_ts,
            "locked_until": now_ts + lock_for,
        }
        return True


class TemporalWindow:
    """Maintains sampled 10-15 frame buffers for persistence checks."""

    def __init__(self, window_s: float = 1.2, max_frames: int = 15, sample_interval_s: float = 0.08) -> None:
        self.window_s = float(window_s)
        self.max_frames = int(max(10, min(15, max_frames)))
        self.sample_interval_s = float(max(0.04, sample_interval_s))
        self.history: Dict[str, deque] = {}
        self._last_sample_ts: Dict[str, float] = {}

    def update(self, key: str, active: bool, now_ts: float) -> None:
        last_ts = self._last_sample_ts.get(key, 0.0)
        if (now_ts - last_ts) < self.sample_interval_s:
            return
        self._last_sample_ts[key] = now_ts

        q = self.history.setdefault(key, deque(maxlen=self.max_frames))
        q.append((now_ts, bool(active)))

        cutoff = now_ts - self.window_s
        while q and q[0][0] < cutoff:
            q.popleft()
        if not q:
            self.history.pop(key, None)
            self._last_sample_ts.pop(key, None)

    def persistent(self, key: str, min_ratio: float = 0.7, min_duration_s: float = 1.0) -> bool:
        q = self.history.get(key)
        if not q or len(q) < 4:
            return False
        span = q[-1][0] - q[0][0]
        if span < min_duration_s:
            return False
        active_samples = sum(1 for _, active in q if active)
        coverage = active_samples / float(len(q))
        return coverage >= float(min_ratio)


class MetricSmoother:
    """Simple EMA smoother for noisy metric streams."""

    def __init__(self, alpha: float = 0.3) -> None:
        self.alpha = float(np.clip(alpha, 0.01, 1.0))
        self._state: Dict[str, float] = {}
        self._max_delta = {
            "knee_angle": 12.0,
            "elbow_angle": 14.0,
            "torso_tilt": 8.0,
            "body_line": 10.0,
            "shoulder_hip_diff": 0.04,
        }

    def update(self, metrics: Dict) -> Dict:
        smoothed: Dict = {}
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and not np.isnan(value):
                current = float(value)
                prev = self._state.get(key, current)
                max_delta = self._max_delta.get(key)
                if max_delta is not None:
                    delta = current - prev
                    if abs(delta) > max_delta:
                        current = prev + np.sign(delta) * max_delta
                new_val = prev + self.alpha * (current - prev)
                self._state[key] = new_val
                smoothed[key] = new_val
            else:
                smoothed[key] = value
        return smoothed


class AsyncObstacleWorker:
    """Runs obstacle detection asynchronously to keep the main loop responsive."""

    def __init__(self, detector: ObstacleManager, stride: int = 3, min_interval_s: float = 0.12) -> None:
        self.detector = detector
        self.stride = max(1, int(stride))
        self.min_interval_s = float(min_interval_s)
        self._latest_frame = None
        self._latest_ts = 0.0
        self._result: Dict = {
            "mode": detector.mode,
            "present": False,
            "alert": False,
            "count": 0,
            "max_area_ratio": 0.0,
            "distance_m": 0.0,
            "summary": "init",
        }
        self._stop = threading.Event()
        self._evt = threading.Event()
        self._idx = 0
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> None:
        if not self._thread.is_alive():
            self._thread.start()

    def close(self) -> None:
        self._stop.set()
        self._evt.set()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)

    def submit(self, frame, now_ts: float) -> None:
        self._idx += 1
        if self._idx % self.stride != 0:
            return
        if (now_ts - self._latest_ts) < self.min_interval_s:
            return
        self._latest_frame = frame.copy() if frame is not None else None
        self._latest_ts = now_ts
        self._evt.set()

    def result(self) -> Dict:
        return self._result

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._evt.wait(0.2)
            self._evt.clear()
            if self._stop.is_set():
                break
            frame = self._latest_frame
            ts = self._latest_ts
            if frame is None:
                continue
            try:
                present, alert, meta = self.detector.detect(frame, ts, enabled=True)
                meta["mode"] = self.detector.mode
                meta["present"] = present
                meta["alert"] = alert
                self._result = meta
            except Exception as exc:  # noqa: BLE001
                self._result = {
                    "mode": self.detector.mode,
                    "present": False,
                    "alert": False,
                    "count": 0,
                    "max_area_ratio": 0.0,
                    "distance_m": 0.0,
                    "summary": f"worker error: {exc}",
                }


class PiAutoStartWatchdog:
    """Keeps Pi camera stream alive by retrying remote start over SSH."""

    def __init__(
        self,
        host: str,
        username: str,
        password: str,
        port: int,
        start_cmd: str,
        retry_interval_s: float = 8.0,
    ) -> None:
        self.enabled = bool(host and username and password)
        self.retry_interval_s = max(3.0, float(retry_interval_s))
        self._pi_alive = False
        self._pi_alive_lock = threading.Lock()
        self._status_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._attempts = 0
        self._last_attempt_ts = 0.0
        self._last_success_ts = 0.0
        self._status = {
            "enabled": self.enabled,
            "ok": False,
            "message": "disabled",
            "attempts": 0,
            "last_attempt_ts": 0.0,
            "last_success_ts": 0.0,
        }

        self._remote = PiCameraRemote(
            PiConfig(
                host=host,
                username=username,
                password=password,
                port=port,
                start_cmd=start_cmd,
            )
        )

        if not self.enabled:
            self._status["message"] = "missing pi ssh credentials"

    def start(self) -> None:
        if not self.enabled:
            return
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._remote.close()

    def set_pi_alive(self, is_alive: bool) -> None:
        with self._pi_alive_lock:
            self._pi_alive = bool(is_alive)

    def status(self) -> Dict:
        with self._status_lock:
            return dict(self._status)

    def _loop(self) -> None:
        while not self._stop.is_set():
            now_ts = time.time()
            with self._pi_alive_lock:
                pi_alive = self._pi_alive

            if pi_alive:
                with self._status_lock:
                    self._status["ok"] = True
                    self._status["message"] = "pi stream healthy"
                time.sleep(1.5)
                continue

            if (now_ts - self._last_attempt_ts) < self.retry_interval_s:
                time.sleep(0.5)
                continue

            self._attempts += 1
            self._last_attempt_ts = now_ts
            ok, message = self._remote.start_stream()

            with self._status_lock:
                self._status["ok"] = bool(ok)
                self._status["message"] = message
                self._status["attempts"] = self._attempts
                self._status["last_attempt_ts"] = self._last_attempt_ts
                if ok:
                    self._last_success_ts = now_ts
                    self._status["last_success_ts"] = self._last_success_ts

            time.sleep(0.6)


class AntardrishtiRuntime:
    """Main runtime loop coordinating vision, gestures, and coaching."""

    def __init__(self, config: RuntimeConfig) -> None:
        self.config = config
        self._state_lock = threading.Lock()
        self._frame_lock = threading.Lock()
        self._latest_frame_jpeg: Optional[bytes] = None
        self._latest_laptop_jpeg: Optional[bytes] = None
        self._latest_pi_jpeg: Optional[bytes] = None
        self._state: Dict = {
            "runtime": "init",
            "state": "INIT",
            "feedback": "Initializing coach runtime.",
            "next_action": "Wait for camera and AI modules to initialize.",
            "gesture": {"control": None, "selection": None},
            "camera_status": {"laptop_alive": False, "pi_alive": False},
        }
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._command_queue: "queue.Queue[Tuple[str, Dict]]" = queue.Queue(maxsize=32)

    # Public API -----------------------------------------------------
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._set_state({"runtime": "stopped"})

    def send_command(self, command: str, payload: Optional[Dict] = None) -> bool:
        payload = payload or {}
        try:
            self._command_queue.put_nowait((command, payload))
            return True
        except queue.Full:
            return False

    def get_state(self) -> Dict:
        with self._state_lock:
            return dict(self._state)

    def latest_frame_jpeg(self) -> Optional[bytes]:
        with self._frame_lock:
            return self._latest_frame_jpeg

    def latest_laptop_jpeg(self) -> Optional[bytes]:
        with self._frame_lock:
            return self._latest_laptop_jpeg

    def latest_pi_jpeg(self) -> Optional[bytes]:
        with self._frame_lock:
            return self._latest_pi_jpeg

    # Internal helpers ----------------------------------------------
    def _set_state(self, state: Dict) -> None:
        with self._state_lock:
            self._state.update(state)

    @staticmethod
    def _safe_pose_quality(pose_data: Optional[Dict]) -> float:
        if not pose_data:
            return 0.0
        try:
            return float(pose_data.get("quality", 0.0))
        except Exception:
            return 0.0

    def _select_pose_source(
        self,
        session_state: str,
        laptop_frame,
        pi_frame,
        laptop_pose: Optional[Dict],
        pi_pose: Optional[Dict],
    ) -> Tuple[str, Dict, str]:
        laptop_q = self._safe_pose_quality(laptop_pose)
        pi_q = self._safe_pose_quality(pi_pose)

        def valid_pose(pose: Optional[Dict], quality: float) -> bool:
            return bool(pose and pose.get("present") and quality >= self.config.fusion_min_quality)

        if session_state == "WORKOUT_PUSHUPS":
            if valid_pose(pi_pose, pi_q):
                return "pi", pi_pose, "pi_primary_pushup"
            if valid_pose(laptop_pose, laptop_q):
                return "laptop", laptop_pose, "laptop_fallback_pushup"
        else:
            if valid_pose(laptop_pose, laptop_q):
                return "laptop", laptop_pose, "laptop_primary"
            if valid_pose(pi_pose, pi_q):
                return "pi", pi_pose, "pi_fallback"

        if laptop_pose and laptop_pose.get("present"):
            return "laptop", laptop_pose, "laptop_low_quality"
        if pi_pose and pi_pose.get("present"):
            return "pi", pi_pose, "pi_low_quality"

        return "none", {"present": False, "quality": 0.0}, "none"

    @staticmethod
    def _draw_people_on_pi(frame, boxes) -> None:
        if frame is None or not boxes:
            return

        for idx, box in enumerate(boxes):
            xyxy = box.get("xyxy", [0, 0, 0, 0])
            if len(xyxy) != 4:
                continue

            x1, y1, x2, y2 = [int(v) for v in xyxy]
            conf = float(box.get("conf", 0.0))
            distance_m = float(box.get("distance_m", 0.0))
            label = box.get("label", "person")

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 255), 2)
            text = f"{label}#{idx + 1} {distance_m:.2f}m conf:{conf:.2f}"
            cv2.putText(
                frame,
                text,
                (x1, max(16, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (0, 220, 255),
                2,
            )

    @staticmethod
    def _normalize_exercise_name(name: str) -> str:
        raw = str(name or "").strip().lower().replace("-", "_").replace(" ", "_")
        aliases = {
            "squat": "squat",
            "squats": "squat",
            "situp": "situp",
            "situps": "situp",
            "sit_up": "situp",
            "sit_ups": "situp",
            "jumping_jack": "jumping_jack",
            "jumping_jacks": "jumping_jack",
            "jack": "jumping_jack",
            "jacks": "jumping_jack",
            "pushup": "pushup",
            "pushups": "pushup",
            "push_up": "pushup",
        }
        return aliases.get(raw, "")

    @staticmethod
    def _state_for_exercise(exercise_name: str) -> str:
        mapping = {
            "squat": "WORKOUT_SQUATS",
            "situp": "WORKOUT_SITUPS",
            "jumping_jack": "WORKOUT_JUMPING_JACKS",
            "pushup": "WORKOUT_PUSHUPS",
        }
        return mapping.get(exercise_name, "WORKOUT_SQUATS")

    @staticmethod
    def _exercise_for_state(state: str) -> str:
        mapping = {
            "WORKOUT_SQUATS": "squat",
            "WORKOUT_SITUPS": "situp",
            "WORKOUT_JUMPING_JACKS": "jumping_jack",
            "WORKOUT_PUSHUPS": "pushup",
        }
        return mapping.get(state, "")

    @staticmethod
    def _exercise_title(exercise_name: str) -> str:
        titles = {
            "squat": "squats",
            "situp": "sit-ups",
            "jumping_jack": "jumping jacks",
            "pushup": "push-ups",
        }
        return titles.get(exercise_name, exercise_name)

    @staticmethod
    def _exercise_intro(exercise_name: str) -> str:
        intros = {
            "squat": (
                "Starting squats now. Stand with feet shoulder-width apart, keep your chest up, "
                "bend your knees, and come back up with control."
            ),
            "situp": (
                "Next is sit-ups. Lie on your back, bend your knees, keep your feet grounded, "
                "then lift your upper body up and lower down slowly."
            ),
            "jumping_jack": (
                "Next is jumping jacks. Start with feet together and hands at your sides. "
                "Jump feet apart while raising both arms up, then return to the start position."
            ),
            "pushup": (
                "Next is push-ups. Keep your body in one straight line, bend your elbows with control, "
                "and push back up."
            ),
        }
        return intros.get(exercise_name, f"Starting {exercise_name}.")

    @staticmethod
    def _situp_signal(joint_angles: Dict, metrics: Dict) -> Optional[float]:
        hip_values = [
            joint_angles.get("left_hip_angle"),
            joint_angles.get("right_hip_angle"),
        ]
        hip_values = [float(v) for v in hip_values if v is not None]
        if hip_values:
            mean_hip = float(np.mean(np.array(hip_values, dtype=np.float32)))
            return float(np.clip(180.0 - mean_hip, 0.0, 180.0))

        torso = metrics.get("torso_tilt")
        if torso is None:
            return None
        return float(np.clip(float(torso), 0.0, 180.0))

    @staticmethod
    def _jumping_jack_open_score(pose_data: Dict) -> Optional[float]:
        if not pose_data or not pose_data.get("present"):
            return None

        keys = [
            "left_ankle",
            "right_ankle",
            "left_shoulder",
            "right_shoulder",
            "left_wrist",
            "right_wrist",
            "nose",
        ]
        points = {k: pose_data.get(k) for k in keys}
        if any(points[k] is None for k in keys):
            return None

        left_ankle = points["left_ankle"]
        right_ankle = points["right_ankle"]
        left_shoulder = points["left_shoulder"]
        right_shoulder = points["right_shoulder"]
        left_wrist = points["left_wrist"]
        right_wrist = points["right_wrist"]
        nose = points["nose"]

        shoulder_width = abs(float(right_shoulder[0]) - float(left_shoulder[0]))
        ankle_width = abs(float(right_ankle[0]) - float(left_ankle[0]))
        width_ratio = ankle_width / max(0.08, shoulder_width)

        hands_above_head = 1.0 if (float(left_wrist[1]) < float(nose[1]) and float(right_wrist[1]) < float(nose[1])) else 0.0
        leg_open_score = float(np.clip((width_ratio - 0.95) / 1.1, 0.0, 1.0))

        return float(np.clip((0.62 * leg_open_score) + (0.38 * hands_above_head), 0.0, 1.0))

    @staticmethod
    def _flow_hint(state: str) -> str:
        hints = {
            "WAIT_USER": "Stand in laptop camera view with full body visible.",
            "WAIT_START_GESTURE": "Show one palm to start. Show two palms to stop.",
            "WAIT_DIFFICULTY": "Show one finger for easy, two for medium, three for tough.",
            "READY": "Difficulty confirmed. Hold steady for one second.",
              "WORKOUT_SQUATS": "Do controlled squats. Keep your chest up and knees aligned. Show four fingers to change exercise.",
              "WORKOUT_SITUPS": "Do controlled sit-ups. Lift up and lower down with control. Show four fingers to change exercise.",
              "WORKOUT_JUMPING_JACKS": "Do jumping jacks with full arm and leg movement. Show four fingers to change exercise.",
              "BREAK": "Take a 10 second break. Breathe and prepare for the next exercise. Show four fingers to skip break.",
              "WORKOUT_PUSHUPS": "Do controlled push-ups. Keep body line straight. Show four fingers to change exercise.",
            "PAUSED": "Session paused. Use dashboard to resume or stop.",
            "END": "Session ended. Show one palm or press Start to begin again.",
            "INIT": "Initializing camera and coach systems.",
        }
        return hints.get(state, "Follow the coach prompts.")

    # Core loop ------------------------------------------------------
    def _run(self) -> None:
        cameras = DualCameraManager(laptop_source=self.config.webcam, pi_udp_url=self.config.pi_url)
        pose_laptop = PoseEstimator(
            model_complexity=self.config.pose_model_laptop,
            min_detection_confidence=self.config.pose_min_detection_conf,
            min_tracking_confidence=self.config.pose_min_tracking_conf,
            preprocess_mode="laptop",
            smooth_alpha=self.config.pose_smooth_alpha,
        )
        pose_pi = PoseEstimator(
            model_complexity=self.config.pose_model_pi,
            min_detection_confidence=self.config.pose_min_detection_conf,
            min_tracking_confidence=self.config.pose_min_tracking_conf,
            preprocess_mode="pi",
            smooth_alpha=self.config.pose_smooth_alpha,
        )
        gesture = GestureDetector()
        activity_detector = ActivityDetector()
        exercise = ExerciseEngine(difficulty="medium")
        obstacle = ObstacleManager(
            mode=self.config.obstacle_mode,
            min_contour_area=self.config.obstacle_area,
            yolo_model_path=self.config.obstacle_yolo_model,
            conf_threshold=self.config.obstacle_yolo_conf,
            min_area_ratio=self.config.obstacle_yolo_min_area,
            alert_cooldown_s=self.config.obstacle_alert_cooldown,
            person_only=self.config.obstacle_person_only,
            ignore_person=self.config.obstacle_ignore_person,
            device=self.config.obstacle_device,
        )
        obstacle_worker = AsyncObstacleWorker(detector=obstacle, stride=3, min_interval_s=0.15)
        pi_watchdog = None
        if self.config.auto_start_pi:
            pi_watchdog = PiAutoStartWatchdog(
                host=self.config.pi_host,
                username=self.config.pi_username,
                password=self.config.pi_password,
                port=self.config.pi_port,
                start_cmd=self.config.pi_start_cmd,
                retry_interval_s=8.0,
            )
        coach = CoachDecisionEngine(
            cooldown_s=max(2.0, min(3.0, self.config.voice_cooldown)),
            dataset_path=self.config.feedback_dataset_path,
        )
        voice = VoiceEngine(
            enabled=not self.config.mute,
            backend=self.config.tts_backend,
            persona=self.config.voice_persona,
            pyttsx3_voice_hint=self.config.pyttsx3_voice_hint,
            edge_voice=self.config.edge_voice,
            edge_rate=self.config.edge_rate,
            edge_pitch=self.config.edge_pitch,
            piper_model=self.config.piper_model,
            piper_bin=self.config.piper_bin,
            piper_speaker=self.config.piper_speaker,
            rephrase_enabled=self.config.rephrase_enabled,
            rephrase_provider=self.config.rephrase_provider,
            rephrase_model=self.config.rephrase_model,
            rephrase_api_key=self.config.rephrase_api_key,
            rephrase_api_base=self.config.rephrase_api_base,
        )
        scorer = ReferenceComparator()
        ai_assist = LocalAIAssistant(
            weights_path=self.config.ai_weights_path,
            min_confidence=self.config.ai_confidence_threshold,
        )
        data_collector = TrainingDataCollector(
            enabled=self.config.collect_training_data,
            output_dir=self.config.training_data_dir,
            flush_every=self.config.training_flush_every,
        )

        interaction = SessionController()
        interaction.transition("WAIT_USER", force=True)
        guard = GestureGuard(confirm_window=1.3)
        error_window = TemporalWindow(window_s=1.2, max_frames=15, sample_interval_s=0.08)
        user_window = TemporalWindow(window_s=1.2, max_frames=15, sample_interval_s=0.08)
        smoother = MetricSmoother(alpha=0.28)

        state_write_interval = 0.25
        last_state_write_ts = 0.0
        last_speak_ts = 0.0
        last_speech_text = ""
        last_spoken_feedback = ""
        last_instruction = ""
        feedback = "Welcome to ANTARDRISHTI. I am ready to guide you."
        feedback_pin_until = 0.0
        gesture_event = ""
        gesture_event_ts = 0.0
        obstacle_enabled = self.config.obstacle_enabled
        frame_idx = 0
        ready_since = 0.0
        current_exercise: Optional[str] = None
        current_plan_index = -1
        base_exercise_plan = ["squat", "situp", "jumping_jack"]
        extra_exercises = []
        exercise_plan = list(base_exercise_plan)
        next_exercise_pending: Optional[str] = None
        break_end_ts = 0.0
        summary_spoken = False
        paused = False
        ended_by_stop = False
        camera_issue_ts = 0.0

        speech_cooldown = max(2.0, min(3.0, self.config.voice_cooldown))
        os.makedirs(self.config.output_dir, exist_ok=True)

        def speak_once(
            text: str,
            force: bool = False,
            tone: str = "neutral",
            exercise_name: str = "general",
            severity: str = "medium",
            sync_feedback: bool = True,
            sync_hold_s: float = 2.2,
            replace_pending: bool = True,
        ) -> None:
            nonlocal last_speak_ts, last_instruction, last_speech_text, last_spoken_feedback, feedback, feedback_pin_until
            if not text:
                return
            now_local = time.time()
            if not force and (now_local - last_speak_ts) < speech_cooldown:
                return
            if not force and text == last_speech_text and (now_local - last_speak_ts) < 6.0:
                return
            voice.speak_structured(
                text,
                tone=tone,
                exercise=exercise_name,
                severity=severity,
                replace_pending=replace_pending,
            )
            last_instruction = text
            last_speech_text = text
            last_spoken_feedback = text
            last_speak_ts = now_local
            if sync_feedback:
                feedback = text
                feedback_pin_until = max(feedback_pin_until, now_local + float(max(0.8, sync_hold_s)))

        def transition_state(new_state: str, force: bool = False, announce: bool = True) -> bool:
            changed = interaction.transition(new_state, force=force)
            if not changed:
                return False
            if announce:
                msg = coach.state_message(time.time(), new_state, force=False)
                if msg:
                    speak_once(msg, tone="guidance", sync_feedback=True, sync_hold_s=2.4)
            return True

        def refresh_plan() -> None:
            nonlocal exercise_plan
            exercise_plan = list(base_exercise_plan) + list(extra_exercises)

        def start_exercise(exercise_name: str, plan_index: Optional[int] = None) -> None:
            nonlocal current_exercise, current_plan_index, feedback, next_exercise_pending, break_end_ts
            normalized = self._normalize_exercise_name(exercise_name)
            if not normalized:
                return

            state_name = self._state_for_exercise(normalized)
            changed = transition_state(state_name, force=True, announce=False)
            if not changed:
                return

            current_exercise = normalized
            if plan_index is not None:
                current_plan_index = int(plan_index)
            next_exercise_pending = None
            break_end_ts = 0.0
            intro = self._exercise_intro(normalized)
            feedback = intro
            speak_once(intro, force=True, tone="guidance", exercise_name=normalized)

        def begin_break(next_exercise: str) -> None:
            nonlocal break_end_ts, next_exercise_pending, feedback
            normalized = self._normalize_exercise_name(next_exercise)
            if not normalized:
                return

            next_exercise_pending = normalized
            transition_state("BREAK", force=True, announce=False)
            break_end_ts = time.time() + float(self.config.break_seconds)
            break_msg = (
                f"Great work. Take {int(self.config.break_seconds)} seconds break. "
                f"Next we will do {self._exercise_title(normalized)}."
            )
            feedback = break_msg
            speak_once(break_msg, force=True, tone="guidance")

        def advance_plan_after_current() -> None:
            nonlocal ended_by_stop
            if current_plan_index + 1 < len(exercise_plan):
                begin_break(exercise_plan[current_plan_index + 1])
            else:
                ended_by_stop = False
                interaction.transition("END", force=True)

        def session_reset() -> None:
            nonlocal summary_spoken, current_exercise, current_plan_index, paused, break_end_ts, ready_since, feedback, feedback_pin_until, ended_by_stop, next_exercise_pending
            exercise.reset()
            scorer.reset()
            coach.reset_memory()
            summary_spoken = False
            current_exercise = None
            current_plan_index = -1
            paused = False
            ended_by_stop = False
            break_end_ts = 0.0
            next_exercise_pending = None
            ready_since = 0.0
            refresh_plan()
            feedback = "Welcome to ANTARDRISHTI. Show one palm to start, two palms to stop."
            feedback_pin_until = 0.0

        def apply_control(command: str, payload: Optional[Dict] = None) -> None:
            nonlocal obstacle_enabled, paused, feedback, current_exercise, current_plan_index, ready_since, ended_by_stop, summary_spoken, break_end_ts, next_exercise_pending
            payload = payload or {}
            if command == "stop":
                # Ignore stop before the session has started or after it has already ended.
                if interaction.state in ("INIT", "WAIT_USER", "WAIT_START_GESTURE", "END"):
                    return
                ended_by_stop = True
                summary_spoken = False
                interaction.transition("END", force=True)
                message = coach.end_message(time.time(), stopped_by_user=True)
                if message:
                    speak_once(message, force=True, tone="safety")
                    feedback = message
                    summary_spoken = True
                return

            if command == "start":
                if interaction.state not in ("WAIT_START_GESTURE", "END"):
                    return
                session_reset()
                transition_state("WAIT_DIFFICULTY")
                return

            if command in ("pause", "pause_toggle"):
                if interaction.state != "PAUSED" and interaction.pause():
                    paused = True
                    msg = coach.state_message(time.time(), "PAUSED")
                    if msg:
                        speak_once(msg, tone="guidance")
                elif interaction.state == "PAUSED" and interaction.resume():
                    paused = False
                    msg = coach.state_message(time.time(), "RESUMED")
                    if msg:
                        speak_once(msg, tone="guidance")
                return

            if command == "resume":
                if interaction.resume():
                    paused = False
                    msg = coach.state_message(time.time(), "RESUMED")
                    if msg:
                        speak_once(msg, tone="guidance")
                return

            if command == "next":
                if interaction.state == "BREAK" and next_exercise_pending:
                    start_exercise(next_exercise_pending, plan_index=current_plan_index + 1)
                    return
                if interaction.state.startswith("WORKOUT_") and (current_plan_index + 1) < len(exercise_plan):
                    begin_break(exercise_plan[current_plan_index + 1])
                return

            if command == "repeat":
                if last_instruction:
                    speak_once(last_instruction, force=True, tone="guidance")
                return

            if command.startswith("difficulty_"):
                selected = command.replace("difficulty_", "")
                if interaction.state == "WAIT_DIFFICULTY" and selected in ("easy", "medium", "tough"):
                    exercise.set_difficulty(selected)
                    feedback = f"Difficulty set to {selected}"
                    transition_state("READY")
                    ready_since = time.time()
                return

            if command == "add_exercise":
                raw_name = payload.get("value")
                if raw_name is None:
                    raw_name = payload.get("exercise")
                normalized = self._normalize_exercise_name(str(raw_name or ""))
                if not normalized:
                    feedback = "Unsupported exercise name. Use squat, situp, jumping_jack, or pushup."
                    speak_once(feedback, force=True, tone="guidance")
                    return

                extra_exercises.append(normalized)
                refresh_plan()
                title = self._exercise_title(normalized)
                feedback = f"Added {title} to your plan."
                speak_once(feedback, force=True, tone="guidance")
                return

            if command == "obstacle_toggle":
                obstacle_enabled = not obstacle_enabled
                speak_once(f"Obstacle detection {'on' if obstacle_enabled else 'off'}.", tone="guidance")
                return
            if command == "obstacle_on":
                obstacle_enabled = True
                speak_once("Obstacle detection on.", tone="guidance")
                return
            if command == "obstacle_off":
                obstacle_enabled = False
                speak_once("Obstacle detection off.", tone="guidance")
                return
            if command == "speak":
                text = str(payload.get("text", "")).strip()
                if text:
                    speak_once(text, force=True, tone="guidance")
                return

        try:
            cameras.start()
            obstacle_worker.start()
            if pi_watchdog is not None:
                pi_watchdog.start()
            self._set_state({"runtime": "running"})
            welcome_text = "Welcome to ANTARDRISHTI. Show one palm to start and two palms to stop."
            feedback = welcome_text
            speak_once(welcome_text, force=True, tone="guidance")

            while not self._stop_event.is_set():
                now_ts = time.time()
                frame_idx += 1

                for _ in range(4):
                    try:
                        cmd, payload = self._command_queue.get_nowait()
                    except queue.Empty:
                        break
                    apply_control(cmd, payload)

                laptop_frame, pi_frame = cameras.get_frames()
                camera_status = cameras.status()
                if pi_watchdog is not None:
                    pi_watchdog.set_pi_alive(bool(camera_status.get("pi_alive", False)))
                pi_annotated = pi_frame.copy() if pi_frame is not None else None

                if laptop_frame is not None:
                    ok, enc = cv2.imencode(".jpg", laptop_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 78])
                    if ok:
                        with self._frame_lock:
                            self._latest_laptop_jpeg = enc.tobytes()

                if obstacle_enabled and pi_frame is not None:
                    obstacle_worker.submit(pi_frame, now_ts)
                    obstacle_meta = obstacle_worker.result()
                    obstacle_present = bool(obstacle_meta.get("present", False))
                else:
                    obstacle_present = False
                    obstacle_meta = {
                        "mode": obstacle.mode,
                        "present": False,
                        "alert": False,
                        "count": 0,
                        "max_area_ratio": 0.0,
                        "distance_m": 0.0,
                        "summary": "disabled",
                        "boxes": [],
                    }

                self._draw_people_on_pi(pi_annotated, obstacle_meta.get("boxes", []))
                if pi_annotated is not None:
                    ok, enc = cv2.imencode(".jpg", pi_annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 78])
                    if ok:
                        with self._frame_lock:
                            self._latest_pi_jpeg = enc.tobytes()

                laptop_pose = pose_laptop.extract(laptop_frame) if laptop_frame is not None else None
                pi_pose = pose_pi.extract(pi_frame) if pi_frame is not None else None

                selected_cam, pose_data, pose_strategy = self._select_pose_source(
                    session_state=interaction.state,
                    laptop_frame=laptop_frame,
                    pi_frame=pi_frame,
                    laptop_pose=laptop_pose,
                    pi_pose=pi_pose,
                )

                if not pose_data.get("present"):
                    if camera_issue_ts == 0.0:
                        camera_issue_ts = now_ts
                    if (now_ts - camera_issue_ts) > 1.0:
                        camera_msg = coach.safety_message(now_ts, "camera_lost", force=False)
                        if camera_msg:
                            speak_once(camera_msg, tone="safety")
                else:
                    camera_issue_ts = 0.0

                activity_info = activity_detector.detect(pose_data)
                activity = activity_info["activity"]
                metrics_raw = activity_info["metrics"]
                metrics = smoother.update(metrics_raw)
                joint_angles = extract_joint_angles(pose_data)
                metrics["situp_signal"] = self._situp_signal(joint_angles, metrics)
                metrics["jumping_jack_open_score"] = self._jumping_jack_open_score(pose_data)
                exercise.calibrate_squat(activity=activity, knee_angle=metrics.get("knee_angle"))

                ai_info = ai_assist.infer(metrics)
                if interaction.state in ("WORKOUT_SQUATS", "WORKOUT_PUSHUPS") and ai_info.get("trusted") and ai_info["label"] in ("standing", "squat", "pushup"):
                    activity = ai_info["label"]

                error_details_now = classify_posture_errors(
                    activity,
                    metrics,
                    strictness=exercise.settings["strictness"],
                )
                errors_now = [str(item.get("code", "")) for item in error_details_now if item.get("code")]
                for key in list(error_window.history.keys()) + errors_now:
                    error_window.update(key, key in errors_now, now_ts)
                persistent_errors = [
                    e for e in error_window.history.keys()
                    if error_window.persistent(e, min_ratio=0.7, min_duration_s=1.0)
                ]

                error_details_map = {
                    str(item.get("code", "")): {
                        "code": str(item.get("code", "")),
                        "severity": str(item.get("severity", "medium")),
                        "deviation": float(item.get("deviation", 0.0)),
                    }
                    for item in error_details_now
                    if item.get("code")
                }
                persistent_error_details = [
                    error_details_map[code]
                    for code in persistent_errors
                    if code in error_details_map
                ]

                data_collector.add_sample(
                    now_ts=now_ts,
                    state=interaction.state,
                    activity=activity,
                    camera=selected_cam,
                    pose_quality=self._safe_pose_quality(pose_data),
                    metrics=metrics,
                    joint_angles=joint_angles,
                    error_codes=errors_now,
                )

                gestures = (
                    gesture.detect(laptop_frame)
                    if laptop_frame is not None
                    else {"control": {"gesture": None}, "selection": {"gesture": None}}
                )
                control_gesture = gestures.get("control", {}).get("gesture")
                selection_gesture = gestures.get("selection", {}).get("gesture")

                if control_gesture == "stop":
                    if interaction.state in (
                        "WAIT_DIFFICULTY",
                        "READY",
                        "WORKOUT_SQUATS",
                        "WORKOUT_SITUPS",
                        "WORKOUT_JUMPING_JACKS",
                        "WORKOUT_PUSHUPS",
                        "BREAK",
                        "PAUSED",
                    ) and guard.check("stop", now_ts, lock_override_s=2.4):
                        gesture_event = "stop"
                        gesture_event_ts = now_ts
                        apply_control("stop", {})
                elif control_gesture == "start":
                    if interaction.state in ("WAIT_START_GESTURE", "END") and guard.check("start", now_ts, lock_override_s=1.2):
                        gesture_event = "start"
                        gesture_event_ts = now_ts
                        apply_control("start", {})
                    elif interaction.state in (
                        "WORKOUT_SQUATS",
                        "WORKOUT_SITUPS",
                        "WORKOUT_JUMPING_JACKS",
                        "WORKOUT_PUSHUPS",
                        "BREAK",
                    ) and guard.check("next", now_ts, lock_override_s=1.6):
                        # A single four-finger palm is reused as next-exercise gesture during workouts.
                        gesture_event = "next"
                        gesture_event_ts = now_ts
                        apply_control("next", {})
                elif control_gesture == "next":
                    if interaction.state in (
                        "WORKOUT_SQUATS",
                        "WORKOUT_SITUPS",
                        "WORKOUT_JUMPING_JACKS",
                        "WORKOUT_PUSHUPS",
                        "BREAK",
                    ) and guard.check("next", now_ts, lock_override_s=1.6):
                        gesture_event = "next"
                        gesture_event_ts = now_ts
                        apply_control("next", {})

                if (
                    selection_gesture
                    and selection_gesture.startswith("difficulty_")
                    and interaction.state == "WAIT_DIFFICULTY"
                    and guard.check(selection_gesture, now_ts, lock_override_s=1.0)
                ):
                    gesture_event = selection_gesture
                    gesture_event_ts = now_ts
                    apply_control(selection_gesture, {})

                user_present = bool(pose_data.get("present"))
                user_window.update("user_present", user_present, now_ts)
                if interaction.state == "INIT":
                    transition_state("WAIT_USER", force=True)

                if interaction.state == "WAIT_USER" and user_window.persistent("user_present", min_ratio=0.7, min_duration_s=1.0):
                    transition_state("WAIT_START_GESTURE")

                if interaction.state == "READY" and ready_since > 0.0 and (now_ts - ready_since) >= 1.0 and user_present:
                    if exercise_plan:
                        start_exercise(exercise_plan[0], plan_index=0)

                obstacle_alert = bool(obstacle_meta.get("alert", False))
                if obstacle_alert:
                    if interaction.state != "PAUSED":
                        if interaction.pause():
                            paused = True
                    safety_msg = coach.safety_message(now_ts, "obstacle", force=True)
                    if safety_msg:
                        feedback = safety_msg
                        speak_once(safety_msg, force=True, tone="safety")

                rep_info = {
                    "exercise": "none",
                    "rep_count": 0,
                    "target": 0,
                    "stage": "none",
                    "event": None,
                }

                if interaction.state == "WORKOUT_SQUATS" and not paused:
                    rep_result = exercise.squat_counter.update(metrics.get("knee_angle"))
                    rep_info = {"exercise": "squat", "target": exercise.target_for("squat"), **rep_result}
                    if rep_info["rep_count"] >= rep_info["target"] > 0:
                        advance_plan_after_current()

                if interaction.state == "WORKOUT_SITUPS" and not paused:
                    rep_result = exercise.situp_counter.update(metrics.get("situp_signal"))
                    rep_info = {"exercise": "situp", "target": exercise.target_for("situp"), **rep_result}
                    if rep_info["rep_count"] >= rep_info["target"] > 0:
                        advance_plan_after_current()

                if interaction.state == "WORKOUT_JUMPING_JACKS" and not paused:
                    rep_result = exercise.jumping_jack_counter.update(metrics.get("jumping_jack_open_score"))
                    rep_info = {"exercise": "jumping_jack", "target": exercise.target_for("jumping_jack"), **rep_result}
                    if rep_info["rep_count"] >= rep_info["target"] > 0:
                        advance_plan_after_current()

                if interaction.state == "WORKOUT_PUSHUPS" and not paused:
                    rep_result = exercise.pushup_counter.update(metrics.get("elbow_angle"))
                    rep_info = {"exercise": "pushup", "target": exercise.target_for("pushup"), **rep_result}
                    if rep_info["rep_count"] >= rep_info["target"] > 0:
                        advance_plan_after_current()

                if interaction.state == "BREAK":
                    if break_end_ts and now_ts >= break_end_ts and next_exercise_pending:
                        start_exercise(next_exercise_pending, plan_index=current_plan_index + 1)

                flow_hint = self._flow_hint(interaction.state)
                if (
                    interaction.state in ("WAIT_USER", "WAIT_START_GESTURE", "WAIT_DIFFICULTY", "READY", "PAUSED")
                    and now_ts >= feedback_pin_until
                ):
                    feedback = flow_hint

                if interaction.state in ("WORKOUT_SQUATS", "WORKOUT_PUSHUPS") and not paused and persistent_error_details:
                    severity_order = {"low": 1, "medium": 2, "high": 3}
                    priority = {
                        "body_not_straight": 0,
                        "back_tilt": 1,
                        "arms_not_bending_enough": 2,
                        "not_enough_depth": 3,
                    }
                    sorted_details = sorted(
                        persistent_error_details,
                        key=lambda item: (
                            -severity_order.get(str(item.get("severity", "medium")), 2),
                            priority.get(str(item.get("code", "")), 99),
                            -float(item.get("deviation", 0.0)),
                        ),
                    )
                    primary_error = sorted_details[0]
                    current_ex = "pushup" if interaction.state == "WORKOUT_PUSHUPS" else "squat"
                    coach_msg = coach.error_message(
                        now_ts=now_ts,
                        exercise=current_ex,
                        error_code=str(primary_error.get("code", "")),
                        severity=str(primary_error.get("severity", "medium")),
                    )
                    if coach_msg:
                        feedback = coach_msg
                        speak_once(
                            coach_msg,
                            tone="corrective",
                            exercise_name=current_ex,
                            severity=str(primary_error.get("severity", "medium")),
                        )

                if interaction.state == "END" and not summary_spoken:
                    summary = coach.end_message(now_ts, stopped_by_user=ended_by_stop)
                    if summary:
                        feedback = summary
                        speak_once(summary, force=True, tone="guidance")
                    summary_spoken = True

                score_info = scorer.update(
                    exercise=rep_info["exercise"],
                    activity=activity,
                    metrics=metrics,
                    rep_event=rep_info.get("event"),
                    errors=persistent_errors,
                    rep_count=rep_info["rep_count"],
                    target=rep_info["target"],
                )

                if selected_cam == "laptop" and laptop_frame is not None:
                    annotated = laptop_frame.copy()
                elif selected_cam == "pi" and pi_annotated is not None:
                    annotated = pi_annotated.copy()
                elif laptop_frame is not None:
                    annotated = laptop_frame.copy()
                elif pi_annotated is not None:
                    annotated = pi_annotated.copy()
                else:
                    annotated = np.zeros((360, 640, 3), dtype=np.uint8)
                if pose_data.get("raw"):
                    (pose_laptop if selected_cam == "laptop" else pose_pi).draw(annotated, pose_data.get("raw"))

                self._draw_overlay(
                    frame=annotated,
                    state=interaction.state,
                    activity=activity,
                    rep_count=rep_info["rep_count"],
                    target=rep_info["target"],
                    difficulty=exercise.difficulty,
                    feedback=feedback,
                    errors=persistent_errors,
                    obstacle=obstacle_present,
                    score=score_info.get("skill_score", 0.0),
                    camera_name=selected_cam,
                    pose_strategy=pose_strategy,
                    laptop_quality=self._safe_pose_quality(laptop_pose),
                    pi_quality=self._safe_pose_quality(pi_pose),
                    obstacle_enabled=obstacle_enabled,
                    obstacle_mode=str(obstacle_meta.get("mode", obstacle.mode)),
                    obstacle_count=int(obstacle_meta.get("count", 0)),
                    obstacle_distance_m=float(obstacle_meta.get("distance_m", 0.0)),
                )

                ok, encoded = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
                if ok:
                    with self._frame_lock:
                        self._latest_frame_jpeg = encoded.tobytes()

                sync_active = bool(now_ts < feedback_pin_until)

                state_update = {
                    "runtime": "running",
                    "state": interaction.state,
                    "activity": activity,
                    "exercise": rep_info["exercise"],
                    "rep_count": rep_info["rep_count"],
                    "target": rep_info["target"],
                    "difficulty": exercise.difficulty,
                    "feedback": feedback,
                    "next_action": flow_hint,
                    "spoken_feedback": last_spoken_feedback if sync_active else "",
                    "feedback_sync_active": sync_active,
                    "exercise_plan": [self._exercise_title(item) for item in exercise_plan],
                    "current_exercise": current_exercise or self._exercise_for_state(interaction.state),
                    "next_exercise": next_exercise_pending or "",
                    "break_remaining_s": max(0, int(round(break_end_ts - now_ts))) if interaction.state == "BREAK" and break_end_ts else 0,
                    "camera": selected_cam,
                    "obstacle": obstacle_present,
                    "obstacle_enabled": obstacle_enabled,
                    "obstacle_mode": obstacle_meta.get("mode", obstacle.mode),
                    "obstacle_count": int(obstacle_meta.get("count", 0)),
                    "obstacle_area": round(float(obstacle_meta.get("max_area_ratio", 0.0)) * 100.0, 2),
                    "obstacle_distance_m": float(obstacle_meta.get("distance_m", 0.0)),
                    "people": [
                        {
                            "label": str(item.get("label", "person")),
                            "distance_m": float(item.get("distance_m", 0.0)),
                            "conf": float(item.get("conf", 0.0)),
                            "xyxy": item.get("xyxy", [0, 0, 0, 0]),
                        }
                        for item in obstacle_meta.get("boxes", [])
                    ],
                    "gesture": {"control": control_gesture, "selection": selection_gesture},
                    "gesture_event": gesture_event if (now_ts - gesture_event_ts) <= 1.8 else "",
                    "gesture_debug": gestures.get("raw", [])[:2],
                    "errors": persistent_errors,
                    "error_details": persistent_error_details,
                    "metrics": {k: round(v, 3) if isinstance(v, float) else v for k, v in metrics.items()},
                    "joint_angles": {
                        key: (round(float(val), 3) if val is not None else None)
                        for key, val in joint_angles.items()
                    },
                    "score": score_info,
                    "ai": ai_info,
                    "voice": voice.status(),
                    "camera_status": camera_status,
                    "pi_control": pi_watchdog.status() if pi_watchdog is not None else {
                        "enabled": False,
                        "ok": False,
                        "message": "not requested",
                        "attempts": 0,
                        "last_attempt_ts": 0.0,
                        "last_success_ts": 0.0,
                    },
                    "pose_sources": {
                        "selected": selected_cam,
                        "strategy": pose_strategy,
                        "laptop_quality": round(self._safe_pose_quality(laptop_pose), 3),
                        "pi_quality": round(self._safe_pose_quality(pi_pose), 3),
                    },
                    "calibration": exercise.calibration_status(),
                    "training_data": {
                        "enabled": self.config.collect_training_data,
                        "path": data_collector.file_path,
                        "total_samples": data_collector.sample_count,
                        "buffered_samples": data_collector.buffered_count,
                    },
                }
                self._set_state(state_update)

                if now_ts - last_state_write_ts >= state_write_interval:
                    write_state(self.get_state(), path=self.config.state_file)
                    last_state_write_ts = now_ts

                time.sleep(0.02)

        except Exception as exc:  # noqa: BLE001
            self._set_state({"runtime": "error", "feedback": f"Runtime error: {exc}"})
        finally:
            self._set_state({"runtime": "stopping"})
            cameras.stop()
            pose_laptop.close()
            pose_pi.close()
            gesture.close()
            voice.close()
            obstacle_worker.close()
            data_collector.close()
            if pi_watchdog is not None:
                pi_watchdog.close()
            terminal_state = self.get_state()
            terminal_feedback = terminal_state.get("feedback", "Runtime stopped")
            if terminal_state.get("runtime") != "error":
                terminal_feedback = "Runtime stopped"
            self._set_state({"runtime": "stopped", "feedback": terminal_feedback})

    def _draw_overlay(
        self,
        frame,
        state: str,
        activity: str,
        rep_count: int,
        target: int,
        difficulty: str,
        feedback: str,
        errors,
        obstacle: bool,
        score: float,
        camera_name: str,
        pose_strategy: str,
        laptop_quality: float,
        pi_quality: float,
        obstacle_enabled: bool,
        obstacle_mode: str,
        obstacle_count: int,
        obstacle_distance_m: float,
    ) -> None:
        color = (255, 255, 255)
        cv2.putText(frame, f"State: {state}", (18, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, f"Activity: {activity}", (18, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, f"Reps: {rep_count}/{target}", (18, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, f"Difficulty: {difficulty}", (18, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, f"Camera: {camera_name}", (18, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
        cv2.putText(frame, f"Fusion: {pose_strategy}", (18, 164), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(
            frame,
            f"Q(L/P): {laptop_quality:.2f}/{pi_quality:.2f}",
            (18, 188),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )
        cv2.putText(
            frame,
            f"Obstacle({obstacle_mode}): {'ON' if obstacle_enabled else 'OFF'} count={obstacle_count}",
            (18, 212),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            color,
            2,
        )
        if obstacle_distance_m:
            cv2.putText(
                frame,
                f"Nearest dist: {obstacle_distance_m:.2f}m",
                (18, 236),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                color,
                2,
            )

        if obstacle:
            cv2.putText(
                frame,
                "Pi camera: person detected. Distance shown on Pi feed.",
                (18, 260),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (0, 180, 255),
                2,
            )

        if errors:
            cv2.putText(
                frame,
                f"Errors: {', '.join(errors[:2])}",
                (18, 284),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (0, 200, 255),
                2,
            )

        safe_feedback = (feedback or "")[:78]
        cv2.putText(
            frame,
            f"Coach: {safe_feedback}",
            (18, frame.shape[0] - 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )
