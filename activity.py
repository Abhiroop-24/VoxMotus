"""Activity understanding from pose landmarks."""

from __future__ import annotations

from typing import Dict, List

from pose import calculate_angle, midpoint


class ActivityDetector:
    def __init__(self) -> None:
        self.last_activity = "idle"

    def detect(self, pose_data: Dict) -> Dict:
        if not pose_data.get("present"):
            self.last_activity = "idle"
            return {
                "activity": "idle",
                "metrics": {},
                "errors": [],
            }

        left_shoulder = pose_data["left_shoulder"]
        right_shoulder = pose_data["right_shoulder"]
        left_hip = pose_data["left_hip"]
        right_hip = pose_data["right_hip"]
        left_knee = pose_data["left_knee"]
        right_knee = pose_data["right_knee"]
        left_ankle = pose_data["left_ankle"]
        right_ankle = pose_data["right_ankle"]
        left_elbow = pose_data["left_elbow"]
        right_elbow = pose_data["right_elbow"]
        left_wrist = pose_data["left_wrist"]
        right_wrist = pose_data["right_wrist"]

        knee_left = calculate_angle(left_hip, left_knee, left_ankle)
        knee_right = calculate_angle(right_hip, right_knee, right_ankle)
        knee_angle = (knee_left + knee_right) / 2.0

        elbow_left = calculate_angle(left_shoulder, left_elbow, left_wrist)
        elbow_right = calculate_angle(right_shoulder, right_elbow, right_wrist)
        elbow_angle = (elbow_left + elbow_right) / 2.0

        shoulder_mid = midpoint(left_shoulder, right_shoulder)
        hip_mid = midpoint(left_hip, right_hip)
        ankle_mid = midpoint(left_ankle, right_ankle)

        torso_tilt = abs(calculate_angle((hip_mid[0], hip_mid[1] - 0.15, 0, 1), hip_mid, shoulder_mid) - 180.0)
        body_line = calculate_angle(shoulder_mid, hip_mid, ankle_mid)

        shoulder_hip_diff = abs(shoulder_mid[1] - hip_mid[1])
        body_horizontal = shoulder_hip_diff < 0.08 and abs(ankle_mid[1] - shoulder_mid[1]) < 0.2

        activity = "standing"

        if body_horizontal and elbow_angle < 175:
            activity = "pushup"
        elif knee_angle < 145:
            activity = "squat"
        elif 145 <= knee_angle <= 160:
            activity = "transition"
        else:
            activity = "standing"

        if activity == self.last_activity:
            smoothed = activity
        else:
            if self.last_activity == "idle" and activity != "idle":
                smoothed = activity
            else:
                smoothed = "transition"
        self.last_activity = smoothed

        metrics = {
            "knee_angle": knee_angle,
            "elbow_angle": elbow_angle,
            "torso_tilt": torso_tilt,
            "body_line": body_line,
            "body_horizontal": body_horizontal,
            "shoulder_hip_diff": shoulder_hip_diff,
        }

        return {
            "activity": smoothed,
            "metrics": metrics,
            "errors": [],
        }


def _severity_from_delta(delta: float) -> str:
    if delta <= 3.0:
        return "low"
    if delta <= 8.0:
        return "medium"
    return "high"


def classify_posture_errors(activity: str, metrics: Dict, strictness: float = 1.0) -> List[Dict[str, float | str]]:
    """Return structured error objects with deterministic severity and deviation."""
    details: List[Dict[str, float | str]] = []

    if not metrics:
        return details

    if activity in ("squat", "transition"):
        max_torso_tilt = 24.0 / strictness
        min_depth_knee_angle = 125.0 * strictness

        torso_tilt = float(metrics.get("torso_tilt", 0.0))
        knee_angle = float(metrics.get("knee_angle", 180.0))

        if torso_tilt > max_torso_tilt:
            delta = torso_tilt - max_torso_tilt
            details.append(
                {
                    "code": "back_tilt",
                    "severity": _severity_from_delta(delta),
                    "deviation": round(delta, 3),
                }
            )

        if activity == "squat" and knee_angle > min_depth_knee_angle:
            delta = knee_angle - min_depth_knee_angle
            details.append(
                {
                    "code": "not_enough_depth",
                    "severity": _severity_from_delta(delta),
                    "deviation": round(delta, 3),
                }
            )

    if activity == "pushup":
        body_line = float(metrics.get("body_line", 180.0))
        elbow_angle = float(metrics.get("elbow_angle", 0.0))

        body_line_threshold = 160.0 * strictness
        elbow_threshold = 100.0 * strictness

        if body_line < body_line_threshold:
            delta = body_line_threshold - body_line
            details.append(
                {
                    "code": "body_not_straight",
                    "severity": _severity_from_delta(delta),
                    "deviation": round(delta, 3),
                }
            )

        if elbow_angle > elbow_threshold:
            delta = elbow_angle - elbow_threshold
            details.append(
                {
                    "code": "arms_not_bending_enough",
                    "severity": _severity_from_delta(delta),
                    "deviation": round(delta, 3),
                }
            )

    return details


def detect_posture_errors(activity: str, metrics: Dict, strictness: float = 1.0) -> List[str]:
    """Backward-compatible helper that returns only error codes."""
    return [item["code"] for item in classify_posture_errors(activity, metrics, strictness=strictness)]
