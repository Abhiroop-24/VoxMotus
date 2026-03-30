"""Deterministic joint-angle mapping utilities for pose data."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

from pose import calculate_angle, midpoint


Point = Tuple[float, float, float, float]


ANGLE_TRIPLETS = {
    "left_knee_angle": ("left_hip", "left_knee", "left_ankle"),
    "right_knee_angle": ("right_hip", "right_knee", "right_ankle"),
    "left_elbow_angle": ("left_shoulder", "left_elbow", "left_wrist"),
    "right_elbow_angle": ("right_shoulder", "right_elbow", "right_wrist"),
    "left_hip_angle": ("left_shoulder", "left_hip", "left_knee"),
    "right_hip_angle": ("right_shoulder", "right_hip", "right_knee"),
    "left_shoulder_angle": ("left_elbow", "left_shoulder", "left_hip"),
    "right_shoulder_angle": ("right_elbow", "right_shoulder", "right_hip"),
}

ANGLE_KEYS = [
    "left_knee_angle",
    "right_knee_angle",
    "mean_knee_angle",
    "left_elbow_angle",
    "right_elbow_angle",
    "mean_elbow_angle",
    "left_hip_angle",
    "right_hip_angle",
    "left_shoulder_angle",
    "right_shoulder_angle",
    "torso_line_angle",
]


def _point(pose_data: Dict, key: str) -> Optional[Point]:
    value = pose_data.get(key)
    if not value:
        return None
    if len(value) < 4:
        return None
    return (
        float(value[0]),
        float(value[1]),
        float(value[2]),
        float(value[3]),
    )


def _safe_angle(a: Optional[Point], b: Optional[Point], c: Optional[Point]) -> Optional[float]:
    if not (a and b and c):
        return None
    return float(calculate_angle(a, b, c))


def extract_joint_angles(pose_data: Dict) -> Dict[str, Optional[float]]:
    """Extract a stable set of joint angles from pose landmarks."""
    if not pose_data or not pose_data.get("present"):
        return {key: None for key in ANGLE_KEYS}

    out: Dict[str, Optional[float]] = {}

    for name, (a_key, b_key, c_key) in ANGLE_TRIPLETS.items():
        out[name] = _safe_angle(_point(pose_data, a_key), _point(pose_data, b_key), _point(pose_data, c_key))

    lk = out.get("left_knee_angle")
    rk = out.get("right_knee_angle")
    out["mean_knee_angle"] = ((lk + rk) / 2.0) if (lk is not None and rk is not None) else (lk if lk is not None else rk)

    le = out.get("left_elbow_angle")
    re = out.get("right_elbow_angle")
    out["mean_elbow_angle"] = ((le + re) / 2.0) if (le is not None and re is not None) else (le if le is not None else re)

    ls = _point(pose_data, "left_shoulder")
    rs = _point(pose_data, "right_shoulder")
    lh = _point(pose_data, "left_hip")
    rh = _point(pose_data, "right_hip")
    la = _point(pose_data, "left_ankle")
    ra = _point(pose_data, "right_ankle")

    if ls and rs and lh and rh and la and ra:
        shoulder_mid = midpoint(ls, rs)
        hip_mid = midpoint(lh, rh)
        ankle_mid = midpoint(la, ra)
        torso_ref = (hip_mid[0], hip_mid[1] - 0.15, 0.0, 1.0)
        out["torso_line_angle"] = float(calculate_angle(torso_ref, hip_mid, shoulder_mid))
    else:
        out["torso_line_angle"] = None

    return out
