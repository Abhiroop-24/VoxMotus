"""Train a deterministic multi-class activity classifier from collected angle datasets."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import time
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np


CLASS_NAMES = ["idle", "standing", "squat", "pushup", "transition"]
FEATURE_NAMES = ["knee_angle", "elbow_angle", "torso_tilt", "body_horizontal", "shoulder_hip_diff"]
CLASS_TO_INDEX = {name: i for i, name in enumerate(CLASS_NAMES)}

# Keep default priors aligned with ai_assist.py so missing classes stay stable.
DEFAULT_W = np.array(
    [
        [-0.6, -0.3, -0.5, -0.4, -0.3],
        [0.2, 0.1, 0.1, -0.2, -0.1],
        [-1.4, -0.1, -0.6, -0.2, -0.2],
        [0.1, -1.3, -0.2, 1.2, -0.3],
        [-0.2, -0.2, 0.4, -0.1, 0.8],
    ],
    dtype=np.float64,
)
DEFAULT_B = np.array([0.6, 1.1, 0.7, 0.2, 0.1], dtype=np.float64)

# MediaPipe Pose landmark indices (0-based, dataset columns are x1..x33).
LANDMARK_INDEX = {
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
}

# Deterministic mapping from external exercise labels to runtime activity classes.
LANDMARK_LABEL_MAP = {
    "idle": "idle",
    "rest": "idle",
    "standing": "standing",
    "stand": "standing",
    "left_bicep": "standing",
    "right_bicep": "standing",
    "left_tricep": "standing",
    "right_tricep": "standing",
    "left_shoulder": "standing",
    "right_shoulder": "standing",
    "squat": "squat",
    "squats": "squat",
    "pushup": "pushup",
    "pushups": "pushup",
    "push_up": "pushup",
    "push-ups": "pushup",
    "transition": "transition",
    "transitions": "transition",
}


def _safe_float(value: str) -> float:
    try:
        x = float(value)
        if math.isnan(x) or math.isinf(x):
            return float("nan")
        return x
    except Exception:
        return float("nan")


def _calculate_angle(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float], c: Tuple[float, float, float, float]) -> float:
    ax, ay = a[:2]
    bx, by = b[:2]
    cx, cy = c[:2]
    radians = math.atan2(cy - by, cx - bx) - math.atan2(ay - by, ax - bx)
    angle = abs(math.degrees(radians))
    if angle > 180.0:
        angle = 360.0 - angle
    return angle


def _midpoint(p1: Tuple[float, float, float, float], p2: Tuple[float, float, float, float]) -> Tuple[float, float]:
    return ((p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0)


def _normalize_label(value: str) -> str:
    return str(value or "").strip().lower().replace(" ", "_")


def _map_landmark_label(raw_label: str) -> str:
    normalized = _normalize_label(raw_label)
    if not normalized:
        return ""
    if normalized in LANDMARK_LABEL_MAP:
        return LANDMARK_LABEL_MAP[normalized]

    # Heuristic fallback is deterministic and intentionally conservative.
    if "squat" in normalized:
        return "squat"
    if "push" in normalized:
        return "pushup"
    if "idle" in normalized or "rest" in normalized:
        return "idle"
    if "transition" in normalized:
        return "transition"
    if (
        "stand" in normalized
        or "bicep" in normalized
        or "tricep" in normalized
        or "shoulder" in normalized
        or "curl" in normalized
        or "raise" in normalized
        or "press" in normalized
    ):
        return "standing"
    return ""


def _landmark_point(row: Dict[str, str], idx0: int) -> Optional[Tuple[float, float, float, float]]:
    idx = int(idx0) + 1
    x = _safe_float(row.get(f"x{idx}", "nan"))
    y = _safe_float(row.get(f"y{idx}", "nan"))
    z = _safe_float(row.get(f"z{idx}", "nan"))
    v = _safe_float(row.get(f"v{idx}", "nan"))
    if not all(np.isfinite(value) for value in (x, y, z, v)):
        return None
    return (float(x), float(y), float(z), float(v))


def _metrics_from_landmark_row(row: Dict[str, str]) -> Optional[Dict[str, float]]:
    points: Dict[str, Tuple[float, float, float, float]] = {}
    for key, idx in LANDMARK_INDEX.items():
        p = _landmark_point(row, idx)
        if p is None:
            return None
        points[key] = p

    knee_left = _calculate_angle(points["left_hip"], points["left_knee"], points["left_ankle"])
    knee_right = _calculate_angle(points["right_hip"], points["right_knee"], points["right_ankle"])
    knee_angle = (knee_left + knee_right) / 2.0

    elbow_left = _calculate_angle(points["left_shoulder"], points["left_elbow"], points["left_wrist"])
    elbow_right = _calculate_angle(points["right_shoulder"], points["right_elbow"], points["right_wrist"])
    elbow_angle = (elbow_left + elbow_right) / 2.0

    shoulder_mid = _midpoint(points["left_shoulder"], points["right_shoulder"])
    hip_mid = _midpoint(points["left_hip"], points["right_hip"])
    ankle_mid = _midpoint(points["left_ankle"], points["right_ankle"])

    torso_ref = (hip_mid[0], hip_mid[1] - 0.15, 0.0, 1.0)
    torso_tilt = abs(_calculate_angle(torso_ref, (hip_mid[0], hip_mid[1], 0.0, 1.0), (shoulder_mid[0], shoulder_mid[1], 0.0, 1.0)) - 180.0)
    body_line = _calculate_angle(
        (shoulder_mid[0], shoulder_mid[1], 0.0, 1.0),
        (hip_mid[0], hip_mid[1], 0.0, 1.0),
        (ankle_mid[0], ankle_mid[1], 0.0, 1.0),
    )

    shoulder_hip_diff = abs(shoulder_mid[1] - hip_mid[1])
    body_horizontal = shoulder_hip_diff < 0.08 and abs(ankle_mid[1] - shoulder_mid[1]) < 0.2

    return {
        "knee_angle": float(knee_angle),
        "elbow_angle": float(elbow_angle),
        "torso_tilt": float(torso_tilt),
        "body_line": float(body_line),
        "body_horizontal": float(1.0 if body_horizontal else 0.0),
        "shoulder_hip_diff": float(shoulder_hip_diff),
    }


def _schema_from_header(fieldnames: List[str]) -> str:
    fields = {str(x).strip().lower() for x in fieldnames}
    if {"activity", "knee_angle", "elbow_angle", "torso_tilt", "body_horizontal", "shoulder_hip_diff"}.issubset(fields):
        return "angles"
    if {"class", "x1", "y1", "z1", "v1", "x33", "y33", "z33", "v33"}.issubset(fields):
        return "landmarks"
    return "unknown"


def _normalized_features(metrics: Dict[str, object]) -> np.ndarray:
    knee = _safe_float(str(metrics.get("knee_angle", "nan")))
    elbow = _safe_float(str(metrics.get("elbow_angle", "nan")))
    torso = _safe_float(str(metrics.get("torso_tilt", "nan")))
    body_horizontal = _safe_float(str(metrics.get("body_horizontal", "nan")))
    shoulder_hip_diff = _safe_float(str(metrics.get("shoulder_hip_diff", "nan")))

    vec = np.array(
        [
            (knee - 120.0) / 40.0,
            (elbow - 120.0) / 40.0,
            -torso / 30.0,
            body_horizontal,
            -shoulder_hip_diff / 0.1,
        ],
        dtype=np.float64,
    )
    return vec


def load_dataset(input_glob: str, strict_unmapped: bool = False) -> Tuple[np.ndarray, np.ndarray, List[str], Counter, Dict]:
    files = sorted(glob.glob(input_glob))
    if not files:
        raise FileNotFoundError(f"No dataset files match: {input_glob}")

    X_rows: List[np.ndarray] = []
    y_rows: List[int] = []
    source_files: List[str] = []
    per_class = Counter()
    schema_per_file: Dict[str, str] = {}
    raw_landmark_labels = Counter()
    mapped_landmark_labels = Counter()
    skipped_unmapped = Counter()
    skipped_invalid = 0
    total_rows = 0

    for file_path in files:
        with open(file_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            header = [str(x) for x in (reader.fieldnames or [])]
            schema = _schema_from_header(header)
            schema_per_file[file_path] = schema
            if schema == "unknown":
                raise RuntimeError(
                    "Unsupported dataset schema in "
                    f"{file_path}. Expected angle schema with 'activity' or "
                    "landmark schema with 'class' and x1..v33 columns."
                )

            for row in reader:
                total_rows += 1

                if schema == "angles":
                    label = _normalize_label(row.get("activity", ""))
                    metrics = row
                else:
                    raw_label = _normalize_label(row.get("class", ""))
                    raw_landmark_labels[raw_label] += 1
                    label = _map_landmark_label(raw_label)
                    metrics = _metrics_from_landmark_row(row)

                if label not in CLASS_TO_INDEX:
                    skipped_unmapped[label or "<empty>"] += 1
                    if strict_unmapped:
                        raise RuntimeError(f"Unmapped label '{label}' in {file_path}")
                    continue

                if schema == "landmarks":
                    mapped_landmark_labels[label] += 1

                if not metrics:
                    skipped_invalid += 1
                    continue

                x = _normalized_features(metrics)
                if np.any(np.isnan(x)) or np.any(np.isinf(x)):
                    skipped_invalid += 1
                    continue

                X_rows.append(x)
                y_idx = CLASS_TO_INDEX[label]
                y_rows.append(y_idx)
                per_class[label] += 1
        source_files.append(file_path)

    if not X_rows:
        raise RuntimeError("No valid samples found after parsing.")

    X = np.vstack(X_rows)
    y = np.array(y_rows, dtype=np.int64)
    parse_report = {
        "rows_total": int(total_rows),
        "rows_used": int(len(X_rows)),
        "rows_skipped_invalid": int(skipped_invalid),
        "rows_skipped_unmapped": int(sum(skipped_unmapped.values())),
        "skipped_unmapped_breakdown": dict(skipped_unmapped),
        "raw_landmark_labels": dict(raw_landmark_labels),
        "mapped_landmark_labels": dict(mapped_landmark_labels),
        "schema_per_file": schema_per_file,
    }
    return X, y, source_files, per_class, parse_report


def deterministic_split(X: np.ndarray, y: np.ndarray, train_ratio: float = 0.8) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_indices: List[int] = []
    val_indices: List[int] = []

    for cls in range(len(CLASS_NAMES)):
        cls_idx = np.where(y == cls)[0]
        if cls_idx.size == 0:
            continue
        cutoff = max(1, int(cls_idx.size * train_ratio))
        if cutoff >= cls_idx.size:
            cutoff = cls_idx.size - 1 if cls_idx.size > 1 else cls_idx.size

        train_indices.extend(cls_idx[:cutoff].tolist())
        val_indices.extend(cls_idx[cutoff:].tolist())

    train_indices = sorted(train_indices)
    val_indices = sorted(val_indices)

    X_train = X[train_indices] if train_indices else X
    y_train = y[train_indices] if train_indices else y
    X_val = X[val_indices] if val_indices else X
    y_val = y[val_indices] if val_indices else y
    return X_train, y_train, X_val, y_val


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


def one_hot(y: np.ndarray, num_classes: int) -> np.ndarray:
    out = np.zeros((y.shape[0], num_classes), dtype=np.float64)
    out[np.arange(y.shape[0]), y] = 1.0
    return out


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int,
    lr: float,
    l2: float,
    init_W: Optional[np.ndarray] = None,
    init_b: Optional[np.ndarray] = None,
    trainable_rows: Optional[List[int]] = None,
) -> Tuple[np.ndarray, np.ndarray, List[float]]:
    num_classes = len(CLASS_NAMES)
    num_features = X_train.shape[1]

    if init_W is None:
        W = np.zeros((num_classes, num_features), dtype=np.float64)
    else:
        W = np.array(init_W, dtype=np.float64)
        if W.shape != (num_classes, num_features):
            raise ValueError(f"Invalid init_W shape {W.shape}, expected {(num_classes, num_features)}")

    if init_b is None:
        b = np.zeros((num_classes,), dtype=np.float64)
    else:
        b = np.array(init_b, dtype=np.float64)
        if b.shape != (num_classes,):
            raise ValueError(f"Invalid init_b shape {b.shape}, expected {(num_classes,)}")

    row_mask = np.ones((num_classes,), dtype=np.float64)
    if trainable_rows is not None:
        row_mask = np.zeros((num_classes,), dtype=np.float64)
        for row_idx in trainable_rows:
            if 0 <= int(row_idx) < num_classes:
                row_mask[int(row_idx)] = 1.0
        if not np.any(row_mask > 0.0):
            raise ValueError("No trainable rows selected")

    Y = one_hot(y_train, num_classes)
    history: List[float] = []

    for _ in range(epochs):
        logits = X_train @ W.T + b
        probs = softmax(logits)

        diff = (probs - Y) / float(X_train.shape[0])
        grad_W = diff.T @ X_train + (l2 * W)
        grad_b = np.sum(diff, axis=0)

        if trainable_rows is not None:
            grad_W = grad_W * row_mask[:, None]
            grad_b = grad_b * row_mask

        W -= lr * grad_W
        b -= lr * grad_b

        loss = -np.mean(np.sum(Y * np.log(np.clip(probs, 1e-12, 1.0)), axis=1))
        loss += 0.5 * l2 * np.sum(W * W)
        history.append(float(loss))

    return W.astype(np.float32), b.astype(np.float32), history


def predict(X: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    logits = X @ W.T + b
    probs = softmax(logits)
    return np.argmax(probs, axis=1)


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0:
        return 0.0
    return float(np.mean(y_true == y_pred))


def ensure_minimum_samples(per_class: Counter, min_per_class: int) -> None:
    missing = [f"{cls}:{per_class.get(cls, 0)}" for cls in CLASS_NAMES if per_class.get(cls, 0) < min_per_class]
    if missing:
        joined = ", ".join(missing)
        raise RuntimeError(
            "Insufficient labeled samples per class. "
            f"Need at least {min_per_class} for each class, got {joined}"
        )


def ensure_minimum_samples_partial(per_class: Counter, min_per_class: int) -> List[int]:
    present = [name for name in CLASS_NAMES if per_class.get(name, 0) > 0]
    if len(present) < 2:
        raise RuntimeError(
            "Partial-class training needs at least two mapped classes. "
            f"Detected: {present}"
        )

    missing = [f"{cls}:{per_class.get(cls, 0)}" for cls in present if per_class.get(cls, 0) < min_per_class]
    if missing:
        joined = ", ".join(missing)
        raise RuntimeError(
            "Insufficient samples in detected classes. "
            f"Need at least {min_per_class} for each present class, got {joined}"
        )

    return [CLASS_TO_INDEX[name] for name in present]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train deterministic activity classifier weights from captured CSV datasets.")
    parser.add_argument("--input-glob", default="outputs/datasets/training_angles_*.csv")
    parser.add_argument("--output", default="models/activity_weights.json")
    parser.add_argument("--epochs", type=int, default=700)
    parser.add_argument("--lr", type=float, default=0.22)
    parser.add_argument("--l2", type=float, default=0.001)
    parser.add_argument("--min-per-class", type=int, default=120)
    parser.add_argument("--allow-partial-classes", action="store_true", default=True)
    parser.add_argument("--require-full-classes", action="store_false", dest="allow_partial_classes")
    parser.add_argument("--strict-unmapped", action="store_true", default=False)
    args = parser.parse_args()

    X, y, files, per_class, parse_report = load_dataset(args.input_glob, strict_unmapped=args.strict_unmapped)

    if args.allow_partial_classes:
        trainable_rows = ensure_minimum_samples_partial(per_class, args.min_per_class)
        init_w = DEFAULT_W
        init_b = DEFAULT_B
    else:
        ensure_minimum_samples(per_class, args.min_per_class)
        trainable_rows = list(range(len(CLASS_NAMES)))
        init_w = None
        init_b = None

    X_train, y_train, X_val, y_val = deterministic_split(X, y, train_ratio=0.8)
    W, b, history = train_model(
        X_train,
        y_train,
        epochs=args.epochs,
        lr=args.lr,
        l2=args.l2,
        init_W=init_w,
        init_b=init_b,
        trainable_rows=trainable_rows,
    )

    train_acc = accuracy(y_train, predict(X_train, W, b))
    val_acc = accuracy(y_val, predict(X_val, W, b))

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    payload = {
        "weights": W.tolist(),
        "bias": b.tolist(),
        "class_names": CLASS_NAMES,
        "feature_names": FEATURE_NAMES,
        "normalization": {
            "knee_angle": "(knee_angle - 120.0) / 40.0",
            "elbow_angle": "(elbow_angle - 120.0) / 40.0",
            "torso_tilt": "-torso_tilt / 30.0",
            "body_horizontal": "float(body_horizontal)",
            "shoulder_hip_diff": "-shoulder_hip_diff / 0.1",
        },
        "training": {
            "timestamp": int(time.time()),
            "epochs": args.epochs,
            "learning_rate": args.lr,
            "l2": args.l2,
            "allow_partial_classes": bool(args.allow_partial_classes),
            "trainable_class_indices": [int(x) for x in trainable_rows],
            "trainable_classes": [CLASS_NAMES[int(x)] for x in trainable_rows],
            "train_samples": int(y_train.size),
            "val_samples": int(y_val.size),
            "train_accuracy": round(train_acc, 6),
            "val_accuracy": round(val_acc, 6),
            "class_counts": dict(per_class),
            "input_files": files,
            "parse_report": parse_report,
            "final_loss": round(history[-1], 8) if history else None,
        },
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)

    print(f"Saved trained weights: {args.output}")
    print(f"Train accuracy: {train_acc:.4f}")
    print(f"Validation accuracy: {val_acc:.4f}")
    print(f"Rows used: {parse_report['rows_used']} / {parse_report['rows_total']}")
    if parse_report["rows_skipped_unmapped"]:
        print(f"Skipped unmapped rows: {parse_report['rows_skipped_unmapped']}")
    if parse_report["rows_skipped_invalid"]:
        print(f"Skipped invalid rows: {parse_report['rows_skipped_invalid']}")


if __name__ == "__main__":
    main()
