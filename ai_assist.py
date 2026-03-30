"""Optional local AI helper with CUDA acceleration when torch is available."""

from __future__ import annotations

import json
import os
from typing import Dict

import numpy as np


class LocalAIAssistant:
    """Lightweight local inference helper for activity confidence estimation."""

    CLASS_NAMES = ["idle", "standing", "squat", "pushup", "transition"]

    DEFAULT_W = [
        [-0.6, -0.3, -0.5, -0.4, -0.3],
        [0.2, 0.1, 0.1, -0.2, -0.1],
        [-1.4, -0.1, -0.6, -0.2, -0.2],
        [0.1, -1.3, -0.2, 1.2, -0.3],
        [-0.2, -0.2, 0.4, -0.1, 0.8],
    ]
    DEFAULT_B = [0.6, 1.1, 0.7, 0.2, 0.1]

    def __init__(self, weights_path: str = "", min_confidence: float = 0.82) -> None:
        self.backend = "numpy"
        self.device = "cpu"
        self.cuda_available = False
        self.gpu_name = ""
        self.model_source = "default"
        self.min_confidence = float(np.clip(min_confidence, 0.5, 0.99))
        self._torch = None
        self._w = None
        self._b = None
        self._numpy_w = np.array(self.DEFAULT_W, dtype=np.float32)
        self._numpy_b = np.array(self.DEFAULT_B, dtype=np.float32)

        try:
            import torch

            self._torch = torch
            self.backend = "torch"
            self.cuda_available = bool(torch.cuda.is_available())
            self.device = "cuda:0" if self.cuda_available else "cpu"
            if self.cuda_available:
                try:
                    self.gpu_name = str(torch.cuda.get_device_name(0))
                except Exception:
                    self.gpu_name = "cuda"
        except Exception:
            self.backend = "numpy"
            self.device = "cpu"
            self.cuda_available = False
            self.gpu_name = ""

        self._load_weights(weights_path)

    def _load_weights(self, weights_path: str) -> None:
        path = (weights_path or "").strip()
        if not path or not os.path.exists(path):
            self._set_model(self._numpy_w, self._numpy_b, source="default")
            return

        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)

            weights = np.array(payload.get("weights", []), dtype=np.float32)
            bias = np.array(payload.get("bias", []), dtype=np.float32)

            expected_shape = (len(self.CLASS_NAMES), 5)
            if weights.shape != expected_shape or bias.shape != (len(self.CLASS_NAMES),):
                raise ValueError(f"Invalid weight shape {weights.shape} or bias shape {bias.shape}")

            self._set_model(weights, bias, source=path)
        except Exception:
            self._set_model(self._numpy_w, self._numpy_b, source="default")

    def _set_model(self, weights: np.ndarray, bias: np.ndarray, source: str) -> None:
        self._numpy_w = np.array(weights, dtype=np.float32)
        self._numpy_b = np.array(bias, dtype=np.float32)
        self.model_source = source
        if self.backend == "torch" and self._torch is not None:
            self._w = self._torch.tensor(self._numpy_w, dtype=self._torch.float32, device=self.device)
            self._b = self._torch.tensor(self._numpy_b, dtype=self._torch.float32, device=self.device)
        else:
            self._w = None
            self._b = None

    def _features(self, metrics: Dict) -> np.ndarray:
        if not metrics:
            return np.zeros(5, dtype=np.float32)

        knee = float(metrics.get("knee_angle", 170.0))
        elbow = float(metrics.get("elbow_angle", 170.0))
        torso_tilt = float(metrics.get("torso_tilt", 0.0))
        body_horizontal = 1.0 if metrics.get("body_horizontal", False) else 0.0
        shoulder_hip_diff = float(metrics.get("shoulder_hip_diff", 0.0))

        return np.array(
            [
                (knee - 120.0) / 40.0,
                (elbow - 120.0) / 40.0,
                -torso_tilt / 30.0,
                body_horizontal,
                -shoulder_hip_diff / 0.1,
            ],
            dtype=np.float32,
        )

    def infer(self, metrics: Dict) -> Dict:
        x = self._features(metrics)

        if self.backend == "torch" and self._torch is not None and self._w is not None:
            tx = self._torch.tensor(x, dtype=self._torch.float32, device=self.device)
            logits = self._w @ tx + self._b
            probs = self._torch.softmax(logits, dim=0)
            probs_np = probs.detach().cpu().numpy()
        else:
            logits = (self._numpy_w @ x) + self._numpy_b
            exp_logits = np.exp(logits - np.max(logits))
            probs_np = exp_logits / np.sum(exp_logits)

        idx = int(np.argmax(probs_np))
        confidence = float(np.round(probs_np[idx], 4))
        trusted = confidence >= self.min_confidence
        return {
            "label": self.CLASS_NAMES[idx],
            "confidence": confidence,
            "trusted": trusted,
            "min_confidence": self.min_confidence,
            "model_source": self.model_source,
            "distribution": {
                name: float(np.round(probs_np[i], 4)) for i, name in enumerate(self.CLASS_NAMES)
            },
            "backend": self.backend,
            "device": self.device,
            "cuda_available": self.cuda_available,
            "gpu_name": self.gpu_name,
        }
