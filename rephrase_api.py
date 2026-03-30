"""Optional safe API rephrasing with non-blocking cache-first behavior."""

from __future__ import annotations

import hashlib
import json
import queue
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class RephraseRequest:
    base_message: str
    tone: str
    exercise: str
    severity: str


class SafeRephraser:
    """Returns cached rephrases immediately and fetches misses in background."""

    def __init__(
        self,
        enabled: bool = False,
        provider: str = "openai",
        model: str = "",
        api_key: str = "",
        api_base: str = "",
    ) -> None:
        self.enabled = bool(enabled and model and api_key)
        self.provider = (provider or "openai").strip().lower()
        self.model = (model or "").strip()
        self.api_key = (api_key or "").strip()
        self.api_base = (api_base or "").strip()
        self._cache: Dict[str, str] = {}
        self._cache_lock = threading.Lock()
        self._queue: "queue.Queue[RephraseRequest]" = queue.Queue(maxsize=96)
        self._running = self.enabled
        self._thread = None
        self._last_error = ""

        if self.enabled:
            self._thread = threading.Thread(target=self._worker, daemon=True)
            self._thread.start()

    def status(self) -> Dict[str, str]:
        return {
            "enabled": str(self.enabled).lower(),
            "provider": self.provider,
            "model": self.model,
            "last_error": self._last_error,
            "cache_size": str(len(self._cache)),
        }

    def _key(self, req: RephraseRequest) -> str:
        canonical = json.dumps(
            {
                "base_message": req.base_message,
                "tone": req.tone,
                "exercise": req.exercise,
                "severity": req.severity,
            },
            sort_keys=True,
            ensure_ascii=True,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @staticmethod
    def _sanitize(sentence: str, fallback: str) -> str:
        text = " ".join((sentence or "").strip().split())
        if not text:
            return fallback
        # Keep output short and one sentence.
        for sep in ("\n", "\r"):
            text = text.replace(sep, " ")
        if len(text) > 130:
            text = text[:130].rstrip()
        if text.count(".") + text.count("!") + text.count("?") > 1:
            parts = [p.strip() for p in text.replace("!", ".").replace("?", ".").split(".") if p.strip()]
            if parts:
                text = parts[0]
        if not text.endswith((".", "!", "?")):
            text += "."
        return text

    def maybe_rephrase(
        self,
        base_message: str,
        tone: str = "neutral",
        exercise: str = "general",
        severity: str = "medium",
    ) -> str:
        base = (base_message or "").strip()
        if not base:
            return ""

        if not self.enabled:
            return base

        req = RephraseRequest(
            base_message=base,
            tone=(tone or "neutral").strip().lower(),
            exercise=(exercise or "general").strip().lower(),
            severity=(severity or "medium").strip().lower(),
        )
        key = self._key(req)

        with self._cache_lock:
            cached = self._cache.get(key)
        if cached:
            return cached

        try:
            self._queue.put_nowait(req)
        except queue.Full:
            pass
        return base

    def _worker(self) -> None:
        while self._running:
            try:
                req = self._queue.get(timeout=0.25)
            except queue.Empty:
                continue

            key = self._key(req)
            with self._cache_lock:
                if key in self._cache:
                    continue

            output = self._call_api(req)
            final = self._sanitize(output, req.base_message)
            with self._cache_lock:
                self._cache[key] = final

    def _endpoint(self) -> str:
        if self.api_base:
            return self.api_base
        if self.provider == "groq":
            return "https://api.groq.com/openai/v1/chat/completions"
        return "https://api.openai.com/v1/chat/completions"

    def _call_api(self, req: RephraseRequest) -> str:
        system_prompt = (
            "You are a professional fitness coach guiding a visually impaired person. "
            "Do not change meaning. Do not add information. Keep one short sentence."
        )
        user_payload = {
            "base_message": req.base_message,
            "tone": req.tone,
            "exercise": req.exercise,
            "severity": req.severity,
        }

        payload = {
            "model": self.model,
            "temperature": 0.2,
            "max_tokens": 60,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=True)},
            ],
        }

        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        request = urllib.request.Request(self._endpoint(), data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=2.6) as resp:
                raw = resp.read().decode("utf-8", errors="ignore")
            parsed = json.loads(raw)
            choices = parsed.get("choices") or []
            if not choices:
                return req.base_message
            message = choices[0].get("message", {})
            content = str(message.get("content", "")).strip()
            self._last_error = ""
            return content or req.base_message
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError) as exc:
            self._last_error = str(exc)
            return req.base_message
        except Exception as exc:  # noqa: BLE001
            self._last_error = str(exc)
            return req.base_message

    def close(self) -> None:
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.2)
