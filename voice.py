"""Voice engine with natural-sounding backend selection.

Priority:
1) Piper (offline neural, if model is configured)
2) Edge TTS (high quality neural, internet)
3) pyttsx3 (offline fallback)
4) print fallback
"""

from __future__ import annotations

import asyncio
import os
import queue
import re
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Dict, Optional

from rephrase_api import SafeRephraser


class VoiceEngine:
    def __init__(
        self,
        enabled: bool = True,
        rate: int = 176,
        volume: float = 1.0,
        backend: str = "auto",
        persona: str = "coach",
        pyttsx3_voice_hint: str = "",
        edge_voice: str = "en-US-AvaNeural",
        edge_rate: str = "+0%",
        edge_pitch: str = "+0Hz",
        piper_model: str = "",
        piper_bin: str = "piper",
        piper_speaker: Optional[int] = None,
        rephrase_enabled: bool = False,
        rephrase_provider: str = "openai",
        rephrase_model: str = "",
        rephrase_api_key: str = "",
        rephrase_api_base: str = "",
    ) -> None:
        self.enabled = enabled
        self.rate = rate
        self.volume = volume
        self.requested_backend = (backend or "auto").strip().lower()
        self.persona = persona
        self.pyttsx3_voice_hint = pyttsx3_voice_hint

        self.edge_voice = edge_voice
        self.edge_rate = edge_rate
        self.edge_pitch = edge_pitch

        self.piper_model = piper_model
        self.piper_bin = piper_bin
        self.piper_speaker = piper_speaker

        self.active_backend = "disabled" if not enabled else "init"
        self.last_text = ""
        self._q: "queue.Queue[dict]" = queue.Queue(maxsize=48)
        self._running = True

        self._ffplay = shutil.which("ffplay")
        self._aplay = shutil.which("aplay")

        self._pyttsx3 = None
        self._pyttsx3_engine = None
        self._edge_tts = None

        self._tmp_dir = Path(tempfile.mkdtemp(prefix="antardrishti_tts_"))
        self._rephraser = SafeRephraser(
            enabled=rephrase_enabled,
            provider=rephrase_provider,
            model=rephrase_model,
            api_key=rephrase_api_key,
            api_base=rephrase_api_base,
        )

        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def status(self) -> Dict[str, object]:
        return {
            "enabled": self.enabled,
            "requested_backend": self.requested_backend,
            "active_backend": self.active_backend,
            "queue_size": int(self._q.qsize()),
            "last_text": self.last_text,
            "edge_voice": self.edge_voice if self.active_backend == "edge" else "",
            "piper_model": self.piper_model if self.active_backend == "piper" else "",
            "rephrase": self._rephraser.status(),
        }

    def _clear_pending(self) -> None:
        while True:
            try:
                self._q.get_nowait()
            except queue.Empty:
                break

    def _worker(self) -> None:
        self._initialize_backend()

        while self._running:
            try:
                payload = self._q.get(timeout=0.2)
            except queue.Empty:
                continue

            if not self._running:
                break

            payload = payload if isinstance(payload, dict) else {"text": str(payload)}
            text = str(payload.get("text", "")).strip()
            if not text:
                continue

            tone = str(payload.get("tone", "neutral") or "neutral")
            exercise = str(payload.get("exercise", "general") or "general")
            severity = str(payload.get("severity", "medium") or "medium")

            natural_text = self._humanize_text(text)
            natural_text = self._rephraser.maybe_rephrase(
                base_message=natural_text,
                tone=tone,
                exercise=exercise,
                severity=severity,
            )
            self.last_text = natural_text

            try:
                if self.active_backend == "piper":
                    self._speak_piper(natural_text)
                elif self.active_backend == "edge":
                    self._speak_edge(natural_text)
                elif self.active_backend == "pyttsx3":
                    self._speak_pyttsx3(natural_text)
                else:
                    self._speak_print(natural_text)
            except Exception:
                self._speak_print(natural_text)

    def _initialize_backend(self) -> None:
        if not self.enabled:
            self.active_backend = "disabled"
            return

        want = self.requested_backend

        if want in ("auto", "piper") and self._can_use_piper():
            self.active_backend = "piper"
            return

        if want in ("auto", "edge") and self._can_use_edge():
            self.active_backend = "edge"
            return

        if want in ("auto", "pyttsx3") and self._ensure_pyttsx3_engine():
            self.active_backend = "pyttsx3"
            return

        self.active_backend = "print"

    def _can_use_piper(self) -> bool:
        if not self.piper_model:
            return False
        if not os.path.exists(self.piper_model):
            return False
        return shutil.which(self.piper_bin) is not None

    def _can_use_edge(self) -> bool:
        if self._ffplay is None:
            return False
        try:
            import edge_tts

            self._edge_tts = edge_tts
            return True
        except Exception:
            return False

    def _ensure_pyttsx3_engine(self) -> bool:
        if self._pyttsx3_engine is not None:
            return True
        try:
            import pyttsx3

            self._pyttsx3 = pyttsx3
            engine = pyttsx3.init()
            engine.setProperty("rate", self.rate)
            engine.setProperty("volume", self.volume)
            self._select_pyttsx3_voice(engine)
            self._pyttsx3_engine = engine
            return True
        except Exception:
            self._pyttsx3_engine = None
            return False

    def _select_pyttsx3_voice(self, engine) -> None:
        try:
            voices = engine.getProperty("voices") or []
        except Exception:
            return

        if not voices:
            return

        chosen_id = None
        hint = (self.pyttsx3_voice_hint or "").strip().lower()
        if hint:
            for voice in voices:
                name = (getattr(voice, "name", "") or "").lower()
                vid = (getattr(voice, "id", "") or "").lower()
                if hint in name or hint in vid:
                    chosen_id = getattr(voice, "id", None)
                    break

        if chosen_id is None:
            priorities = ["en-us", "female", "english", "mbrola"]
            for key in priorities:
                for voice in voices:
                    blob = f"{getattr(voice, 'name', '')} {getattr(voice, 'id', '')}".lower()
                    if key in blob:
                        chosen_id = getattr(voice, "id", None)
                        break
                if chosen_id:
                    break

        if chosen_id:
            try:
                engine.setProperty("voice", chosen_id)
            except Exception:
                pass

    def _humanize_text(self, text: str) -> str:
        text = re.sub(r"\s+", " ", text.strip())
        if not text:
            return text

        rewrites = {
            "Starting squats": "Alright, starting squats now. I am right here with you",
            "Starting squats now.": "Alright, starting squats now. I am right here with you",
            "Session paused": "Okay, pausing here. Catch your breath",
            "Session resumed": "Nice, we are back. Continue with control",
            "Workout complete": "Workout complete. You did really well today",
            "Get into push-up position": "Great. Move into push-up position when you are ready",
        }
        if text in rewrites:
            text = rewrites[text]

        if not text.endswith((".", "!", "?")):
            text += "."

        text = text.replace("...", ", ")
        text = text.replace("..", ".")
        return text

    def _speak_piper(self, text: str) -> None:
        wav_path = self._tmp_dir / f"utt_{int(time.time() * 1000)}.wav"
        cmd = [self.piper_bin, "--model", self.piper_model, "--output_file", str(wav_path)]
        if self.piper_speaker is not None:
            cmd.extend(["--speaker", str(self.piper_speaker)])

        proc = subprocess.run(
            cmd,
            input=text.encode("utf-8"),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if proc.returncode != 0 or not wav_path.exists():
            self._speak_print(text)
            return

        self._play_file(wav_path)
        self._delete_file(wav_path)

    async def _edge_save(self, text: str, out_file: Path) -> None:
        communicate = self._edge_tts.Communicate(
            text=text,
            voice=self.edge_voice,
            rate=self.edge_rate,
            pitch=self.edge_pitch,
        )
        await communicate.save(str(out_file))

    def _speak_edge(self, text: str) -> None:
        if self._edge_tts is None:
            if not self._can_use_edge():
                if self._ensure_pyttsx3_engine():
                    self.active_backend = "pyttsx3"
                    self._speak_pyttsx3(text)
                else:
                    self._speak_print(text)
                return

        mp3_path = self._tmp_dir / f"utt_{int(time.time() * 1000)}.mp3"
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self._edge_save(text, mp3_path))
            loop.close()
            asyncio.set_event_loop(None)
        except Exception:
            if self._ensure_pyttsx3_engine():
                self.active_backend = "pyttsx3"
                self._speak_pyttsx3(text)
            else:
                self._speak_print(text)
            self._delete_file(mp3_path)
            return

        if mp3_path.exists():
            self._play_file(mp3_path)
            self._delete_file(mp3_path)
        else:
            self._speak_print(text)

    def _speak_pyttsx3(self, text: str) -> None:
        if not self._ensure_pyttsx3_engine():
            self._speak_print(text)
            return
        try:
            self._pyttsx3_engine.say(text)
            self._pyttsx3_engine.runAndWait()
        except Exception:
            self._speak_print(text)

    def _play_file(self, path: Path) -> None:
        if self._ffplay:
            subprocess.run(
                [self._ffplay, "-nodisp", "-autoexit", "-loglevel", "quiet", str(path)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            return

        if self._aplay and path.suffix.lower() == ".wav":
            subprocess.run(
                [self._aplay, "-q", str(path)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            return

        self._speak_print(f"Audio generated: {path.name}")

    def _speak_print(self, text: str) -> None:
        print(f"[VOICE] {text}")
        time.sleep(min(1.3, max(0.45, len(text) / 35.0)))

    def _delete_file(self, path: Path) -> None:
        try:
            if path.exists():
                path.unlink()
        except Exception:
            pass

    def speak(self, text: str, replace_pending: bool = False) -> None:
        if not text:
            return
        payload = {
            "text": str(text),
            "tone": "neutral",
            "exercise": "general",
            "severity": "medium",
        }
        if replace_pending:
            self._clear_pending()
        try:
            self._q.put_nowait(payload)
        except queue.Full:
            try:
                self._q.get_nowait()
            except queue.Empty:
                pass
            try:
                self._q.put_nowait(payload)
            except queue.Full:
                pass

    def speak_structured(
        self,
        text: str,
        tone: str = "neutral",
        exercise: str = "general",
        severity: str = "medium",
        replace_pending: bool = True,
    ) -> None:
        if not text:
            return
        payload = {
            "text": str(text),
            "tone": str(tone or "neutral"),
            "exercise": str(exercise or "general"),
            "severity": str(severity or "medium"),
        }
        if replace_pending:
            self._clear_pending()
        try:
            self._q.put_nowait(payload)
        except queue.Full:
            try:
                self._q.get_nowait()
            except queue.Empty:
                pass
            try:
                self._q.put_nowait(payload)
            except queue.Full:
                pass

    def close(self) -> None:
        self._running = False
        try:
            self._q.put_nowait({"text": ""})
        except queue.Full:
            pass
        self._thread.join(timeout=2.0)
        self._rephraser.close()

        try:
            if self._pyttsx3_engine is not None:
                self._pyttsx3_engine.stop()
        except Exception:
            pass

        try:
            for file_path in self._tmp_dir.glob("*"):
                self._delete_file(file_path)
            self._tmp_dir.rmdir()
        except Exception:
            pass
