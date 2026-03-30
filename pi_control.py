"""Remote Raspberry Pi stream control over SSH."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class PiConfig:
    host: str = ""
    username: str = ""
    password: str = ""
    port: int = 22
    start_cmd: str = "cd ~ && ./start_cam.sh > /tmp/drishti_stream.log 2>&1 &"


class PiCameraRemote:
    def __init__(self, config: PiConfig) -> None:
        self.config = config
        self._client = None

    def start_stream(self) -> Tuple[bool, str]:
        if not (self.config.host and self.config.username and self.config.password):
            return False, "Missing Pi credentials"

        try:
            import paramiko

            if self._client is not None:
                try:
                    self._client.close()
                except Exception:
                    pass
                self._client = None

            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            client.connect(
                hostname=self.config.host,
                port=self.config.port,
                username=self.config.username,
                password=self.config.password,
                timeout=6,
                banner_timeout=6,
                auth_timeout=6,
            )
            client.exec_command(self.config.start_cmd)
            self._client = client
            try:
                client.close()
            except Exception:
                pass
            self._client = None
            return True, "Pi stream start command sent over SSH"
        except Exception as exc:
            return False, f"Pi stream start failed: {exc}"

    def stop_stream(self) -> Tuple[bool, str]:
        if self._client is None:
            return False, "No Pi SSH session"
        try:
            self._client.exec_command("pkill -f 'rpicam-vid|ffmpeg' || true")
            return True, "Pi stream stop command sent"
        except Exception as exc:
            return False, f"Pi stream stop failed: {exc}"

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None
