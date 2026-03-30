"""State helpers and optional console monitor."""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from typing import Dict

STATE_FILE = "state.json"


def write_state(state: Dict, path: str = STATE_FILE) -> None:
    state = dict(state)
    state["updated_at"] = datetime.utcnow().isoformat()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def read_state(path: str = STATE_FILE) -> Dict:
    if not os.path.exists(path):
        return {
            "state": "booting",
            "activity": "idle",
            "exercise": "none",
            "rep_count": 0,
            "target": 0,
            "difficulty": "medium",
            "feedback": "waiting for system",
            "camera": "none",
            "obstacle": False,
        }

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_console_dashboard(path: str = STATE_FILE, refresh_s: float = 0.7) -> None:
    """Simple terminal monitor for environments without web UI access."""
    while True:
        state = read_state(path)
        print(
            f"[{state.get('updated_at', 'n/a')}] "
            f"state={state.get('state', 'n/a')} "
            f"activity={state.get('activity', 'n/a')} "
            f"reps={state.get('rep_count', 0)}/{state.get('target', 0)} "
            f"score={state.get('score', {}).get('skill_score', 0)} "
            f"feedback={state.get('feedback', '')}"
        )
        time.sleep(refresh_s)


if __name__ == "__main__":
    run_console_dashboard()
