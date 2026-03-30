"""Flask product entrypoint for ANTARDRISHTI.

Run with one command:
flask --app app run --host 0.0.0.0 --port 5000 --no-reload
"""

from __future__ import annotations

import atexit
import logging
import os
import socket
import time
from typing import Dict

from flask import Flask, Response, jsonify, render_template_string, request

from runtime import AntardrishtiRuntime, RuntimeConfig


def _load_local_env(path: str = ".env") -> None:
  """Load simple KEY=VALUE pairs from local .env when present."""
  if not os.path.exists(path):
    return

  try:
    with open(path, "r", encoding="utf-8") as f:
      for raw_line in f:
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
          continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
          continue

        # Keep explicit shell-provided env vars higher priority.
        if key in os.environ:
          continue

        if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
          value = value[1:-1]
        os.environ[key] = value
  except Exception:
    # Runtime can still run with shell-provided env vars.
    return


def _bool_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def _configure_http_logging() -> None:
  """Reduce noisy request logs while preserving errors."""
  if not _bool_env("APP_QUIET_HTTP_LOGS", True):
    return

  werkzeug_logger = logging.getLogger("werkzeug")
  werkzeug_logger.setLevel(logging.ERROR)
  werkzeug_logger.propagate = False

  flask_logger = logging.getLogger("flask.app")
  flask_logger.setLevel(logging.ERROR)
  flask_logger.propagate = False


def _resolve_lan_ip() -> str:
  """Best-effort LAN IP detection for terminal startup hints."""
  try:
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    probe.connect(("8.8.8.8", 80))
    ip = probe.getsockname()[0]
    probe.close()
    return ip
  except Exception:
    return ""


def _print_startup_urls(host: str, port: int) -> None:
  host = str(host or "").strip()
  port = int(port)

  if host in ("", "0.0.0.0", "::"):
    print(f"Open in browser: http://127.0.0.1:{port}", flush=True)
    lan_ip = _resolve_lan_ip()
    if lan_ip and lan_ip not in ("127.0.0.1", "0.0.0.0"):
      print(f"LAN URL: http://{lan_ip}:{port}", flush=True)
    return

  print(f"Open in browser: http://{host}:{port}", flush=True)


def _build_config() -> RuntimeConfig:
    piper_speaker_raw = os.getenv("APP_PIPER_SPEAKER", "").strip()
    try:
        piper_speaker = int(piper_speaker_raw) if piper_speaker_raw else None
    except ValueError:
        piper_speaker = None

    return RuntimeConfig(
        webcam=int(os.getenv("APP_WEBCAM", "0")),
        pi_url=os.getenv("APP_PI_URL", "udp://@:8080"),
        state_file=os.getenv("APP_STATE_FILE", "state.json"),
      break_seconds=int(os.getenv("APP_BREAK_SECONDS", "10")),
        obstacle_area=int(os.getenv("APP_OBSTACLE_AREA", "7000")),
        voice_cooldown=float(os.getenv("APP_VOICE_COOLDOWN", "2.4")),
        mute=_bool_env("APP_MUTE", False),
        tts_backend=os.getenv("APP_TTS_BACKEND", "auto"),
        voice_persona=os.getenv("APP_VOICE_PERSONA", "coach"),
        pyttsx3_voice_hint=os.getenv("APP_PYTTSX3_VOICE_HINT", ""),
        edge_voice=os.getenv("APP_EDGE_TTS_VOICE", "en-US-AvaNeural"),
        edge_rate=os.getenv("APP_EDGE_TTS_RATE", "+0%"),
        edge_pitch=os.getenv("APP_EDGE_TTS_PITCH", "+0Hz"),
        piper_model=os.getenv("APP_PIPER_MODEL", ""),
        piper_bin=os.getenv("APP_PIPER_BIN", "piper"),
        piper_speaker=piper_speaker,
        output_dir=os.getenv("APP_OUTPUT_DIR", "outputs"),
        record_annotated=_bool_env("APP_RECORD_ANNOTATED", True),
        output_fps=int(os.getenv("APP_OUTPUT_FPS", "20")),
        auto_start_pi=_bool_env("PI_AUTO_START", False),
        pi_host=os.getenv("PI_HOST", ""),
        pi_username=os.getenv("PI_USER", ""),
        pi_password=os.getenv("PI_PASSWORD", ""),
        pi_port=int(os.getenv("PI_PORT", "22")),
        pi_start_cmd=os.getenv(
            "PI_START_CMD",
            "cd ~ && ./start_cam.sh > /tmp/drishti_stream.log 2>&1 &",
        ),
        obstacle_enabled=_bool_env("APP_OBSTACLE_ENABLED", True),
        obstacle_mode=os.getenv("APP_OBSTACLE_MODE", "yolo"),
        obstacle_yolo_model=os.getenv("APP_OBSTACLE_YOLO_MODEL", "yolov8n.pt"),
        obstacle_yolo_conf=float(os.getenv("APP_OBSTACLE_YOLO_CONF", "0.35")),
        obstacle_yolo_min_area=float(os.getenv("APP_OBSTACLE_YOLO_MIN_AREA", "0.06")),
        obstacle_alert_cooldown=float(os.getenv("APP_OBSTACLE_ALERT_COOLDOWN", "6.0")),
        obstacle_person_only=_bool_env("APP_OBSTACLE_PERSON_ONLY", True),
        obstacle_ignore_person=_bool_env("APP_OBSTACLE_IGNORE_PERSON", False),
        obstacle_device=os.getenv("APP_OBSTACLE_DEVICE", "cuda"),
        pose_model_laptop=int(os.getenv("APP_POSE_MODEL_LAPTOP", "1")),
        pose_model_pi=int(os.getenv("APP_POSE_MODEL_PI", "2")),
        pose_min_detection_conf=float(os.getenv("APP_POSE_MIN_DETECTION_CONF", "0.55")),
        pose_min_tracking_conf=float(os.getenv("APP_POSE_MIN_TRACKING_CONF", "0.55")),
        pose_smooth_alpha=float(os.getenv("APP_POSE_SMOOTH_ALPHA", "0.35")),
        fusion_min_quality=float(os.getenv("APP_FUSION_MIN_QUALITY", "0.18")),
        feedback_dataset_path=os.getenv("APP_FEEDBACK_DATASET", "feedback_dataset.json"),
        rephrase_enabled=_bool_env("APP_REPHRASE_ENABLED", False),
        rephrase_provider=os.getenv("APP_REPHRASE_PROVIDER", "openai"),
        rephrase_model=os.getenv("APP_REPHRASE_MODEL", ""),
        rephrase_api_key=os.getenv("APP_REPHRASE_API_KEY", os.getenv("OPENAI_API_KEY", "")),
        rephrase_api_base=os.getenv("APP_REPHRASE_API_BASE", ""),
        ai_weights_path=os.getenv("APP_AI_WEIGHTS_PATH", "models/activity_weights.json"),
        ai_confidence_threshold=float(os.getenv("APP_AI_CONFIDENCE", "0.82")),
        collect_training_data=_bool_env("APP_COLLECT_TRAINING_DATA", False),
        training_data_dir=os.getenv("APP_TRAINING_DATA_DIR", "outputs/datasets"),
        training_flush_every=int(os.getenv("APP_TRAINING_FLUSH_EVERY", "250")),
    )


def _should_start_runtime() -> bool:
    debug_enabled = _bool_env("FLASK_DEBUG", False)
    if not debug_enabled:
        return True
    return os.getenv("WERKZEUG_RUN_MAIN") == "true"


def _control_to_runtime_command(payload: Dict) -> Dict:
    cmd = str(payload.get("command", "")).strip().lower()
    value = payload.get("value")

    if cmd in (
      "start",
      "pause",
      "pause_toggle",
      "resume",
      "stop",
      "next",
      "repeat",
      "add_exercise",
      "obstacle_toggle",
      "obstacle_on",
      "obstacle_off",
    ):
        return {"command": cmd, "payload": payload}

    if cmd == "speak":
        text = payload.get("text")
        if text is None:
            text = value
        return {"command": "speak", "payload": {"text": text}}

    if cmd in ("easy", "medium", "tough"):
        return {"command": f"difficulty_{cmd}", "payload": payload}

    if cmd == "difficulty":
        return {"command": "difficulty", "payload": {"value": value}}

    return {"command": "", "payload": {}}


DASHBOARD_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>ANTARDRISHTI</title>
  <style>
    html, body {
      height: 100%;
    }
    :root {
      --bg: #030405;
      --bg-2: #07090b;
      --fg: #f2f5f7;
      --muted: #a5aeb7;
      --line: #222a31;
      --line-strong: #32404a;
      --panel: #0a0f13;
      --panel-2: #0d1318;
      --alert: #ff6868;
      --accent: #6ce0ff;
      --accent-2: #8cffc5;
      --shadow: rgba(0, 0, 0, 0.45);
    }
    body {
      margin: 0;
      background:
        radial-gradient(1200px 500px at 100% -20%, #122331 0%, transparent 58%),
        radial-gradient(900px 420px at -10% 0%, #1a1111 0%, transparent 52%),
        linear-gradient(180deg, var(--bg-2) 0%, var(--bg) 55%, #020303 100%);
      color: var(--fg);
      font-family: "IBM Plex Mono", "Courier New", monospace;
      overflow: hidden;
    }
    .wrap {
      width: 100vw;
      height: 100vh;
      box-sizing: border-box;
      margin: 0;
      padding: 10px;
      display: grid;
      grid-template-columns: 1.2fr 1fr;
      grid-template-rows: auto 1fr;
      gap: 14px;
    }
    .card {
      border: 1px solid var(--line);
      border-radius: 14px;
      background: linear-gradient(180deg, var(--panel-2) 0%, var(--panel) 100%);
      padding: 14px;
      box-shadow: 0 14px 36px var(--shadow), inset 0 0 24px rgba(255, 255, 255, 0.03);
      position: relative;
      overflow: hidden;
      min-height: 0;
    }
    .card::after {
      content: "";
      position: absolute;
      left: 0;
      top: 0;
      width: 100%;
      height: 1px;
      background: linear-gradient(90deg, transparent 0%, var(--line-strong) 30%, transparent 100%);
      opacity: 0.8;
    }
    .title {
      font-weight: 700;
      letter-spacing: 1px;
      margin-bottom: 10px;
      display: flex;
      align-items: center;
      justify-content: space-between;
    }
    .hero {
      grid-column: 1 / -1;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      padding: 10px 14px;
      border-color: var(--line-strong);
      background: linear-gradient(180deg, #0f161c 0%, #0a0f13 100%);
    }
    .hero-left {
      display: flex;
      align-items: center;
      gap: 10px;
    }
    .hero-id {
      font-size: 12px;
      color: var(--accent);
      border: 1px solid #2f4a58;
      border-radius: 999px;
      padding: 4px 10px;
      background: #0b171f;
    }
    .hero-text {
      font-size: 12px;
      color: var(--muted);
    }
    .badge-row {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }
    .badge {
      font-size: 11px;
      border: 1px solid var(--line-strong);
      border-radius: 999px;
      padding: 4px 9px;
      color: var(--muted);
      background: #0a1319;
    }
    .badge.ok {
      color: var(--accent-2);
      border-color: #2d5a46;
      background: #0a1712;
    }
    .badge.warn {
      color: #ffcf78;
      border-color: #5e4a2b;
      background: #191308;
    }
    .video-shell {
      border: 1px solid var(--line-strong);
      border-radius: 10px;
      padding: 8px;
      background: #05090c;
      flex: 1;
      min-height: 0;
      display: flex;
    }
    .video {
      width: 100%;
      height: 100%;
      border-radius: 8px;
      border: 1px solid var(--line);
      background: #000;
      object-fit: contain;
    }
    .video-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px;
      margin-top: 8px;
    }
    .video-small {
      width: 100%;
      border-radius: 6px;
      border: 1px solid var(--line);
      background: #000;
      height: 250px;
      object-fit: cover;
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px;
      margin: 8px 0;
    }
    .metric {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 8px;
      background: #080d11;
    }
    .label {
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 6px;
    }
    .value {
      font-size: 19px;
      font-weight: 700;
    }
    .feedback {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #050a0d;
      padding: 10px;
      min-height: 70px;
      line-height: 1.45;
    }
    .video-caption {
      color: var(--muted);
      font-size: 12px;
      margin-top: 8px;
    }
    .controls {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 8px;
      margin-top: 10px;
    }
    button {
      border: 1px solid #343434;
      border-radius: 8px;
      background: #10161c;
      color: var(--fg);
      padding: 9px 10px;
      font-family: inherit;
      cursor: pointer;
      transition: border-color 120ms ease, transform 120ms ease, background 120ms ease;
    }
    button:hover {
      border-color: var(--accent);
      background: #121d25;
      transform: translateY(-1px);
    }
    button.primary {
      border-color: #2d616f;
      background: linear-gradient(180deg, #12303a 0%, #0f242c 100%);
      color: #d7f5ff;
    }
    button.wide {
      grid-column: span 3;
    }
    .mono-row {
      display: grid;
      gap: 4px;
      margin-top: 8px;
    }
    .muted {
      color: var(--muted);
      font-size: 12px;
      margin-top: 6px;
    }
    .alert {
      color: var(--alert);
      font-weight: 700;
    }
    input, select {
      border: 1px solid #343434;
      border-radius: 8px;
      background: #0f141a;
      color: var(--fg);
      padding: 8px 10px;
      font-family: inherit;
    }
    .panel {
      height: calc(100vh - 86px);
      display: flex;
      flex-direction: column;
      overflow: auto;
    }
    .add-exercise {
      margin-top: 10px;
      display: grid;
      grid-template-columns: 1fr 140px;
      gap: 8px;
    }
    @media (max-width: 980px) {
      body { overflow: auto; }
      .wrap { grid-template-columns: 1fr; }
      .controls { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      button.wide { grid-column: span 2; }
      .panel { height: auto; }
      .add-exercise { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="wrap">
    <header class="card hero">
      <div class="hero-left">
        <div class="hero-id">ANTARDRISHTI</div>
        <div class="hero-text">Blind-first AI Coach • Pose Mapping + Voice Guidance + Pi Person Distance</div>
      </div>
      <div class="badge-row">
        <div class="badge" id="runtimeBadge">runtime: booting</div>
        <div class="badge" id="gpuBadge">compute: probing</div>
        <div class="badge" id="piBadge">pi stream: checking</div>
      </div>
    </header>

    <section class="card panel">
      <div class="title">ANTARDRISHTI // LIVE ANNOTATED FEED <span class="muted">Laptop Pose is Primary</span></div>
      <div class="video-shell">
        <img class="video" src="/video_feed" alt="Live feed" />
      </div>
      <div class="video-grid">
        <div>
          <div class="video-caption">Laptop Camera (MediaPipe pose source)</div>
          <img class="video-small" src="/video_laptop" alt="Laptop feed" />
        </div>
        <div>
          <div class="video-caption">Pi Camera (YOLO people boxes + distance)</div>
          <img class="video-small" src="/video_pi" alt="Pi feed" />
        </div>
      </div>
      <div class="mono-row">
        <div class="muted" id="runtimeStatus">Runtime: booting</div>
        <div class="muted" id="cameraStatus">Cameras: unknown</div>
        <div class="muted" id="poseStatus">Pose fusion: n/a</div>
        <div class="muted" id="videoPath">Output video: n/a</div>
      </div>
    </section>

    <section class="card panel">
      <div class="title">COACH CONSOLE</div>
      <div class="grid">
        <div class="metric"><div class="label">STATE</div><div class="value" id="state">-</div></div>
        <div class="metric"><div class="label">ACTIVITY</div><div class="value" id="activity">-</div></div>
        <div class="metric"><div class="label">DIFFICULTY</div><div class="value" id="difficulty">-</div></div>
        <div class="metric"><div class="label">CAMERA</div><div class="value" id="camera">-</div></div>
        <div class="metric"><div class="label">REPS</div><div class="value" id="reps">0/0</div></div>
        <div class="metric"><div class="label">SKILL SCORE</div><div class="value" id="score">0</div></div>
      </div>
      <div class="feedback" id="feedback">Waiting for runtime...</div>
      <div class="muted" id="obstacleText">Obstacle: NO</div>
      <div class="muted" id="peopleText">People: none</div>
      <div class="muted" id="flowText">Flow: waiting for coach state...</div>
      <div class="muted" id="planText">Plan: squats -> sit-ups -> jumping jacks</div>
      <div class="muted" id="piControlText">Pi auto-start: idle</div>
      <div class="muted" id="errorDetailText">Errors: none</div>
      <div class="muted" id="aiText">Local AI: initializing</div>
      <div class="muted" id="modelText">Model source: default</div>
      <div class="muted" id="voiceText">Voice: initializing</div>
      <div class="muted" id="syncText">Sync: waiting for runtime...</div>
      <div class="muted" id="trainingText">Dataset capture: off</div>
      <div class="muted" id="calibText">Calibration: pending</div>
      <div class="controls">
        <button class="primary" onclick="sendControl('start')">Start</button>
        <button onclick="sendControl('pause_toggle')">Pause/Resume</button>
        <button onclick="sendControl('stop')">Stop</button>
        <button onclick="sendControl('easy')">Easy</button>
        <button onclick="sendControl('medium')">Medium</button>
        <button onclick="sendControl('tough')">Tough</button>
        <button onclick="sendControl('next')">👉 Next Exercise</button>
        <button onclick="sendControl('repeat')">Repeat</button>
        <button onclick="speakLine()">Speak</button>
        <button class="wide" onclick="sendControl('obstacle_toggle')">Toggle Pi Person Detector</button>
      </div>
      <div class="add-exercise">
        <input id="addExerciseInput" placeholder="add exercise: squat, situp, jumping_jack, pushup" />
        <button onclick="addExercise()">Add Exercise</button>
      </div>
      <div class="muted">PS4 alignment: pose estimation, joint analysis, reference comparison, quantified feedback.</div>
    </section>
  </div>

  <script>
    async function sendControl(command, value = null) {
      try {
        const body = value === null ? { command } : { command, value };
        await fetch('/api/control', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body)
        });
      } catch (e) {
        console.error(e);
      }
    }

    function speakLine() {
      const text = prompt('Coach line to speak:');
      if (!text) return;
      sendControl('speak', text);
    }

    function addExercise() {
      const input = document.getElementById('addExerciseInput');
      const value = String(input.value || '').trim();
      if (!value) return;
      sendControl('add_exercise', value);
      input.value = '';
    }

    function txt(id, value) {
      document.getElementById(id).textContent = value;
    }

    async function pollState() {
      try {
        const res = await fetch('/api/state');
        const data = await res.json();

        txt('state', String(data.state || '-').toUpperCase());
        txt('activity', String(data.activity || '-').toUpperCase());
        txt('difficulty', String(data.difficulty || '-').toUpperCase());
        txt('camera', String(data.camera || '-').toUpperCase());
        txt('reps', `${data.rep_count || 0}/${data.target || 0}`);
        txt('score', Math.round((data.score || {}).skill_score || 0));
          const spokenFeedback = String(data.spoken_feedback || '');
          const syncLocked = Boolean(data.feedback_sync_active);
          const feedbackLine = (syncLocked && spokenFeedback) ? spokenFeedback : String(data.feedback || '');
          txt('feedback', feedbackLine);

        const obstacle = data.obstacle ? 'YES' : 'NO';
        const obstacleEnabled = data.obstacle_enabled ? 'ON' : 'OFF';
        const obstacleMode = data.obstacle_mode || '-';
        const obstacleCount = data.obstacle_count || 0;
        const obstacleDist = data.obstacle_distance_m ? `${data.obstacle_distance_m.toFixed(2)}m` : 'n/a';
        const obstacleEl = document.getElementById('obstacleText');
        obstacleEl.textContent = `Pi detector: people=${obstacle} | mode=${obstacleMode}:${obstacleEnabled} | count=${obstacleCount} | nearest=${obstacleDist}`;
        obstacleEl.className = data.obstacle ? 'muted alert' : 'muted';

        const people = Array.isArray(data.people) ? data.people : [];
        if (people.length > 0) {
          const compact = people.slice(0, 3).map((p, i) => `P${i + 1}:${(p.distance_m || 0).toFixed(2)}m`).join(' | ');
          txt('peopleText', `People distances: ${compact}`);
        } else {
          txt('peopleText', 'People: none');
        }

        txt('flowText', `Flow: ${data.next_action || 'Follow dashboard prompts and gesture hints.'}`);
        const plan = Array.isArray(data.exercise_plan) ? data.exercise_plan : [];
        const nextExercise = data.next_exercise || '-';
        const breakRemaining = Number(data.break_remaining_s || 0);
        const breakText = breakRemaining > 0 ? ` | break ${breakRemaining}s` : '';
        txt('planText', `Plan: ${plan.length ? plan.join(' -> ') : '-'} | Next: ${nextExercise}${breakText}`);

        const ai = data.ai || {};
        const gpuInfo = ai.gpu_name ? ` | GPU=${ai.gpu_name}` : '';
        txt('aiText', `Local AI: ${ai.label || '-'} (${ai.confidence || 0}), ${ai.backend || '-'}:${ai.device || '-'}${gpuInfo}`);
        txt('modelText', `Model source: ${ai.model_source || 'default'} | trusted=${ai.trusted ? 'yes' : 'no'} | min_conf=${ai.min_confidence || '-'}`);

        const voice = data.voice || {};
          const voiceQueue = Number(voice.queue_size || 0);
        const rephrase = voice.rephrase || {};
        const rephraseLabel = rephrase.enabled === 'true' ? `rephrase=${rephrase.provider || '-'}:${rephrase.model || '-'}` : 'rephrase=off';
          txt('voiceText', `Voice: ${voice.active_backend || '-'} (requested=${voice.requested_backend || '-'}) | queue=${voiceQueue} | ${rephraseLabel}`);

          const gestureEvent = data.gesture_event || '-';
          const voiceLast = String(voice.last_text || '').trim();
          const activeSpeech = syncLocked || voiceQueue > 0;
          const spokenNow = activeSpeech ? (spokenFeedback || voiceLast || '-') : '-';
          txt('syncText', `Sync: gesture=${gestureEvent} | spoken=${spokenNow} | lock=${syncLocked ? 'on' : 'off'}`);

        const errDetails = Array.isArray(data.error_details) ? data.error_details : [];
        if (errDetails.length > 0) {
          const top = errDetails[0];
          txt('errorDetailText', `Errors: ${top.code || '-'} [${String(top.severity || '').toUpperCase()}]`);
        } else {
          txt('errorDetailText', 'Errors: none');
        }

        const train = data.training_data || {};
        txt('trainingText', `Dataset capture: ${train.enabled ? 'on' : 'off'} | total=${train.total_samples || 0} | buffer=${train.buffered_samples || 0} | file=${train.path || '-'}`);

        const calib = data.calibration || {};
        txt('calibText', `Calibration: squat=${calib.squat_calibrated ? 'ready' : 'learning'} thresholds=${calib.squat_down_threshold || 0}/${calib.squat_up_threshold || 0}`);

        const cameraStatus = data.camera_status || {};
        txt('cameraStatus', `Cameras: laptop=${cameraStatus.laptop_alive ? 'on' : 'off'}, pi=${cameraStatus.pi_alive ? 'on' : 'off'}`);
        const piCtl = data.pi_control || {};
        txt('piControlText', `Pi auto-start: ${piCtl.enabled ? 'enabled' : 'disabled'} | ok=${piCtl.ok ? 'yes' : 'no'} | attempts=${piCtl.attempts || 0} | ${piCtl.message || ''}`);
        const pose = data.pose_sources || {};
        txt('poseStatus', `Pose fusion: src=${pose.selected || '-'} strategy=${pose.strategy || '-'} q(l/p)=${pose.laptop_quality || 0}/${pose.pi_quality || 0}`);
        txt('runtimeStatus', `Runtime: ${data.runtime || 'unknown'} | Updated: ${data.updated_at || ''}`);
        txt('videoPath', `Output video: ${data.output_video || 'n/a'}`);

        const runtimeBadge = document.getElementById('runtimeBadge');
        runtimeBadge.textContent = `runtime: ${data.runtime || 'unknown'}`;
        runtimeBadge.className = `badge ${(data.runtime === 'running') ? 'ok' : 'warn'}`;

        const gpuBadge = document.getElementById('gpuBadge');
        gpuBadge.textContent = `compute: ${(ai.backend || '-')} / ${(ai.device || '-')}`;
          gpuBadge.className = `badge ${(String(ai.device || '').toLowerCase().startsWith('cuda')) ? 'ok' : 'warn'}`;

        const piBadge = document.getElementById('piBadge');
        piBadge.textContent = `pi stream: ${(cameraStatus.pi_alive ? 'online' : 'offline')}`;
        piBadge.className = `badge ${(cameraStatus.pi_alive ? 'ok' : 'warn')}`;
      } catch (e) {
        console.error(e);
      }
    }

    setInterval(pollState, 650);
    pollState();
  </script>
</body>
</html>
"""


def create_app() -> Flask:
    app = Flask(__name__)

    _load_local_env()
    _configure_http_logging()
    config = _build_config()
    runtime = AntardrishtiRuntime(config)
    app.config["runtime"] = runtime

    if _should_start_runtime():
        runtime.start()

    @app.get("/")
    def index():
        return render_template_string(DASHBOARD_HTML)

    @app.get("/api/health")
    def health():
        state = runtime.get_state()
        return jsonify({"ok": True, "runtime": state.get("runtime", "unknown")})

    @app.get("/api/state")
    def state():
        return jsonify(runtime.get_state())

    @app.post("/api/control")
    def control():
        payload = request.get_json(silent=True) or {}
        mapped = _control_to_runtime_command(payload)
        command = mapped.get("command", "")

        if not command:
            return jsonify({"ok": False, "error": "Unknown command"}), 400

        accepted = runtime.send_command(command, mapped.get("payload", {}))
        status = 200 if accepted else 429
        return jsonify({"ok": accepted, "queued": accepted, "command": command}), status

    @app.get("/video_feed")
    def video_feed():
        def generate():
            while True:
                frame = runtime.latest_frame_jpeg()
                if not frame:
                    time.sleep(0.05)
                    continue
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
                )
                time.sleep(0.03)

        return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.get("/video_laptop")
    def video_laptop():
        def generate():
            while True:
                frame = runtime.latest_laptop_jpeg()
                if not frame:
                    time.sleep(0.05)
                    continue
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
                )
                time.sleep(0.04)

        return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.get("/video_pi")
    def video_pi():
        def generate():
            while True:
                frame = runtime.latest_pi_jpeg()
                if not frame:
                    time.sleep(0.05)
                    continue
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
                )
                time.sleep(0.04)

        return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @atexit.register
    def _cleanup_runtime() -> None:
        runtime.stop()

    return app


app = create_app()


if __name__ == "__main__":
  host = os.getenv("APP_HOST", "0.0.0.0")
  port = int(os.getenv("APP_PORT", "5000"))
  _print_startup_urls(host=host, port=port)
  app.run(host=host, port=port, debug=False)
