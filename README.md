# ANTARDRISHTI

Adaptive human-like AI coach for visually impaired users or for a normal user

## Product Goals
- Blind-first audio coaching with natural feedback and correction priority
- Real-time human action analysis from dual camera input
- Quantitative movement scoring (0-100) with reference comparison
- One-command product runtime using Flask
- Auto-control option for Raspberry Pi camera stream over SSH

## Deterministic Safety Architecture
Pipeline (strict order):
1. Sensors (laptop cam + Pi cam + gestures)
2. Pose smoothing and metric filtering (EMA + spike clamp)
3. Strict state machine
4. Deterministic rule evaluation
5. Time-based validation (10-15 frame buffers, >=1 second persistence)
6. Structured feedback dataset lookup
7. Optional API rephrasing (cache-first, non-blocking)
8. Voice output

Behavior guarantees:
- No random coaching phrase selection
- Gesture commands gated by hold + lock and state validity
- Stop gesture overrides all states
- Obstacle alert has highest priority and pauses workout flow
- If camera/pose is lost, coach gives fail-safe audio instead of silent freeze
- AI API is optional and can only rephrase existing messages, never decide logic

## PS4 Objective Alignment
1. Pose Estimation
   - MediaPipe Pose on real-time video
2. Joint Angle Analysis
   - Knee, elbow, torso tilt, body line, shoulder-hip relation
3. Reference Comparison
   - Per-rep reference trajectory comparison for squat and push-up
4. Feedback
   - Voice coaching, quantified skill score, errors, and annotated video

## Current Capabilities
- Dual camera ingestion:
  - Laptop webcam
  - Pi UDP stream
- Dynamic camera intelligence:
  - Laptop camera is primary for all activity mapping and rep counting
  - Pi stream is dedicated to person detection overlays only
  - YOLOv8n on Pi stream draws person-only boxes with distance on Pi video (no person warning speech)
- Activity detection:
  - idle, standing, squat, pushup, transition
- Rep counting:
  - squat, sit-up, and jumping-jack cycles
  - squat thresholds auto-calibrate online from user motion
- Error detection:
  - squat: back tilt, depth
  - push-up: body straightness, elbow bend depth
- Gesture controls:
  - one palm start, two palms stop
  - one/two/three fingers for easy/medium/tough
- Safety context:
  - Pi person mapping is visual-only and does not interrupt coaching speech
- Humanized coaching:
  - varied phrases, cooldown, escalation on repeated error
- Local AI assist:
  - optional CUDA path (torch if available) for confidence estimation
- Product output:
  - live Flask dashboard
  - MJPEG annotated stream
  - recorded annotated session video
  - API state and control endpoints

## Project Structure
- app.py: Flask app, dashboard, API, video stream
- runtime.py: production runtime loop and orchestration
- camera.py: webcam + UDP camera management
- pose.py: pose landmarks and angle helpers
- gesture.py: hand gesture detection and mapping
- activity.py: activity understanding and posture errors
- exercise.py: rep counting and difficulty presets
- scoring.py: reference comparison and skill scoring (0-100)
- ai_assist.py: local AI confidence helper with optional CUDA backend
- joint_mapping.py: full joint-angle extraction map
- training_data.py: CSV collector for angle-labeled runtime samples
- train_activity_classifier.py: deterministic trainer exporting model weights JSON
- obstacle.py: Pi person detection (YOLO) and distance metadata
- coach.py: decision engine and session flow
- voice.py: queued speech engine
- memory.py: short-term memory for feedback adaptation
- pi_control.py: SSH auto-start for Pi stream
- datasets/registry.json: curated dataset registry
- scripts/fetch_pose_datasets.py: dataset downloader/organizer
- main.py: CLI fallback runner
- run_product.sh: one-command launcher

## One Command Run (Recommended)
```bash
bash run_product.sh
```

Then open:
http://localhost:5000

Compatibility note:
- requirements pin mediapipe==0.10.14 because newer builds may not expose the solutions API needed for pose and hand tracking.

## Human-like Voice Modes
Exact proprietary Gemini voice cloning is not available locally, but this project now supports much more natural alternatives.

1. Edge neural voice mode (most natural, internet needed)
  - in .env set APP_TTS_BACKEND=edge
  - choose voice with APP_EDGE_TTS_VOICE (default en-US-AvaNeural)

2. Piper neural voice mode (offline)
  - install Piper binary and a model locally
  - in .env set:
    - APP_TTS_BACKEND=piper
    - APP_PIPER_MODEL=/absolute/path/to/model.onnx

3. Auto mode (recommended default)
  - APP_TTS_BACKEND=auto
  - fallback chain: Piper -> Edge -> pyttsx3

Current voice backend is shown in the Flask dashboard.

## Configuration
1. Copy env template:
```bash
cp .env.example .env
```
2. Edit .env values for your setup.

To auto-start Pi camera stream from laptop:
- set PI_AUTO_START=true
- set PI_HOST, PI_USER, PI_PASSWORD

For better Pi pose quality and squat mapping:
- APP_POSE_MODEL_PI=2
- APP_POSE_MIN_DETECTION_CONF=0.55 (increase to 0.6 if noisy)
- APP_POSE_SMOOTH_ALPHA=0.35
- APP_FUSION_MIN_QUALITY=0.18

Pi person detector settings:
- APP_OBSTACLE_MODE=yolo
- APP_OBSTACLE_ENABLED=true
- APP_OBSTACLE_YOLO_MODEL=yolov8n.pt
- APP_OBSTACLE_ALERT_COOLDOWN=6.0
- APP_OBSTACLE_PERSON_ONLY=true
- APP_OBSTACLE_IGNORE_PERSON=false
- APP_OBSTACLE_DEVICE=cuda

Dashboard feeds:
- Annotated coaching stream (laptop primary overlays)
- Raw laptop camera
- Raw Pi camera (YOLO person-only with distance estimate)

You can toggle Pi person detection live from dashboard using the Toggle Pi Person Detector button.

## Manual Pi Stream (Alternative)
On Raspberry Pi:
```bash
./start_cam.sh
```

On laptop, stream test:
```bash
ffplay -fflags nobuffer -flags low_delay -framedrop udp://@:8080
```

## Flask API
- GET /api/health
- GET /api/state
- POST /api/control
  - body examples:
    - {"command": "start"}
    - {"command": "pause_toggle"}
    - {"command": "easy"}
    - {"command": "add_exercise", "value": "situp"}
    - {"command": "speak", "value": "Great control"}

## Coach Workflow (Clear Flow)
1. WAIT_USER
  - System waits until full body is visible in laptop camera.
2. WAIT_START_GESTURE
  - Show one palm (or dashboard Start) to begin.
  - Show two palms to stop at any time.
3. WAIT_DIFFICULTY
  - One finger easy, two medium, three tough.
4. READY
  - User holds steady briefly; system transitions to squats.
5. WORKOUT_SQUATS
  - Rep counting + posture coaching.
6. BREAK
  - 10 second guided recovery break.
7. WORKOUT_SITUPS
  - Rep counting + human-style spoken explanation.
8. BREAK
  - 10 second guided recovery break.
9. WORKOUT_JUMPING_JACKS
  - Rep counting + human-style spoken explanation.
10. END
  - Session summary.
  - Show one palm or press Start to restart.

Dashboard note:
- You can add more exercises from the dashboard using the Add Exercise field.

## Gesture Map
- one open palm: start
- two open palms: stop
- one finger: easy
- two fingers: medium
- three fingers: tough

## Local AI (Optional GPU)
If torch with CUDA is installed, runtime auto-uses CUDA path in ai_assist.py.
The active backend/device/GPU name are exposed in API state and dashboard.
If torch is missing, it falls back to numpy.

## Dataset-Driven Activity Training
This project now supports deterministic, non-hallucinating training from preset datasets.

1. Capture your own angle-labeled data:
- set APP_COLLECT_TRAINING_DATA=true in .env
- run the app and perform balanced reps across idle/standing/squat/pushup/transition
- collected CSV files are written to APP_TRAINING_DATA_DIR (default outputs/datasets)

2. Train classifier weights:
```bash
python train_activity_classifier.py --input-glob "outputs/datasets/training_angles_*.csv" --output models/activity_weights.json
```

Kaggle landmark CSV path (for datasets like Multi-Class Exercise Poses with x1..v33 columns):
```bash
python train_activity_classifier.py --input-glob "datasets/dataset_all_points.csv" --output models/activity_weights.json
```

Landmark label mapping is deterministic:
- rest -> idle
- left/right bicep, tricep, shoulder -> standing
- squat/pushup/transition labels map directly when present

If some runtime classes are missing in the external dataset, trainer keeps stable default priors for missing classes instead of producing undefined behavior.

3. Use trained weights in runtime:
- set APP_AI_WEIGHTS_PATH=models/activity_weights.json
- set APP_AI_CONFIDENCE=0.82 (raise for stricter trust gating)

4. Download external datasets (optional):
```bash
python scripts/fetch_pose_datasets.py --list
python scripts/fetch_pose_datasets.py --dataset coco_keypoints_2017
```

Design notes:
- Inference is deterministic (fixed model weights + fixed normalization)
- AI predictions are trust-gated; low-confidence outputs do not override rule logic
- Structured correction messages remain dataset-driven and non-generative

## Outputs
- Live annotated feed in browser
- State JSON in APP_STATE_FILE
- Recorded annotated video in APP_OUTPUT_DIR
- Skill score and component scores in dashboard and API

## Hackathon Demo Script
1. Start app with bash run_product.sh
2. User enters frame, coach asks difficulty
3. Squat guidance with rep count and corrections
4. Break, then push-up guidance with Pi primary camera
5. Pi people distance demo (person box + distance on Pi feed)
6. Session complete summary with score

## Troubleshooting
- If Pi camera feed is missing:
  - verify UDP source with ffplay
  - verify APP_PI_URL in .env
- If no speech:
  - verify local audio output and pyttsx3 backend
- If duplicate runtime appears:
  - run Flask with --no-reload (already set in run_product.sh)
