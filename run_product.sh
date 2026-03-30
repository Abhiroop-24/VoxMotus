#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi

if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi

source .venv/bin/activate
python -m pip install --upgrade pip >/dev/null

USE_GPU="${APP_USE_GPU:-true}"
if [[ "${USE_GPU,,}" == "true" ]] && command -v nvidia-smi >/dev/null 2>&1; then
  echo "NVIDIA GPU detected. Installing CUDA-enabled torch stack..."
  python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 >/dev/null
fi

python -m pip install -r requirements.txt >/dev/null

export FLASK_DEBUG=0

PORT="${APP_PORT:-5000}"
HOST="${APP_HOST:-0.0.0.0}"

echo "Starting ANTARDRISHTI on ${HOST}:${PORT}"
python -m flask --app app run --host "${HOST}" --port "${PORT}" --no-reload
