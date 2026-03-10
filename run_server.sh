#!/bin/bash
cd /workspace/telugu-voice-agent
source venv/bin/activate

# Set your HuggingFace token — either export HF_TOKEN before running,
# or create a .env file with: HF_TOKEN=hf_your_token_here
if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
fi
export HF_TOKEN="${HF_TOKEN:-}"
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"

export PYTHONPATH="/workspace/telugu-voice-agent"
export PYTHONUNBUFFERED=1
export COQUI_TOS_AGREED=1

mkdir -p logs
python3 -u -m uvicorn backend.main:app \
  --host 0.0.0.0 \
  --port 8000 \
  --log-level info \
  --no-access-log \
  --ws-ping-interval 20 \
  --ws-ping-timeout 30 \
  >> /workspace/telugu-voice-agent/logs/server.log 2>&1
