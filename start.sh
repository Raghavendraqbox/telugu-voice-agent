#!/usr/bin/env bash
# ============================================================
# Telugu Voice Agent — startup script
# ============================================================
# Run from /workspace:
#   chmod +x start.sh && ./start.sh
#
# Assumes:
#   - Python 3.10+ with venv at /workspace/venv
#   - NVIDIA GPU with CUDA 12+
#   - All pip packages installed (see install.sh)
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================================"
echo "  Telugu Voice Agent"
echo "============================================================"

# Verify GPU
if ! command -v nvidia-smi &>/dev/null; then
  echo "ERROR: nvidia-smi not found. A CUDA GPU is required."
  exit 1
fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

# Activate virtual environment if it exists
if [ -d "venv" ]; then
  echo "Activating venv..."
  source venv/bin/activate
fi

# Verify Python imports
python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available!'; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}')"

mkdir -p logs

# Accept Coqui TTS Terms of Service non-interactively
export COQUI_TOS_AGREED=1

echo "Starting FastAPI server on http://0.0.0.0:8000"
echo "Open http://localhost:8000 in your browser to use the voice agent."
echo "Press Ctrl+C to stop."
echo ""

PYTHONPATH="$SCRIPT_DIR" python3 -m uvicorn backend.main:app \
  --host 0.0.0.0 \
  --port 8000 \
  --log-level info \
  --no-access-log \
  --ws-ping-interval 20 \
  --ws-ping-timeout 30
