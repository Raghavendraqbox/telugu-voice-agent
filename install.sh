#!/usr/bin/env bash
# ============================================================
# Telugu Voice Agent — dependency installation script
# ============================================================
# Run from /workspace:
#   chmod +x install.sh && ./install.sh
#
# This script:
#   1. Creates a Python 3.10+ virtual environment
#   2. Installs PyTorch with CUDA 12.1 support
#   3. Installs vLLM (needs CUDA 12.1 torch first)
#   4. Installs remaining requirements
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON=${PYTHON:-python3}
VENV_DIR="$SCRIPT_DIR/venv"

echo "============================================================"
echo "  Telugu Voice Agent — Installation"
echo "============================================================"

# ── 1. Create virtual environment ──────────────────────────
if [ ! -d "$VENV_DIR" ]; then
  echo "[1/5] Creating virtual environment..."
  $PYTHON -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
pip install --upgrade pip wheel setuptools -q

# ── 2. PyTorch + torchaudio with CUDA 12.1 ─────────────────
echo "[2/5] Installing PyTorch (CUDA 12.1)..."
pip install \
  torch==2.5.1+cu121 \
  torchaudio==2.5.1+cu121 \
  --extra-index-url https://download.pytorch.org/whl/cu121 \
  -q

# Verify CUDA
python3 -c "import torch; print(f'  PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"

# ── 3. vLLM ────────────────────────────────────────────────
echo "[3/5] Installing vLLM..."
pip install vllm==0.6.6.post1 -q

# ── 4. faster-whisper ──────────────────────────────────────
echo "[4/5] Installing faster-whisper and Silero VAD..."
pip install faster-whisper==1.1.0 -q
pip install silero-vad==5.1.2 -q

# ── 5. Remaining dependencies ──────────────────────────────
echo "[5/5] Installing remaining dependencies..."
pip install \
  fastapi==0.115.6 \
  "uvicorn[standard]==0.34.0" \
  websockets==14.1 \
  python-multipart==0.0.20 \
  aiofiles==24.1.0 \
  numpy==1.26.4 \
  scipy==1.13.1 \
  soundfile==0.12.1 \
  librosa==0.10.2 \
  pydub==0.25.1 \
  av==13.1.0 \
  TTS==0.22.0 \
  transformers==4.47.1 \
  accelerate==1.2.1 \
  "huggingface-hub==0.27.1" \
  tokenizers==0.21.0 \
  pydantic==2.10.4 \
  pydantic-settings==2.7.0 \
  python-dotenv==1.0.1 \
  loguru==0.7.3 \
  httpx==0.28.1 \
  anyio==4.7.0 \
  -q

echo ""
echo "============================================================"
echo "  Installation complete!"
echo "============================================================"
echo ""
echo "Next steps:"
echo "  1. HF_TOKEN is baked into config.py as the default."
echo "     Override with: export HF_TOKEN=hf_... if needed."
echo "  2. Start the server: ./start.sh"
echo "  3. Open http://localhost:8000 in your browser"
echo "  4. Click Connect — the mic starts immediately (full-duplex phone-call mode)"
