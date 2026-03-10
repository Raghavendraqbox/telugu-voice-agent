# Telugu Voice Agent

Real-time full-duplex Telugu conversational voice agent. Speak Telugu into your microphone and receive a spoken Telugu response in under 1 second.

**Tech Stack:**
- **VAD:** Silero VAD v5 (CPU, real-time voice activity detection)
- **STT:** `vasista22/whisper-telugu-base` via faster-whisper (GPU, CTranslate2 format)
- **LLM:** `Qwen/Qwen2.5-7B-Instruct` via vLLM async streaming (GPU)
- **TTS:** `facebook/mms-tts-tel` — native Telugu TTS, 16 kHz (GPU)
- **Backend:** FastAPI + WebSocket (Python 3.11)
- **Frontend:** Vanilla JS, Web Audio API (AudioWorklet), WebSocket

---

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | 24 GB VRAM | 40–48 GB (A40/A100) |
| RAM | 32 GB | 64 GB |
| Disk | 40 GB free | 80 GB |
| CUDA | 12.x | 12.4+ |

> RunPod recommended: **A40 (48 GB)** pod with PyTorch 2.x template.

---

## Fresh RunPod Setup (Step by Step)

### Step 1 — Open a terminal on your RunPod pod

In the RunPod dashboard, click **Connect → SSH** or use the **Jupyter terminal**.

### Step 2 — Clone this repository

```bash
git clone https://github.com/Raghavendraqbox/telugu-voice-agent.git /workspace/telugu-voice-agent
cd /workspace/telugu-voice-agent
```

### Step 3 — Create Python virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip setuptools wheel
```

### Step 4 — Install all dependencies

```bash
pip install -r requirements.txt
```

> This installs: vLLM, faster-whisper, ctranslate2, transformers, TTS (Coqui), silero-vad, FastAPI, uvicorn, numpy, librosa, soundfile, loguru, pydantic-settings.
>
> **Takes 10–20 minutes on first run.**

### Step 5 — Convert the Whisper Telugu STT model to CTranslate2 format

The STT model must be in CTranslate2 format (faster-whisper requirement). Run once:

```bash
source venv/bin/activate
mkdir -p models/whisper-telugu-base-ct2

ct2-transformers-converter \
  --model vasista22/whisper-telugu-base \
  --output_dir models/whisper-telugu-base-ct2 \
  --quantization float16 \
  --force
```

**If you see an error like** `got unexpected keyword argument 'dtype'`:

```bash
# Patch the converter for transformers >= 4.47
python3 - <<'PATCH'
import re, pathlib
f = pathlib.Path("venv/lib/python3.11/site-packages/ctranslate2/converters/transformers.py")
txt = f.read_text()
# Replace dtype= with torch_dtype=
old = '''            kwargs = {
                "dtype": (torch.float16 if self._load_as_float16 else None)
            }'''
new = '''            _dtype = torch.float16 if self._load_as_float16 else None
            kwargs = {"torch_dtype": _dtype} if _dtype is not None else {}'''
if old in txt:
    f.write_text(txt.replace(old, new))
    print("Patched successfully")
else:
    print("Pattern not found — may already be patched or different version")
PATCH

# Now re-run the conversion
ct2-transformers-converter \
  --model vasista22/whisper-telugu-base \
  --output_dir models/whisper-telugu-base-ct2 \
  --quantization float16 \
  --force
```

Verify it worked — you should see `model.bin` (≈ 139 MB):
```bash
ls -lh models/whisper-telugu-base-ct2/
# Expected: config.json  model.bin  tokenizer files  vocabulary.json
```

### Step 6 — Pre-download the MMS-TTS Telugu model

```bash
source venv/bin/activate
python3 -c "
from transformers import VitsModel, AutoTokenizer
print('Downloading MMS-TTS Telugu...')
AutoTokenizer.from_pretrained('facebook/mms-tts-tel')
VitsModel.from_pretrained('facebook/mms-tts-tel')
print('Done!')
"
```

### Step 7 — Pre-download the Qwen 2.5-7B LLM

```bash
source venv/bin/activate
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    'Qwen/Qwen2.5-7B-Instruct',
    ignore_patterns=['*.gguf'],
)
print('Done!')
"
```

> No HuggingFace token needed for Qwen — it is a public model.

### Step 8 — Set your environment variables

Create a `.env` file (optional, the server script also exports them inline):

```bash
cat > /workspace/telugu-voice-agent/.env <<EOF
HF_TOKEN=your_hf_token_here
COQUI_TOS_AGREED=1
EOF
```

---

## Starting the Server

```bash
cd /workspace/telugu-voice-agent
bash run_server.sh
```

The server starts in the background and logs to `logs/server.log`.

**Watch startup progress:**
```bash
tail -f /workspace/telugu-voice-agent/logs/server.log
```

**Startup sequence (takes 3–5 minutes):**
```
[1/4] Loading Silero VAD...       ← ~1s
[2/4] Loading STT model...        ← ~2s
[3/4] Loading LLM engine (vLLM).. ← 2–4 minutes (Qwen 7B)
[4/4] Loading TTS model...        ← ~1s (MMS-TTS)
Telugu Voice Agent ready          ← server is now accepting connections
```

### Alternative: start manually (foreground, useful for debugging)

```bash
cd /workspace/telugu-voice-agent
source venv/bin/activate

export HF_TOKEN="your_hf_token_here"
export HUGGING_FACE_HUB_TOKEN="your_hf_token_here"
export PYTHONPATH="/workspace/telugu-voice-agent"
export COQUI_TOS_AGREED=1
export PYTHONUNBUFFERED=1

python3 -m uvicorn backend.main:app \
  --host 0.0.0.0 \
  --port 8000 \
  --log-level info
```

---

## Accessing the UI

### On RunPod

1. In RunPod dashboard → your pod → **Connect** tab
2. Expose **HTTP port 8000**
3. Your URL will be: `https://YOUR_POD_ID-8000.proxy.runpod.net/`
4. Open this URL in **Chrome** (Firefox may have mic issues with AudioWorklet)

### Locally (if running on your own machine)

Open `http://localhost:8000/`

---

## Using the Voice Agent

1. Open the URL in Chrome
2. Click the **green phone button** at the bottom
3. Allow microphone access when the browser asks
4. Wait for status: **"Ready — speak in Telugu"**
5. Speak a Telugu sentence — your words appear in the center transcript
6. The agent thinks and responds in spoken Telugu + shows text

**To end the call:** click the red phone button.

### Microphone Setup (Windows)

If your mic is not being detected (RMS stays near 0):

1. Press `Win + R` → type `mmsys.cpl` → Enter
2. Click **Recording** tab
3. Double-click **your microphone**
4. Click **Levels** tab
5. Set volume to **100**
6. Optionally add **+20 dB** boost
7. Click OK, then retry

---

## Monitoring & Logs

```bash
# Live log stream
tail -f /workspace/telugu-voice-agent/logs/server.log

# Check if server process is running
ps aux | grep uvicorn

# Health check (shows GPU usage, model status)
curl http://localhost:8000/health

# Stop the server
pkill -f "uvicorn backend.main"
```

**What to look for in logs when speaking:**
```
VAD: IDLE→LISTENING (prob=0.92)          ← speech detected
VAD: LISTENING→IDLE (silence timeout...) ← you stopped speaking
STT (95ms): 'నమస్కారం అని చెప్పండి'     ← your words transcribed
LLM first token in 820ms                 ← response generating
TTS first chunk ready in 180ms           ← audio being sent
```

---

## Architecture

```
User's Browser (Chrome)
  │
  │  getUserMedia (mic, echoCancellation=true)
  │  ↓
  │  AudioWorklet (PCMProcessor)
  │    - browser native rate (44100/48000 Hz)
  │    - linear-interpolation resample → 16000 Hz
  │    - emit 320-sample (20ms) Int16 chunks
  │  ↓
  │  WebSocket (wss://)  ← binary Int16 PCM, 16kHz mono
  │
  ▼
RunPod Server (FastAPI, port 8000)
  │
  ├─ /ws/audio  WebSocket endpoint
  │    │
  │    ├─ receive_loop  ──────────────────────────────────┐
  │    │   Silero VAD (CPU, per 512-sample frame)         │
  │    │   State: IDLE → LISTENING → PROCESSING           │
  │    │   Sends vad_state JSON to frontend for UI        │
  │    │                                                  │
  │    └─ pipeline_loop ◄──────────────────────────────── ┘
  │         │  (complete utterance bytes from VAD)
  │         │
  │         ├─ STT: faster-whisper (GPU, CTranslate2)
  │         │   vasista22/whisper-telugu-base
  │         │   Input: float32 16kHz PCM
  │         │   Output: Telugu text string
  │         │
  │         ├─ LLM: vLLM AsyncLLMEngine (GPU)
  │         │   Qwen/Qwen2.5-7B-Instruct (ChatML format)
  │         │   Streaming token generation
  │         │   System prompt: respond in Telugu
  │         │
  │         └─ TTS: facebook/mms-tts-tel (GPU)
  │             Native Telugu VITS model
  │             Input: text string (sentence chunks)
  │             Output: float32 → Int16 PCM @ 16kHz
  │             Streamed sentence-by-sentence
  │
  ├─ /        Static frontend files (HTML, CSS, JS)
  └─ /health  JSON health + GPU info
```

---

## Configuration

All settings are in `backend/config.py`. Key values:

```python
STT_MODEL = "/workspace/telugu-voice-agent/models/whisper-telugu-base-ct2"
LLM_MODEL = "Qwen/Qwen2.5-7B-Instruct"

# VAD thresholds
VAD_SPEECH_THRESHOLD = 0.15    # probability to trigger LISTENING
VAD_SILENCE_THRESHOLD = 0.05   # probability to count as silence
VAD_MIN_SPEECH_DURATION_MS = 200   # ms of speech needed to confirm utterance
VAD_SILENCE_DURATION_MS = 300      # ms of silence to end utterance

# LLM
LLM_MAX_TOKENS = 150           # keep responses short for low latency
LLM_TEMPERATURE = 0.7

# Audio
STT_SAMPLE_RATE = 16000        # Hz
TTS_SAMPLE_RATE = 16000        # Hz (MMS-TTS native)
AUDIO_CHUNK_MS = 20            # ms per WebSocket frame
```

Override any setting with environment variables (uppercase, same name):
```bash
export VAD_SPEECH_THRESHOLD=0.2
export LLM_MAX_TOKENS=100
```

---

## Troubleshooting

### Server won't start
```bash
# Check CUDA
python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# Check port is free
ss -tlnp | grep 8000
```

### "Model still loading" WebSocket error
The models take 3–5 minutes to load. Wait for `Telugu Voice Agent ready` in the log before connecting.

### No voice output (audio doesn't play)
- Did you click the **green button** before speaking? The AudioContext requires a user click.
- Check browser console (F12 → Console) for errors.
- Make sure you're using **Chrome** (not Firefox/Safari).

### Voice sounds wrong / not Telugu
- The server must be using `backend/tts/mms_tts.py` — check `backend/main.py` imports `from backend.tts.mms_tts import TeluguTTS`
- Check the log at startup: should say `Loading TTS model: facebook/mms-tts-tel`

### Barge-in / response immediately cut off
- Normal: if you speak while the agent is responding, it stops (barge-in).
- If triggering when you're silent: echo cancellation may be off. Check `frontend/app.js` — `echoCancellation: true` should be set.

### Out of memory (OOM)
```bash
# Check VRAM usage
nvidia-smi

# Reduce LLM memory fraction in config.py
VLLM_GPU_MEMORY_UTILIZATION = 0.45  # default 0.55
```

### Commit / push after pod restart (if you make changes)
```bash
cd /workspace/telugu-voice-agent
git config user.email "raghavendraqbox@gmail.com"
git config user.name "Raghavendraqbox"
git add -A
git commit -m "your message"
git push origin main
```

---

## File Structure

```
telugu-voice-agent/
├── README.md
├── requirements.txt
├── run_server.sh                  ← start the server
├── start.sh                       ← alternative start script
├── .env                           ← HF_TOKEN (create this yourself)
│
├── backend/
│   ├── main.py                    ← FastAPI app, model loading, WebSocket
│   ├── config.py                  ← all configuration
│   ├── vad/
│   │   └── silero_vad_engine.py   ← Silero VAD v5 wrapper
│   ├── stt/
│   │   └── whisper_stt.py         ← faster-whisper Telugu STT
│   ├── llm/
│   │   └── vllm_engine.py         ← vLLM async streaming engine
│   ├── tts/
│   │   ├── mms_tts.py             ← MMS-TTS native Telugu (ACTIVE)
│   │   └── xtts_tts.py            ← XTTS v2 (legacy, not used)
│   └── pipeline/
│       └── voice_pipeline.py      ← VAD→STT→LLM→TTS orchestration
│
├── frontend/
│   ├── index.html                 ← phone-call UI
│   ├── style.css                  ← dark theme styles
│   └── app.js                     ← WebSocket + Web Audio API
│
├── models/
│   └── whisper-telugu-base-ct2/   ← CTranslate2 model (create in Step 5)
│       ├── model.bin              ← 139 MB weights
│       ├── config.json
│       └── vocabulary.json
│
└── logs/
    └── server.log                 ← server output (created at runtime)
```

---

## Latency Profile

| Stage | Target | Typical on A40 |
|-------|--------|----------------|
| VAD detection | < 50ms | ~30ms |
| STT (Whisper) | < 200ms | ~100ms |
| LLM first token | < 300ms | ~800ms |
| TTS first chunk | < 200ms | ~200ms |
| **Total (E2E)** | **< 700ms** | **~1.1s** |

The LLM is the main bottleneck. To reduce further:
- Use `Qwen/Qwen2.5-1.5B-Instruct` (much faster, slightly lower quality)
- Enable AWQ quantization: `model_name = "Qwen/Qwen2.5-7B-Instruct-AWQ"`
- Reduce `LLM_MAX_TOKENS` to 80–100

---

## Credits

- [vasista22/whisper-telugu-base](https://huggingface.co/vasista22/whisper-telugu-base) — Telugu Whisper fine-tune
- [facebook/mms-tts-tel](https://huggingface.co/facebook/mms-tts-tel) — MMS Telugu TTS
- [Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) — LLM
- [silero-vad](https://github.com/snakers4/silero-models) — Voice Activity Detection
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — CTranslate2-accelerated Whisper
- [vLLM](https://github.com/vllm-project/vllm) — Fast LLM inference engine
