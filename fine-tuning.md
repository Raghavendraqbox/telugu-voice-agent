# 🗣️ Telugu Full-Duplex Speech Model — Complete Guide

This guide walks you through **every step** to fine-tune Moshi/PersonaPlex to speak Telugu, giving you a full-duplex (simultaneous listen + speak) Telugu conversational AI.

---

## 📋 Overview

```
Current State:  PersonaPlex (English-only, full-duplex)
Target State:   PersonaPlex (Telugu-speaking, full-duplex)
Method:         Fine-tune Moshi using kyutai-labs/moshi-finetune with Telugu data
```

### High-Level Steps

```
Step 1: Set up hardware & environment
Step 2: Collect/create Telugu conversation audio data
Step 3: Prepare data in moshi-finetune format (stereo WAV + transcripts)
Step 4: Set up moshi-finetune
Step 5: Configure training for Telugu
Step 6: Train the model
Step 7: Add LoRA support to PersonaPlex server (or use full fine-tune weights)
Step 8: Deploy & test
```

---

## Step 1: Hardware & Environment Requirements

### GPU Requirements

| Option | GPU | VRAM | Training Time (est.) |
|--------|-----|------|---------------------|
| Minimum | 1× NVIDIA A100 | 80 GB | ~24-48 hours |
| Recommended | 4× A100 | 80 GB each | ~6-12 hours |
| Best | 8× A100 / H100 | 80 GB each | ~3-6 hours |

> **Where to get GPUs:**
> - [Google Cloud (A100)](https://cloud.google.com/compute/docs/gpus) — ~$3.67/hr per A100
> - [Lambda Labs](https://lambdalabs.com/) — ~$1.10/hr per A100
> - [RunPod](https://www.runpod.io/) — ~$1.64/hr per A100
> - [Vast.ai](https://vast.ai/) — cheapest, variable pricing
> - Google Colab Pro+ (1× A100, limited hours)

### Software Requirements

```bash
# Python 3.10+
# CUDA 11.8 or 12.x
# PyTorch 2.1+

# Core dependencies
pip install torch torchvision torchaudio
pip install sentencepiece sphn safetensors huggingface_hub
pip install wandb  # optional, for training monitoring
```

### Accounts Needed

- [ ] **Hugging Face account** — to download Moshi base weights
  - Accept license at: https://huggingface.co/kyutai/moshiko-pytorch-bf16
  - Get API token: https://huggingface.co/settings/tokens
- [ ] **Weights & Biases account** (optional) — to monitor training
  - Sign up at: https://wandb.ai

---

## Step 2: Collect Telugu Conversation Audio Data

This is the **most important step**. You need Telugu conversational speech data.

### Minimum Data Requirements

| Quality Level | Hours of Audio | Expected Result |
|--------------|----------------|-----------------|
| Bare minimum | 10-20 hours | Basic Telugu, many errors |
| Decent | 50-100 hours | Usable Telugu conversations |
| Good | 200+ hours | Natural-sounding Telugu |

### Option A: Use Existing Telugu Datasets (Recommended to start)

#### 1. AI4Bharat IndicVoices (best for conversational)
- **What:** 23,700 hours across 22 Indian languages, 15% conversational
- **Telugu portion:** ~150+ hours
- **Download:**
```python
from huggingface_hub import snapshot_download
snapshot_download(
    "ai4bharat/IndicVoices",
    repo_type="dataset",
    local_dir="./indicvoices"
)
```

#### 2. AI4Bharat Kathbath
- **What:** 1,684 hours of labeled speech, 12 Indian languages
- **Telugu portion:** ~155 hours
- **Download:**
```python
from huggingface_hub import snapshot_download
snapshot_download(
    "ai4bharat/kathbath",
    repo_type="dataset",
    local_dir="./kathbath"
)
```

#### 3. AI4Bharat Rasa (TTS quality)
- **What:** Expressive speech with male + female Telugu speakers
- **Telugu portion:** ~52 hours (27h female + 25h male)
- **Download:**
```python
from huggingface_hub import snapshot_download
snapshot_download(
    "ai4bharat/Rasa",
    repo_type="dataset",
    local_dir="./rasa"
)
```

#### 4. Mozilla Common Voice
- **What:** Crowd-sourced read speech
- **Download:** https://commonvoice.mozilla.org/te/datasets

### Option B: Create Synthetic Telugu Conversations

If you don't have enough conversational data, you can generate synthetic conversations:

#### Step B.1: Generate Telugu conversation scripts using an LLM

```python
# Use Google Gemini or GPT-4 to generate Telugu dialogues
import google.generativeai as genai

genai.configure(api_key="YOUR_GEMINI_API_KEY")
model = genai.GenerativeModel("gemini-2.0-flash")

prompt = """
Generate a natural Telugu conversation between two people (Speaker A and Speaker B).
The conversation should be about everyday topics like food, weather, travel, family.
Write the conversation in Telugu script.
Include 10-15 turns per speaker.
Format each line as:
Speaker A: [Telugu text]
Speaker B: [Telugu text]
"""

response = model.generate_content(prompt)
print(response.text)
```

#### Step B.2: Convert text to speech using Sarvam AI Bulbul V3

```python
# Sarvam AI Bulbul V3 supports Telugu TTS
# Sign up at: https://www.sarvam.ai/
# API docs: https://docs.sarvam.ai/

import requests

API_KEY = "your_sarvam_api_key"

def generate_telugu_speech(text, speaker="meera", output_path="output.wav"):
    """Generate Telugu speech using Sarvam Bulbul V3"""
    response = requests.post(
        "https://api.sarvam.ai/text-to-speech",
        headers={"API-Subscription-Key": API_KEY},
        json={
            "inputs": [text],
            "target_language_code": "te-IN",
            "speaker": speaker,  # Check docs for Telugu voices
            "model": "bulbul:v2",
        }
    )
    # Save the audio
    with open(output_path, "wb") as f:
        f.write(response.content)
```

#### Step B.3: Combine into stereo conversation files

```python
import numpy as np
import soundfile as sf

def make_stereo_conversation(speaker_a_wav, speaker_b_wav, output_path):
    """
    Combine two mono WAV files into a stereo file.
    Left channel  = Speaker A (this becomes Moshi's voice)
    Right channel = Speaker B (this becomes the user's voice)
    """
    audio_a, sr_a = sf.read(speaker_a_wav)
    audio_b, sr_b = sf.read(speaker_b_wav)

    assert sr_a == sr_b, "Sample rates must match"

    # Pad shorter audio with silence
    max_len = max(len(audio_a), len(audio_b))
    audio_a = np.pad(audio_a, (0, max_len - len(audio_a)))
    audio_b = np.pad(audio_b, (0, max_len - len(audio_b)))

    # Stack as stereo: left=Moshi, right=User
    stereo = np.column_stack([audio_a, audio_b])
    sf.write(output_path, stereo, sr_a)
```

### Option C: Record Your Own Conversations

Record real Telugu conversations with two people:
- Use a setup that captures each speaker on a separate channel
- Or record with two microphones and combine into stereo
- Aim for clear audio, minimal background noise
- Sample rate: **24000 Hz** (Moshi's native sample rate)

---

## Step 3: Prepare Data in Moshi-Finetune Format

### Required Format

```
telugu_data/
├── telugu_conversations.jsonl       # List of all audio files
└── audio_stereo/
    ├── conversation_001.wav         # Stereo WAV (left=moshi, right=user)
    ├── conversation_001.json        # Word-level transcript with timestamps
    ├── conversation_002.wav
    ├── conversation_002.json
    └── ...
```

### 3.1: Create the JSONL index file

```python
import sphn
import json
from pathlib import Path

wav_dir = "telugu_data/audio_stereo"
paths = [str(f) for f in Path(wav_dir).glob("*.wav")]
durations = sphn.durations(paths)

with open("telugu_data/telugu_conversations.jsonl", "w") as fobj:
    for p, d in zip(paths, durations):
        if d is None:
            continue
        json.dump({"path": p, "duration": d}, fobj)
        fobj.write("\n")
```

### 3.2: Create word-level transcript JSONs

Each `.wav` file needs a matching `.json` with this format:

```json
[
  {"word": "నమస్కారం", "start": 0.0, "end": 0.8, "speaker": "A"},
  {"word": "మీరు", "start": 1.2, "end": 1.5, "speaker": "B"},
  {"word": "ఎలా", "start": 1.5, "end": 1.8, "speaker": "B"},
  {"word": "ఉన్నారు", "start": 1.8, "end": 2.3, "speaker": "B"}
]
```

**Auto-generate transcripts using the annotation script:**

```bash
# This uses Whisper internally for transcription
# Whisper supports Telugu (language code: "te")
python annotate.py telugu_data/telugu_conversations.jsonl
```

> **Note:** The default annotate.py may use English Whisper. You may need to modify it to specify Telugu language. See Step 5 for details.

### 3.3: Ensure correct audio format

All WAV files must be:
- **Stereo** (2 channels)
- **Sample rate:** 24000 Hz
- **Format:** 16-bit PCM or 32-bit float

```python
import soundfile as sf
import librosa

def convert_to_moshi_format(input_path, output_path):
    """Convert any audio to Moshi's required format"""
    audio, sr = sf.read(input_path)

    # Resample to 24000 Hz if needed
    if sr != 24000:
        if audio.ndim == 1:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=24000)
        else:
            audio = np.column_stack([
                librosa.resample(audio[:, 0], orig_sr=sr, target_sr=24000),
                librosa.resample(audio[:, 1], orig_sr=sr, target_sr=24000),
            ])

    sf.write(output_path, audio, 24000)
```

---

## Step 4: Set Up moshi-finetune

### 4.1: Clone the repository

```bash
git clone https://github.com/kyutai-labs/moshi-finetune.git
cd moshi-finetune
```

### 4.2: Install dependencies

```bash
pip install -r requirements.txt
```

### 4.3: Set Hugging Face token

```bash
export HF_TOKEN=your_huggingface_token
```

### 4.4: Download base model

The base model will be downloaded automatically during training if you specify the HF repo in the config. But you can pre-download:

```python
from huggingface_hub import snapshot_download
snapshot_download("kyutai/moshiko-pytorch-bf16", local_dir="./moshiko-base")
```

---

## Step 5: Configure Training for Telugu

### 5.1: Create Telugu training config

Create a file `telugu_config.yaml`:

```yaml
# Model configuration
moshi_paths:
  hf_repo_id: "kyutai/moshiko-pytorch-bf16"

# Output directory
run_dir: "./telugu_finetune_output"

# Data configuration
data:
  train_data: "telugu_data/telugu_conversations.jsonl"
  eval_data: "telugu_data/telugu_eval.jsonl"    # optional
  shuffle: true

# Training parameters
duration_sec: 100          # Duration of audio chunks in seconds
batch_size: 4              # Reduce if OOM (out of memory)
max_steps: 5000            # More steps for a new language
seed: 42

# LoRA configuration (recommended for single GPU)
lora:
  enable: true
  rank: 128                # Higher rank = more capacity for new language
  scaling: 2.0
  ft_embed: true           # IMPORTANT: fine-tune embeddings for new language tokens

# Optimizer
optim:
  lr: 1.0e-4               # Learning rate
  weight_decay: 0.1
  pct_start: 0.1

# Loss weights
first_codebook_weight_multiplier: 5.0
text_padding_weight: 0.01

# Checkpointing
ckpt_freq: 500             # Save checkpoint every 500 steps
log_freq: 10

# Evaluation
eval_freq: 500
no_eval: false

# Gradient checkpointing (saves memory)
gradient_checkpointing: true

# Save LoRA adapters only (smaller files)
save_adapters: true

# Weights & Biases (optional)
# wandb:
#   key: "your_wandb_api_key"
#   project: "telugu-moshi-finetune"
```

### 5.2: Telugu-specific tokenizer adjustments

Since Telugu uses non-Latin script, you may need these flags when running training:

```bash
# Telugu doesn't always use spaces between words like English
# These flags adjust how the tokenizer handles text

--text_padding_id 3          # Keep same as PersonaPlex default
--end_of_text_padding_id 0   # Keep same as PersonaPlex default
--no_whitespace_before_word   # IMPORTANT for Telugu script
```

### 5.3: Modify annotate.py for Telugu (if auto-transcribing)

If using the auto-annotation script, ensure Whisper transcribes in Telugu:

```python
# In annotate.py, find the Whisper model loading section and add:
# language="te" to force Telugu transcription

# Example modification:
result = model.transcribe(
    audio_path,
    language="te",          # <-- Add this for Telugu
    word_timestamps=True,   # Required for word-level timing
)
```

---

## Step 6: Train the Model

### 6.1: Start training

**Single GPU:**
```bash
torchrun --nproc-per-node 1 -m train telugu_config.yaml
```

**Multi-GPU (4 GPUs):**
```bash
torchrun --nproc-per-node 4 --master_port $RANDOM -m train telugu_config.yaml
```

**Multi-GPU (8 GPUs):**
```bash
torchrun --nproc-per-node 8 --master_port $RANDOM -m train telugu_config.yaml
```

### 6.2: Monitor training

- **Terminal:** Watch the loss values — they should decrease steadily
- **W&B:** If configured, check your W&B dashboard for training curves
- **Key metrics to watch:**
  - `train_loss` — should decrease over time
  - `text_loss` — how well the model learns Telugu text
  - `audio_loss` — how well the model generates Telugu audio

### 6.3: Expected training timeline

| Steps | What to Expect |
|-------|---------------|
| 0-500 | Loss drops rapidly, model starts learning patterns |
| 500-2000 | Loss stabilizes, model producing some Telugu sounds |
| 2000-5000 | Model improves Telugu coherence |
| 5000+ | Diminishing returns, evaluate and decide |

### 6.4: Checkpoints

Checkpoints are saved to:
```
telugu_finetune_output/checkpoints/
├── checkpoint_000500/
│   └── consolidated/
│       ├── lora.safetensors          # LoRA weights
│       └── config.json               # Model config
├── checkpoint_001000/
├── checkpoint_001500/
└── ...
```

---

## Step 7: Deploy with PersonaPlex

### Option A: Full Fine-Tuning Weights (simplest)

If you trained with `full_finetuning: True` and `save_adapters: False`:

```bash
cd /path/to/mememates-persona

SSL_DIR=$(mktemp -d)
python -m moshi.server \
  --moshi-weight=/path/to/checkpoint/consolidated/consolidated.safetensors \
  --ssl "$SSL_DIR"
```

This works **out of the box** with the current PersonaPlex code.

### Option B: LoRA Weights (needs code change)

If you trained with LoRA (recommended), you need to add LoRA loading support to PersonaPlex's `server.py`. The changes needed are:

#### 7.1: Add `--lora-weight` argument to `server.py`

In `moshi/moshi/server.py`, add this argument in the `main()` function:

```python
parser.add_argument("--lora-weight", type=str,
                    help="Path to a LoRA adapter safetensors file.")
```

#### 7.2: Add LoRA loading in `loaders.py`

In `moshi/moshi/models/loaders.py`, add a function to merge LoRA weights:

```python
def apply_lora_weights(model: LMModel, lora_path: str):
    """Load and merge LoRA adapter weights into the base model."""
    from safetensors.torch import load_file
    lora_state = load_file(lora_path)

    model_sd = model.state_dict()
    for key, value in lora_state.items():
        if key in model_sd:
            model_sd[key] = model_sd[key] + value
    model.load_state_dict(model_sd, strict=False)
    return model
```

#### 7.3: Call it in server.py after model loading

```python
# After: lm = loaders.get_moshi_lm(...)
if args.lora_weight:
    lm = loaders.apply_lora_weights(lm, args.lora_weight)
```

Then run:
```bash
SSL_DIR=$(mktemp -d)
python -m moshi.server \
  --lora-weight=/path/to/checkpoint/consolidated/lora.safetensors \
  --ssl "$SSL_DIR"
```

---

## Step 8: Create Telugu Voice Prompts

The pre-packaged voices (NATF0, NATM0, etc.) are English speakers. For Telugu, you need Telugu voice embeddings.

### 8.1: Record a Telugu voice prompt

Record ~10-30 seconds of clear Telugu speech from one speaker:
- Save as mono WAV, 24000 Hz
- Clear pronunciation, no background noise

### 8.2: Generate voice prompt embedding

```bash
python -m moshi.offline \
  --voice-prompt "path/to/telugu_voice.wav" \
  --input-wav "path/to/test_input.wav" \
  --output-wav "test_output.wav" \
  --save-voice-prompt-embeddings
```

This saves a `.pt` file with the voice embedding.

### 8.3: Add to voice prompt directory

Copy the `.pt` file to your voice prompts directory and update the client UI to include Telugu voice options.

---

## Step 9: Update Client UI for Telugu

### 9.1: Add Telugu text prompt presets

In `client/src/pages/Queue/Queue.tsx`, add Telugu presets:

```typescript
const TEXT_PROMPT_PRESETS = [
  // ... existing English presets ...
  {
    label: "Telugu Assistant",
    text: "మీరు ఒక తెలివైన మరియు స్నేహపూర్వక ఉపాధ్యాయుడు. ప్రశ్నలకు సమాధానం ఇవ్వండి లేదా స్పష్టంగా మరియు ఆకర్షణీయంగా సలహా ఇవ్వండి.",
  },
  {
    label: "Telugu Casual",
    text: "మీరు మంచి సంభాషణ చేయడం ఆనందిస్తారు.",
  },
];
```

### 9.2: Add Telugu voice options (after creating voice prompts)

```typescript
const VOICE_OPTIONS = [
  // ... existing English voices ...
  "TELUGU_F0.pt",  // Telugu female voice
  "TELUGU_M0.pt",  // Telugu male voice
];
```

---

## Quick Reference: Useful Commands

```bash
# Set up environment
export HF_TOKEN=your_token

# Start training (single GPU)
cd moshi-finetune
torchrun --nproc-per-node 1 -m train telugu_config.yaml

# Start training (multi GPU)
torchrun --nproc-per-node 4 --master_port $RANDOM -m train telugu_config.yaml

# Run PersonaPlex with fine-tuned model (full weights)
cd mememates-persona
SSL_DIR=$(mktemp -d)
python -m moshi.server \
  --moshi-weight=/path/to/consolidated.safetensors \
  --ssl "$SSL_DIR"

# Offline test
python -m moshi.offline \
  --moshi-weight=/path/to/consolidated.safetensors \
  --voice-prompt "TELUGU_F0.pt" \
  --text-prompt "మీరు మంచి సంభాషణ చేయడం ఆనందిస్తారు." \
  --input-wav "telugu_test_input.wav" \
  --output-wav "telugu_test_output.wav"
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Out of memory (OOM) | Reduce `batch_size` to 1-2, enable `gradient_checkpointing` |
| Model outputs English | Need more Telugu data, train for more steps |
| Garbled Telugu audio | Check stereo channel assignment (left=moshi, right=user) |
| Tokenizer errors | Use `--no_whitespace_before_word` flag |
| Training loss not decreasing | Try lower learning rate (5e-5), check data quality |
| Model goes silent | Increase `duration_sec` in config |

---

## Estimated Cost & Timeline

| Item | Cost | Time |
|------|------|------|
| Telugu data collection | Free (AI4Bharat) or ~$50-200 (synthetic via Sarvam) | 1-3 days |
| Data preparation | Free | 1-2 days |
| GPU rental (1× A100, 48 hrs) | ~$55-175 | 2 days |
| Testing & iteration | Variable | 1-2 days |
| **Total** | **~$55-375** | **~5-9 days** |

---

## Resources & Links

- **moshi-finetune:** https://github.com/kyutai-labs/moshi-finetune
- **Moshi base model:** https://huggingface.co/kyutai/moshiko-pytorch-bf16
- **AI4Bharat datasets:** https://ai4bharat.iitm.ac.in/
- **Sarvam AI (Telugu TTS):** https://www.sarvam.ai/
- **Whisper (Telugu ASR):** https://github.com/openai/whisper
- **IndicVoices dataset:** https://huggingface.co/datasets/ai4bharat/IndicVoices
- **Common Voice Telugu:** https://commonvoice.mozilla.org/te/datasets
