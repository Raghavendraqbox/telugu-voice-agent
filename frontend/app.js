/**
 * Telugu Voice Agent — Full-Duplex Phone Call Frontend
 *
 * Audio capture pipeline:
 *   getUserMedia (16kHz, mono)
 *   → AudioContext → AudioWorklet (PCMProcessor, 128-sample callbacks)
 *   → accumulate 320 samples (20ms at 16kHz)
 *   → convert Float32 → Int16
 *   → send as binary WebSocket frame (640 bytes)
 *
 * Audio playback pipeline:
 *   WebSocket binary frames (raw Int16 PCM, 22050Hz)
 *   → convert Int16 → Float32
 *   → AudioContext (22050Hz) → scheduled BufferSource nodes
 *   → speakers (gapless, pre-scheduled 80ms ahead)
 *
 * Barge-in:
 *   When agent is speaking (playbackActive == true) and VAD ring goes active
 *   (server sends vad_state "listening"), stop playback immediately and
 *   send {"type":"interrupt"} to server.
 *
 * WebSocket protocol:
 *   Client → Server  binary: 640-byte Int16 PCM frames (20ms, 16kHz, mono)
 *   Client → Server  text:   JSON {"type":"interrupt"}
 *   Server → Client  binary: raw Int16 PCM (TTS, 22050Hz, mono, no header)
 *   Server → Client  text:   JSON {type:"ready"|"vad_state"|"transcript"|"error"}
 */

'use strict';

// ──────────────────────────────────────────────────────────────────────────────
// Constants
// ──────────────────────────────────────────────────────────────────────────────
const WS_URL              = `ws://${location.host}/ws/audio`;
const CAPTURE_SAMPLE_RATE = 16000;
const PLAYBACK_SAMPLE_RATE = 22050;
const CHUNK_SAMPLES       = 320;    // 20ms at 16kHz
const CHUNK_BYTES         = 640;    // 320 samples * 2 bytes

// ──────────────────────────────────────────────────────────────────────────────
// AudioWorklet processor source (inlined as Blob URL to avoid separate file)
// ──────────────────────────────────────────────────────────────────────────────
const WORKLET_CODE = `
class PCMProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this._buffer = new Float32Array(0);
  }
  process(inputs) {
    const input = inputs[0];
    if (!input || !input[0]) return true;
    const samples = input[0]; // Float32Array, 128 samples per callback
    // Append to internal buffer
    const merged = new Float32Array(this._buffer.length + samples.length);
    merged.set(this._buffer, 0);
    merged.set(samples, this._buffer.length);
    this._buffer = merged;
    // Emit complete 320-sample (20ms) chunks
    const chunkSize = 320;
    while (this._buffer.length >= chunkSize) {
      const chunk = this._buffer.slice(0, chunkSize);
      this._buffer = this._buffer.slice(chunkSize);
      this.port.postMessage(chunk.buffer, [chunk.buffer]);
    }
    return true;
  }
}
registerProcessor('pcm-processor', PCMProcessor);
`;

// ──────────────────────────────────────────────────────────────────────────────
// DOM elements
// ──────────────────────────────────────────────────────────────────────────────
const $connectBtn   = document.getElementById('connect-btn');
const $connDot      = document.getElementById('connection-dot');
const $statusText   = document.getElementById('status-text');
const $transcript   = document.getElementById('transcript');
const $clearBtn     = document.getElementById('clear-btn');
const $canvas       = document.getElementById('visualizer');
const $latStt       = document.getElementById('lat-stt');
const $latLlm       = document.getElementById('lat-llm');
const $latTts       = document.getElementById('lat-tts');
const $latTotal     = document.getElementById('lat-total');
const $agentAvatar  = document.getElementById('agent-avatar');
const $vadRing      = document.getElementById('vad-ring');
const $vadLabel     = document.getElementById('vad-label');

// ──────────────────────────────────────────────────────────────────────────────
// State
// ──────────────────────────────────────────────────────────────────────────────
let ws               = null;
let captureCtx       = null;   // AudioContext for mic capture (16kHz)
let workletNode      = null;
let analyserNode     = null;
let micStream        = null;
let vizAnimId        = null;

// Playback
let playbackCtx      = null;   // AudioContext for TTS playback (22050Hz)
let nextPlayTime     = 0;
let playbackActive   = false;

// Agent/pipeline state
let agentState       = 'idle'; // idle | listening | processing | speaking
let vadActive        = false;  // true when server reports voice detected

// Turn timing for latency display
let turnStartTime    = null;
let sttEndTime       = null;
let ttsFirstChunk    = null;

// Transcript UI
let agentBubbleEl    = null;
let agentText        = '';

// Canvas context
const canvasCtx = $canvas.getContext('2d');

// ──────────────────────────────────────────────────────────────────────────────
// Connection management
// ──────────────────────────────────────────────────────────────────────────────

function connect() {
  setConnectionState('connecting');
  $connectBtn.disabled = true;

  ws = new WebSocket(WS_URL);
  ws.binaryType = 'arraybuffer';

  ws.addEventListener('open', onWsOpen);
  ws.addEventListener('message', onWsMessage);
  ws.addEventListener('close', onWsClose);
  ws.addEventListener('error', onWsError);
}

function disconnect() {
  stopCapture();
  stopPlayback();
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.close(1000, 'User disconnect');
  }
  ws = null;
  setConnectionState('disconnected');
  setAgentState('idle');
  setVadActive(false);
  stopVisualizer();
}

async function onWsOpen() {
  console.log('[WS] Connected');
  setConnectionState('connected');
  setAgentState('idle');
  $connectBtn.textContent = 'Disconnect';
  $connectBtn.className = 'btn btn-disconnect';
  $connectBtn.disabled = false;

  // Start mic + worklet immediately — phone call mode, always on
  await startCapture();
}

function onWsClose(evt) {
  console.log(`[WS] Closed: ${evt.code} ${evt.reason}`);
  setConnectionState('disconnected');
  setAgentState('idle');
  setVadActive(false);
  $connectBtn.textContent = 'Connect';
  $connectBtn.className = 'btn btn-connect';
  $connectBtn.disabled = false;
  stopCapture();
  stopPlayback();
  stopVisualizer();
}

function onWsError(evt) {
  console.error('[WS] Error', evt);
  setConnectionState('error');
  showError('WebSocket connection error. Is the server running?');
}

// ──────────────────────────────────────────────────────────────────────────────
// Message handling
// ──────────────────────────────────────────────────────────────────────────────

function onWsMessage(evt) {
  if (evt.data instanceof ArrayBuffer) {
    handleIncomingAudio(evt.data);
  } else {
    let msg;
    try { msg = JSON.parse(evt.data); } catch { return; }

    switch (msg.type) {
      case 'ready':
        handleReady(msg);
        break;
      case 'vad_state':
        handleVadState(msg);
        break;
      case 'transcript':
        handleTranscript(msg);
        break;
      case 'error':
        showError(msg.message);
        break;
    }
  }
}

function handleReady(msg) {
  console.log('[Agent] Ready, audio format:', msg.audio_format);
  setAgentState('idle');
  updateStatusText('Ready — speak in Telugu');
}

function handleVadState(msg) {
  const state = msg.state; // "listening" | "silent" | "processing" | "speaking"

  switch (state) {
    case 'listening':
      // User voice detected — VAD ring active
      setVadActive(true);
      setAgentState('listening');
      updateStatusText('Listening...');
      turnStartTime = performance.now();
      sttEndTime = null;
      ttsFirstChunk = null;
      agentBubbleEl = null;
      agentText = '';
      // Barge-in: if agent was speaking, stop playback + send interrupt
      if (playbackActive) {
        console.log('[Barge-in] User spoke while agent talking — interrupting');
        stopPlayback();
        sendInterrupt();
      }
      break;

    case 'silent':
      // Voice went quiet (VAD gate closed), still waiting for silence timeout
      setVadActive(false);
      if (agentState === 'listening') {
        updateStatusText('...');
      }
      break;

    case 'processing':
      setVadActive(false);
      setAgentState('processing');
      updateStatusText('Processing...');
      sttEndTime = performance.now();
      if (turnStartTime) {
        updateLatency('stt', sttEndTime - turnStartTime);
      }
      showAgentTyping();
      break;

    case 'speaking':
      setAgentState('speaking');
      updateStatusText('Agent speaking...');
      if (ttsFirstChunk === null) {
        ttsFirstChunk = performance.now();
        if (sttEndTime) {
          updateLatency('llm', ttsFirstChunk - sttEndTime);
        }
        if (turnStartTime) {
          updateLatency('total', ttsFirstChunk - turnStartTime);
        }
      }
      break;
  }
}

function handleTranscript(msg) {
  if (msg.role === 'user') {
    appendUserBubble(msg.text);
  } else if (msg.role === 'agent') {
    finaliseAgentBubble(msg.text);
    if (turnStartTime) {
      updateLatency('total', performance.now() - turnStartTime);
    }
  }
}

// ──────────────────────────────────────────────────────────────────────────────
// Audio capture — AudioWorklet (low-latency raw PCM)
// ──────────────────────────────────────────────────────────────────────────────

async function startCapture() {
  // Request microphone
  try {
    micStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        sampleRate: CAPTURE_SAMPLE_RATE,
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
    });
  } catch (err) {
    showError(`Microphone access denied: ${err.message}`);
    return;
  }

  // Create AudioContext for capture at 16kHz
  captureCtx = new AudioContext({ sampleRate: CAPTURE_SAMPLE_RATE });

  // Resume on user gesture (browser autoplay policy)
  if (captureCtx.state === 'suspended') {
    await captureCtx.resume();
  }

  // Analyser for visualiser (waveform of mic input)
  analyserNode = captureCtx.createAnalyser();
  analyserNode.fftSize = 256;
  const sourceNode = captureCtx.createMediaStreamSource(micStream);
  sourceNode.connect(analyserNode);

  // Load AudioWorklet (inlined as Blob URL)
  const blob = new Blob([WORKLET_CODE], { type: 'application/javascript' });
  const workletUrl = URL.createObjectURL(blob);
  try {
    await captureCtx.audioWorklet.addModule(workletUrl);
  } catch (err) {
    showError(`AudioWorklet failed to load: ${err.message}`);
    return;
  } finally {
    URL.revokeObjectURL(workletUrl);
  }

  workletNode = new AudioWorkletNode(captureCtx, 'pcm-processor');

  // Handle 20ms PCM chunks from worklet
  workletNode.port.onmessage = (evt) => {
    const float32Array = new Float32Array(evt.data);
    sendPcmChunk(float32Array);
  };

  // Connect: source → analyser, source → worklet (worklet output disconnected)
  sourceNode.connect(workletNode);
  // workletNode has no output consumers — it only posts messages

  startVisualizer();
  console.log('[Capture] AudioWorklet started, streaming 20ms PCM chunks');
}

function stopCapture() {
  if (workletNode) {
    workletNode.port.onmessage = null;
    workletNode.disconnect();
    workletNode = null;
  }
  if (analyserNode) {
    analyserNode.disconnect();
    analyserNode = null;
  }
  if (captureCtx && captureCtx.state !== 'closed') {
    captureCtx.close();
    captureCtx = null;
  }
  if (micStream) {
    micStream.getTracks().forEach(t => t.stop());
    micStream = null;
  }
}

/**
 * Convert a Float32 PCM chunk to Int16 and send over WebSocket.
 * @param {Float32Array} float32Chunk - 320-sample (20ms) mono audio
 */
function sendPcmChunk(float32Chunk) {
  if (!ws || ws.readyState !== WebSocket.OPEN) return;

  const int16 = new Int16Array(float32Chunk.length);
  for (let i = 0; i < float32Chunk.length; i++) {
    // Clamp then scale to int16 range
    const clamped = Math.max(-1.0, Math.min(1.0, float32Chunk[i]));
    int16[i] = clamped < 0
      ? Math.round(clamped * 32768)
      : Math.round(clamped * 32767);
  }
  ws.send(int16.buffer);
}

// ──────────────────────────────────────────────────────────────────────────────
// Audio playback — raw Int16 PCM scheduling
// ──────────────────────────────────────────────────────────────────────────────

function ensurePlaybackCtx() {
  if (!playbackCtx || playbackCtx.state === 'closed') {
    playbackCtx = new AudioContext({ sampleRate: PLAYBACK_SAMPLE_RATE });
    nextPlayTime = 0;
    playbackActive = false;
  }
  // Resume if suspended (autoplay policy)
  if (playbackCtx.state === 'suspended') {
    playbackCtx.resume();
  }
  return playbackCtx;
}

function handleIncomingAudio(arrayBuffer) {
  if (!arrayBuffer || arrayBuffer.byteLength === 0) return;

  // Track TTS latency on first chunk per turn
  if (ttsFirstChunk === null) {
    ttsFirstChunk = performance.now();
    if (sttEndTime) {
      updateLatency('tts', ttsFirstChunk - sttEndTime);
    }
    if (turnStartTime) {
      updateLatency('total', ttsFirstChunk - turnStartTime);
    }
  }

  const ctx = ensurePlaybackCtx();

  // Decode raw Int16 PCM → Float32 AudioBuffer
  const int16 = new Int16Array(arrayBuffer);
  const numSamples = int16.length;
  if (numSamples === 0) return;

  const audioBuffer = ctx.createBuffer(1, numSamples, PLAYBACK_SAMPLE_RATE);
  const channelData = audioBuffer.getChannelData(0);
  for (let i = 0; i < numSamples; i++) {
    channelData[i] = int16[i] / 32768.0;
  }

  scheduleBuffer(ctx, audioBuffer);
}

function scheduleBuffer(ctx, buffer) {
  const source = ctx.createBufferSource();
  source.buffer = buffer;
  source.connect(ctx.destination);

  const now = ctx.currentTime;
  // Keep 80ms ahead to avoid under-runs without adding noticeable latency
  if (nextPlayTime < now + 0.08) {
    nextPlayTime = now + 0.08;
  }
  source.start(nextPlayTime);
  source.onended = onBufferEnded;
  nextPlayTime += buffer.duration;
  playbackActive = true;
}

function onBufferEnded() {
  // Check if playback queue has been drained
  if (!playbackCtx) return;
  const now = playbackCtx.currentTime;
  if (nextPlayTime <= now + 0.05) {
    playbackActive = false;
  }
}

function stopPlayback() {
  if (playbackCtx && playbackCtx.state !== 'closed') {
    playbackCtx.close();
    playbackCtx = null;
  }
  nextPlayTime = 0;
  playbackActive = false;
}

// ──────────────────────────────────────────────────────────────────────────────
// Interrupt (barge-in)
// ──────────────────────────────────────────────────────────────────────────────

function sendInterrupt() {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: 'interrupt' }));
    console.log('[Interrupt] Sent to server');
  }
}

// ──────────────────────────────────────────────────────────────────────────────
// Visualizer (waveform of mic input)
// ──────────────────────────────────────────────────────────────────────────────

function startVisualizer() {
  if (vizAnimId) cancelAnimationFrame(vizAnimId);
  drawVisualizer();
}

function stopVisualizer() {
  if (vizAnimId) cancelAnimationFrame(vizAnimId);
  vizAnimId = null;
  clearCanvas();
}

function clearCanvas() {
  canvasCtx.fillStyle = '#242424';
  canvasCtx.fillRect(0, 0, $canvas.width, $canvas.height);
}

function drawVisualizer() {
  vizAnimId = requestAnimationFrame(drawVisualizer);
  if (!analyserNode) { clearCanvas(); return; }

  const bufLen = analyserNode.frequencyBinCount;
  const data   = new Uint8Array(bufLen);
  analyserNode.getByteTimeDomainData(data);

  const W = $canvas.width;
  const H = $canvas.height;

  // Background
  canvasCtx.fillStyle = '#242424';
  canvasCtx.fillRect(0, 0, W, H);

  // Waveform colour by agent state
  const colours = {
    idle:       '#3a3a3a',
    listening:  '#FF9933',
    processing: '#f0c040',
    speaking:   '#50c080',
  };
  canvasCtx.strokeStyle = colours[agentState] || colours.idle;
  canvasCtx.lineWidth   = 2;
  canvasCtx.beginPath();

  const sliceW = W / bufLen;
  let x = 0;
  for (let i = 0; i < bufLen; i++) {
    const v = data[i] / 128.0;
    const y = (v * H) / 2;
    if (i === 0) canvasCtx.moveTo(x, y);
    else         canvasCtx.lineTo(x, y);
    x += sliceW;
  }
  canvasCtx.lineTo(W, H / 2);
  canvasCtx.stroke();
}

// ──────────────────────────────────────────────────────────────────────────────
// UI state management
// ──────────────────────────────────────────────────────────────────────────────

function setConnectionState(state) {
  $connDot.className = `conn-dot ${state}`;
}

function setAgentState(state) {
  agentState = state;

  // Avatar classes
  $agentAvatar.classList.remove('user-speaking', 'agent-speaking', 'processing');
  $statusText.classList.remove('listening', 'processing', 'speaking', 'error');

  if (state === 'listening') {
    $agentAvatar.classList.add('user-speaking');
    $statusText.classList.add('listening');
  } else if (state === 'processing') {
    $agentAvatar.classList.add('processing');
    $statusText.classList.add('processing');
  } else if (state === 'speaking') {
    $agentAvatar.classList.add('agent-speaking');
    $statusText.classList.add('speaking');
  }
}

function setVadActive(active) {
  vadActive = active;
  if (active) {
    $vadRing.classList.add('active');
    $vadLabel.classList.add('active');
    $vadLabel.textContent = 'Voice detected';
  } else {
    $vadRing.classList.remove('active');
    $vadLabel.classList.remove('active');
    $vadLabel.textContent = agentState === 'idle' || agentState === 'listening'
      ? 'Mic active'
      : 'Mic active';
  }
}

function updateStatusText(text) {
  $statusText.textContent = text;
}

function showError(msg) {
  $statusText.textContent = `Error: ${msg}`;
  $statusText.classList.remove('listening', 'processing', 'speaking');
  $statusText.classList.add('error');
  console.error('[Error]', msg);
}

// ──────────────────────────────────────────────────────────────────────────────
// Transcript helpers
// ──────────────────────────────────────────────────────────────────────────────

function clearEmptyPlaceholder() {
  const empty = $transcript.querySelector('.transcript-empty');
  if (empty) empty.remove();
}

function appendUserBubble(text) {
  clearEmptyPlaceholder();
  const turn = document.createElement('div');
  turn.className = 'chat-turn user';
  turn.innerHTML = `
    <span class="bubble-role">You</span>
    <div class="bubble">${escHtml(text)}</div>
  `;
  $transcript.appendChild(turn);
  scrollTranscript();
}

function showAgentTyping() {
  clearEmptyPlaceholder();
  if (agentBubbleEl) return;
  const turn = document.createElement('div');
  turn.className = 'chat-turn agent';
  const bubble = document.createElement('div');
  bubble.className = 'bubble typing';
  bubble.innerHTML = `
    <span class="typing-dot"></span>
    <span class="typing-dot"></span>
    <span class="typing-dot"></span>
  `;
  turn.appendChild(bubble);
  $transcript.appendChild(turn);
  agentBubbleEl = bubble;
  scrollTranscript();
}

function finaliseAgentBubble(text) {
  if (agentBubbleEl) {
    agentBubbleEl.className = 'bubble';
    agentBubbleEl.textContent = text;
    agentBubbleEl = null;
  } else {
    clearEmptyPlaceholder();
    const turn = document.createElement('div');
    turn.className = 'chat-turn agent';
    turn.innerHTML = `
      <span class="bubble-role">Agent</span>
      <div class="bubble">${escHtml(text)}</div>
    `;
    $transcript.appendChild(turn);
  }
  agentText = '';
  scrollTranscript();
}

function scrollTranscript() {
  $transcript.scrollTop = $transcript.scrollHeight;
}

function escHtml(str) {
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

// ──────────────────────────────────────────────────────────────────────────────
// Latency display
// ──────────────────────────────────────────────────────────────────────────────

function updateLatency(key, ms) {
  const formatted = `${Math.round(ms)}ms`;
  const map = { stt: $latStt, llm: $latLlm, tts: $latTts, total: $latTotal };
  const el = map[key];
  if (!el) return;
  el.textContent = formatted;
  const thresholds = { stt: 200, llm: 300, tts: 200, total: 700 };
  const thresh = thresholds[key];
  el.style.color = ms <= thresh ? '#50c080'
                  : ms <= thresh * 1.5 ? '#f0c040'
                  : '#e05050';
}

// ──────────────────────────────────────────────────────────────────────────────
// Event listeners
// ──────────────────────────────────────────────────────────────────────────────

$connectBtn.addEventListener('click', () => {
  if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) {
    disconnect();
  } else {
    connect();
  }
});

$clearBtn.addEventListener('click', () => {
  $transcript.innerHTML = '<div class="transcript-empty">Connect and start speaking in Telugu…</div>';
  agentBubbleEl = null;
  agentText = '';
});

// Resume AudioContext on user gesture (browser autoplay policy)
document.addEventListener('click', () => {
  if (captureCtx && captureCtx.state === 'suspended') captureCtx.resume();
  if (playbackCtx && playbackCtx.state === 'suspended') playbackCtx.resume();
}, { capture: true });

window.addEventListener('beforeunload', () => {
  if (ws) ws.close(1001, 'Page unload');
});

// ──────────────────────────────────────────────────────────────────────────────
// Initialise
// ──────────────────────────────────────────────────────────────────────────────

(function init() {
  clearCanvas();
  $statusText.textContent = 'Disconnected';
  agentState = 'idle';
  setVadActive(false);
  $vadLabel.textContent = 'Mic inactive';

  if (!window.WebSocket) {
    showError('WebSocket not supported in this browser.');
    $connectBtn.disabled = true;
    return;
  }

  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    showError('Microphone access not supported. Use Chrome, Firefox, or Edge.');
    $connectBtn.disabled = true;
    return;
  }

  if (typeof AudioWorkletNode === 'undefined') {
    showError('AudioWorklet not supported in this browser. Use Chrome 66+ or Firefox 76+.');
    $connectBtn.disabled = true;
    return;
  }

  console.log('[Init] Telugu Voice Agent (full-duplex) ready');
})();
