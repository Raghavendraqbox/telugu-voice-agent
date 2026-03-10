'use strict';

// ── Constants ──────────────────────────────────────────────────────────────────
const WS_URL               = `${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws/audio`;
const PLAYBACK_SAMPLE_RATE = 16000;

// ── AudioWorklet: runs at browser native rate, resamples to 16kHz, emits 320-sample chunks ──
const WORKLET_CODE = `
class PCMProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this._buf   = [];
    this._phase = 0.0;
    this._TARGET = 16000;
    this._CHUNK  = 320;
  }
  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (!ch || ch.length === 0) return true;
    const ratio = sampleRate / this._TARGET;
    while (this._phase < ch.length) {
      const i  = Math.floor(this._phase);
      const f  = this._phase - i;
      const s0 = ch[i]   !== undefined ? ch[i]   : 0.0;
      const s1 = ch[i+1] !== undefined ? ch[i+1] : s0;
      this._buf.push(s0 + (s1 - s0) * f);
      this._phase += ratio;
    }
    this._phase -= ch.length;
    while (this._buf.length >= this._CHUNK) {
      const chunk = new Float32Array(this._buf.splice(0, this._CHUNK));
      this.port.postMessage(chunk.buffer, [chunk.buffer]);
    }
    return true;
  }
}
registerProcessor('pcm-processor', PCMProcessor);
`;

// ── DOM refs ───────────────────────────────────────────────────────────────────
const $connDot     = document.getElementById('connection-dot');
const $statusText  = document.getElementById('status-text');
const $transcript  = document.getElementById('transcript');
const $canvas      = document.getElementById('visualizer');
const $latStt      = document.getElementById('lat-stt');
const $latLlm      = document.getElementById('lat-llm');
const $latTts      = document.getElementById('lat-tts');
const $latTotal    = document.getElementById('lat-total');
const $agentAv     = document.getElementById('agent-avatar');
const $vadRing     = document.getElementById('vad-ring');
const $startBtn    = document.getElementById('start-call-btn');
const $endBtn      = document.getElementById('end-call-btn');
const canvasCtx    = $canvas.getContext('2d');

// ── State ──────────────────────────────────────────────────────────────────────
let ws             = null;
let captureCtx     = null;   // created INSIDE click handler (user gesture)
let workletNode    = null;
let analyserNode   = null;
let micStream      = null;
let vizAnimId      = null;
let playbackCtx    = null;
let nextPlayTime   = 0;
let playbackActive = false;
let agentState     = 'idle';
let turnStartTime  = null;
let sttEndTime     = null;
let ttsFirstChunk  = null;
let agentBubbleEl  = null;
let callActive     = false;

// ── Start / End call ───────────────────────────────────────────────────────────

// Called from the GREEN button click — user gesture ensures AudioContext works
async function startCall() {
  $startBtn.style.display = 'none';
  $endBtn.style.display   = '';
  callActive = true;

  setConnState('connecting');
  setStatusText('Connecting…');

  // --- Create AudioContexts INSIDE user gesture ---
  captureCtx = new AudioContext();
  await captureCtx.resume();

  // Pre-create playback context here (user gesture) so it's not suspended later
  if (playbackCtx && playbackCtx.state !== 'closed') playbackCtx.close();
  playbackCtx = new AudioContext({ sampleRate: PLAYBACK_SAMPLE_RATE });
  await playbackCtx.resume();
  nextPlayTime = 0; playbackActive = false;

  // --- Request microphone ---
  // echoCancellation ON: prevents TTS speaker audio from triggering barge-in
  try {
    micStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      }
    });
  } catch (err) {
    try {
      micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch (err2) {
      setStatusText(`Mic error: ${err2.message}`);
      endCall();
      return;
    }
  }

  // Log which device was selected to aid debugging
  const track = micStream.getAudioTracks()[0];
  console.log('[mic] device:', track.label, track.getSettings());

  // --- Set up audio graph ---
  analyserNode = captureCtx.createAnalyser();
  analyserNode.fftSize = 256;
  const src = captureCtx.createMediaStreamSource(micStream);

  // Mild pre-amplify — 2x boost; OS mic should already be at 100%
  const preGain = captureCtx.createGain();
  preGain.gain.value = 2.0;
  src.connect(preGain);
  preGain.connect(analyserNode);

  const blob = new Blob([WORKLET_CODE], { type: 'application/javascript' });
  const url  = URL.createObjectURL(blob);
  try {
    await captureCtx.audioWorklet.addModule(url);
  } catch (err) {
    setStatusText(`AudioWorklet error: ${err.message}`);
    endCall();
    return;
  } finally {
    URL.revokeObjectURL(url);
  }

  workletNode = new AudioWorkletNode(captureCtx, 'pcm-processor');
  workletNode.port.onmessage = (e) => sendPcm(new Float32Array(e.data));
  preGain.connect(workletNode);

  // REQUIRED: path to destination so Chrome pulls live audio through the graph
  const silentGain = captureCtx.createGain();
  silentGain.gain.value = 0;
  workletNode.connect(silentGain);
  silentGain.connect(captureCtx.destination);

  startViz();

  // --- Open WebSocket ---
  ws = new WebSocket(WS_URL);
  ws.binaryType = 'arraybuffer';
  ws.addEventListener('open',    onOpen);
  ws.addEventListener('message', onMessage);
  ws.addEventListener('close',   onClose);
  ws.addEventListener('error',   onError);
}

function endCall() {
  callActive = false;
  stopCapture();
  stopPlayback();
  stopViz();
  if (ws && ws.readyState <= WebSocket.OPEN) ws.close(1000, 'User hangup');
  ws = null;
  setConnState('disconnected');
  setAgentState('idle');
  setVad(false);
  setStatusText('Call ended — tap green to call again');
  $endBtn.style.display   = 'none';
  $startBtn.style.display = '';
  // Clear transcript
  $transcript.innerHTML = '<div class="transcript-empty">కాల్ ప్రారంభించడానికి క్రింది బటన్ నొక్కండి…</div>';
  agentBubbleEl = null;
}

// ── WebSocket handlers ─────────────────────────────────────────────────────────

function onOpen() {
  setConnState('connected');
  setStatusText('Ready — speak in Telugu');
}

function onClose(e) {
  if (!callActive) return;
  setConnState('disconnected');
  setAgentState('idle');
  setVad(false);
  if (e.code !== 1000) {
    setStatusText('Reconnecting…');
    setTimeout(() => {
      if (!callActive) return;
      ws = new WebSocket(WS_URL);
      ws.binaryType = 'arraybuffer';
      ws.addEventListener('open',    onOpen);
      ws.addEventListener('message', onMessage);
      ws.addEventListener('close',   onClose);
      ws.addEventListener('error',   onError);
    }, 2000);
  }
}

function onError() {
  setConnState('error');
  setStatusText('Connection error — retrying…');
}

// ── Messages ───────────────────────────────────────────────────────────────────

function onMessage(evt) {
  if (evt.data instanceof ArrayBuffer) {
    handleAudio(evt.data);
    return;
  }
  let msg;
  try { msg = JSON.parse(evt.data); } catch { return; }

  switch (msg.type) {
    case 'ready':
      setStatusText('Ready — speak in Telugu');
      setAgentState('idle');
      clearPlaceholder();
      break;
    case 'vad_state':
      handleVadState(msg.state);
      break;
    case 'transcript':
      if (msg.role === 'user')  appendUser(msg.text);
      if (msg.role === 'agent') finaliseAgent(msg.text);
      break;
    case 'error':
      setStatusText(`Error: ${msg.message}`);
      break;
  }
}

function handleVadState(state) {
  switch (state) {
    case 'listening':
      setVad(true);
      setAgentState('listening');
      setStatusText('Listening…');
      turnStartTime = performance.now();
      sttEndTime = null; ttsFirstChunk = null; agentBubbleEl = null;
      if (playbackActive) { stopPlayback(); sendInterrupt(); }
      break;
    case 'silent':
      setVad(false);
      if (agentState === 'listening') setStatusText('…');
      break;
    case 'processing':
      setVad(false);
      setAgentState('processing');
      setStatusText('Processing…');
      sttEndTime = performance.now();
      if (turnStartTime) updateLat('stt', sttEndTime - turnStartTime);
      showTyping();
      break;
    case 'speaking':
      setAgentState('speaking');
      setStatusText('Agent speaking…');
      if (ttsFirstChunk === null) {
        ttsFirstChunk = performance.now();
        if (sttEndTime)    updateLat('llm',   ttsFirstChunk - sttEndTime);
        if (turnStartTime) updateLat('total', ttsFirstChunk - turnStartTime);
      }
      break;
  }
}

// ── Audio capture helpers ──────────────────────────────────────────────────────

function stopCapture() {
  if (workletNode) { workletNode.port.onmessage = null; workletNode.disconnect(); workletNode = null; }
  if (analyserNode) { analyserNode.disconnect(); analyserNode = null; }
  if (captureCtx && captureCtx.state !== 'closed') { captureCtx.close(); captureCtx = null; }
  if (micStream) { micStream.getTracks().forEach(t => t.stop()); micStream = null; }
}

let _dbgChunk = 0;
function sendPcm(f32) {
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  // Log RMS every 50 chunks (~1s) to browser console for diagnostics
  if (++_dbgChunk % 50 === 0) {
    let sum = 0; for (let i = 0; i < f32.length; i++) sum += f32[i]*f32[i];
    console.log(`[mic] rms=${Math.sqrt(sum/f32.length).toFixed(5)} chunk#${_dbgChunk}`);
  }
  const i16 = new Int16Array(f32.length);
  for (let i = 0; i < f32.length; i++) {
    const c = Math.max(-1, Math.min(1, f32[i]));
    i16[i] = c < 0 ? Math.round(c * 32768) : Math.round(c * 32767);
  }
  ws.send(i16.buffer);
}

// ── Audio playback ─────────────────────────────────────────────────────────────

function ensurePlayCtx() {
  if (!playbackCtx || playbackCtx.state === 'closed') {
    playbackCtx = new AudioContext({ sampleRate: PLAYBACK_SAMPLE_RATE });
    nextPlayTime = 0; playbackActive = false;
  }
  if (playbackCtx.state === 'suspended') playbackCtx.resume();
  return playbackCtx;
}

function handleAudio(buf) {
  if (!buf || buf.byteLength === 0) return;
  if (ttsFirstChunk === null) {
    ttsFirstChunk = performance.now();
    if (sttEndTime)    updateLat('tts',   ttsFirstChunk - sttEndTime);
    if (turnStartTime) updateLat('total', ttsFirstChunk - turnStartTime);
  }
  const ctx = ensurePlayCtx();
  const i16 = new Int16Array(buf);
  if (!i16.length) return;
  const abuf = ctx.createBuffer(1, i16.length, PLAYBACK_SAMPLE_RATE);
  const ch   = abuf.getChannelData(0);
  for (let i = 0; i < i16.length; i++) ch[i] = i16[i] / 32768;
  const src = ctx.createBufferSource();
  src.buffer = abuf;
  src.connect(ctx.destination);
  const now = ctx.currentTime;
  if (nextPlayTime < now + 0.05) nextPlayTime = now + 0.05;
  src.start(nextPlayTime);
  src.onended = () => {
    if (playbackCtx && nextPlayTime <= playbackCtx.currentTime + 0.05)
      playbackActive = false;
  };
  nextPlayTime += abuf.duration;
  playbackActive = true;
}

function stopPlayback() {
  if (playbackCtx && playbackCtx.state !== 'closed') { playbackCtx.close(); playbackCtx = null; }
  nextPlayTime = 0; playbackActive = false;
}

function sendInterrupt() {
  if (ws && ws.readyState === WebSocket.OPEN)
    ws.send(JSON.stringify({ type: 'interrupt' }));
}

// ── Visualizer ─────────────────────────────────────────────────────────────────

function startViz() {
  if (vizAnimId) cancelAnimationFrame(vizAnimId);
  drawViz();
}
function stopViz() {
  if (vizAnimId) { cancelAnimationFrame(vizAnimId); vizAnimId = null; }
  canvasCtx.fillStyle = '#141414';
  canvasCtx.fillRect(0, 0, $canvas.width, $canvas.height);
}
function drawViz() {
  vizAnimId = requestAnimationFrame(drawViz);
  if (!analyserNode) { stopViz(); return; }
  const d  = new Uint8Array(analyserNode.frequencyBinCount);
  analyserNode.getByteTimeDomainData(d);
  const W  = $canvas.width, H = $canvas.height;
  canvasCtx.fillStyle = '#141414';
  canvasCtx.fillRect(0, 0, W, H);
  const cols = { idle:'#333', listening:'#FF9933', processing:'#f0c040', speaking:'#4caf7d' };
  canvasCtx.strokeStyle = cols[agentState] || '#333';
  canvasCtx.lineWidth = 2;
  canvasCtx.beginPath();
  const sw = W / d.length;
  for (let i = 0; i < d.length; i++) {
    const y = (d[i] / 128) * (H / 2);
    if (i === 0) canvasCtx.moveTo(0, y);
    else         canvasCtx.lineTo(i * sw, y);
  }
  canvasCtx.lineTo(W, H / 2);
  canvasCtx.stroke();
}

// ── UI helpers ─────────────────────────────────────────────────────────────────

function setConnState(s) { $connDot.className = `conn-dot ${s}`; }
function setStatusText(t) { $statusText.textContent = t; }

function setAgentState(s) {
  agentState = s;
  $agentAv.className = 'avatar';
  if (s === 'listening')  $agentAv.classList.add('user-speaking');
  if (s === 'processing') $agentAv.classList.add('processing');
  if (s === 'speaking')   $agentAv.classList.add('agent-speaking');
}
function setVad(active) {
  $vadRing.className = `avatar user-avatar${active ? ' active' : ''}`;
}
function updateLat(k, ms) {
  const m = { stt: $latStt, llm: $latLlm, tts: $latTts, total: $latTotal };
  const el = m[k]; if (!el) return;
  el.textContent = `${Math.round(ms)}ms`;
  const thr = { stt:200, llm:300, tts:200, total:700 }[k];
  el.style.color = ms <= thr ? '#4caf7d' : ms <= thr*1.5 ? '#f0c040' : '#e53935';
}

// ── Transcript ─────────────────────────────────────────────────────────────────

function clearPlaceholder() {
  const e = $transcript.querySelector('.transcript-empty');
  if (e) e.remove();
}
function appendUser(text) {
  clearPlaceholder();
  const d = document.createElement('div');
  d.className = 'chat-turn user';
  d.innerHTML = `<span class="bubble-role">మీరు</span><div class="bubble">${esc(text)}</div>`;
  $transcript.appendChild(d);
  $transcript.scrollTop = $transcript.scrollHeight;
}
function showTyping() {
  clearPlaceholder();
  if (agentBubbleEl) return;
  const d = document.createElement('div');
  d.className = 'chat-turn agent';
  const b = document.createElement('div');
  b.className = 'bubble typing';
  b.innerHTML = '<span class="typing-dot"></span><span class="typing-dot"></span><span class="typing-dot"></span>';
  d.appendChild(b);
  $transcript.appendChild(d);
  agentBubbleEl = b;
  $transcript.scrollTop = $transcript.scrollHeight;
}
function finaliseAgent(text) {
  if (agentBubbleEl) {
    agentBubbleEl.className = 'bubble';
    agentBubbleEl.textContent = text;
    agentBubbleEl = null;
  } else {
    clearPlaceholder();
    const d = document.createElement('div');
    d.className = 'chat-turn agent';
    d.innerHTML = `<span class="bubble-role">అసిస్టెంట్</span><div class="bubble">${esc(text)}</div>`;
    $transcript.appendChild(d);
  }
  $transcript.scrollTop = $transcript.scrollHeight;
  if (turnStartTime) updateLat('total', performance.now() - turnStartTime);
}
function esc(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

// ── Button handlers ────────────────────────────────────────────────────────────

$startBtn.addEventListener('click', () => startCall());
$endBtn.addEventListener('click',   () => endCall());

window.addEventListener('beforeunload', () => {
  if (ws) ws.close(1001, 'unload');
});

// ── Init ───────────────────────────────────────────────────────────────────────
stopViz();
