// tamil_talkies/static/app.js
const micBtn = document.getElementById("mic-btn");
const thinkToggle = document.getElementById("think-toggle");
const personaInput = document.getElementById("persona-input");
const statusEl = document.getElementById("status");
const conversationEl = document.getElementById("conversation");
const waveformCanvas = document.getElementById("waveform");
const waveformCtx = waveformCanvas.getContext("2d");

let ws = null;
let audioCtx = null;
let processorNode = null;
let sourceNode = null;
let mediaStream = null;
let recording = false;

// Playback state for the current turn's TTS reply.
let playbackCtx = null;
let playbackTime = 0;
let analyserNode = null;
let waveformRAF = null;

// The <div class="turn assistant"> for the in-progress reply, and the
// currently-"speaking" sentence <span> inside it, for this turn.
let currentAssistantTurn = null;
let currentSpeakingSpan = null;

function scrollToBottom() {
  conversationEl.scrollTop = conversationEl.scrollHeight;
}

function addTurn(role, text) {
  const div = document.createElement("div");
  div.className = `turn ${role}`;
  div.textContent = text;
  conversationEl.appendChild(div);
  scrollToBottom();
}

function startAssistantTurn() {
  currentAssistantTurn = document.createElement("div");
  currentAssistantTurn.className = "turn assistant";
  conversationEl.appendChild(currentAssistantTurn);
}

function addResponseSentence(text) {
  if (!currentAssistantTurn) startAssistantTurn();
  if (currentSpeakingSpan) currentSpeakingSpan.classList.remove("speaking");
  const span = document.createElement("span");
  span.className = "speaking";
  span.textContent = text + " ";
  currentAssistantTurn.appendChild(span);
  currentSpeakingSpan = span;
  scrollToBottom();
}

function endAssistantTurn() {
  if (currentSpeakingSpan) currentSpeakingSpan.classList.remove("speaking");
  currentAssistantTurn = null;
  currentSpeakingSpan = null;
}

function ensureSocket() {
  if (ws && ws.readyState === WebSocket.OPEN) return;
  const wsScheme = location.protocol === "https:" ? "wss:" : "ws:";
  ws = new WebSocket(`${wsScheme}//${location.host}/talk`);
  ws.binaryType = "arraybuffer";
  ws.onmessage = (event) => {
    if (typeof event.data === "string") {
      const evt = JSON.parse(event.data);
      handleEvent(evt);
    } else {
      playChunk(new Int16Array(event.data));
    }
  };
  ws.onclose = () => { statusEl.textContent = "Disconnected — reload to reconnect"; };
}

function reenableMicNow() {
  micBtn.disabled = false;
  stopWaveform();
}

function reenableMicWhenPlaybackEnds() {
  if (!playbackCtx) {
    reenableMicNow();
    return;
  }
  const remaining = playbackTime - playbackCtx.currentTime;
  if (remaining <= 0) {
    reenableMicNow();
  } else {
    setTimeout(reenableMicNow, remaining * 1000);
  }
}

function handleEvent(evt) {
  if (evt.event === "transcript") {
    addTurn("user", evt.text);
  } else if (evt.event === "response_sentence") {
    addResponseSentence(evt.text);
  } else if (evt.event === "audio_start") {
    if (playbackCtx && playbackCtx.state !== "closed") {
      playbackCtx.close();
    }
    playbackCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: evt.sample_rate });
    playbackTime = playbackCtx.currentTime;
    analyserNode = playbackCtx.createAnalyser();
    analyserNode.fftSize = 256;
    analyserNode.connect(playbackCtx.destination);
    startWaveform();
  } else if (evt.event === "done") {
    endAssistantTurn();
    statusEl.textContent = "Click and hold to talk";
    reenableMicWhenPlaybackEnds();
  } else if (evt.event === "error") {
    statusEl.textContent = `Error: ${evt.detail}`;
    endAssistantTurn();
    reenableMicNow();
  }
}

function startWaveform() {
  if (waveformRAF) cancelAnimationFrame(waveformRAF);
  const data = new Uint8Array(analyserNode.frequencyBinCount);
  const draw = () => {
    analyserNode.getByteTimeDomainData(data);
    const w = waveformCanvas.width = waveformCanvas.clientWidth;
    const h = waveformCanvas.height = waveformCanvas.clientHeight;
    waveformCtx.clearRect(0, 0, w, h);

    const cx = w / 2;
    const cy = h / 2;
    const baseRadius = Math.min(w, h) / 2 * 0.5;
    const amplitude = Math.min(w, h) / 2 * 0.42;

    waveformCtx.beginPath();
    for (let i = 0; i <= data.length; i++) {
      const idx = i % data.length;
      const amp = (data[idx] - 128) / 128;
      const angle = (i / data.length) * Math.PI * 2 - Math.PI / 2;
      const radius = baseRadius + amp * amplitude;
      const x = cx + radius * Math.cos(angle);
      const y = cy + radius * Math.sin(angle);
      if (i === 0) waveformCtx.moveTo(x, y);
      else waveformCtx.lineTo(x, y);
    }
    waveformCtx.closePath();
    waveformCtx.lineWidth = 3;
    waveformCtx.strokeStyle = "#4f4";
    waveformCtx.shadowColor = "#4f4";
    waveformCtx.shadowBlur = 14;
    waveformCtx.stroke();
    waveformCtx.shadowBlur = 0;

    waveformRAF = requestAnimationFrame(draw);
  };
  draw();
}

function stopWaveform() {
  if (waveformRAF) {
    cancelAnimationFrame(waveformRAF);
    waveformRAF = null;
  }
  waveformCtx.clearRect(0, 0, waveformCanvas.width, waveformCanvas.height);
}

function playChunk(int16) {
  if (!playbackCtx) return;
  const float32 = new Float32Array(int16.length);
  for (let i = 0; i < int16.length; i++) float32[i] = int16[i] / 32768;
  const buffer = playbackCtx.createBuffer(1, float32.length, playbackCtx.sampleRate);
  buffer.copyToChannel(float32, 0);
  const src = playbackCtx.createBufferSource();
  src.buffer = buffer;
  src.connect(analyserNode || playbackCtx.destination);
  const startAt = Math.max(playbackTime, playbackCtx.currentTime);
  src.start(startAt);
  playbackTime = startAt + buffer.duration;
}

async function startRecording() {
  ensureSocket();
  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (err) {
    statusEl.textContent = `Mic error: ${err.message}`;
    return;
  }
  audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  sourceNode = audioCtx.createMediaStreamSource(mediaStream);
  processorNode = audioCtx.createScriptProcessor(4096, 1, 1);
  processorNode.onaudioprocess = (e) => {
    if (!recording || ws.readyState !== WebSocket.OPEN) return;
    const input = e.inputBuffer.getChannelData(0);
    const int16 = new Int16Array(input.length);
    for (let i = 0; i < input.length; i++) {
      const s = Math.max(-1, Math.min(1, input[i]));
      int16[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
    }
    ws.send(int16.buffer);
  };
  sourceNode.connect(processorNode);
  processorNode.connect(audioCtx.destination);
  recording = true;
  micBtn.classList.add("recording");
  statusEl.textContent = "Listening...";
}

function sendEndEvent() {
  ws.send(JSON.stringify({
    event: "end",
    think: thinkToggle.checked,
    sample_rate: audioCtx.sampleRate,
    system_prompt: personaInput.value,
  }));
}

function stopRecording() {
  if (!recording) return;
  recording = false;
  micBtn.classList.remove("recording");
  micBtn.disabled = true;
  statusEl.textContent = "Thinking...";
  processorNode.disconnect();
  sourceNode.disconnect();
  mediaStream.getTracks().forEach((t) => t.stop());
  if (ws.readyState === WebSocket.OPEN) {
    sendEndEvent();
  } else if (ws.readyState === WebSocket.CONNECTING) {
    // Fast tap: the WS may not have finished connecting yet (this is most
    // likely on the very first press, before any socket exists to reuse).
    // Send "end" as soon as it opens instead of dropping the turn silently.
    ws.addEventListener("open", sendEndEvent, { once: true });
  } else {
    statusEl.textContent = "Connection lost — reload to try again";
    micBtn.disabled = false;
  }
  audioCtx.close();
}

micBtn.addEventListener("mousedown", startRecording);
micBtn.addEventListener("touchstart", (e) => { e.preventDefault(); startRecording(); });
micBtn.addEventListener("mouseup", stopRecording);
micBtn.addEventListener("mouseleave", () => { if (recording) stopRecording(); });
micBtn.addEventListener("touchend", (e) => { e.preventDefault(); stopRecording(); });
