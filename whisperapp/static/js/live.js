import { api } from './api.js';
import { renderWaveform } from './shell.js';

let _session = null;
let _mediaStream = null;
let _audioCtx = null;
let _processor = null;
let _elapsed = 0;
let _elapsedTimer = null;
let _transcript = [];
let _partial = '';
let _waveAnimFrame = null;

export function initLive(container) {
  container.innerHTML = `
    <div class="wa-topbar">
      <div>
        <h1>Live</h1>
        <div class="wa-topbar-sub mono" id="live-sub">Select a microphone and start recording</div>
      </div>
      <div class="wa-topbar-meta" id="live-meta"></div>
    </div>
    <div class="wa-content" style="display:flex;flex-direction:column;gap:20px">
      <!-- Waveform hero -->
      <div class="wa-card" style="padding:22px">
        <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:16px">
          <div>
            <div class="wa-card-sub">Input level</div>
            <div class="mono" style="font-size:22px;font-weight:500;margin-top:4px;letter-spacing:-0.01em">
              <span id="db-value">—</span>
              <span style="color:var(--ink-4);font-size:13px"> dB</span>
            </div>
          </div>
          <div style="display:flex;gap:8px;align-items:center">
            <select class="wa-select" id="device-select" style="width:260px">
              <option value="">Loading devices…</option>
            </select>
            <button class="wa-btn" id="check-levels-btn">Check levels</button>
          </div>
        </div>
        <div id="waveform-container"></div>
        <div class="mono" style="display:flex;justify-content:space-between;margin-top:8px;font-size:10.5px;color:var(--ink-4)">
          <span>00:00</span><span id="elapsed-mid">—</span><span id="elapsed-total">—</span>
        </div>
      </div>

      <!-- Transcript + side panel -->
      <div style="display:grid;grid-template-columns:1.7fr 1fr;gap:20px">
        <div class="wa-card" style="padding:0;overflow:hidden;display:flex;flex-direction:column;min-height:360px">
          <div style="display:flex;align-items:center;justify-content:space-between;padding:14px 20px;border-bottom:1px solid var(--rule)">
            <div class="wa-card-title">Live transcript</div>
            <div class="mono" style="font-size:10.5px;color:var(--ink-4)" id="vad-meta">VAD · ready</div>
          </div>
          <div id="transcript-body" style="padding:18px 22px;display:flex;flex-direction:column;gap:14px;font-size:14px;line-height:1.55;color:var(--ink);flex:1;overflow-y:auto">
            <p style="color:var(--ink-4);font-style:italic">Transcript will appear here…</p>
          </div>
        </div>

        <div style="display:flex;flex-direction:column;gap:20px">
          <div class="wa-card">
            <div class="wa-card-head">
              <div class="wa-card-title">Session</div>
              <div class="wa-card-sub" id="session-sub">—</div>
            </div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;font-size:12px">
              <div><div class="wa-label">Latency</div><div class="mono" style="font-size:13px" id="stat-latency">—</div></div>
              <div><div class="wa-label">Words</div><div class="mono" style="font-size:13px" id="stat-words">0</div></div>
              <div><div class="wa-label">Segments</div><div class="mono" style="font-size:13px" id="stat-segs">0</div></div>
              <div><div class="wa-label">Output</div><div class="mono" style="font-size:13px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" id="stat-output">~/Downloads</div></div>
            </div>
            <div style="margin-top:16px">
              <span class="wa-label">Save formats</span>
              <div style="display:flex;gap:6px;flex-wrap:wrap" id="live-formats">
                ${['txt','srt','vtt','json','tsv'].map((f,i)=>`<span class="wa-chip${i===0?' on':''}" data-fmt="${f}">${f}</span>`).join('')}
              </div>
            </div>
          </div>

          <div class="wa-card">
            <div class="wa-card-head"><div class="wa-card-title">Controls</div></div>
            <div style="display:flex;flex-direction:column;gap:8px">
              <button class="wa-btn wa-btn-primary" id="record-btn" style="height:44px;font-size:14px">
                Start recording
              </button>
              <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px">
                <button class="wa-btn" id="pause-btn" disabled>Pause</button>
                <button class="wa-btn" id="clear-btn">Clear</button>
              </div>
              <button class="wa-btn" id="polish-btn" disabled style="height:40px">
                Polish · align + diarize
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  `;

  // Initial waveform render
  renderWaveform(container.querySelector('#waveform-container'), { height: 64, bars: 120 });

  // Load devices using browser's MediaDevices API for correct deviceId strings
  if (navigator.mediaDevices && navigator.mediaDevices.enumerateDevices) {
    navigator.mediaDevices.enumerateDevices()
      .then(devices => {
        const audioInputs = devices.filter(d => d.kind === 'audioinput');
        const sel = container.querySelector('#device-select');
        sel.innerHTML = audioInputs.length
          ? audioInputs.map(d => `<option value="${d.deviceId}">${d.label || 'Microphone ' + d.deviceId.slice(0,8)}</option>`).join('')
          : `<option value="">No input devices found</option>`;
        updateSubTitle(sel.options[sel.selectedIndex]?.text || '');
      })
      .catch(() => {
        // Fall back to label-only option
        const sel = container.querySelector('#device-select');
        sel.innerHTML = `<option value="">Default microphone</option>`;
      });
  }

  // Format chips
  const liveFmts = new Set(['txt']);
  container.querySelector('#live-formats').addEventListener('click', e => {
    const chip = e.target.closest('[data-fmt]');
    if (!chip) return;
    const f = chip.dataset.fmt;
    if (liveFmts.has(f)) liveFmts.delete(f); else liveFmts.add(f);
    chip.classList.toggle('on', liveFmts.has(f));
  });

  // Record button
  container.querySelector('#record-btn').addEventListener('click', () => {
    if (_session) stopRecording(container);
    else startRecording(container, liveFmts);
  });

  container.querySelector('#clear-btn').addEventListener('click', () => {
    _transcript = []; _partial = '';
    updateTranscriptUI(container);
  });
}

function updateSubTitle(deviceName) {
  const el = document.getElementById('live-sub');
  if (el && deviceName) el.textContent = deviceName;
}

async function startRecording(container, formats) {
  const deviceId = container.querySelector('#device-select').value;

  const streamRes = await api.streamStart({ model: 'base' });
  _session = streamRes.session_id;
  _elapsed = 0;

  const btn = container.querySelector('#record-btn');
  btn.className = 'wa-btn wa-btn-danger';
  btn.style.height = '44px';
  btn.style.fontSize = '14px';
  btn.innerHTML = `<span style="width:10px;height:10px;border-radius:2px;background:var(--signal-rec)"></span> Stop &amp; save`;

  container.querySelector('#pause-btn').disabled = false;
  container.querySelector('#polish-btn').disabled = true;

  // Update meta with recording indicator
  document.getElementById('live-meta').innerHTML = `
    <span style="display:inline-flex;align-items:center;gap:6px">
      <span style="width:8px;height:8px;border-radius:50%;background:var(--signal-rec);
        box-shadow:0 0 0 4px color-mix(in oklch,var(--signal-rec) 18%,transparent);
        animation:pulse-ring 1.6s ease-in-out infinite"></span>
      <span style="color:var(--signal-rec);font-weight:500">Recording</span>
    </span>
    <span style="color:var(--ink-4)">·</span>
    <span class="mono" id="elapsed-display">00:00:00</span>
  `;

  _elapsedTimer = setInterval(() => {
    _elapsed++;
    const h = String(Math.floor(_elapsed/3600)).padStart(2,'0');
    const m = String(Math.floor((_elapsed%3600)/60)).padStart(2,'0');
    const s = String(_elapsed%60).padStart(2,'0');
    const el = document.getElementById('elapsed-display');
    if (el) el.textContent = `${h}:${m}:${s}`;
  }, 1000);

  // Get mic stream
  try {
    const audioConstraints = deviceId ? { audio: { deviceId: { exact: deviceId } } } : { audio: true };
    _mediaStream = await navigator.mediaDevices.getUserMedia(audioConstraints);
    _audioCtx = new AudioContext({ sampleRate: 16000 });
    const src = _audioCtx.createMediaStreamSource(_mediaStream);
    _processor = _audioCtx.createScriptProcessor(4096, 1, 1);
    src.connect(_processor);
    _processor.connect(_audioCtx.destination);
    _processor.onaudioprocess = async e => {
      if (!_session) return;
      const pcm = e.inputBuffer.getChannelData(0);
      const b64 = float32ToBase64(pcm);
      try {
        const res = await api.streamChunk({ session_id: _session, audio_b64: b64 });
        if (res.new_text) {
          _transcript.push({ t: formatTime(_elapsed), txt: res.new_text });
          _partial = '';
        }
        if (res.partial) _partial = res.partial;
        updateTranscriptUI(container);
        const words = _transcript.reduce((n, l) => n + l.txt.split(/\s+/).length, 0);
        const segs  = _transcript.length;
        document.getElementById('stat-words').textContent = words;
        document.getElementById('stat-segs').textContent  = segs;
      } catch { /* chunk errors are non-fatal */ }
    };
  } catch (err) {
    alert('Microphone access denied: ' + err.message);
    await stopRecording(container);
  }
}

async function stopRecording(container) {
  clearInterval(_elapsedTimer);
  if (_processor) { _processor.disconnect(); _processor = null; }
  if (_audioCtx)  { await _audioCtx.close(); _audioCtx = null; }
  if (_mediaStream) { _mediaStream.getTracks().forEach(t => t.stop()); _mediaStream = null; }

  const sid = _session;
  _session = null;

  if (sid) {
    try { await api.streamStop({ session_id: sid }); } catch { /* best effort */ }
  }

  const btn = container.querySelector('#record-btn');
  btn.className = 'wa-btn wa-btn-primary';
  btn.innerHTML = 'Start recording';
  container.querySelector('#pause-btn').disabled = true;
  container.querySelector('#polish-btn').disabled = false;
  document.getElementById('live-meta').innerHTML = '';
}

function updateTranscriptUI(container) {
  const body = container.querySelector('#transcript-body');
  if (!body) return;
  const lines = _transcript.map(l => `
    <div class="transcript-line">
      <span class="transcript-ts mono">${l.t}</span>
      <span>${escHtml(l.txt)}</span>
    </div>`).join('');
  const partial = _partial
    ? `<p style="color:var(--ink-3);font-style:italic">
        …<span class="transcript-partial-word">${escHtml(_partial)}</span>
        <span class="transcript-typing"> ·typing·</span>
       </p>`
    : '';
  body.innerHTML = lines + partial || `<p style="color:var(--ink-4);font-style:italic">Transcript will appear here…</p>`;
  body.scrollTop = body.scrollHeight;
}

function float32ToBase64(f32) {
  const buf = new Uint8Array(f32.buffer);
  let bin = '';
  for (let i = 0; i < buf.length; i++) bin += String.fromCharCode(buf[i]);
  return btoa(bin);
}

function formatTime(secs) {
  const h = String(Math.floor(secs/3600)).padStart(2,'0');
  const m = String(Math.floor((secs%3600)/60)).padStart(2,'0');
  const s = String(secs%60).padStart(2,'0');
  return `${h}:${m}:${s}`;
}

function escHtml(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}
