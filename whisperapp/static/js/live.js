import { api } from './api.js';
import { renderWaveform } from './shell.js';

// ── Recording state ──────────────────────────────────────────────────────────
let _session = null;
let _lastSessionId = null;   // kept after stop so polish can use it
let _mediaStream = null;
let _audioCtx = null;
let _processor = null;
let _elapsed = 0;
let _elapsedTimer = null;
let _transcript = [];        // [{id, t, txt, speaker, review}]
let _partial = '';
let _pcmQueue = [];
let _draining = false;
let _segIdCounter = 0;
let _speakerNames = new Map(); // SPEAKER_00 → user-friendly name

// ── Always-on device meter ───────────────────────────────────────────────────
let _meterStream = null;
let _meterCtx = null;
let _meterProc = null;
let _meterFrame = null;
let _meterRms = 0;
let _meterData = new Float32Array(2048);

// ── Recording waveform data (from onaudioprocess) ────────────────────────────
let _recRms = 0;
let _recData = new Float32Array(4096);
let _recMeterFrame = null;

// ── Check levels panel (all devices simultaneously) ──────────────────────────
let _checkPanelMap = new Map(); // deviceId → {stream, ctx, proc, rms}
let _checkPanelFrame = null;

// ── Config cache ─────────────────────────────────────────────────────────────
let _liveCfg = { streaming_show_listening: true, diarize_by_default: true, hf_token: '' };

// ── Main container ref (needed in async callbacks) ───────────────────────────
let _liveContainer = null;

export function initLive(container) {
  _liveContainer = container;
  api.getConfig().then(cfg => { _liveCfg = cfg; }).catch(() => {});

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

      <!-- Check levels panel (hidden until button clicked) -->
      <div id="check-levels-panel" style="display:none"></div>

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
              <div><div class="wa-label">Audio</div><div class="mono" style="font-size:11px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" id="stat-output">—</div></div>
            </div>
          </div>

          <div class="wa-card">
            <div class="wa-card-head"><div class="wa-card-title">Save formats</div></div>
            <div style="display:flex;gap:6px;flex-wrap:wrap" id="live-formats">
              ${['txt','srt','vtt','json','tsv','docx','pdf'].map((f,i)=>`<span class="wa-chip${i===0?' on':''}" data-fmt="${f}">${f}</span>`).join('')}
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

  renderWaveform(container.querySelector('#waveform-container'), { height: 64, bars: 120, color: 'var(--ink-2)' });

  _initDevices(container);

  const liveFmts = new Set(['txt']);
  container.querySelector('#live-formats').addEventListener('click', e => {
    const chip = e.target.closest('[data-fmt]');
    if (!chip) return;
    const f = chip.dataset.fmt;
    if (liveFmts.has(f)) liveFmts.delete(f); else liveFmts.add(f);
    chip.classList.toggle('on', liveFmts.has(f));
  });

  container.querySelector('#check-levels-btn').addEventListener('click', () => {
    const panel = container.querySelector('#check-levels-panel');
    if (panel.style.display === 'none') openCheckLevelsPanel(container);
    else closeCheckLevelsPanel(container);
  });

  container.querySelector('#record-btn').addEventListener('click', () => {
    if (_session) stopRecording(container);
    else startRecording(container, liveFmts);
  });

  container.querySelector('#clear-btn').addEventListener('click', () => {
    _transcript = []; _partial = '';
    updateTranscriptUI(container);
  });

  container.querySelector('#polish-btn').addEventListener('click', () => {
    if (_lastSessionId) runPolish(container, _lastSessionId);
  });
}

// ── Device initialisation ────────────────────────────────────────────────────

async function _initDevices(container) {
  const sel = container.querySelector('#device-select');
  try {
    // First call may return devices without labels — request permission first
    await navigator.mediaDevices.getUserMedia({ audio: true }).then(s => s.getTracks().forEach(t => t.stop())).catch(() => {});
    const devices = await navigator.mediaDevices.enumerateDevices();
    const inputs = devices.filter(d => d.kind === 'audioinput');
    if (!inputs.length) {
      sel.innerHTML = `<option value="">No input devices found</option>`;
      return;
    }
    sel.innerHTML = inputs.map(d =>
      `<option value="${escHtml(d.deviceId)}">${escHtml(d.label || 'Mic ' + d.deviceId.slice(0, 8))}</option>`
    ).join('');

    // Restore last used device
    const last = localStorage.getItem('whisperapp_device_id');
    if (last && inputs.find(d => d.deviceId === last)) sel.value = last;

    _updateSubTitle(sel.options[sel.selectedIndex]?.text || '');
    startDeviceMeter(container, sel.value);
  } catch {
    sel.innerHTML = `<option value="">Default microphone</option>`;
  }

  sel.addEventListener('change', () => {
    localStorage.setItem('whisperapp_device_id', sel.value);
    _updateSubTitle(sel.options[sel.selectedIndex]?.text || '');
    if (!_session) {
      stopDeviceMeter();
      startDeviceMeter(container, sel.value);
    }
  });
}

function _updateSubTitle(name) {
  const el = document.getElementById('live-sub');
  if (el && name) el.textContent = name;
}

// ── Always-on single-device meter ────────────────────────────────────────────

async function startDeviceMeter(container, deviceId) {
  stopDeviceMeter();
  try {
    _meterCtx = new AudioContext();
    const constraints = deviceId ? { audio: { deviceId: { exact: deviceId }, echoCancellation: false, noiseSuppression: false, autoGainControl: false } } : { audio: true };
    _meterStream = await navigator.mediaDevices.getUserMedia(constraints);
    await _meterCtx.resume();
    const src = _meterCtx.createMediaStreamSource(_meterStream);
    _meterProc = _meterCtx.createScriptProcessor(2048, 1, 1);
    const sink = _meterCtx.createMediaStreamDestination();
    _meterProc.onaudioprocess = e => {
      const d = e.inputBuffer.getChannelData(0);
      _meterData = d.slice();
      let s = 0;
      for (let i = 0; i < d.length; i++) s += d[i] * d[i];
      _meterRms = Math.sqrt(s / d.length);
    };
    src.connect(_meterProc);
    _meterProc.connect(sink);
    _meterTick(container);
  } catch { /* device unavailable */ }
}

function stopDeviceMeter() {
  if (_meterFrame) { cancelAnimationFrame(_meterFrame); _meterFrame = null; }
  if (_meterProc) { _meterProc.disconnect(); _meterProc = null; }
  if (_meterCtx) { _meterCtx.close().catch(() => {}); _meterCtx = null; }
  if (_meterStream) { _meterStream.getTracks().forEach(t => t.stop()); _meterStream = null; }
  _meterRms = 0;
}

function _meterTick(container) {
  _meterFrame = requestAnimationFrame(() => _meterTick(container));
  _drawMeter(container, _meterRms, _meterData);
}

function _drawMeter(container, rms, data) {
  const db = rms > 1e-7 ? 20 * Math.log10(rms) : null;
  const hasSignal = rms > 1e-4;

  const dbEl = container.querySelector('#db-value');
  if (dbEl) {
    dbEl.textContent = db != null ? db.toFixed(1) : '−∞';
    dbEl.style.color = hasSignal ? 'var(--accent)' : '';
  }
  const vadMeta = container.querySelector('#vad-meta');
  if (vadMeta && !_session) {
    vadMeta.textContent = hasSignal ? 'VAD · signal ✓' : 'VAD · no signal';
  }

  const wc = container.querySelector('#waveform-container');
  if (!wc) return;
  const bars = wc.querySelectorAll('.wa-waveform-bar');
  const step = Math.max(1, Math.floor(data.length / bars.length));
  bars.forEach((bar, i) => {
    const amp = Math.abs(data[i * step] || 0);
    const h = Math.max(0.06, Math.min(1, 0.12 + amp * 4));
    bar.style.height = (h * 100) + '%';
    bar.style.background = hasSignal ? '#4caf7d' : 'var(--accent)';
    bar.style.opacity = 0.3 + amp * 0.7;
  });
}

// ── Check levels panel (all devices simultaneously) ──────────────────────────

async function openCheckLevelsPanel(container) {
  const panel = container.querySelector('#check-levels-panel');
  const btn = container.querySelector('#check-levels-btn');
  btn.textContent = 'Close levels';
  btn.classList.add('active');

  panel.style.display = '';
  panel.innerHTML = `
    <div class="wa-card" style="padding:18px 22px">
      <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:14px">
        <div class="wa-card-title">All input levels</div>
        <div class="mono" style="font-size:11px;color:var(--ink-4)">Click a row to select that device</div>
      </div>
      <div id="check-panel-rows" style="display:flex;flex-direction:column;gap:10px">
        <div style="color:var(--ink-4);font-size:12.5px">Opening devices…</div>
      </div>
    </div>`;

  let devices = [];
  try {
    const all = await navigator.mediaDevices.enumerateDevices();
    devices = all.filter(d => d.kind === 'audioinput');
  } catch { return; }

  const rowsEl = panel.querySelector('#check-panel-rows');
  rowsEl.innerHTML = devices.map(d => `
    <div class="check-level-row" data-device-id="${escHtml(d.deviceId)}"
      style="display:grid;grid-template-columns:220px 1fr 56px 64px;gap:12px;align-items:center;padding:8px 10px;border-radius:6px;cursor:pointer;transition:background 0.15s"
      onmouseover="this.style.background='var(--paper-3)'" onmouseout="this.style.background=''">
      <div style="font-size:12.5px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="${escHtml(d.label)}">${escHtml(d.label || 'Mic ' + d.deviceId.slice(0, 8))}</div>
      <div style="background:var(--paper-3);border-radius:3px;height:8px;overflow:hidden;position:relative">
        <div id="bar-${escHtml(d.deviceId)}" style="width:0%;height:100%;background:var(--ink-4);border-radius:3px;transition:width 0.05s"></div>
      </div>
      <div class="mono" style="font-size:11px;color:var(--ink-3);text-align:right" id="db-${escHtml(d.deviceId)}">—</div>
      <button class="wa-btn" style="font-size:11px;padding:0 8px;height:26px" data-use-device="${escHtml(d.deviceId)}">Use</button>
    </div>`).join('');

  // Wire "Use" buttons
  rowsEl.querySelectorAll('[data-use-device]').forEach(btn => {
    btn.addEventListener('click', e => {
      e.stopPropagation();
      const id = btn.dataset.useDevice;
      const sel = container.querySelector('#device-select');
      if (sel) sel.value = id;
      localStorage.setItem('whisperapp_device_id', id);
      _updateSubTitle(sel?.options[sel.selectedIndex]?.text || '');
      closeCheckLevelsPanel(container);
      if (!_session) {
        stopDeviceMeter();
        startDeviceMeter(container, id);
      }
    });
  });

  // Open all devices concurrently
  await Promise.all(devices.map(async d => {
    try {
      const ctx = new AudioContext();
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: { deviceId: { exact: d.deviceId }, echoCancellation: false, noiseSuppression: false, autoGainControl: false }
      });
      await ctx.resume();
      const src = ctx.createMediaStreamSource(stream);
      const proc = ctx.createScriptProcessor(2048, 1, 1);
      const sink = ctx.createMediaStreamDestination();
      let rms = 0;
      proc.onaudioprocess = e => {
        const data = e.inputBuffer.getChannelData(0);
        let s = 0;
        for (let i = 0; i < data.length; i++) s += data[i] * data[i];
        rms = Math.sqrt(s / data.length);
      };
      src.connect(proc); proc.connect(sink);
      _checkPanelMap.set(d.deviceId, { stream, ctx, proc, get rms() { return rms; } });
    } catch {
      // Device couldn't be opened — row stays with "—" level
    }
  }));

  // Animate level bars
  function _panelTick() {
    _checkPanelFrame = requestAnimationFrame(_panelTick);
    for (const [id, state] of _checkPanelMap) {
      const bar = document.getElementById('bar-' + id);
      const dbEl = document.getElementById('db-' + id);
      if (!bar) continue;
      const r = state.rms;
      const pct = Math.min(100, r * 500);
      bar.style.width = pct + '%';
      bar.style.background = r > 1e-4 ? '#4caf7d' : 'var(--ink-4)';
      if (dbEl) {
        const db = r > 1e-7 ? 20 * Math.log10(r) : null;
        dbEl.textContent = db != null ? db.toFixed(1) : '—';
        dbEl.style.color = r > 1e-4 ? 'var(--accent)' : 'var(--ink-3)';
      }
    }
  }
  _panelTick();
}

function closeCheckLevelsPanel(container) {
  if (_checkPanelFrame) { cancelAnimationFrame(_checkPanelFrame); _checkPanelFrame = null; }
  for (const { stream, ctx, proc } of _checkPanelMap.values()) {
    try { proc.disconnect(); } catch {}
    try { stream.getTracks().forEach(t => t.stop()); } catch {}
    try { ctx.close(); } catch {}
  }
  _checkPanelMap.clear();

  const panel = container.querySelector('#check-levels-panel');
  if (panel) { panel.style.display = 'none'; panel.innerHTML = ''; }
  const btn = container.querySelector('#check-levels-btn');
  if (btn) { btn.textContent = 'Check levels'; btn.classList.remove('active'); }
}

// ── Recording ────────────────────────────────────────────────────────────────

async function startRecording(container, formats) {
  const deviceId = container.querySelector('#device-select').value;

  // Close check levels panel and stop always-on meter
  closeCheckLevelsPanel(container);
  stopDeviceMeter();

  _audioCtx = new AudioContext();

  try {
    _mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: false,
        noiseSuppression: false,
        autoGainControl: false,
        ...(deviceId ? { deviceId: { exact: deviceId } } : {}),
      }
    });
    await _audioCtx.resume();

    const src = _audioCtx.createMediaStreamSource(_mediaStream);
    _processor = _audioCtx.createScriptProcessor(4096, 1, 1);
    const sink = _audioCtx.createMediaStreamDestination();
    src.connect(_processor);
    _processor.connect(sink);

    _processor.onaudioprocess = e => {
      const d = e.inputBuffer.getChannelData(0);
      // Update recording waveform
      _recData = d.slice();
      let s = 0;
      for (let i = 0; i < d.length; i++) s += d[i] * d[i];
      _recRms = Math.sqrt(s / d.length);

      if (!_session) return;
      _pcmQueue.push(d.slice());
      if (!_draining) _drainChunkQueue(container);
    };

    // Drive waveform during recording
    function _recMeterTick() {
      _recMeterFrame = requestAnimationFrame(_recMeterTick);
      if (!_session) return;
      _drawMeter(container, _recRms, _recData);
    }
    _recMeterTick();

  } catch (err) {
    alert('Microphone access denied: ' + err.message);
    await stopRecording(container);
    return;
  }

  api.pauseWorker().catch(() => {});
  const streamRes = await api.streamStart({ model: _liveCfg.streaming_model || 'base' });
  _session = streamRes.session_id;
  _lastSessionId = _session;
  _elapsed = 0;

  const btn = container.querySelector('#record-btn');
  btn.className = 'wa-btn wa-btn-danger';
  btn.style.height = '44px';
  btn.style.fontSize = '14px';
  btn.innerHTML = `<span style="width:10px;height:10px;border-radius:2px;background:var(--signal-rec)"></span> Stop &amp; save`;

  container.querySelector('#pause-btn').disabled = false;
  container.querySelector('#polish-btn').disabled = true;

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
    const h = String(Math.floor(_elapsed / 3600)).padStart(2, '0');
    const m = String(Math.floor((_elapsed % 3600) / 60)).padStart(2, '0');
    const s = String(_elapsed % 60).padStart(2, '0');
    const el = document.getElementById('elapsed-display');
    if (el) el.textContent = `${h}:${m}:${s}`;
  }, 1000);

  updateTranscriptUI(container);
}

async function stopRecording(container) {
  clearInterval(_elapsedTimer);
  if (_recMeterFrame) { cancelAnimationFrame(_recMeterFrame); _recMeterFrame = null; }
  if (_processor) { _processor.disconnect(); _processor = null; }
  if (_audioCtx) { await _audioCtx.close(); _audioCtx = null; }
  if (_mediaStream) { _mediaStream.getTracks().forEach(t => t.stop()); _mediaStream = null; }

  api.resumeWorker().catch(() => {});

  const sid = _session;
  _session = null;

  // Wait for chunk drain before stopping
  if (_draining) {
    const deadline = Date.now() + 5000;
    while (_draining && Date.now() < deadline)
      await new Promise(r => setTimeout(r, 50));
  }
  _pcmQueue.length = 0;

  if (sid) {
    try {
      const result = await api.streamStop({ session_id: sid });
      if (result && result.text && result.text.trim()) {
        _transcript.push({ id: _segIdCounter++, t: formatTime(_elapsed), txt: result.text.trim(), speaker: null, review: false });
        updateTranscriptUI(container);
      }
      if (result && result.audio_path) {
        const name = result.audio_path.replace(/.*[/\\]/, '');
        const el = document.getElementById('stat-output');
        if (el) el.textContent = name;
      }
    } catch { /* best effort */ }
  }

  const btn = container.querySelector('#record-btn');
  btn.className = 'wa-btn wa-btn-primary';
  btn.innerHTML = 'Start recording';
  container.querySelector('#pause-btn').disabled = true;
  document.getElementById('live-meta').innerHTML = '';

  // Auto-polish if configured
  if (sid && _liveCfg.diarize_by_default !== false && _liveCfg.hf_token) {
    runPolish(container, sid);
  } else {
    container.querySelector('#polish-btn').disabled = false;
  }

  // Restart always-on meter
  const sel = container.querySelector('#device-select');
  startDeviceMeter(container, sel?.value || '');
}

// ── Chunk drain ──────────────────────────────────────────────────────────────

async function _drainChunkQueue(container) {
  _draining = true;
  while (_pcmQueue.length > 0) {
    const sid = _session;
    if (!sid) break;

    const batch = _pcmQueue.splice(0);
    const totalLen = batch.reduce((n, a) => n + a.length, 0);
    const merged = new Float32Array(totalLen);
    let off = 0;
    for (const a of batch) { merged.set(a, off); off += a.length; }

    try {
      const sr = _audioCtx ? _audioCtx.sampleRate : 48000;
      const t0 = Date.now();
      const res = await api.streamChunk({ session_id: sid, audio_b64: float32ToBase64(merged), sample_rate: sr });
      const latency = Date.now() - t0;
      const latEl = document.getElementById('stat-latency');
      if (latEl) latEl.textContent = latency + ' ms';

      if (res.new_text) {
        _transcript.push({ id: _segIdCounter++, t: formatTime(_elapsed), txt: res.new_text, speaker: null, review: false });
        _partial = '';
      }
      if (res.partial) _partial = res.partial;
      updateTranscriptUI(container);
      document.getElementById('stat-words').textContent =
        _transcript.reduce((n, l) => n + l.txt.split(/\s+/).length, 0);
      document.getElementById('stat-segs').textContent = _transcript.length;

      const vadMeta = container.querySelector('#vad-meta');
      if (vadMeta) vadMeta.textContent = res.new_text ? 'VAD · speech ✓' : 'VAD · listening';
    } catch (err) {
      const vadMeta = container.querySelector('#vad-meta');
      if (vadMeta) vadMeta.textContent = `chunk error: ${err.detail || err.message || err}`;
      console.error('stream/chunk error:', err);
    }
  }
  _draining = false;
}

// ── Polish (align + diarize) ─────────────────────────────────────────────────

async function runPolish(container, sessionId) {
  const polishBtn = container.querySelector('#polish-btn');
  polishBtn.disabled = true;
  polishBtn.textContent = 'Polishing…';

  const vadMeta = container.querySelector('#vad-meta');
  if (vadMeta) vadMeta.textContent = 'Polishing…';

  try {
    const result = await api.streamPolish({ session_id: sessionId, hf_token: _liveCfg.hf_token || '' });
    if (result && result.segments && result.segments.length) {
      _applyPolishResult(result, container);
      if (vadMeta) vadMeta.textContent = 'Polish · done';
    }
  } catch {
    if (vadMeta) vadMeta.textContent = 'Polish · failed';
  } finally {
    polishBtn.textContent = 'Polish · align + diarize';
    polishBtn.disabled = false;
    _lastSessionId = null;
  }
}

function _applyPolishResult(result, container) {
  const segs = result.segments || [];
  if (!segs.length) return;

  _transcript = segs
    .filter(s => (s.text || '').trim())
    .map(s => ({
      id: _segIdCounter++,
      t: formatTime(Math.floor(s.start || 0)),
      txt: (s.text || '').trim(),
      speaker: s.speaker || null,
      review: false,
    }));

  updateTranscriptUI(container);
  document.getElementById('stat-words').textContent =
    _transcript.reduce((n, l) => n + l.txt.split(/\s+/).length, 0);
  document.getElementById('stat-segs').textContent = _transcript.length;
}

// ── Transcript UI ────────────────────────────────────────────────────────────

function updateTranscriptUI(container) {
  const body = container.querySelector('#transcript-body');
  if (!body) return;

  if (_transcript.length === 0 && !_partial) {
    if (_session && _liveCfg.streaming_show_listening !== false) {
      body.innerHTML = `
        <p style="display:flex;align-items:center;gap:8px;color:var(--ink-3);font-style:italic">
          <span style="width:8px;height:8px;border-radius:50%;background:var(--accent);flex-shrink:0;
            animation:pulse-ring 1.6s ease-in-out infinite;display:inline-block"></span>
          Listening…
        </p>`;
    } else if (!_session) {
      body.innerHTML = `<p style="color:var(--ink-4);font-style:italic">Transcript will appear here…</p>`;
    }
    return;
  }

  const lines = _transcript.map(l => {
    const speakerHtml = l.speaker
      ? `<span class="transcript-speaker" data-speaker="${escHtml(l.speaker)}"
           style="font-size:10.5px;font-weight:600;padding:1px 6px;border-radius:4px;
                  background:var(--paper-3);color:var(--ink-2);cursor:pointer;white-space:nowrap"
           title="Click to rename">${escHtml(_speakerNames.get(l.speaker) || l.speaker)}</span>`
      : '';
    return `
      <div class="transcript-line${l.review ? ' review-flagged' : ''}" data-seg-id="${l.id}"
           style="display:flex;align-items:baseline;gap:6px;${l.review ? 'background:color-mix(in oklch,var(--signal-wait) 12%,transparent);border-radius:4px;padding:4px 8px;margin:-4px -8px;' : ''}">
        <button class="review-flag-btn" data-flag="${l.id}"
          style="flex:0 0 auto;background:none;border:none;cursor:pointer;opacity:${l.review ? 1 : 0.3};
                 font-size:13px;color:${l.review ? 'var(--signal-wait)' : 'var(--ink-3)'};padding:0;line-height:1"
          title="${l.review ? 'Remove review flag' : 'Flag for review'}">⚑</button>
        <span class="transcript-ts mono">${l.t}</span>
        ${speakerHtml}
        <span>${escHtml(l.txt)}</span>
      </div>`;
  }).join('');

  const partial = _partial
    ? `<p style="color:var(--ink-3);font-style:italic">…${escHtml(_partial)}<span class="transcript-typing"> ·typing·</span></p>`
    : '';

  body.innerHTML = lines + partial;
  body.scrollTop = body.scrollHeight;

  // Review flag buttons
  body.querySelectorAll('.review-flag-btn').forEach(btn => {
    btn.addEventListener('click', e => {
      e.stopPropagation();
      const id = parseInt(btn.dataset.flag);
      const seg = _transcript.find(s => s.id === id);
      if (seg) { seg.review = !seg.review; updateTranscriptUI(container); }
    });
  });

  // Speaker rename on click
  body.querySelectorAll('.transcript-speaker').forEach(el => {
    el.addEventListener('click', () => {
      const orig = el.dataset.speaker;
      const current = _speakerNames.get(orig) || orig;
      const next = prompt(`Rename "${current}" across all segments:`, current);
      if (next && next !== current) {
        _speakerNames.set(orig, next);
        updateTranscriptUI(container);
      }
    });
  });
}

// ── Helpers ──────────────────────────────────────────────────────────────────

function float32ToBase64(f32) {
  const buf = new Uint8Array(f32.buffer);
  let bin = '';
  for (let i = 0; i < buf.length; i++) bin += String.fromCharCode(buf[i]);
  return btoa(bin);
}

function formatTime(secs) {
  const h = String(Math.floor(secs / 3600)).padStart(2, '0');
  const m = String(Math.floor((secs % 3600) / 60)).padStart(2, '0');
  const s = String(secs % 60).padStart(2, '0');
  return `${h}:${m}:${s}`;
}

function escHtml(s) {
  return String(s ?? '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}
