import { api } from './api.js';
import { NAV_ICONS } from './marks.js';

const MODELS = ['tiny', 'base', 'small', 'medium', 'large-v2', 'large-v3'];
const FORMATS = ['txt', 'srt', 'vtt', 'json', 'tsv', 'docx', 'pdf'];
let _pollTimer = null;

export function initTranscribe(container) {
  let selectedFile = null;
  let selectedFormats = new Set(['txt', 'srt', 'vtt', 'json']);

  container.innerHTML = `
    <div class="wa-topbar">
      <div>
        <h1>Transcribe</h1>
        <div class="wa-topbar-sub mono">Drop a file · or paste a path</div>
      </div>
      <div class="wa-topbar-meta mono" id="tx-meta"></div>
    </div>
    <div class="wa-content" style="display:grid;grid-template-columns:1.4fr 1fr;gap:24px;align-content:start">
      <!-- Left -->
      <div style="display:flex;flex-direction:column;gap:20px">
        <div class="wa-drop" id="drop-zone">
          <div style="width:40px;height:40px;color:var(--ink-3)">${NAV_ICONS.upload}</div>
          <div class="wa-drop-title">Drop audio or video</div>
          <div class="wa-drop-sub">.mp3 · .wav · .m4a · .mp4 · .mov · up to 4 GB</div>
          <div id="drop-file-name" style="font-family:var(--font-mono);font-size:11px;color:var(--accent);display:none"></div>
          <div style="display:flex;gap:8px;margin-top:6px">
            <label class="wa-btn" style="cursor:pointer">
              Choose file…
              <input type="file" id="file-picker" accept=".mp3,.wav,.m4a,.mp4,.mov,.flac,.ogg,.webm" style="display:none">
            </label>
            <button class="wa-btn" id="paste-path-btn" style="background:transparent">Paste path</button>
          </div>
        </div>

        <div class="wa-card">
          <div class="wa-card-head">
            <div class="wa-card-title">Pipeline</div>
            <div class="wa-card-sub" id="pipeline-sub">3 steps</div>
          </div>
          <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px">
            <div>
              <span class="wa-label">Model</span>
              <select class="wa-select" id="model-select">
                ${MODELS.map(m => `<option value="${m}"${m==='large-v2'?' selected':''}>${m}</option>`).join('')}
              </select>
            </div>
            <div>
              <span class="wa-label">Language</span>
              <select class="wa-select" id="lang-select">
                <option value="">Auto-detect</option>
                <option>en</option><option>fr</option><option>de</option>
                <option>es</option><option>it</option><option>ja</option>
                <option>zh</option><option>pt</option>
              </select>
            </div>
            <div style="grid-column:1/-1">
              <span class="wa-label">Output path</span>
              <input class="wa-input" id="output-path" placeholder="~/Downloads/transcripts">
            </div>
          </div>

          <div style="margin-top:20px;display:flex;flex-direction:column;gap:12px">
            <span class="wa-label">Stages</span>
            <div style="display:flex;flex-direction:column;gap:8px">
              <div class="pipeline-stage" id="stage-transcribe">
                <span class="pipeline-stage-idx mono">01</span>
                <div style="flex:1">
                  <div style="font-size:13px;font-weight:500">Transcribe</div>
                  <div class="pipeline-stage-detail mono">WhisperX · word-level timestamps</div>
                </div>
                <span class="wa-toggle on" data-stage="transcribe"></span>
              </div>
              <div class="pipeline-stage" id="stage-align">
                <span class="pipeline-stage-idx mono">02</span>
                <div style="flex:1">
                  <div style="font-size:13px;font-weight:500">Align</div>
                  <div class="pipeline-stage-detail mono">phoneme · forced alignment</div>
                </div>
                <span class="wa-toggle on" data-stage="align"></span>
              </div>
              <div class="pipeline-stage" id="stage-diarize">
                <span class="pipeline-stage-idx mono">03</span>
                <div style="flex:1">
                  <div style="font-size:13px;font-weight:500">Diarize</div>
                  <div class="pipeline-stage-detail mono">pyannote · who-said-what</div>
                </div>
                <span class="wa-toggle on" data-stage="diarize"></span>
              </div>
            </div>
          </div>

          <div style="margin-top:20px">
            <span class="wa-label">Output formats</span>
            <div style="display:flex;gap:6px;flex-wrap:wrap" id="format-chips">
              ${FORMATS.map(f => `
                <span class="wa-chip${selectedFormats.has(f)?' on':''}" data-fmt="${f}">${f}</span>
              `).join('')}
            </div>
          </div>
        </div>
      </div>

      <!-- Right: processing + queue -->
      <div style="display:flex;flex-direction:column;gap:20px">
        <!-- Processing card -->
        <div class="wa-card" id="processing-card">
          <div class="wa-card-head">
            <div class="wa-card-title">Processing</div>
            <button class="wa-btn" id="pause-btn" style="height:26px;font-size:11.5px;padding:0 10px">Pause</button>
          </div>
          <div id="processing-list">
            <div style="color:var(--ink-4);font-size:12.5px">Idle — nothing processing.</div>
          </div>
        </div>

        <!-- Queue card -->
        <div class="wa-card">
          <div class="wa-card-head">
            <div class="wa-card-title">Queue</div>
            <div class="wa-card-sub" id="queue-sub">—</div>
          </div>
          <div id="queue-list" style="display:flex;flex-direction:column;gap:8px">
            <div style="color:var(--ink-4);font-size:12.5px">No jobs queued.</div>
          </div>
          <div id="history-list" style="display:flex;flex-direction:column;gap:6px"></div>
          <div style="display:flex;gap:8px;margin-top:16px;padding-top:16px;border-top:1px solid var(--rule)">
            <button class="wa-btn wa-btn-primary" id="add-btn" style="flex:1">Add to queue</button>
            <button class="wa-btn" id="clear-btn">Clear history</button>
          </div>
        </div>
      </div>
    </div>
  `;

  // Load saved config
  api.getConfig().then(cfg => {
    document.getElementById('output-path').value = cfg.default_output_path || '';
    if (!cfg.diarize_by_default) {
      const tog = container.querySelector('[data-stage="diarize"]');
      if (tog) { tog.classList.remove('on'); container.querySelector('#stage-diarize').classList.add('off'); }
    }
  }).catch(() => {});

  // Wire up drop zone
  const zone = container.querySelector('#drop-zone');
  zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('dragover'); });
  zone.addEventListener('dragleave', () => zone.classList.remove('dragover'));
  zone.addEventListener('drop', e => {
    e.preventDefault();
    zone.classList.remove('dragover');
    const file = e.dataTransfer.files[0];
    if (file) selectFile(file);
  });

  container.querySelector('#file-picker').addEventListener('change', e => {
    if (e.target.files[0]) selectFile(e.target.files[0]);
  });

  container.querySelector('#paste-path-btn').addEventListener('click', () => {
    const p = prompt('Paste file path:');
    if (p) { selectedFile = { type: 'path', value: p.trim() }; showFileName(p.trim()); }
  });

  function selectFile(file) {
    selectedFile = { type: 'file', value: file };
    showFileName(file.name);
  }

  function showFileName(name) {
    const el = container.querySelector('#drop-file-name');
    el.textContent = name;
    el.style.display = 'block';
  }

  // Format chip toggles
  container.querySelector('#format-chips').addEventListener('click', e => {
    const chip = e.target.closest('[data-fmt]');
    if (!chip) return;
    const fmt = chip.dataset.fmt;
    if (selectedFormats.has(fmt)) selectedFormats.delete(fmt);
    else selectedFormats.add(fmt);
    chip.classList.toggle('on', selectedFormats.has(fmt));
  });

  // Stage toggles
  container.querySelectorAll('[data-stage]').forEach(tog => {
    tog.addEventListener('click', () => {
      const on = tog.classList.toggle('on');
      tog.closest('.pipeline-stage').classList.toggle('off', !on);
    });
  });

  // Add to queue
  container.querySelector('#add-btn').addEventListener('click', async () => {
    if (!selectedFile) { alert('Select a file first.'); return; }

    // Check: emotion enabled but no models configured
    try {
      const cfg = await api.getConfig();
      if (cfg.emotion_enabled && (!cfg.emotion_model_ids || cfg.emotion_model_ids.length === 0)) {
        alert('Emotion detection is on but no models are enabled. Go to Settings → Emotion Models to download one first.');
        return;
      }
    } catch { /* server may not be ready, continue anyway */ }

    const btn = container.querySelector('#add-btn');
    btn.disabled = true;
    btn.textContent = 'Adding…';
    try {
      let filePath;
      if (selectedFile.type === 'path') {
        filePath = selectedFile.value;
      } else {
        const up = await api.upload(selectedFile.value);
        filePath = up.path;
      }
      const diarize = container.querySelector('[data-stage="diarize"]').classList.contains('on');
      await api.transcribe({
        file_path: filePath,
        output_path: container.querySelector('#output-path').value || null,
        model: container.querySelector('#model-select').value,
        diarize,
        formats: [...selectedFormats],
      });
      selectedFile = null;
      container.querySelector('#drop-file-name').style.display = 'none';
      refreshQueue();
    } catch (err) {
      alert('Error: ' + err.message);
    } finally {
      btn.disabled = false;
      btn.textContent = 'Add to queue';
    }
  });

  // Clear history
  container.querySelector('#clear-btn').addEventListener('click', async () => {
    await api.clearJobs().catch(() => {});
    refreshQueue();
  });

  // Pause/resume
  container.querySelector('#pause-btn').addEventListener('click', async () => {
    const btn = container.querySelector('#pause-btn');
    const paused = btn.dataset.paused === 'true';
    if (paused) {
      await api.resumeWorker().catch(() => {});
      btn.dataset.paused = 'false';
      btn.textContent = 'Pause';
    } else {
      await api.pauseWorker().catch(() => {});
      btn.dataset.paused = 'true';
      btn.textContent = 'Resume';
    }
  });

  // Sync initial pause state
  api.workerStatus().then(s => {
    const btn = container.querySelector('#pause-btn');
    if (btn && s.paused) { btn.dataset.paused = 'true'; btn.textContent = 'Resume'; }
  }).catch(() => {});

  // Poll queue
  refreshQueue();
  _pollTimer = setInterval(refreshQueue, 3000);
}

async function refreshQueue() {
  let jobs;
  try { jobs = await api.listJobs({ limit: 50 }); } catch { return; }
  const processingList = document.getElementById('processing-list');
  const queueList      = document.getElementById('queue-list');
  const historyList    = document.getElementById('history-list');
  const sub            = document.getElementById('queue-sub');
  if (!queueList) return;

  const running  = jobs.filter(j => j.status === 'running' || j.status === 'speaker_review');
  const queued   = jobs.filter(j => j.status === 'queued')
                       .sort((a, b) => a.order_idx - b.order_idx);
  const history  = jobs.filter(j => ['done','failed','cancelled'].includes(j.status));

  // Processing card
  if (processingList) {
    processingList.innerHTML = running.length
      ? running.map(j => renderRunningItem(j)).join('')
      : `<div style="color:var(--ink-4);font-size:12.5px">Idle — nothing processing.</div>`;
  }

  // Queue card
  sub.textContent = queued.length ? `${queued.length} waiting` : '—';
  queueList.innerHTML = queued.length
    ? queued.map((j, i) => renderQueuedItem(j, i === 0, i === queued.length - 1)).join('')
    : `<div style="color:var(--ink-4);font-size:12.5px">No jobs queued.</div>`;

  // History
  if (historyList) {
    historyList.innerHTML = history.length
      ? `<div style="margin-top:10px;padding-top:10px;border-top:1px solid var(--rule);display:flex;flex-direction:column;gap:6px">${
          history.map(j => renderHistoryItem(j)).join('')
        }</div>`
      : '';
  }

  // Wire up speaker-review deep-link badges in the Processing card
  if (processingList) {
    processingList.querySelectorAll('[data-review]').forEach(btn => {
      btn.addEventListener('click', e => {
        e.stopPropagation();
        const jobId = btn.dataset.review;
        // Tell the speakers screen which job to open, then navigate.
        window.dispatchEvent(new CustomEvent('wa-open-speaker-review', { detail: { jobId } }));
        location.hash = 'speakers';
      });
    });
  }

  // Wire up queue reorder + delete buttons
  queueList.querySelectorAll('[data-move-up]').forEach(btn => {
    btn.addEventListener('click', async e => {
      e.stopPropagation();
      await api.moveJobUp(btn.dataset.moveUp).catch(() => {});
      refreshQueue();
    });
  });
  queueList.querySelectorAll('[data-move-down]').forEach(btn => {
    btn.addEventListener('click', async e => {
      e.stopPropagation();
      await api.moveJobDown(btn.dataset.moveDown).catch(() => {});
      refreshQueue();
    });
  });
  queueList.querySelectorAll('[data-delete]').forEach(btn => {
    btn.addEventListener('click', async e => {
      e.stopPropagation();
      await api.deleteJob(btn.dataset.delete).catch(() => {});
      refreshQueue();
    });
  });
  if (historyList) {
    historyList.querySelectorAll('[data-delete]').forEach(btn => {
      btn.addEventListener('click', async e => {
        e.stopPropagation();
        await api.deleteJob(btn.dataset.delete).catch(() => {});
        refreshQueue();
      });
    });
  }
}

function renderRunningItem(j) {
  const pct = j.progress ?? 0;
  const stage = j.stage || j.status;
  // When the worker has stopped on `speaker_review`, the badge becomes a
  // deep-link button so the user can hop straight to that job's review pane.
  const badge = j.status === 'speaker_review'
    ? `<button class="queue-badge queue-badge-link" data-review="${j.id}" title="Open speaker review">${j.status} ▸</button>`
    : `<span class="queue-badge">${j.status}</span>`;
  return `
    <div class="queue-item running" data-job-id="${j.id}">
      <div class="queue-item-header">
        <span class="queue-dot running"></span>
        <span class="queue-item-title">${escHtml(j.file_name || j.file_path)}</span>
        ${badge}
      </div>
      <div class="queue-meta mono">${j.model}${j.diarize ? ' · diarize' : ''}</div>
      <div class="queue-progress">
        <div class="queue-progress-bar"><div class="queue-progress-fill" style="width:${pct}%"></div></div>
        <div class="queue-progress-meta mono">
          <span>${escHtml(stage)}</span><span>${pct}%</span>
        </div>
      </div>
    </div>`;
}

function renderQueuedItem(j, isFirst, isLast) {
  return `
    <div class="queue-item queued" data-job-id="${j.id}">
      <div class="queue-item-header">
        <div class="queue-reorder-col">
          <button class="queue-reorder-btn" data-move-up="${j.id}" ${isFirst ? 'disabled' : ''} title="Move up">▲</button>
          <button class="queue-reorder-btn" data-move-down="${j.id}" ${isLast ? 'disabled' : ''} title="Move down">▼</button>
        </div>
        <span class="queue-dot queued"></span>
        <span class="queue-item-title">${escHtml(j.file_name || j.file_path)}</span>
        <span class="queue-badge">queued</span>
        <button class="queue-remove-btn" data-delete="${j.id}" title="Remove">✕</button>
      </div>
      <div class="queue-meta mono" style="padding-left:28px">${j.model}${j.diarize ? ' · diarize' : ''}</div>
    </div>`;
}

function renderHistoryItem(j) {
  const icon = j.status === 'done' ? '✓' : j.status === 'failed' ? '✗' : '—';
  const color = j.status === 'done' ? 'var(--signal-ok)' : j.status === 'failed' ? 'var(--signal-rec)' : 'var(--ink-4)';
  return `
    <div style="display:flex;align-items:center;gap:8px;font-size:12px;color:var(--ink-3)">
      <span style="color:${color};font-size:11px;width:12px;text-align:center;flex-shrink:0">${icon}</span>
      <span style="flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${escHtml(j.file_name || j.file_path)}</span>
      <span class="mono" style="color:var(--ink-4);font-size:10.5px">${j.status}</span>
      <button class="queue-remove-btn" data-delete="${j.id}" title="Remove" style="font-size:10px">✕</button>
    </div>`;
}

function escHtml(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}
