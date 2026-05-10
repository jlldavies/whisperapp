import { api } from './api.js';

const IS_MAC = /Mac|iPhone|iPad/i.test(navigator.platform || navigator.userAgent || '');
const REVEAL_LABEL = IS_MAC ? 'Reveal in Finder' : 'Show in Explorer';

const SECTIONS = ['Transcription', 'AI features', 'Acoustic', 'Models', 'Storage', 'Startup', 'API & CLI', 'About'];
const PROVIDERS = [
  { id: 'none',   label: 'None',   sub: 'manual labels only' },
  { id: 'claude', label: 'Claude', sub: 'anthropic · cloud' },
  { id: 'openai', label: 'OpenAI', sub: 'gpt-4o · cloud' },
  { id: 'ollama', label: 'Ollama', sub: 'local · no key needed' },
];

export function initSettings(container) {
  container.innerHTML = `
    <div class="wa-topbar">
      <div>
        <h1>Settings</h1>
        <div class="wa-topbar-sub mono">~/.whisperapp/config.json</div>
      </div>
      <div class="wa-topbar-meta mono" id="settings-meta"></div>
    </div>
    <div class="wa-content" style="display:grid;grid-template-columns:180px 1fr;gap:28px;max-width:980px;align-content:start">
      <div class="settings-subnav" id="settings-subnav">
        ${SECTIONS.map((s, i) => `
          <div class="settings-subnav-item${i===0?' active':''}" data-section="${i}">
            <span>${s}</span>${s === 'Models' ? '<span class="settings-update-pip" id="models-update-pip" style="display:none">*</span>' : ''}
          </div>
        `).join('')}
      </div>
      <div id="settings-body" style="display:flex;flex-direction:column;gap:24px"></div>
    </div>
  `;

  let cfg = {};
  let activeSection = 0;

  async function load() {
    try {
      cfg = await api.getConfig();
      render();
    } catch { /* server may not be ready */ }
    // Background: fetch the catalogue once on settings open so the Models tab
    // can show its red `*` even before the user clicks into it. This is a
    // lightweight read — the heavy PyPI lookups are cached server-side.
    refreshUpdatePip();
  }

  async function refreshUpdatePip() {
    const pip = container.querySelector('#models-update-pip');
    if (!pip) return;
    try {
      const cat = await api.getCatalogue();
      const n = _countPendingUpdates(cat);
      pip.style.display = n > 0 ? 'inline' : 'none';
      pip.title = n ? `${n} update${n === 1 ? '' : 's'} available` : '';
    } catch { /* ignore */ }
  }
  // Re-poll every 30s so the user notices when an update appears mid-session.
  setInterval(refreshUpdatePip, 30000);

  function render() {
    const body = container.querySelector('#settings-body');
    body.innerHTML = '';

    if (activeSection === 0) renderTranscription(body, cfg);
    if (activeSection === 1) renderAI(body, cfg);
    if (activeSection === 2) renderAcoustic(body, cfg);
    if (activeSection === 3) renderModels(body, cfg);
    if (activeSection === 4) renderStorage(body, cfg);
    if (activeSection === 5) renderStartup(body, cfg);
    if (activeSection === 6) renderAPICLI(body);
    if (activeSection === 7) renderAbout(body);

    if (activeSection < 3 || (activeSection > 3 && activeSection < 6)) renderSaveRow(body, cfg);
  }

  container.querySelector('#settings-subnav').addEventListener('click', e => {
    const item = e.target.closest('[data-section]');
    if (!item) return;
    activeSection = parseInt(item.dataset.section);
    container.querySelectorAll('.settings-subnav-item').forEach((el, i) => {
      el.classList.toggle('active', i === activeSection);
    });
    render();
  });

  load();
}

function field(label, inputHtml, hint = '') {
  return `
    <div>
      <span class="wa-label">${label}</span>
      ${inputHtml}
      ${hint ? `<div class="wa-field-hint">${hint}</div>` : ''}
    </div>`;
}

// Count rows across the catalogue that are in the `update` state. Used by
// the Settings → Models nav to show a red `*` when something needs attention.
function _countPendingUpdates(catalogue) {
  if (!catalogue) return 0;
  let n = 0;
  for (const cat of Object.values(catalogue)) {
    for (const m of (cat.models || [])) if (m.state === 'update') n++;
  }
  return n;
}

function toggleRow(label, checked, key) {
  return `
    <label class="toggle-row">
      <span class="wa-toggle${checked?' on':''}" data-toggle="${key}"></span>
      <span>${label}</span>
    </label>`;
}

function renderTranscription(body, cfg) {
  body.innerHTML += `
    <section class="settings-section">
      <header class="settings-section-head">
        <h2>Transcription</h2>
        <span class="wa-card-sub mono">WhisperX &amp; diarization</span>
      </header>
      ${field('HuggingFace token',
        `<input class="wa-input" type="password" data-cfg="hf_token" value="${escHtml(cfg.hf_token||'')}">`,
        'Required for pyannote.audio diarization. Stored locally — never sent off-device.'
      )}
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:14px">
        ${field('Default model',
          `<select class="wa-select" data-cfg="default_model">
            ${['tiny','base','small','medium','large-v2','large-v3'].map(m=>
              `<option${cfg.default_model===m?' selected':''}>${m}</option>`).join('')}
          </select>`
        )}
        ${field('Streaming model',
          `<select class="wa-select" data-cfg="streaming_model">
            ${['tiny','base','small'].map(m=>
              `<option${cfg.streaming_model===m?' selected':''}>${m}</option>`).join('')}
          </select>`
        )}
      </div>
      ${toggleRow('Diarization on by default', cfg.diarize_by_default, 'diarize_by_default')}
    </section>
    <section class="settings-section">
      <header class="settings-section-head">
        <h2>Pauses</h2>
        <span class="wa-card-sub mono">Silence markers in transcripts</span>
      </header>
      ${toggleRow('Enable pause detection', cfg.pause_detection !== false, 'pause_detection')}
      <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:14px">
        ${field('Pause threshold (s)',
          `<input class="wa-input" type="number" min="1" data-cfg="pause_threshold" value="${escHtml(cfg.pause_threshold ?? 3)}">`,
          'e.g. 3 seconds'
        )}
        ${field('Long pause threshold (s)',
          `<input class="wa-input" type="number" min="1" data-cfg="long_pause_threshold" value="${escHtml(cfg.long_pause_threshold ?? 10)}">`,
          'e.g. 10 seconds'
        )}
        ${field('Gap threshold (s)',
          `<input class="wa-input" type="number" min="1" data-cfg="gap_threshold" value="${escHtml(cfg.gap_threshold ?? 30)}">`,
          'e.g. 30 seconds'
        )}
      </div>
    </section>`;

  wireToggles(body, cfg);
  wireInputs(body, cfg);
}

function renderAI(body, cfg) {
  body.innerHTML += `
    <section class="settings-section">
      <header class="settings-section-head">
        <h2>AI features</h2>
        <span class="wa-card-sub mono">Optional · speaker ID &amp; meeting notes</span>
      </header>
      <p style="font-size:12.5px;color:var(--ink-3);line-height:1.5">
        Enable a provider for automatic speaker identification, meeting notes, and live summaries.
        All transcription works without AI.
      </p>
      <div class="provider-grid" id="provider-grid">
        ${PROVIDERS.map(p => `
          <div class="provider-card${cfg.ai_provider===p.id?' active':''}" data-provider="${p.id}">
            <div class="provider-card-name">${p.label}</div>
            <div class="provider-card-sub mono">${p.sub}</div>
          </div>`).join('')}
      </div>
      ${field('API key',
        `<input class="wa-input" type="password" data-cfg="ai_api_key" value="${escHtml(cfg.ai_api_key||'')}">`
      )}
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:14px">
        ${field('Model', `<input class="wa-input" data-cfg="ai_model" value="${escHtml(cfg.ai_model||'')}" placeholder="(provider default)">`)}
        ${field('Base URL', `<input class="wa-input" data-cfg="ai_base_url" value="${escHtml(cfg.ai_base_url||'')}" placeholder="https://api.anthropic.com">`)}
      </div>
    </section>`;

  wireInputs(body, cfg);

  body.querySelector('#provider-grid').addEventListener('click', e => {
    const card = e.target.closest('[data-provider]');
    if (!card) return;
    cfg.ai_provider = card.dataset.provider;
    body.querySelectorAll('.provider-card').forEach(c => {
      c.classList.toggle('active', c.dataset.provider === cfg.ai_provider);
    });
  });
}

function renderStorage(body, cfg) {
  body.innerHTML += `
    <section class="settings-section">
      <header class="settings-section-head">
        <h2>Storage</h2>
        <span class="wa-card-sub mono">Outputs &amp; queue</span>
      </header>
      ${field('Default output path',
        `<input class="wa-input" data-cfg="default_output_path" value="${escHtml(cfg.default_output_path||'')}">`,
        'Transcripts and exports are saved here unless overridden per-job.'
      )}
      <div>
        <span class="wa-label">Config directory</span>
        <div class="wa-card" style="background:var(--paper-2);padding:10px 12px;font-family:var(--font-mono);font-size:12px;color:var(--ink-2)">~/.whisperapp/</div>
      </div>
      <div>
        <span class="wa-label">Queue</span>
        <div style="display:flex;gap:8px;align-items:center">
          <button class="wa-btn wa-btn-danger" id="clear-jobs-btn">Clear completed jobs</button>
          <span class="mono" style="font-size:11px;color:var(--ink-4)" id="clear-jobs-result"></span>
        </div>
      </div>
    </section>

    <section class="settings-section">
      <header class="settings-section-head">
        <h2>Output templates</h2>
        <span class="wa-card-sub mono">DOCX &amp; PDF</span>
      </header>
      <p style="font-size:12.5px;color:var(--ink-3);line-height:1.6;margin-bottom:12px">
        DOCX and PDF outputs use a template file with
        <span class="mono" style="background:var(--paper-2);padding:1px 5px;border-radius:3px">{{field_name}}</span>
        markers. Leave blank to use the built-in default. Download the defaults below,
        edit them, then point the path here to use your version.
      </p>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:16px">
        <div>
          <span class="wa-label">DOCX template path</span>
          <input class="wa-input" data-cfg="output_template_docx"
            placeholder="~/.whisperapp/templates/custom.docx"
            value="${escHtml(cfg.output_template_docx||'')}">
          <div style="font-size:11px;color:var(--ink-4);margin-top:4px">
            Leave blank to use the built-in default layout
          </div>
        </div>
        <div>
          <span class="wa-label">PDF template path</span>
          <input class="wa-input" data-cfg="output_template_pdf"
            placeholder="~/.whisperapp/templates/custom.html"
            value="${escHtml(cfg.output_template_pdf||'')}">
          <div style="font-size:11px;color:var(--ink-4);margin-top:4px">
            HTML file — use CSS @page for print margins
          </div>
        </div>
      </div>
      <div style="display:flex;gap:8px;flex-wrap:wrap">
        <a class="wa-btn" href="http://127.0.0.1:7861/templates/download-docx"
           download="whisperapp_template.docx">Download DOCX template</a>
        <a class="wa-btn" href="http://127.0.0.1:7861/templates/download-html"
           download="whisperapp_pdf_template.html">Download HTML template</a>
        <a class="wa-btn" href="http://127.0.0.1:7861/templates/download-legal-html"
           download="whisperapp_legal_template.html">Download Legal template</a>
      </div>
      <div style="margin-top:16px">
        <span class="wa-label" style="display:block;margin-bottom:8px">Available markers</span>
        <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:4px 12px;font-size:11px">
          ${[
            ['{{filename}}','Source file name'],
            ['{{datetime}}','Date and time'],
            ['{{duration}}','Audio duration'],
            ['{{speakers}}','Speaker names'],
            ['{{word_count}}','Word count'],
            ['{{model}}','Whisper model'],
            ['{{transcript}}','Plain transcript'],
            ['{{diarized_transcript}}','With speaker labels'],
            ['{{segments}}','Timestamped segments'],
            ['{{meeting_notes}}','AI meeting notes'],
            ['{{language}}','Detected language'],
            ['{{filepath}}','Full source path'],
            ['{{source_hash}}','SHA-256 of source file'],
            ['{{file_size_bytes}}','Source file size'],
            ['{{legal_transcript}}','Line-numbered transcript'],
          ].map(([marker, desc]) => `
            <div style="display:flex;flex-direction:column;gap:2px;padding:4px 0">
              <span class="mono" style="font-size:10.5px;color:var(--accent)">${marker}</span>
              <span style="color:var(--ink-3)">${desc}</span>
            </div>`).join('')}
        </div>
      </div>
    </section>`;

  wireInputs(body, cfg);

  body.querySelector('#clear-jobs-btn').addEventListener('click', async () => {
    const btn = body.querySelector('#clear-jobs-btn');
    const result = body.querySelector('#clear-jobs-result');
    btn.disabled = true;
    try {
      const { cleared } = await api.clearJobs();
      result.textContent = `${cleared} job${cleared !== 1 ? 's' : ''} removed`;
      setTimeout(() => { result.textContent = ''; }, 3000);
    } catch {
      result.textContent = 'Error';
    } finally {
      btn.disabled = false;
    }
  });
}

function renderStartup(body, cfg) {
  // The two toggles target *different* persistence layers:
  //
  //   - "Launch on login" is an OS-level registration (Windows registry HKCU
  //     Run key on Windows, ~/Library/LaunchAgents plist on macOS). Driven
  //     directly by /startup/enable + /startup/disable, never via cfg.
  //
  //   - "Auto-update WhisperX on startup" is a config field (cfg.auto_update)
  //     which __main__.py reads to decide whether to spin the update thread.
  body.insertAdjacentHTML('beforeend', `
    <section class="settings-section">
      <header class="settings-section-head"><h2>Startup</h2></header>
      <p style="font-size:12.5px;color:var(--ink-3);line-height:1.5">
        Launch-on-login uses the OS registration mechanism for your platform —
        the Windows HKCU Run key, or a macOS LaunchAgent plist. Stored
        per-machine, not synced.
      </p>
      <div id="startup-launch-row">${toggleRow('Launch on login', false, '__launch_on_login')}</div>
      ${toggleRow('Auto-update WhisperX on startup', cfg.auto_update !== false, 'auto_update')}
    </section>`);

  // The cfg-backed Auto-update toggle uses the standard wireToggles handler,
  // which flips cfg[key] on click; the Save button persists it via /config.
  wireToggles(body, cfg);

  // The OS-level Launch-on-login toggle is wired separately because it's a
  // direct API call (not a cfg flip) and needs the initial state fetched
  // from the daemon.
  const launchTog = body.querySelector('[data-toggle="__launch_on_login"]');
  if (launchTog) {
    launchTog.classList.remove('on');  // start off until we hear from the server
    launchTog.dataset.busy = '';
    api.startupStatus().then(({ registered }) => {
      launchTog.classList.toggle('on', !!registered);
    }).catch(() => {});

    launchTog.addEventListener('click', async () => {
      // Optimistic flip, revert on error. Disable while in flight to stop
      // double-clicks from racing.
      if (launchTog.dataset.busy === '1') return;
      const wasOn = launchTog.classList.contains('on');
      const wantOn = !wasOn;
      launchTog.dataset.busy = '1';
      launchTog.classList.toggle('on', wantOn);
      try {
        const r = wantOn ? await api.startupEnable() : await api.startupDisable();
        launchTog.classList.toggle('on', !!r.registered);
      } catch (e) {
        launchTog.classList.toggle('on', wasOn);  // revert
        console.warn('startup toggle failed:', e);
      } finally {
        launchTog.dataset.busy = '';
      }
    });
  }
}

function renderAPICLI(body) {
  body.innerHTML += `
    <section class="settings-section">
      <header class="settings-section-head">
        <h2>API &amp; CLI</h2>
        <span class="wa-card-sub mono">Local-only · 127.0.0.1</span>
      </header>
      <div style="display:flex;flex-direction:column;gap:8px;font-size:13px;color:var(--ink-2)">
        ${field('REST API', `<input class="wa-input" value="http://127.0.0.1:7861" readonly>`)}
        ${field('API docs', `<a href="http://127.0.0.1:7861/docs" target="_blank" class="wa-btn" style="display:inline-flex;margin-top:4px">Open API docs →</a>`)}
      </div>
      <div style="margin-top:12px">
        <span class="wa-label">CLI usage</span>
        <div class="wa-card" style="background:var(--paper-2);padding:14px;font-family:var(--font-mono);font-size:12px;color:var(--ink-2);line-height:1.7">
          whisperapp transcribe recording.mp3 -m large-v2 --diarize<br>
          whisperapp status &lt;job-id&gt;<br>
          whisperapp list
        </div>
      </div>
    </section>`;
}

function renderAbout(body) {
  body.innerHTML += `
    <section class="settings-section">
      <header class="settings-section-head"><h2>About</h2></header>
      <div style="display:flex;flex-direction:column;gap:8px;font-size:13px;color:var(--ink-2)">
        <div><span style="color:var(--ink-3);font-family:var(--font-mono);font-size:11px">VERSION</span><br>WhisperApp v1.1.0</div>
        <div><span style="color:var(--ink-3);font-family:var(--font-mono);font-size:11px">ENGINE</span><br>WhisperX · pyannote.audio · faster-whisper</div>
        <div><span style="color:var(--ink-3);font-family:var(--font-mono);font-size:11px">CONFIG</span><br><span class="mono">~/.whisperapp/config.json</span></div>
      </div>
    </section>`;
}

function renderAcoustic(body, cfg) {
  const enabled = cfg.acoustic_enabled === true;
  body.innerHTML += `
    <section class="settings-section">
      <header class="settings-section-head">
        <h2>Acoustic</h2>
        <span class="wa-card-sub mono">Volume, pitch &amp; speaking rate markers</span>
      </header>
      <p style="font-size:12.5px;color:var(--ink-3);line-height:1.5">
        Detects vocal emphasis using audio analysis. Inserts markers like [Loud], [Raised voice],
        [Rapid] into txt/srt/vtt transcripts. Pure DSP — no AI provider needed.
      </p>
      ${toggleRow('Enable acoustic features', enabled, 'acoustic_enabled')}
      <div id="acoustic-sub" style="display:${enabled ? 'flex' : 'none'};flex-direction:column;gap:16px;margin-top:8px;padding-left:12px;border-left:2px solid var(--rule)">
        <div>
          <div style="font-weight:600;font-size:12.5px;margin-bottom:8px">Volume / emphasis</div>
          ${toggleRow('Detect loud/quiet segments', cfg.acoustic_volume_enabled !== false, 'acoustic_volume_enabled')}
          ${field('Threshold (multiplier)',
            `<input class="wa-input" type="range" min="1.2" max="3.0" step="0.1"
               data-cfg="acoustic_volume_threshold" value="${escHtml(cfg.acoustic_volume_threshold ?? 1.8)}"
               oninput="this.nextElementSibling.textContent=this.value">
             <span class="mono" style="font-size:11px;color:var(--ink-3)">${escHtml(cfg.acoustic_volume_threshold ?? 1.8)}×</span>`,
            'Ratio above/below baseline to trigger [Loud] or [Quiet]. Default: 1.8×'
          )}
        </div>
        <div>
          <div style="font-weight:600;font-size:12.5px;margin-bottom:8px">Pitch variation</div>
          ${toggleRow('Detect pitch changes', cfg.acoustic_pitch_enabled !== false, 'acoustic_pitch_enabled')}
          ${field('Std dev threshold (Hz)',
            `<input class="wa-input" type="number" min="20" max="100"
               data-cfg="acoustic_pitch_threshold" value="${escHtml(cfg.acoustic_pitch_threshold ?? 40)}">`,
            'Standard deviation of F0 to trigger [Animated]. Default: 40 Hz'
          )}
        </div>
        <div>
          <div style="font-weight:600;font-size:12.5px;margin-bottom:8px">Speaking rate</div>
          ${toggleRow('Detect speaking rate', cfg.acoustic_rate_enabled !== false, 'acoustic_rate_enabled')}
          <div style="display:grid;grid-template-columns:1fr 1fr;gap:14px">
            ${field('Fast threshold (WPM)',
              `<input class="wa-input" type="number" min="100" max="300"
                 data-cfg="acoustic_rate_fast_wpm" value="${escHtml(cfg.acoustic_rate_fast_wpm ?? 180)}">`,
              'Above this WPM → [Rapid]'
            )}
            ${field('Slow threshold (WPM)',
              `<input class="wa-input" type="number" min="20" max="150"
                 data-cfg="acoustic_rate_slow_wpm" value="${escHtml(cfg.acoustic_rate_slow_wpm ?? 90)}">`,
              'Below this WPM → [Slow]'
            )}
          </div>
        </div>
      </div>
    </section>`;

  wireToggles(body, cfg);
  wireInputs(body, cfg);

  // Show/hide sub-controls based on master toggle
  const masterTog = body.querySelector('[data-toggle="acoustic_enabled"]');
  const subDiv = body.querySelector('#acoustic-sub');
  if (masterTog && subDiv) {
    masterTog.addEventListener('click', () => {
      subDiv.style.display = masterTog.classList.contains('on') ? 'flex' : 'none';
    });
  }
}

// ══════════════════════════════════════════════════════════════════════════
// Models settings page — unified catalogue for all 5 model categories
// ══════════════════════════════════════════════════════════════════════════

const LANG_LIST = [
  { code: 'en', name: 'English' },
  { code: 'fr', name: 'French' },
  { code: 'de', name: 'German' },
  { code: 'es', name: 'Spanish' },
  { code: 'it', name: 'Italian' },
  { code: 'pt', name: 'Portuguese' },
  { code: 'nl', name: 'Dutch' },
  { code: 'ja', name: 'Japanese' },
  { code: 'zh', name: 'Chinese' },
];

const CAT_COLORS = [
  'oklch(60% 0.13 45)',   // transcription · accent-ish
  'oklch(58% 0.11 220)',  // streaming · slate
  'oklch(60% 0.10 145)',  // alignment · sage
  'oklch(58% 0.09 290)',  // diarization · mauve
  'oklch(65% 0.11 60)',   // emotion · warm
  'oklch(58% 0.05 30)',   // capabilities · neutral grey-warm
];

const CAT_KEYS = ['transcription', 'streaming', 'alignment', 'diarization', 'emotion', 'capabilities'];

// Local state — persists within the settings page session
const _modelsState = {
  search: '',
  langFilter: 'all',
  // All categories collapsed by default — the Models page lists ~30 entries
  // across 6 sections, so opening everything makes the page very busy. Users
  // open the section they care about; the toggle state persists for the
  // lifetime of the Settings page session.
  drawerOpen: {
    transcription: false, alignment: false, streaming: false,
    diarization: false, emotion: false, capabilities: false,
  },
  catalogue: null,
  diskSummary: null,
  loading: true,
  pollTimer: null,
};

function _mBtn(label, variant, size, attrs = '') {
  const hMap = { sm: 26, md: 32 };
  const h = hMap[size] || 26;
  const pad = size === 'sm' ? '0 10px' : '0 14px';
  const fs = size === 'sm' ? 12 : 13;
  const br = size === 'sm' ? 5 : 6;
  const styles = {
    default: `background:var(--paper-3);color:var(--ink);border:1px solid var(--rule)`,
    primary: `background:var(--ink);color:var(--paper);border:1px solid var(--ink)`,
    accent:  `background:var(--accent);color:#fff;border:1px solid var(--accent)`,
    ghost:   `background:transparent;color:var(--ink-3);border:1px solid transparent`,
    danger:  `background:transparent;color:var(--signal-rec);border:1px solid color-mix(in oklch,var(--signal-rec) 25%,transparent)`,
    warn:    `background:var(--signal-wait);color:#fff;border:1px solid var(--signal-wait)`,
  }[variant] || '';
  return `<button style="height:${h}px;padding:${pad};border-radius:${br}px;font-size:${fs}px;font-weight:500;font-family:inherit;cursor:pointer;display:inline-flex;align-items:center;gap:6px;white-space:nowrap;${styles}" ${attrs}>${label}</button>`;
}

function _statusDotHtml(state) {
  const map = {
    active:      { bg: 'var(--signal-go)', glow: true },
    installed:   { bg: 'var(--signal-go)', glow: false },
    update:      { bg: 'var(--signal-wait)', glow: true },
    downloading: { bg: 'var(--signal-wait)', glow: true },
    available:   { bg: 'var(--ink-4)', glow: false },
  };
  const v = map[state] || map.available;
  const shadow = v.glow ? `box-shadow:0 0 0 4px color-mix(in oklch,${v.bg} 22%,transparent)` : '';
  return `<span style="width:8px;height:8px;border-radius:50%;background:${v.bg};${shadow};flex-shrink:0;display:inline-block"></span>`;
}

function _sizeBadge(size) {
  return `<span style="font-family:var(--font-mono);font-size:11px;color:var(--ink-3);background:var(--paper-3);padding:2px 7px;border-radius:4px">${escHtml(size)}</span>`;
}

function _recommendedChip() {
  return `<span style="font-size:10.5px;font-weight:500;padding:2px 7px;border-radius:999px;background:color-mix(in oklch,var(--accent) 14%,transparent);color:var(--accent);display:inline-flex;align-items:center;gap:4px"><svg width="9" height="9" viewBox="0 0 12 12" fill="currentColor"><path d="M6 0l1.5 4.2L12 4.5l-3.5 2.7L9.7 12 6 9.4 2.3 12l1.2-4.8L0 4.5l4.5-.3z"/></svg>Recommended</span>`;
}

function _activeRowHtml(m) {
  return `
    <div class="models-active-row" data-active-row="${escHtml(m.id)}">
      ${_statusDotHtml(m.state)}
      <div style="min-width:0">
        <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap">
          <span style="font-size:13.5px;font-weight:500">${escHtml(m.name)}</span>
          ${_sizeBadge(m.size || '')}
          ${m.role ? `<span style="font-size:10.5px;font-weight:500;padding:2px 7px;border-radius:999px;background:var(--ink);color:var(--paper)">${escHtml(m.role)}</span>` : ''}
        </div>
        <div class="mono" style="font-size:11px;color:var(--ink-3);margin-top:3px">${escHtml(m.langs || '')}</div>
      </div>
      ${_mBtn(`<svg width="13" height="13" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"><path d="M3 8h10M8 3l5 5-5 5"/></svg> Set default`, 'ghost', 'sm', `data-set-default="${escHtml(m.id)}" style="padding:0 8px"`)}
      ${_mBtn(`<svg width="13" height="13" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"><line x1="4" y1="4" x2="12" y2="12"/><line x1="12" y1="4" x2="4" y2="12"/></svg>`, 'ghost', 'sm', `data-deactivate="${escHtml(m.id)}" title="Disable" style="padding:0 7px"`)}
    </div>`;
}

function _stateMetaText(m) {
  if (m.state === 'update') return { color: 'var(--signal-wait)', text: `UPDATE AVAILABLE · ${escHtml(m.current || '')} → ${escHtml(m.available || '')}` };
  if (m.state === 'downloading') return { color: 'var(--signal-wait)', text: `DOWNLOADING · ${Math.round((m.progress || 0) * 100)}%` };
  if (m.state === 'installed')   return { color: 'var(--signal-go)',   text: 'INSTALLED' };
  if (m.state === 'active')      return { color: 'var(--signal-go)',   text: 'ACTIVE' };
  return { color: 'var(--ink-3)', text: 'NOT DOWNLOADED' };
}

function _actionCluster(m, catKey) {
  const isCap = catKey === 'capabilities';
  if (m.state === 'downloading') {
    if (isCap) return _mBtn('Working…', 'ghost', 'sm', 'disabled');
    return _mBtn(`Cancel · ${Math.round((m.progress || 0) * 100)}%`, 'warn', 'sm', `data-cancel-dl="${escHtml(m.id)}"`);
  }
  // `state === 'update'` means "installed AND a newer version is available".
  // Show the active Update button alongside Uninstall.
  if (m.state === 'update') {
    return `<div style="display:flex;gap:6px">${_mBtn('Update','accent','sm',`data-update-model="${escHtml(m.id)}"`)
      }${_mBtn(isCap ? 'Uninstall' : 'Remove','danger','sm',`data-remove-model="${escHtml(m.id)}"`)}</div>`;
  }
  if (m.state === 'active' || m.state === 'installed') {
    // Installed and on the latest version we know about: keep Update visible
    // for affordance but disabled, with a hover-explained tooltip. This makes
    // the "what does Update do?" question answer itself.
    if (isCap) {
      return `<div style="display:flex;gap:6px">${
        _mBtn('Update','ghost','sm','disabled title="Already up to date"')
      }${_mBtn('Uninstall','danger','sm',`data-remove-model="${escHtml(m.id)}"`)}</div>`;
    }
    return _mBtn('Remove','danger','sm',`data-remove-model="${escHtml(m.id)}"`);
  }
  return _mBtn(isCap ? 'Install' : 'Download','default','sm',`data-download-model="${escHtml(m.id)}"`);
}

function _libraryRowHtml(m, isLast, highlight, catKey) {
  const meta = _stateMetaText(m);
  const bg = highlight ? 'background:color-mix(in oklch,var(--accent) 8%,var(--paper))' : '';
  const borderBottom = isLast ? '' : 'border-bottom:1px solid var(--rule)';
  return `
    <div style="display:grid;grid-template-columns:auto 1fr auto;gap:14px;align-items:center;padding:12px 14px;${borderBottom};${bg}" data-library-row="${escHtml(m.id)}">
      ${_statusDotHtml(m.state)}
      <div style="min-width:0">
        <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap">
          <span style="font-size:13.5px;font-weight:500">${escHtml(m.name)}${
            m.state === 'update'
              ? ` <span class="settings-update-pip" title="Update available">*</span>`
              : ''
          }</span>
          ${_sizeBadge(m.size || '')}
          <span class="mono" style="font-size:10.5px;color:var(--ink-3)">· ${escHtml(m.langs || '')}</span>
          ${m.recommended ? _recommendedChip() : ''}
        </div>
        <div style="font-size:12px;color:var(--ink-3);margin-top:4px;line-height:1.4">${escHtml(m.desc || '')}</div>
        ${m.state === 'downloading' ? `<div class="models-progress-bar"><div class="models-progress-fill" style="width:${Math.round((m.progress || 0) * 100)}%"></div></div>` : ''}
        <div class="mono" style="font-size:10.5px;color:${meta.color};margin-top:6px;text-transform:uppercase;letter-spacing:0.06em">${meta.text}</div>
      </div>
      <div style="flex-shrink:0">${_actionCluster(m, catKey)}</div>
    </div>`;
}

function _parseSize(sizeStr) {
  if (!sizeStr) return 0;
  const m = String(sizeStr).match(/([\d.]+)\s*(MB|GB|KB)/i);
  if (!m) return 0;
  const n = parseFloat(m[1]);
  const u = m[2].toUpperCase();
  if (u === 'GB') return n * 1024;
  if (u === 'KB') return n / 1024;
  return n;
}

function _formatMB(mb) {
  if (mb >= 1024) return (mb / 1024).toFixed(2) + ' GB';
  return Math.round(mb) + ' MB';
}

function _diskSummaryHtml(catalogue, diskSummary) {
  // Build totals from catalogue data (client-side fallback if diskSummary is null)
  const catKeys = CAT_KEYS;
  const totals = {};
  if (diskSummary) {
    for (const [k, v] of Object.entries(diskSummary)) {
      totals[k] = { bytes: v.bytes, count: v.count, label: v.label };
    }
  } else if (catalogue) {
    catKeys.forEach((key, i) => {
      const cat = catalogue[key];
      if (!cat) return;
      let bytes = 0, count = 0;
      for (const m of cat.models || []) {
        if (['active','installed','update','downloading'].includes(m.state)) {
          bytes += _parseSize(m.size) * 1024 * 1024;
          count++;
        }
      }
      totals[key] = { bytes, count, label: cat.title };
    });
  }

  const grandBytes = Object.values(totals).reduce((s,t) => s + t.bytes, 0);
  const grandCount = Object.values(totals).reduce((s,t) => s + t.count, 0);
  const grandMB = grandBytes / (1024 * 1024);

  const barSegs = catKeys.map((key, i) => {
    const t = totals[key];
    if (!t) return '';
    const pct = grandBytes > 0 ? (t.bytes / grandBytes * 100) : 0;
    return `<div style="width:${pct}%;height:100%;background:${CAT_COLORS[i]}"></div>`;
  }).join('');

  const legend = catKeys.map((key, i) => {
    const t = totals[key];
    if (!t) return '';
    return `<div style="display:flex;align-items:center;gap:7px;font-size:12px">
      <span style="width:8px;height:8px;border-radius:2px;background:${CAT_COLORS[i]};flex-shrink:0"></span>
      <span style="color:var(--ink-2)">${escHtml(t.label)}</span>
      <span class="mono" style="color:var(--ink-3)">${_formatMB(t.bytes / (1024*1024))}</span>
    </div>`;
  }).join('');

  return `
    <div class="models-disk-summary">
      <div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:4px">
        <div>
          <div class="mono" style="font-size:10.5px;color:var(--ink-4);text-transform:uppercase;letter-spacing:0.08em">On-device storage</div>
          <div style="font-size:24px;font-weight:600;margin-top:4px;letter-spacing:-0.02em">
            ${escHtml(_formatMB(grandMB))}
            <span style="font-size:14px;color:var(--ink-3);font-weight:400">· ${grandCount} model${grandCount !== 1 ? 's' : ''} on disk</span>
          </div>
          <div class="mono" style="font-size:11px;color:var(--ink-3);margin-top:2px">~/.cache/whisperx · ~/.cache/huggingface</div>
        </div>
        <div style="display:flex;gap:6px">
          ${_mBtn(REVEAL_LABEL,'default','md','id="models-reveal-btn"')}
          ${_mBtn('Check for updates','default','md','id="models-check-updates-btn"')}
        </div>
      </div>
      <div class="models-stacked-bar">${barSegs}</div>
      <div style="display:flex;flex-wrap:wrap;gap:14px;margin-top:12px">${legend}</div>
    </div>`;
}

function _matchesSearch(m, q) {
  if (!q) return true;
  return [m.name, m.langs, m.desc].some(v => (v || '').toLowerCase().includes(q));
}

function _categoryCardHtml(catKey, cat, langFilter, drawerOpen, catIndex) {
  const models = cat.models || [];
  const active = models.filter(m => m.state === 'active');

  // Library models: for alignment, apply lang filter (active always shown)
  let library = models;
  if (cat.language_filterable && langFilter !== 'all') {
    library = models.filter(m => m.lang === langFilter || m.state === 'active');
  }

  const installed = models.filter(m => ['active','installed','update','downloading'].includes(m.state));
  const diskMB = installed.reduce((s,m) => s + _parseSize(m.size || ''), 0);

  const headerBorder = (drawerOpen || active.length > 0) ? 'border-bottom:1px solid var(--rule)' : '';

  const langSelectHtml = cat.language_filterable ? `
    <select id="lang-filter-${catKey}" style="height:28px;padding:0 10px;border:1px solid var(--rule);border-radius:6px;background:var(--paper-2);font-family:var(--font-mono);font-size:12px;color:var(--ink)">
      <option value="all"${langFilter === 'all' ? ' selected' : ''}>All languages</option>
      ${LANG_LIST.map(l => `<option value="${l.code}"${langFilter === l.code ? ' selected' : ''}>${l.name}</option>`).join('')}
    </select>` : '';

  const activeRowsHtml = active.length > 0 ? `
    <div class="models-active-band" style="border-bottom:1px solid var(--rule)">
      ${active.map(m => _activeRowHtml(m)).join('')}
    </div>` : '';

  const drawerToggleBorder = active.length > 0 ? 'border-top:1px solid var(--rule)' : '';
  const chevronClass = drawerOpen ? 'open' : '';

  const drawerHtml = drawerOpen ? `
    <div data-drawer-content="${catKey}">
      ${library.length === 0
        ? `<div style="padding:20px;text-align:center;color:var(--ink-3);font-size:13px">No models match this filter.</div>`
        : library.map((m, i) => {
            const isLast = i === library.length - 1;
            // Highlight recommended model if category has none installed
            const noneInstalled = !models.some(x => ['active','installed'].includes(x.state));
            const highlight = m.recommended && noneInstalled;
            return _libraryRowHtml(m, isLast, highlight, catKey);
          }).join('')
      }
    </div>` : '';

  return `
    <div class="models-category-card" data-cat-key="${catKey}">
      <header class="models-category-header" style="${headerBorder}">
        <div style="flex:1">
          <div style="display:flex;align-items:baseline;gap:10px">
            <h3 style="font-size:15px;font-weight:600">${escHtml(cat.title)}</h3>
            <span class="mono" style="font-size:11px;color:var(--ink-3)">${active.length} active · ${installed.length} on disk · ${_formatMB(diskMB)}</span>
          </div>
          <p style="font-size:12px;color:var(--ink-3);margin-top:3px">${escHtml(cat.sub || '')}</p>
        </div>
        ${langSelectHtml}
      </header>
      ${activeRowsHtml}
      <button class="models-drawer-toggle" style="${drawerToggleBorder}" data-drawer-toggle="${catKey}">
        <svg class="models-chevron ${chevronClass}" width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M6 4l4 4-4 4"/></svg>
        <span style="font-weight:500">Library</span>
        <span class="mono" style="font-size:11px;color:var(--ink-3)">${library.length} model${library.length === 1 ? '' : 's'}</span>
        <span style="margin-left:auto;font-size:12px;color:var(--ink-3)">${drawerOpen ? 'Hide' : 'Browse'}</span>
      </button>
      ${drawerHtml}
    </div>`;
}

async function _loadModelsData(container) {
  try {
    const [catalogue, diskSummary] = await Promise.all([
      api.getCatalogue(),
      api.getDiskSummary().catch(() => null),
    ]);
    _modelsState.catalogue = catalogue;
    _modelsState.diskSummary = diskSummary;
    _modelsState.loading = false;
    _renderModelsContent(container);
  } catch (e) {
    _modelsState.loading = false;
    _renderModelsContent(container);
  }
}

function _renderModelsContent(container) {
  const body = container.querySelector('#models-body');
  if (!body) return;

  const { catalogue, diskSummary, loading, search, langFilter, drawerOpen } = _modelsState;

  if (loading) {
    body.innerHTML = `<div style="padding:20px;color:var(--ink-3);font-size:13px">Loading models…</div>`;
    return;
  }

  if (!catalogue) {
    body.innerHTML = `<div style="padding:20px;color:var(--ink-3);font-size:13px">Could not load model catalogue (server not running?)</div>`;
    return;
  }

  const q = search.trim().toLowerCase();
  const catKeys = CAT_KEYS;

  // Filter catalogue by search
  const filteredCatalogue = {};
  for (const key of catKeys) {
    const cat = catalogue[key];
    if (!cat) continue;
    const filteredModels = q
      ? (cat.models || []).filter(m => _matchesSearch(m, q))
      : cat.models || [];
    if (q && filteredModels.length === 0) continue;
    filteredCatalogue[key] = { ...cat, models: filteredModels };
  }

  const catCardsHtml = Object.entries(filteredCatalogue).map(([key, cat], i) =>
    _categoryCardHtml(key, cat, langFilter, drawerOpen[key], catKeys.indexOf(key))
  ).join('');

  body.innerHTML = `
    ${_diskSummaryHtml(filteredCatalogue, diskSummary)}
    <div style="display:flex;gap:8px;margin-bottom:18px">
      <div style="position:relative;flex:1">
        <svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" style="position:absolute;left:12px;top:50%;transform:translateY(-50%);color:var(--ink-4);pointer-events:none">
          <circle cx="7" cy="7" r="5"/><path d="M11 11l3 3"/>
        </svg>
        <input id="models-search" type="search" value="${escHtml(search)}" placeholder="Search models, languages, descriptions…"
          style="width:100%;height:36px;padding:0 12px 0 34px;border:1px solid var(--rule);border-radius:8px;background:var(--paper);font-family:var(--font-mono);font-size:12.5px;color:var(--ink);outline:none">
      </div>
      ${_mBtn('+ Add from HuggingFace','default','md','id="models-add-hf-btn"')}
      ${_mBtn('Import .bin…','default','md','id="models-import-btn"')}
    </div>
    ${catCardsHtml}
    <div class="models-footer-note">
      <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="var(--ink-3)" stroke-width="1.5" stroke-linecap="round">
        <circle cx="8" cy="8" r="6"/>
        <line x1="8" y1="7" x2="8" y2="11.5"/>
        <circle cx="8" cy="5" r="0.6" fill="var(--ink-3)"/>
      </svg>
      <span>Models download once and are shared across all jobs. Removing a model frees disk but does not affect saved transcripts.</span>
    </div>`;

  _wireModelsEvents(container);
}

function _wireModelsEvents(container) {
  const body = container.querySelector('#models-body');
  if (!body) return;

  // Search
  const searchInput = body.querySelector('#models-search');
  if (searchInput) {
    searchInput.addEventListener('input', () => {
      _modelsState.search = searchInput.value;
      _renderModelsContent(container);
    });
  }

  // Reveal storage directory
  const revealBtn = body.querySelector('#models-reveal-btn');
  if (revealBtn) {
    revealBtn.addEventListener('click', async () => {
      revealBtn.disabled = true;
      try { await api.revealStorage(); } catch {}
      revealBtn.disabled = false;
    });
  }

  // Check for updates — clears server-side caches and re-renders the
  // catalogue. Any model/capability with a newer version available will then
  // come back as state="update" with current/available fields populated.
  const checkBtn = body.querySelector('#models-check-updates-btn');
  if (checkBtn) {
    checkBtn.addEventListener('click', async () => {
      checkBtn.disabled = true;
      checkBtn.textContent = 'Checking…';
      try {
        await api.checkModelUpdates();
        // Re-fetch and re-render so any new "update" states surface.
        await _reloadCatalogue(container);
        const updates = _countPendingUpdates(_modelsState.catalogue);
        checkBtn.textContent = updates ? `${updates} update${updates === 1 ? '' : 's'} available` : 'Up to date ✓';
        setTimeout(() => { checkBtn.disabled = false; checkBtn.textContent = 'Check for updates'; }, 2500);
      } catch {
        checkBtn.textContent = 'Error';
        setTimeout(() => { checkBtn.disabled = false; checkBtn.textContent = 'Check for updates'; }, 2000);
      }
    });
  }

  // Add from HuggingFace
  const addHfBtn = body.querySelector('#models-add-hf-btn');
  if (addHfBtn) {
    addHfBtn.addEventListener('click', async () => {
      const repoId = prompt('Enter a HuggingFace repo ID (e.g. openai/whisper-large-v2):');
      if (!repoId) return;
      try {
        const res = await api.addCustomModel();
        alert(res.message || 'Custom model import coming soon');
      } catch { alert('Custom model import coming soon'); }
    });
  }

  // Import .bin
  const importBtn = body.querySelector('#models-import-btn');
  if (importBtn) {
    importBtn.addEventListener('click', () => {
      alert('Import from .bin file: coming soon');
    });
  }

  // Drawer toggles
  body.querySelectorAll('[data-drawer-toggle]').forEach(btn => {
    btn.addEventListener('click', () => {
      const key = btn.dataset.drawerToggle;
      _modelsState.drawerOpen[key] = !_modelsState.drawerOpen[key];
      _renderModelsContent(container);
    });
  });

  // Language filter
  body.querySelectorAll('[id^="lang-filter-"]').forEach(sel => {
    sel.addEventListener('change', () => {
      _modelsState.langFilter = sel.value;
      _renderModelsContent(container);
    });
  });

  // Set default (activate)
  body.querySelectorAll('[data-set-default]').forEach(btn => {
    btn.addEventListener('click', async () => {
      const id = btn.dataset.setDefault;
      btn.disabled = true;
      try {
        await api.activateModel(id);
        await _reloadCatalogue(container);
      } catch { btn.disabled = false; }
    });
  });

  // Deactivate (×)
  body.querySelectorAll('[data-deactivate]').forEach(btn => {
    btn.addEventListener('click', async () => {
      const id = btn.dataset.deactivate;
      btn.disabled = true;
      try {
        await api.deactivateModel(id);
        await _reloadCatalogue(container);
      } catch { btn.disabled = false; }
    });
  });

  // Download
  body.querySelectorAll('[data-download-model]').forEach(btn => {
    btn.addEventListener('click', async () => {
      const id = btn.dataset.downloadModel;
      btn.disabled = true; btn.textContent = 'Starting…';
      try {
        await api.downloadModel(id);
        await _reloadCatalogue(container);
      } catch {
        btn.textContent = 'Error';
        setTimeout(() => { btn.disabled = false; btn.textContent = 'Download'; }, 2000);
      }
    });
  });

  // Remove
  body.querySelectorAll('[data-remove-model]').forEach(btn => {
    btn.addEventListener('click', async () => {
      const id = btn.dataset.removeModel;
      if (!confirm(`Remove ${id}? You can re-download it later.`)) return;
      btn.disabled = true;
      try {
        await api.deleteModel(id);
        await _reloadCatalogue(container);
      } catch { btn.disabled = false; }
    });
  });

  // Update
  body.querySelectorAll('[data-update-model]').forEach(btn => {
    btn.addEventListener('click', async () => {
      const id = btn.dataset.updateModel;
      btn.disabled = true; btn.textContent = 'Updating…';
      try {
        await api.updateModel(id);
        await _reloadCatalogue(container);
      } catch {
        btn.textContent = 'Error';
        setTimeout(() => { btn.disabled = false; btn.textContent = 'Update'; }, 2000);
      }
    });
  });

  // Cancel download
  body.querySelectorAll('[data-cancel-dl]').forEach(btn => {
    btn.addEventListener('click', async () => {
      // Not yet implemented server-side; just refresh
      await _reloadCatalogue(container);
    });
  });
}

async function _reloadCatalogue(container) {
  try {
    const [catalogue, diskSummary] = await Promise.all([
      api.getCatalogue(),
      api.getDiskSummary().catch(() => null),
    ]);
    _modelsState.catalogue = catalogue;
    _modelsState.diskSummary = diskSummary;
    _renderModelsContent(container);
  } catch {}
}

function renderModels(container, cfg) {
  container.innerHTML = `<div id="models-body" style="max-width:860px"></div>`;

  // Clear any existing poll timer
  if (_modelsState.pollTimer) {
    clearInterval(_modelsState.pollTimer);
    _modelsState.pollTimer = null;
  }
  _modelsState.loading = true;

  _loadModelsData(container);

  // Poll every 5 seconds for download progress updates
  _modelsState.pollTimer = setInterval(() => {
    _reloadCatalogue(container);
  }, 5000);

  // Cleanup on container removal (best-effort)
  const obs = new MutationObserver(() => {
    if (!document.contains(container)) {
      clearInterval(_modelsState.pollTimer);
      obs.disconnect();
    }
  });
  obs.observe(document.body, { childList: true, subtree: true });
}

function renderEmotionModels(body, cfg) {
  body.innerHTML += `
    <section class="settings-section" id="emotion-section">
      <header class="settings-section-head">
        <h2>Emotion Models</h2>
        <span class="wa-card-sub mono">Optional · per-segment emotion detection</span>
      </header>
      <p style="font-size:12.5px;color:var(--ink-3);line-height:1.5">
        Download a model below, then enable it. Emotion markers ([angry], [happy], etc.) are
        inserted into txt/srt/vtt transcripts. Models are stored in ~/.whisperapp/emotion_models/.
      </p>
      <div id="emotion-model-library" style="display:flex;flex-direction:column;gap:10px">
        <div class="wa-card" style="padding:12px;color:var(--ink-3);font-size:12.5px">Loading models…</div>
      </div>
      <div id="emotion-active-section" style="display:none;margin-top:20px">
        <header class="settings-section-head" style="margin-bottom:12px">
          <h2 style="font-size:14px">Active models</h2>
        </header>
        ${toggleRow('Enable emotion detection', cfg.emotion_enabled === true, 'emotion_enabled')}
        ${field('Confidence threshold',
          `<input class="wa-input" type="range" min="0.4" max="0.95" step="0.05"
             data-cfg="emotion_confidence_threshold"
             value="${escHtml(cfg.emotion_confidence_threshold ?? 0.65)}"
             oninput="this.nextElementSibling.textContent=(+this.value).toFixed(2)">
           <span class="mono" style="font-size:11px;color:var(--ink-3)">${escHtml((cfg.emotion_confidence_threshold ?? 0.65).toFixed(2))}</span>`,
          'Suppress labels below this confidence. Default: 0.65'
        )}
        <div id="emotion-ai-combine-row" style="display:none">
          ${toggleRow('Combine with AI provider', cfg.emotion_combine_with_ai === true, 'emotion_combine_with_ai')}
          <div class="wa-field-hint">Uses your configured AI provider to synthesise conflicting model outputs</div>
        </div>
        <div id="emotion-no-models-warning" style="display:none;background:var(--signal-warn-bg,#fffbe6);border:1px solid var(--signal-warn,#f5a623);border-radius:6px;padding:10px 12px;font-size:12.5px;color:var(--ink-2);margin-top:8px">
          Download and enable at least one model above
        </div>
      </div>
    </section>`;

  wireToggles(body, cfg);
  wireInputs(body, cfg);

  // Show AI combine row if AI is configured
  if (cfg.ai_provider && cfg.ai_provider !== 'none') {
    const aiRow = body.querySelector('#emotion-ai-combine-row');
    if (aiRow) aiRow.style.display = '';
  }

  // Load model library
  _renderEmotionLibrary(body, cfg);
}

async function _renderEmotionLibrary(body, cfg) {
  const library = body.querySelector('#emotion-model-library');
  const activeSection = body.querySelector('#emotion-active-section');
  if (!library) return;

  let models = [];
  try {
    models = await api.listModels();
  } catch {
    library.innerHTML = `<div class="wa-card" style="padding:12px;color:var(--ink-3);font-size:12.5px">Could not load models (server not running?)</div>`;
    return;
  }

  const hasDownloaded = models.some(m => m.downloaded);
  if (activeSection) activeSection.style.display = hasDownloaded ? '' : 'none';

  // Check if emotion_enabled but no model_ids
  const noModelsWarning = body.querySelector('#emotion-no-models-warning');
  if (noModelsWarning) {
    const enabledIds = Array.isArray(cfg.emotion_model_ids) ? cfg.emotion_model_ids : [];
    noModelsWarning.style.display = (cfg.emotion_enabled && enabledIds.length === 0) ? '' : 'none';
  }

  library.innerHTML = models.map(m => `
    <div class="wa-card" style="padding:14px 16px" data-model-id="${escHtml(m.id)}">
      <div style="display:flex;align-items:flex-start;gap:10px">
        <div style="flex:1">
          <div style="font-weight:600;font-size:13px">${escHtml(m.name)}
            ${m.recommended ? '<span style="background:var(--signal-ok,#22c55e);color:#fff;font-size:10px;padding:1px 6px;border-radius:4px;margin-left:6px">Recommended</span>' : ''}
          </div>
          <div style="font-size:12px;color:var(--ink-3);margin-top:2px">${escHtml(m.description)}</div>
          <div class="mono" style="font-size:11px;color:var(--ink-4);margin-top:4px">
            ${escHtml(m.size_mb)} MB &nbsp;·&nbsp; ${escHtml((m.labels || []).join(', '))}
          </div>
        </div>
        <div style="display:flex;flex-direction:column;align-items:flex-end;gap:6px;flex-shrink:0;min-width:120px">
          <div class="model-status mono" style="font-size:11px;color:var(--ink-3)">
            ${m.downloading ? `Downloading ${m.progress}%` : m.downloaded ? 'Downloaded' : 'Not downloaded'}
          </div>
          ${_modelActionHtml(m, cfg)}
        </div>
      </div>
      ${m.downloading ? `<div style="margin-top:8px;background:var(--paper-3);border-radius:3px;height:4px"><div style="background:var(--ink);height:4px;border-radius:3px;width:${m.progress}%;transition:width 0.3s"></div></div>` : ''}
    </div>
  `).join('');

  // Wire action buttons
  library.querySelectorAll('[data-download]').forEach(btn => {
    btn.addEventListener('click', async () => {
      const mid = btn.dataset.download;
      btn.disabled = true;
      btn.textContent = 'Starting…';
      try {
        await api.downloadModel(mid);
        _pollModelProgress(mid, body, cfg);
      } catch {
        btn.textContent = 'Error';
        setTimeout(() => { btn.disabled = false; btn.textContent = 'Download'; }, 2000);
      }
    });
  });

  library.querySelectorAll('[data-delete-model]').forEach(btn => {
    btn.addEventListener('click', async () => {
      const mid = btn.dataset.deleteModel;
      if (!confirm(`Delete ${mid}? You can re-download it later.`)) return;
      btn.disabled = true;
      try {
        await api.deleteModel(mid);
        // Remove from enabled list
        if (Array.isArray(cfg.emotion_model_ids)) {
          cfg.emotion_model_ids = cfg.emotion_model_ids.filter(id => id !== mid);
        }
        _renderEmotionLibrary(body, cfg);
      } catch {
        btn.disabled = false;
      }
    });
  });

  library.querySelectorAll('[data-toggle-model]').forEach(tog => {
    const mid = tog.dataset.toggleModel;
    tog.addEventListener('click', () => {
      const on = tog.classList.toggle('on');
      if (!Array.isArray(cfg.emotion_model_ids)) cfg.emotion_model_ids = [];
      if (on) {
        if (!cfg.emotion_model_ids.includes(mid)) cfg.emotion_model_ids.push(mid);
      } else {
        cfg.emotion_model_ids = cfg.emotion_model_ids.filter(id => id !== mid);
      }
      // Update warning
      const noModelsWarning = body.querySelector('#emotion-no-models-warning');
      if (noModelsWarning) {
        noModelsWarning.style.display = (cfg.emotion_enabled && cfg.emotion_model_ids.length === 0) ? '' : 'none';
      }
    });
  });
}

function _modelActionHtml(m, cfg) {
  const enabledIds = Array.isArray(cfg.emotion_model_ids) ? cfg.emotion_model_ids : [];
  if (m.downloading) {
    return `<button class="wa-btn" disabled style="font-size:11px">Downloading…</button>`;
  }
  if (m.downloaded) {
    const isEnabled = enabledIds.includes(m.id);
    return `
      <div style="display:flex;gap:6px;align-items:center">
        <label style="display:flex;align-items:center;gap:5px;font-size:12px">
          <span class="wa-toggle${isEnabled ? ' on' : ''}" data-toggle-model="${escHtml(m.id)}"></span>
          Enable
        </label>
        <button class="wa-btn wa-btn-danger" data-delete-model="${escHtml(m.id)}" style="font-size:11px;padding:4px 8px">Delete</button>
      </div>`;
  }
  return `<button class="wa-btn wa-btn-primary" data-download="${escHtml(m.id)}" style="font-size:11px">Download</button>`;
}

function _pollModelProgress(modelId, body, cfg) {
  const timer = setInterval(async () => {
    try {
      const status = await api.modelProgress(modelId);
      if (!status.downloading) {
        clearInterval(timer);
        _renderEmotionLibrary(body, cfg);
      } else {
        // Update progress bar in place
        const card = body.querySelector(`[data-model-id="${modelId}"]`);
        if (card) {
          const statusEl = card.querySelector('.model-status');
          if (statusEl) statusEl.textContent = `Downloading ${status.progress}%`;
          const bar = card.querySelector('[style*="height:4px;border-radius:3px;width"]');
          if (bar) bar.style.width = `${status.progress}%`;
        }
      }
    } catch {
      clearInterval(timer);
    }
  }, 2000);
}

function renderSaveRow(body, cfg) {
  // insertAdjacentHTML appends without re-parsing the existing DOM, so the
  // event listeners that the renderXxx() above just attached survive.
  // Using `body.innerHTML += ...` here would silently wipe them — that bug
  // made the Acoustic master toggle (and others) inert.
  body.insertAdjacentHTML('beforeend', `
    <div style="display:flex;gap:8px;padding-top:8px;border-top:1px solid var(--rule)">
      <button class="wa-btn wa-btn-primary" id="save-btn">Save settings</button>
      <button class="wa-btn" id="test-ai-btn">Test AI connection</button>
      <button class="wa-btn" id="reset-btn" style="margin-left:auto;color:var(--signal-rec)">Reset to defaults</button>
    </div>`);

  body.querySelector('#save-btn').addEventListener('click', async () => {
    const btn = body.querySelector('#save-btn');
    btn.disabled = true; btn.textContent = 'Saving…';
    try {
      await api.updateConfig(cfg);
      btn.textContent = 'Saved ✓';
      setTimeout(() => { btn.disabled = false; btn.textContent = 'Save settings'; }, 1500);
    } catch {
      btn.textContent = 'Error saving';
      setTimeout(() => { btn.disabled = false; btn.textContent = 'Save settings'; }, 2000);
    }
  });

  body.querySelector('#test-ai-btn').addEventListener('click', async () => {
    try {
      const res = await api.health();
      alert(`AI: ${res.ai_provider} — ${res.ai_available ? 'available' : 'not configured'}`);
    } catch {
      alert('Could not reach REST API at :7861');
    }
  });

  body.querySelector('#reset-btn').addEventListener('click', () => {
    if (confirm('Reset all settings to defaults?')) {
      api.updateConfig({
        default_model: 'large-v2',
        streaming_model: 'base',
        diarize_by_default: true,
        default_output_path: '',
        ai_provider: 'none',
        ai_api_key: '',
        ai_model: '',
        ai_base_url: '',
      }).then(() => location.reload()).catch(() => alert('Error resetting settings'));
    }
  });
}

function wireInputs(body, cfg) {
  body.querySelectorAll('[data-cfg]').forEach(el => {
    el.addEventListener('input', () => { cfg[el.dataset.cfg] = el.value; });
    el.addEventListener('change', () => { cfg[el.dataset.cfg] = el.value; });
  });
}

function wireToggles(body, cfg) {
  body.querySelectorAll('[data-toggle]').forEach(tog => {
    tog.addEventListener('click', () => {
      const on = tog.classList.toggle('on');
      const key = tog.dataset.toggle;
      if (key in cfg) cfg[key] = on;
    });
  });
}

function escHtml(s) {
  return String(s ?? '').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}
