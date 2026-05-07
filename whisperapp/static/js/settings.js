import { api } from './api.js';

const SECTIONS = ['Transcription', 'AI features', 'Startup', 'API & CLI', 'About'];
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
          <div class="settings-subnav-item${i===0?' active':''}" data-section="${i}">${s}</div>
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
  }

  function render() {
    const body = container.querySelector('#settings-body');
    body.innerHTML = '';

    if (activeSection === 0) renderTranscription(body, cfg);
    if (activeSection === 1) renderAI(body, cfg);
    if (activeSection === 2) renderStartup(body, cfg);
    if (activeSection === 3) renderAPICLI(body);
    if (activeSection === 4) renderAbout(body);

    if (activeSection < 3) renderSaveRow(body, cfg);
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
      ${field('Default output path',
        `<input class="wa-input" data-cfg="default_output_path" value="${escHtml(cfg.default_output_path||'')}">`
      )}
      ${toggleRow('Diarization on by default', cfg.diarize_by_default, 'diarize_by_default')}
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

function renderStartup(body, cfg) {
  body.innerHTML += `
    <section class="settings-section">
      <header class="settings-section-head"><h2>Startup</h2></header>
      ${toggleRow('Launch on login', false, 'launch_on_login')}
      ${toggleRow('Auto-update WhisperX on startup', true, 'auto_update')}
    </section>`;
  // Note: launch_on_login and auto_update are not yet persisted in config.
  // Toggles are interactive visually but not wired to cfg until implemented.
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

function renderSaveRow(body, cfg) {
  body.innerHTML += `
    <div style="display:flex;gap:8px;padding-top:8px;border-top:1px solid var(--rule)">
      <button class="wa-btn wa-btn-primary" id="save-btn">Save settings</button>
      <button class="wa-btn" id="test-ai-btn">Test AI connection</button>
      <button class="wa-btn" id="reset-btn" style="margin-left:auto;color:var(--signal-rec)">Reset to defaults</button>
    </div>`;

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
