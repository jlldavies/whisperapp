import { api } from './api.js';

const SPEAKER_COLORS = [
  'oklch(60% 0.13 45)',
  'oklch(58% 0.11 220)',
  'oklch(60% 0.10 145)',
  'oklch(58% 0.09 290)',
];

export function initSpeakers(container) {
  container.innerHTML = `
    <div class="wa-topbar">
      <div>
        <h1>Speakers</h1>
        <div class="wa-topbar-sub mono" id="speakers-sub">Select a completed job with speaker review pending</div>
      </div>
      <div class="wa-topbar-meta mono" id="speakers-meta"></div>
    </div>
    <div class="wa-content" id="speakers-content">
      <div id="speakers-job-picker" style="display:flex;flex-direction:column;gap:12px;max-width:520px">
        <p style="color:var(--ink-3);font-size:13px">Jobs awaiting speaker review:</p>
        <div id="speakers-job-list"><div style="color:var(--ink-4);font-size:12.5px">Loading…</div></div>
      </div>
      <div id="speakers-review" style="display:none;grid-template-columns:320px 1fr;gap:24px"></div>
    </div>
  `;

  loadPendingJobs(container);

  // Deep-link from the Transcribe Processing card: when the user clicks the
  // `speaker_review` badge there, transcribe.js dispatches this event and
  // navigates to #speakers. We listen at module-init time so subsequent
  // visits to the screen also pick up new requests.
  window.addEventListener('wa-open-speaker-review', e => {
    const jobId = e.detail?.jobId;
    if (jobId) openJob(container, jobId);
  });
}

async function loadPendingJobs(container) {
  try {
    const jobs = await api.listJobs('speaker_review');
    const list = container.querySelector('#speakers-job-list');
    if (!jobs.length) {
      list.innerHTML = `<div style="color:var(--ink-4);font-size:12.5px">No jobs awaiting speaker review.</div>`;
      return;
    }
    list.innerHTML = jobs.map(j => `
      <div class="queue-item" style="cursor:pointer" data-job-id="${j.id}">
        <div class="queue-item-header">
          <span class="queue-dot done"></span>
          <span class="queue-item-title">${escHtml(j.file_name || j.id)}</span>
          <span class="queue-badge mono">${j.id.slice(0,8)}</span>
        </div>
        <div class="queue-meta mono">${j.model}${j.diarize?' · diarize':''}</div>
      </div>
    `).join('');
    list.querySelectorAll('[data-job-id]').forEach(el => {
      el.addEventListener('click', () => openJob(container, el.dataset.jobId));
    });
  } catch { /* server may not be up yet */ }
}

async function openJob(container, jobId) {
  let job, data;
  try {
    job = await api.getJob(jobId);
    data = await api.getSpeakers(jobId);
  } catch (err) {
    const list = container.querySelector('#speakers-job-list');
    if (!list) return;
    if (err.status === 409 || err.status === 404) {
      // Job left the queue or finished elsewhere — drop this entry and
      // refresh quietly so the user just sees the up-to-date list.
      list.innerHTML = `<div style="color:var(--ink-3);font-size:12.5px">That job has already been completed. Refreshing list…</div>`;
      setTimeout(() => loadPendingJobs(container), 800);
      return;
    }
    list.innerHTML = `<div style="color:var(--signal-rec);font-size:12.5px">Error loading job: ${escHtml(String(err.message))}</div>`;
    return;
  }

  // Server returns `{ speakers: { SPEAKER_00: ["snippet text", ...], ... } }`
  // — an object keyed by speaker id where each value is an array of strings.
  // The UI works in terms of `[{id, name, lines, snippets:[{text,start,end}]}]`,
  // so adapt here. (Tolerate the array shape too, in case the server contract
  // changes later.)
  let speakers = [];
  const raw = data.speakers;
  if (Array.isArray(raw)) {
    speakers = raw;
  } else if (raw && typeof raw === 'object') {
    speakers = Object.entries(raw)
      .sort(([a], [b]) => a.localeCompare(b))
      .map(([id, snippets]) => ({
        id,
        name: '',
        lines: Array.isArray(snippets) ? snippets.length : 0,
        snippets: (snippets || []).map(s =>
          typeof s === 'string'
            ? { text: s, start: '', end: '' }
            : { text: s.text ?? '', start: s.start ?? '', end: s.end ?? '' }
        ),
      }));
  }

  container.querySelector('#speakers-sub').textContent =
    `${job.file_name || jobId} · ${speakers.length} speakers detected`;
  container.querySelector('#speakers-meta').textContent =
    `${job.id.slice(0,8)} · ${job.model}`;

  container.querySelector('#speakers-job-picker').style.display = 'none';
  const review = container.querySelector('#speakers-review');
  review.style.display = 'grid';

  const names = Object.fromEntries(speakers.map(s => [s.id, s.name || '']));
  let activeSpeaker = speakers[0]?.id;

  function render() {
    review.innerHTML = `
      <!-- Voices column -->
      <div class="wa-card" style="padding:0;overflow:hidden;display:flex;flex-direction:column">
        <div style="padding:16px 18px;border-bottom:1px solid var(--rule);display:flex;justify-content:space-between;align-items:baseline">
          <div class="wa-card-title">Voices</div>
          <div class="wa-card-sub">${speakers.length}</div>
        </div>
        <div style="flex:1;overflow-y:auto">
          ${speakers.map((s, i) => {
            const color = SPEAKER_COLORS[i % SPEAKER_COLORS.length];
            const name  = names[s.id] || '';
            const initial = name ? name[0].toUpperCase() : '?';
            return `
              <div class="speaker-row${s.id === activeSpeaker ? ' active' : ''}" data-speaker="${s.id}">
                <div class="speaker-avatar" style="background:${color}">${initial}</div>
                <div style="flex:1;min-width:0">
                  <div style="font-size:13.5px;font-weight:500">
                    ${name || `<span style="color:var(--ink-4);font-style:italic;font-weight:400">Unnamed</span>`}
                  </div>
                  <div class="mono" style="font-size:10.5px;color:var(--ink-3);margin-top:1px">
                    ${s.id} · ${s.lines ?? '?'} lines
                  </div>
                </div>
                ${s.id === activeSpeaker ? `<svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="var(--ink)" stroke-width="1.6" stroke-linecap="round"><path d="M5 8l2 2 4-4"/></svg>` : ''}
              </div>
            `;
          }).join('')}
        </div>
        <div style="padding:14px;border-top:1px solid var(--rule);display:flex;flex-direction:column;gap:8px">
          <div id="confirm-status" style="font-size:11.5px;color:var(--ink-3);min-height:14px"></div>
          <div style="display:flex;gap:8px">
            <button class="wa-btn" id="skip-btn" style="flex:1">Skip</button>
            <button class="wa-btn wa-btn-primary" id="confirm-btn" style="flex:2">Confirm names</button>
          </div>
        </div>
      </div>

      <!-- Review pane -->
      <div class="wa-card" style="padding:0;overflow:hidden;display:flex;flex-direction:column">
        ${renderReviewPane(speakers.find(s => s.id === activeSpeaker), names, activeSpeaker)}
      </div>
    `;

    // Wire up speaker row clicks
    review.querySelectorAll('[data-speaker]').forEach(el => {
      el.addEventListener('click', () => {
        activeSpeaker = el.dataset.speaker;
        render();
      });
    });

    // Name editing in review pane — with ghost-text autocomplete from the
    // local cache of previously-confirmed names. Tab completes the suggestion.
    const nameInput  = review.querySelector('#speaker-name-input');
    const ghostTyped = review.querySelector('.ghost-typed');
    const ghostSugg  = review.querySelector('.ghost-suggest');

    function suggestionFor(prefix) {
      if (!prefix) return '';
      const lc = prefix.toLowerCase();
      const cache = loadNameCache();
      const hit = cache.find(n => n.toLowerCase().startsWith(lc) && n.length > prefix.length);
      return hit ? hit.slice(prefix.length) : '';
    }
    function refreshGhost() {
      if (!nameInput || !ghostTyped || !ghostSugg) return;
      const v = nameInput.value;
      ghostTyped.textContent = v;
      ghostSugg.textContent  = suggestionFor(v);
    }
    if (nameInput) {
      // Refresh ghost on initial render so a pre-filled name shows no ghost.
      refreshGhost();
      nameInput.addEventListener('input', e => {
        names[activeSpeaker] = e.target.value;
        refreshGhost();
      });
      nameInput.addEventListener('keydown', e => {
        if (e.key === 'Tab' && ghostSugg.textContent) {
          e.preventDefault();
          nameInput.value      = nameInput.value + ghostSugg.textContent;
          names[activeSpeaker] = nameInput.value;
          refreshGhost();
          updateHint();
        }
      });
    }

    const confirmBtn = review.querySelector('#confirm-btn');
    const skipBtn    = review.querySelector('#skip-btn');
    const statusEl   = review.querySelector('#confirm-status');

    // Hint to the user that empty names are fine — the server keeps the
    // SPEAKER_XX placeholder for any speaker left unnamed.
    function updateHint() {
      const total  = speakers.length;
      const filled = speakers.filter(s => (names[s.id] || '').trim()).length;
      if (filled === 0) {
        statusEl.textContent = `0 / ${total} named — confirming will keep SPEAKER_XX labels.`;
      } else if (filled < total) {
        statusEl.textContent = `${filled} / ${total} named — unnamed speakers keep SPEAKER_XX.`;
      } else {
        statusEl.textContent = `${filled} / ${total} named.`;
      }
    }
    updateHint();
    if (nameInput) {
      nameInput.addEventListener('input', updateHint);
    }

    confirmBtn.addEventListener('click', async () => {
      confirmBtn.disabled = true;
      skipBtn.disabled    = true;
      const original      = confirmBtn.textContent;
      confirmBtn.textContent = 'Saving…';
      statusEl.style.color = 'var(--ink-3)';
      statusEl.textContent = 'Saving names and finishing the job…';
      try {
        await api.confirmSpeakers(jobId, names);
        // Remember any names the user actually filled in so they ghost-complete
        // next time. Empty / placeholder values don't pollute the cache.
        Object.values(names).forEach(n => addToNameCache(n));
        review.style.display = 'none';
        container.querySelector('#speakers-job-picker').style.display = 'flex';
        loadPendingJobs(container);
      } catch (err) {
        // 409 = job left speaker_review between opening the pane and clicking
        // Confirm (e.g. confirmed in another window, or completed externally).
        // Names the user typed are still worth keeping for autocomplete.
        if (err.status === 409) {
          Object.values(names).forEach(n => addToNameCache(n));
          statusEl.style.color = 'var(--ink-3)';
          statusEl.textContent =
            'This job has already been completed elsewhere. Returning to the list…';
          setTimeout(() => {
            review.style.display = 'none';
            container.querySelector('#speakers-job-picker').style.display = 'flex';
            loadPendingJobs(container);
          }, 1400);
          return;
        }
        statusEl.style.color = 'var(--signal-rec)';
        statusEl.textContent = `Confirm failed: ${err.message}. Try again or Skip.`;
        confirmBtn.disabled    = false;
        skipBtn.disabled       = false;
        confirmBtn.textContent = original;
      }
    });

    skipBtn.addEventListener('click', () => {
      review.style.display = 'none';
      container.querySelector('#speakers-job-picker').style.display = 'flex';
      loadPendingJobs(container);
    });

    // Wire snippet play buttons. We reuse a single HTMLAudioElement across
    // snippets — the browser caches the file once, then a click just seeks
    // to `start` and pauses at `end`. Any play in flight is stopped first.
    review.querySelectorAll('.snippet-play').forEach(btn => {
      btn.addEventListener('click', () => playSnippet(btn));
    });
  }

  let _audio = null;
  let _stopTimer = null;
  let _activePlayBtn = null;

  function playSnippet(btn) {
    const start = Number(btn.dataset.start) || 0;
    const end   = Number(btn.dataset.end)   || 0;
    if (!_audio) {
      _audio = new Audio(api.jobAudioUrl(jobId));
      _audio.preload = 'auto';
      _audio.addEventListener('error', () => {
        if (_activePlayBtn) _activePlayBtn.classList.remove('playing');
        console.warn('Audio load failed for', jobId);
      });
    }
    // Cancel any in-flight playback from a previous click.
    if (_stopTimer) { clearTimeout(_stopTimer); _stopTimer = null; }
    if (_activePlayBtn && _activePlayBtn !== btn) {
      _activePlayBtn.classList.remove('playing');
    }
    if (!_audio.paused) _audio.pause();

    _activePlayBtn = btn;
    btn.classList.add('playing');
    _audio.currentTime = start;
    _audio.play().catch(err => {
      btn.classList.remove('playing');
      console.warn('Audio play rejected:', err);
    });

    if (end > start) {
      const ms = Math.max(200, (end - start) * 1000 + 50);
      _stopTimer = setTimeout(() => {
        _audio.pause();
        btn.classList.remove('playing');
        _stopTimer = null;
      }, ms);
    }
  }

  render();
}

function formatTs(seconds) {
  if (!isFinite(seconds) || seconds < 0) return '00:00.0';
  const m = Math.floor(seconds / 60);
  const s = seconds - m * 60;
  return String(m).padStart(2, '0') + ':' + s.toFixed(1).padStart(4, '0');
}

// ─── Speaker-name cache ─────────────────────────────────────────────────────
// Stored in localStorage under "wa-speaker-names" as a JSON array, MRU first.
// Used by the review pane to ghost-complete names the user has typed before.

const NAME_CACHE_KEY = 'wa-speaker-names';
const NAME_CACHE_MAX = 200;

function loadNameCache() {
  try {
    const raw = localStorage.getItem(NAME_CACHE_KEY);
    const arr = raw ? JSON.parse(raw) : [];
    return Array.isArray(arr) ? arr : [];
  } catch {
    return [];
  }
}

function addToNameCache(name) {
  const clean = String(name || '').trim();
  if (!clean) return;
  // Don't cache the SPEAKER_XX placeholders — those are noise.
  if (/^speaker[_\s-]?\d+$/i.test(clean)) return;
  const cache = loadNameCache();
  // Promote to front, dedupe case-insensitively.
  const filtered = cache.filter(n => n.toLowerCase() !== clean.toLowerCase());
  filtered.unshift(clean);
  if (filtered.length > NAME_CACHE_MAX) filtered.length = NAME_CACHE_MAX;
  try { localStorage.setItem(NAME_CACHE_KEY, JSON.stringify(filtered)); } catch {}
}

function renderReviewPane(speaker, names, speakerId) {
  if (!speaker) return `<div style="padding:20px;color:var(--ink-4)">Select a speaker</div>`;
  const name    = names[speakerId] || '';
  const snippets = speaker.snippets || [];
  return `
    <div style="padding:16px 22px;border-bottom:1px solid var(--rule);display:flex;align-items:center;justify-content:space-between">
      <div>
        <div class="wa-card-title">
          <span class="speaker-name-wrap">
            <span class="speaker-name-ghost" id="speaker-name-ghost" aria-hidden="true">
              <span class="ghost-typed"></span><span class="ghost-suggest"></span>
            </span>
            <input id="speaker-name-input" value="${escHtml(name)}" placeholder="Enter name…"
              autocomplete="off" spellcheck="false">
          </span>
          <span style="font-weight:400;color:var(--ink-3)"> · ${speakerId}</span>
        </div>
        <div class="mono" style="font-size:10.5px;color:var(--ink-3);margin-top:2px">
          ${snippets.length} representative snippet${snippets.length !== 1 ? 's' : ''}
        </div>
      </div>
    </div>
    <div style="padding:20px 22px;display:flex;flex-direction:column;gap:14px;flex:1;overflow-y:auto">
      ${snippets.map(s => {
        const start = Number(s.start) || 0;
        const end   = Number(s.end)   || 0;
        return `
        <div class="speaker-snippet">
          <button class="snippet-play" title="Play snippet"
                  data-start="${start}" data-end="${end}">
            <svg width="10" height="10" viewBox="0 0 16 16" fill="currentColor"><path d="M5 3.5v9l8-4.5z"/></svg>
          </button>
          <div style="flex:1">
            <div class="mono" style="font-size:10.5px;color:var(--ink-3);margin-bottom:4px">${formatTs(start)} → ${formatTs(end)}</div>
            <div style="font-size:13.5px;line-height:1.5">${escHtml(s.text ?? '')}</div>
          </div>
        </div>`;
      }).join('')}
    </div>
    <div style="margin-top:auto;padding:14px 22px;border-top:1px solid var(--rule)">
      <div class="mono" style="font-size:11px;color:var(--ink-3)">
        Tip: paste <span style="color:var(--ink-2)">SPEAKER_00=Alice</span> lines to rename in bulk
      </div>
    </div>
  `;
}

function escHtml(s) {
  return String(s ?? '').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}
