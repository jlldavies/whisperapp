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
    if (list) list.innerHTML = `<div style="color:var(--signal-rec);font-size:12.5px">Error loading job: ${escHtml(String(err.message))}</div>`;
    return;
  }
  const speakers = data.speakers || [];

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
        <div style="padding:14px;border-top:1px solid var(--rule);display:flex;gap:8px">
          <button class="wa-btn" id="skip-btn" style="flex:1">Skip</button>
          <button class="wa-btn wa-btn-primary" id="confirm-btn" style="flex:2">Confirm names</button>
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

    // Name editing in review pane
    const nameInput = review.querySelector('#speaker-name-input');
    if (nameInput) {
      nameInput.addEventListener('input', e => {
        names[activeSpeaker] = e.target.value;
      });
    }

    review.querySelector('#confirm-btn').addEventListener('click', async () => {
      await api.confirmSpeakers(jobId, names);
      review.style.display = 'none';
      container.querySelector('#speakers-job-picker').style.display = 'flex';
      loadPendingJobs(container);
    });

    review.querySelector('#skip-btn').addEventListener('click', () => {
      review.style.display = 'none';
      container.querySelector('#speakers-job-picker').style.display = 'flex';
      loadPendingJobs(container);
    });
  }

  render();
}

function renderReviewPane(speaker, names, speakerId) {
  if (!speaker) return `<div style="padding:20px;color:var(--ink-4)">Select a speaker</div>`;
  const name    = names[speakerId] || '';
  const snippets = speaker.snippets || [];
  return `
    <div style="padding:16px 22px;border-bottom:1px solid var(--rule);display:flex;align-items:center;justify-content:space-between">
      <div>
        <div class="wa-card-title">
          <input id="speaker-name-input" value="${escHtml(name)}" placeholder="Enter name…"
            style="border:none;background:transparent;font-size:13px;font-weight:600;color:var(--ink);outline:none;width:160px">
          <span style="font-weight:400;color:var(--ink-3)"> · ${speakerId}</span>
        </div>
        <div class="mono" style="font-size:10.5px;color:var(--ink-3);margin-top:2px">
          ${snippets.length} representative snippet${snippets.length !== 1 ? 's' : ''}
        </div>
      </div>
    </div>
    <div style="padding:20px 22px;display:flex;flex-direction:column;gap:14px;flex:1;overflow-y:auto">
      ${snippets.map(s => `
        <div class="speaker-snippet">
          <button class="snippet-play" title="Play">
            <svg width="10" height="10" viewBox="0 0 16 16" fill="currentColor"><path d="M5 3.5v9l8-4.5z"/></svg>
          </button>
          <div style="flex:1">
            <div class="mono" style="font-size:10.5px;color:var(--ink-3);margin-bottom:4px">${escHtml(s.start ?? '')} → ${escHtml(s.end ?? '')}</div>
            <div style="font-size:13.5px;line-height:1.5">${escHtml(s.text ?? '')}</div>
          </div>
        </div>
      `).join('')}
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
