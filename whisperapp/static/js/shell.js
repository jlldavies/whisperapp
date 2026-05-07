import { LOGO_SVG, NAV_ICONS } from './marks.js';

const NAV_ITEMS = [
  { id: 'transcribe', label: 'Transcribe', kbd: '⌘1' },
  { id: 'live',       label: 'Live',       kbd: '⌘2' },
  { id: 'speakers',   label: 'Speakers',   kbd: '⌘3' },
  { id: 'settings',   label: 'Settings',   kbd: '⌘,' },
];

export function renderSidebar(container, activeId, onNavigate) {
  const recentJobs = JSON.parse(localStorage.getItem('wa-recent') || '[]');

  container.innerHTML = `
    <div class="wa-brand">
      <div class="wa-brand-mark">${LOGO_SVG}</div>
      <div class="wa-brand-name">Whisper</div>
      <div class="wa-brand-sub">v1.1</div>
    </div>

    <nav class="wa-nav">
      <div class="wa-nav-section">Workspace</div>
      ${NAV_ITEMS.map(item => `
        <div class="wa-nav-item${item.id === activeId ? ' active' : ''}" data-nav="${item.id}">
          <span class="wa-nav-icon">${NAV_ICONS[item.id]}</span>
          <span>${item.label}</span>
          <span class="wa-nav-kbd">${item.kbd}</span>
        </div>
      `).join('')}
    </nav>

    ${recentJobs.length ? `
    <div class="wa-nav">
      <div class="wa-nav-section">Recent</div>
      ${recentJobs.slice(0, 3).map(j => `
        <div class="wa-nav-item" style="font-size:12.5px" data-nav-job="${j.id}">
          <span class="wa-recent-dot"></span>
          <span style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${j.name}</span>
        </div>
      `).join('')}
    </div>` : ''}

    <div class="wa-status-bar" id="status-bar">
      <div class="wa-status-row">
        <span><span class="wa-dot"></span>Daemon</span>
        <span>:7860</span>
      </div>
      <div class="wa-status-row">
        <span><span class="wa-dot"></span>REST API</span>
        <span>:7861</span>
      </div>
      <div class="wa-status-row" id="gpu-row">
        <span style="color:var(--ink-4)">GPU</span>
        <span>—</span>
      </div>
    </div>
  `;

  container.querySelectorAll('[data-nav]').forEach(el => {
    el.addEventListener('click', () => onNavigate(el.dataset.nav));
  });

  // Fetch /info to populate GPU row
  fetch('http://127.0.0.1:7861/info')
    .then(r => r.json())
    .then(info => {
      const row = document.getElementById('gpu-row');
      if (row) row.innerHTML = `
        <span style="color:var(--ink-4)">GPU</span>
        <span>${info.acceleration}</span>
      `;
    })
    .catch(() => {});
}

export function renderTopBar(container, { title, sub = '', right = '' }) {
  container.innerHTML = `
    <div class="wa-topbar">
      <div>
        <h1>${title}</h1>
        ${sub ? `<div class="wa-topbar-sub mono">${sub}</div>` : ''}
      </div>
      <div class="wa-topbar-meta">${right}</div>
    </div>
  `;
}

export function renderWaveform(container, { height = 56, bars = 80, playhead = null, color = 'var(--ink-3)' }) {
  const heights = Array.from({ length: bars }, (_, i) => {
    const x = i / bars;
    const v = Math.sin(x * 22) * 0.4 + Math.sin(x * 6 + 1.2) * 0.5 + Math.sin(x * 53) * 0.15;
    return Math.max(0.15, Math.min(1, 0.55 + v * 0.45));
  });
  container.style.height = height + 'px';
  container.className = 'wa-waveform';
  container.innerHTML = heights.map((h, i) => {
    const past = playhead != null && i / bars < playhead;
    return `<div class="wa-waveform-bar" style="
      height:${h * 100}%;
      background:${past ? 'var(--accent)' : color};
      opacity:${past ? 1 : 0.55}
    "></div>`;
  }).join('');
}
