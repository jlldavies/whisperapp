import { renderSidebar } from './shell.js';
import { Router } from './router.js';
import { initTranscribe } from './transcribe.js';
import { initLive } from './live.js';
import { initSpeakers } from './speakers.js';
import { initSettings } from './settings.js';

const SCREENS = {
  transcribe: document.getElementById('screen-transcribe'),
  live:       document.getElementById('screen-live'),
  speakers:   document.getElementById('screen-speakers'),
  settings:   document.getElementById('screen-settings'),
};

let _router;

function onNavigate(id) {
  renderSidebar(document.getElementById('sidebar'), id, id => _router.navigate(id));
}

// Theme
const themeBtn = document.getElementById('theme-toggle');
themeBtn.addEventListener('click', () => {
  const html = document.documentElement;
  const next = html.dataset.theme === 'dark' ? 'light' : 'dark';
  html.dataset.theme = next;
  localStorage.setItem('wa-theme', next);
});
const saved = localStorage.getItem('wa-theme');
if (saved) document.documentElement.dataset.theme = saved;

// Init screens (idempotent — called once each)
initTranscribe(SCREENS.transcribe);
initLive(SCREENS.live);
initSpeakers(SCREENS.speakers);
initSettings(SCREENS.settings);

// Sidebar initial render
renderSidebar(document.getElementById('sidebar'), 'transcribe', id => _router.navigate(id));

// Router
_router = new Router(SCREENS, onNavigate);
_router.start('transcribe');
