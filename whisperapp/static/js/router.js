export class Router {
  constructor(screens, onNavigate) {
    this._screens = screens; // { id: HTMLElement }
    this._onNavigate = onNavigate;
    this._current = null;
    window.addEventListener('hashchange', () => this._apply());
  }

  navigate(id) {
    window.location.hash = id;
  }

  start(defaultId) {
    const id = window.location.hash.slice(1) || defaultId;
    this._show(id);
  }

  _apply() {
    const id = window.location.hash.slice(1);
    if (id) this._show(id);
  }

  _show(id) {
    if (!this._screens[id]) return;
    Object.entries(this._screens).forEach(([sid, el]) => {
      el.classList.toggle('active', sid === id);
    });
    this._current = id;
    this._onNavigate(id);
  }

  get current() { return this._current; }
}
