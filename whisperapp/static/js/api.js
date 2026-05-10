const API = 'http://127.0.0.1:7861';

async function req(method, path, body) {
  const opts = { method, headers: {} };
  if (body !== undefined) {
    opts.headers['Content-Type'] = 'application/json';
    opts.body = JSON.stringify(body);
  }
  const r = await fetch(API + path, opts);
  if (!r.ok) {
    const err = new Error(`${method} ${path} → ${r.status}`);
    err.status = r.status;
    try { err.detail = (await r.json()).detail; } catch {}
    throw err;
  }
  return r.json();
}

export const api = {
  // Health
  health: ()           => req('GET', '/health'),
  info:   ()           => req('GET', '/info'),

  // Jobs
  listJobs: (opts) => {
    if (!opts) return req('GET', '/jobs');
    if (typeof opts === 'string') return req('GET', `/jobs?status=${opts}`);
    const qs = new URLSearchParams(opts).toString();
    return req('GET', '/jobs' + (qs ? '?' + qs : ''));
  },
  getJob:   (id)       => req('GET', `/jobs/${id}`),
  cancelJob:    (id)   => req('POST', `/jobs/${id}/cancel`),
  deleteJob:    (id)   => req('DELETE', `/jobs/${id}`),
  moveJobUp:    (id)   => req('POST', `/jobs/${id}/move-up`),
  moveJobDown:  (id)   => req('POST', `/jobs/${id}/move-down`),
  clearJobs:    ()     => req('POST', '/jobs/clear'),
  getTranscript: (id, fmt) => req('GET', `/jobs/${id}/transcript?format=${fmt}`),
  getSegments:   (id, offset=0, limit=500) => req('GET', `/jobs/${id}/segments?offset=${offset}&limit=${limit}`),

  // Worker
  workerStatus: ()     => req('GET',  '/worker/status'),
  pauseWorker:  ()     => req('POST', '/worker/pause'),
  resumeWorker: ()     => req('POST', '/worker/resume'),

  // Transcribe
  transcribe: (body)   => req('POST', '/transcribe', body),
  upload: async (file) => {
    const fd = new FormData();
    fd.append('file', file);
    const r = await fetch(API + '/upload', { method: 'POST', body: fd });
    if (!r.ok) throw new Error('Upload failed');
    return r.json();
  },

  // Speakers
  getSpeakers:     (id)        => req('GET',  `/jobs/${id}/speakers`),
  confirmSpeakers: (id, names) => req('POST', `/jobs/${id}/speakers`, { names }),
  jobAudioUrl:     (id)        => `${API}/jobs/${id}/audio`,

  // Config
  getConfig:    ()     => req('GET',  '/config'),
  updateConfig: (body) => req('POST', '/config', body),

  // Emotion models (existing)
  listModels:    ()    => req('GET',  '/models'),
  getModel:      (id)  => req('GET',  `/models/${id}`),
  downloadModel: (id)  => req('POST', `/models/${id}/download`),
  deleteModel:   (id)  => req('DELETE', `/models/${id}`),
  modelProgress: (id)  => req('GET',  `/models/${id}/progress`),

  // Model catalogue (unified — all categories)
  getCatalogue:      ()    => req('GET',  '/models/catalogue'),
  getDiskSummary:    ()    => req('GET',  '/models/disk-summary'),
  activateModel:     (id)  => req('POST', `/models/${id}/activate`),
  deactivateModel:   (id)  => req('POST', `/models/${id}/deactivate`),
  updateModel:       (id)  => req('POST', `/models/${id}/update`),
  revealStorage:     ()    => req('POST', '/storage/reveal'),
  checkModelUpdates: ()    => req('POST', '/models/check-updates'),
  addCustomModel:    ()    => req('POST', '/models/add-custom'),

  // Audio
  getAudioDevices: ()  => req('GET', '/audio/devices'),

  // OS-level startup registration (launch on login)
  startupStatus:   ()  => req('GET',  '/startup/status'),
  startupEnable:   ()  => req('POST', '/startup/enable'),
  startupDisable:  ()  => req('POST', '/startup/disable'),

  // Streaming
  streamStart: (body)  => req('POST', '/stream/start', body),
  streamChunk: (body)  => req('POST', '/stream/chunk', body),
  streamStop:  (body)  => req('POST', '/stream/stop', body),
  streamPolish:(body)  => req('POST', '/stream/polish', body),
};
