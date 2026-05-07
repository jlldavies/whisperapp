const API = 'http://127.0.0.1:7861';

async function req(method, path, body) {
  const opts = { method, headers: {} };
  if (body !== undefined) {
    opts.headers['Content-Type'] = 'application/json';
    opts.body = JSON.stringify(body);
  }
  const r = await fetch(API + path, opts);
  if (!r.ok) throw new Error(`${method} ${path} → ${r.status}`);
  return r.json();
}

export const api = {
  // Health
  health: ()           => req('GET', '/health'),
  info:   ()           => req('GET', '/info'),

  // Jobs
  listJobs: (status)   => req('GET', '/jobs' + (status ? `?status=${status}` : '')),
  getJob:   (id)       => req('GET', `/jobs/${id}`),
  cancelJob:(id)       => req('POST', `/jobs/${id}/cancel`),
  getTranscript: (id, fmt) => req('GET', `/jobs/${id}/transcript?format=${fmt}`),

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
  getSpeakers:     (id)       => req('GET',  `/jobs/${id}/speakers`),
  confirmSpeakers: (id, names) => req('POST', `/jobs/${id}/speakers`, { names }),

  // Config
  getConfig:    ()     => req('GET',  '/config'),
  updateConfig: (body) => req('POST', '/config', body),

  // Audio
  getAudioDevices: ()  => req('GET', '/audio/devices'),

  // Streaming
  streamStart: (body)  => req('POST', '/stream/start', body),
  streamChunk: (body)  => req('POST', '/stream/chunk', body),
  streamStop:  (body)  => req('POST', '/stream/stop', body),
  streamPolish:(body)  => req('POST', '/stream/polish', body),
};
