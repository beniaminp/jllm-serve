'use strict';
// jllm console: shared helpers, navigation, status, theme. Plain JavaScript, no build step.

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[c]));
const store = {
  get(k, d) { try { const v = localStorage.getItem(k); return v == null ? d : JSON.parse(v); } catch { return d; } },
  set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch { /* private mode or full */ } },
};
const fmtBytes = n => n == null || n < 0 ? '?' : n >= 1e9 ? (n / 1e9).toFixed(2) + ' GB' : n >= 1e6 ? (n / 1e6).toFixed(1) + ' MB' : n >= 1e3 ? (n / 1e3).toFixed(0) + ' kB' : n + ' B';
const fmtTime = ms => new Date(ms).toLocaleTimeString([], {hour: '2-digit', minute: '2-digit'});
const fmtDur = ms => { const s = Math.floor(ms / 1000); return s < 60 ? s + 's' : s < 3600 ? Math.floor(s / 60) + 'm' : Math.floor(s / 3600) + 'h ' + Math.floor(s % 3600 / 60) + 'm'; };
const fmtTok = n => n >= 1000 ? (n / 1000).toFixed(n >= 10000 ? 0 : 1) + 'k' : String(n);

function toast(msg, kind = 'info', ms = 4500) {
  const t = document.createElement('div');
  t.className = 'toast ' + kind;
  t.textContent = msg;
  $('#toasts').appendChild(t);
  setTimeout(() => t.remove(), ms);
}

function authHeaders() {
  const key = store.get('jllm.apiKey', '');
  return key ? {Authorization: 'Bearer ' + key} : {};
}

async function api(path, opts = {}) {
  const headers = {...authHeaders(), ...(opts.body ? {'Content-Type': 'application/json'} : {}), ...(opts.headers || {})};
  const res = await fetch(path, {...opts, headers, body: opts.body && typeof opts.body !== 'string' ? JSON.stringify(opts.body) : opts.body});
  const text = await res.text();
  let data = null;
  try { data = text ? JSON.parse(text) : null; } catch { data = text; }
  if (res.status === 401) {
    showTab('api');
    $('#api-key-note').innerHTML = '<span class="error">This server requires an API key: enter it below.</span>';
  }
  if (!res.ok) {
    const msg = data && (data.error?.message || data.error || data.message) || res.statusText;
    throw new Error(typeof msg === 'string' ? msg : JSON.stringify(msg));
  }
  return data;
}

function download(name, text, type = 'text/markdown') {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([text], {type}));
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

async function copyText(text, el) {
  try {
    await navigator.clipboard.writeText(text);
  } catch {
    const ta = document.createElement('textarea');
    ta.value = text; document.body.appendChild(ta); ta.select(); document.execCommand('copy'); ta.remove();
  }
  if (el) { const old = el.textContent; el.textContent = 'copied'; setTimeout(() => { el.textContent = old; }, 1200); }
}

// ------------------------------------------------------------------ navigation

const TABS = ['chat', 'models', 'tools', 'settings', 'logs', 'api', 'images' /* images */, 'video' /* video */];
const onShow = {};  // tab -> callback when it becomes visible
const onTick = {};  // tab -> callback polled while visible
let activeTab = 'chat';

function showTab(name) {
  if (!TABS.includes(name)) name = 'chat';
  activeTab = name;
  $$('.rail nav button').forEach(b => b.classList.toggle('active', b.dataset.tab === name));
  $$('.tab').forEach(t => t.classList.toggle('active', t.id === 'tab-' + name));
  if (location.hash.slice(1) !== name) history.replaceState(null, '', '#' + name);
  onShow[name]?.();
}
$$('.rail nav button').forEach(b => b.addEventListener('click', () => showTab(b.dataset.tab)));
window.addEventListener('hashchange', () => showTab(location.hash.slice(1)));

// ------------------------------------------------------------------ theme

const THEMES = ['auto', 'light', 'dark'];
function applyTheme(t) {
  if (t === 'auto') delete document.documentElement.dataset.theme; else document.documentElement.dataset.theme = t;
  $('#theme-label').textContent = t;
  store.set('jllm.theme', t);
}
$('#theme-toggle').addEventListener('click', () => applyTheme(THEMES[(THEMES.indexOf(store.get('jllm.theme', 'auto')) + 1) % 3]));

// ------------------------------------------------------------------ status

let serverStatus = null;
async function refreshStatus() {
  try {
    const s = serverStatus = await api('/api/status');
    $('#status').innerHTML = `<b>${s.loadedModels}</b> model${s.loadedModels === 1 ? '' : 's'} loaded<br>weights ${fmtBytes(s.weightBytes)}<br>`
      + `heap ${fmtBytes(s.heapUsed)} / ${fmtBytes(s.heapMax)}<br>${s.processors} cpus · up ${fmtDur(s.uptimeMs)}`
      + (s.thermalState >= 0 ? `<br>thermal: ${['nominal', 'fair', '<b class="error">serious (GPU slowed)</b>',
        '<b class="error">critical (GPU slowed)</b>'][s.thermalState] || s.thermalState}` : '')
      + (s.runningJobs ? `<br><b>${s.runningJobs}</b> job${s.runningJobs === 1 ? '' : 's'} running` : '');
    $('#settings-file').textContent = s.settingsFile;
    $('#api-key-note').innerHTML = s.apiKeyRequired ? 'This server requires <code>Authorization: Bearer &lt;key&gt;</code> (<code>jllm.web.api-key</code>).'
      : 'No API key is required (start with <code>--jllm.web.api-key=…</code> to require one).';
  } catch (e) { $('#status').textContent = 'server unreachable'; }
}

// loaded models, shared by the chat and the models page
let loadedModels = [];
const modelListeners = [];
async function refreshLoaded() {
  try { loadedModels = await api('/api/models/loaded'); } catch { return loadedModels; }
  modelListeners.forEach(f => f(loadedModels));
  return loadedModels;
}

function startApp() {
  applyTheme(store.get('jllm.theme', 'auto'));
  refreshStatus(); refreshLoaded();
  setInterval(refreshStatus, 5000);
  setInterval(() => onTick[activeTab]?.(), 1500);
  setInterval(() => { if (activeTab !== 'models') refreshLoaded(); }, 6000);
  showTab(location.hash.slice(1) || 'chat');
}
