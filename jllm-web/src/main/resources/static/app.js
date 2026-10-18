'use strict';
// jllm console: plain JavaScript, no build step. Talks to /api (management) and /v1 (OpenAI-compatible API).

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[c]));
const store = {
  get(k, d) { try { const v = localStorage.getItem(k); return v == null ? d : JSON.parse(v); } catch { return d; } },
  set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch { /* private mode */ } },
};
const fmtBytes = n => n == null || n < 0 ? '?' : n >= 1e9 ? (n / 1e9).toFixed(2) + ' GB' : n >= 1e6 ? (n / 1e6).toFixed(1) + ' MB' : n >= 1e3 ? (n / 1e3).toFixed(0) + ' kB' : n + ' B';
const fmtTime = ms => new Date(ms).toLocaleTimeString();
const fmtDur = ms => { const s = Math.floor(ms / 1000); return s < 60 ? s + 's' : s < 3600 ? Math.floor(s / 60) + 'm' : Math.floor(s / 3600) + 'h ' + Math.floor(s % 3600 / 60) + 'm'; };

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
  if (res.status === 401) { showTab('api'); $('#api-key-note').innerHTML = '<span class="error">The server requires an API key: enter it below.</span>'; }
  if (!res.ok) {
    const msg = data && (data.error?.message || data.error || data.message) || res.statusText;
    throw new Error(typeof msg === 'string' ? msg : JSON.stringify(msg));
  }
  return data;
}

// ------------------------------------------------------------------ tabs and status

let activeTab = 'chat';
function showTab(name) {
  activeTab = name;
  $$('nav button').forEach(b => b.classList.toggle('active', b.dataset.tab === name));
  $$('.tab').forEach(t => t.classList.toggle('active', t.id === 'tab-' + name));
  if (name === 'models') refreshModels(true);
  if (name === 'settings') loadSettings();
  if (name === 'logs') pollLogs();
  location.hash = name;
}
$$('nav button').forEach(b => b.addEventListener('click', () => showTab(b.dataset.tab)));

async function refreshStatus() {
  try {
    const s = await api('/api/status');
    $('#status').textContent = `${s.loadedModels} model${s.loadedModels === 1 ? '' : 's'} loaded · weights ${fmtBytes(s.weightBytes)} · heap ${fmtBytes(s.heapUsed)} / ${fmtBytes(s.heapMax)} · ${s.processors} cpus · up ${fmtDur(s.uptimeMs)}`
      + (s.runningJobs ? ` · ${s.runningJobs} job${s.runningJobs === 1 ? '' : 's'} running` : '');
    $('#settings-file').textContent = s.settingsFile;
    $('#api-key-note').innerHTML = s.apiKeyRequired ? 'This server requires <code>Authorization: Bearer &lt;key&gt;</code> (<code>jllm.web.api-key</code>).' : 'No API key is required (set <code>jllm.web.api-key</code> to require one).';
  } catch (e) { $('#status').textContent = 'server unreachable: ' + e.message; }
}

// ------------------------------------------------------------------ chat

let convs = store.get('jllm.convs', []);
let currentId = store.get('jllm.current', null);
let loadedModels = [];
let abort = null;

const saveConvs = () => { store.set('jllm.convs', convs); store.set('jllm.current', currentId); };
const current = () => convs.find(c => c.id === currentId);

function newConv() {
  const c = {id: Date.now().toString(36), title: 'New chat', messages: []};
  convs.unshift(c); currentId = c.id; saveConvs(); renderConvs(); renderMessages();
  $('#input').focus();
}

function renderConvs() {
  const ul = $('#conv-list');
  ul.innerHTML = '';
  for (const c of convs) {
    const li = document.createElement('li');
    li.className = c.id === currentId ? 'active' : '';
    li.innerHTML = `<span title="${esc(c.title)}">${esc(c.title)}</span><button title="delete">✕</button>`;
    li.addEventListener('click', () => { currentId = c.id; saveConvs(); renderConvs(); renderMessages(); });
    li.querySelector('button').addEventListener('click', ev => {
      ev.stopPropagation();
      convs = convs.filter(x => x.id !== c.id);
      if (currentId === c.id) currentId = convs[0]?.id ?? null;
      saveConvs(); renderConvs(); renderMessages();
    });
    ul.appendChild(li);
  }
}

// a small, safe markdown subset: fenced code, inline code, bold, italics, links, headings, lists, paragraphs
function markdown(src) {
  const parts = src.split(/(```[^\n]*\n[\s\S]*?(?:```|$))/g);
  return parts.map(part => {
    const fence = part.match(/^```([^\n]*)\n([\s\S]*?)(?:```)?$/);
    if (fence) return `<pre><code>${esc(fence[2])}</code></pre>`;
    return esc(part).split(/\n{2,}/).map(block => {
      let b = block
        .replace(/`([^`\n]+)`/g, '<code>$1</code>')
        .replace(/\*\*([^*\n]+)\*\*/g, '<strong>$1</strong>')
        .replace(/(^|[^*])\*([^*\n]+)\*/g, '$1<em>$2</em>')
        .replace(/\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
      if (/^#{1,6} /.test(b)) return `<p><strong>${b.replace(/^#{1,6} /, '')}</strong></p>`;
      if (/^\s*([-*]|\d+\.) /.test(b)) {
        const items = b.split('\n').map(l => l.replace(/^\s*([-*]|\d+\.) /, '')).map(l => `<li>${l}</li>`).join('');
        return /^\s*\d+\. /.test(b) ? `<ol>${items}</ol>` : `<ul>${items}</ul>`;
      }
      return b.trim() ? `<p>${b.replace(/\n/g, '<br>')}</p>` : '';
    }).join('');
  }).join('');
}

// Qwen3 / DeepSeek-R1 put their reasoning between <think> and </think>
function renderAssistant(text, streaming) {
  let html = '';
  let rest = text;
  const open = rest.indexOf('<think>');
  if (open >= 0 && rest.slice(0, open).trim() === '') {
    const close = rest.indexOf('</think>');
    const thought = close >= 0 ? rest.slice(open + 7, close) : rest.slice(open + 7);
    rest = close >= 0 ? rest.slice(close + 8) : '';
    if (thought.trim()) html += `<details class="think" ${close < 0 ? 'open' : ''}><summary>${close < 0 ? 'thinking…' : 'thoughts'}</summary>${esc(thought.trim())}</details>`;
  }
  html += markdown(rest.replace(/^\s+/, ''));
  return html + (streaming ? '<span class="cursor"></span>' : '');
}

function statsLine(u) {
  if (!u) return '';
  return `${u.prompt_tokens} prompt tok (${u.cached_tokens} cached) · ${u.completion_tokens} generated · ${u.tokens_per_second} tok/s`
    + (u.drafted_tokens ? ` · ${u.accepted_draft_tokens}/${u.drafted_tokens} drafted accepted` : '');
}

function renderMessages() {
  const box = $('#messages');
  box.innerHTML = '';
  const c = current();
  if (!c || !c.messages.length) {
    box.innerHTML = `<div class="msg empty">${loadedModels.length ? 'Start a conversation.' : 'No model is loaded yet: open the <a href="#models" onclick="showTab(\'models\');return false">Models</a> tab to download and load one.'}</div>`;
    return;
  }
  for (const m of c.messages) box.appendChild(messageEl(m));
  box.scrollTop = box.scrollHeight;
}

function messageEl(m) {
  const div = document.createElement('div');
  div.className = 'msg ' + m.role;
  div.innerHTML = `<div class="who">${m.role === 'user' ? 'You' : esc(m.model || 'assistant')}</div><div class="body"></div><div class="stats"></div>`;
  const body = div.querySelector('.body');
  if (m.role === 'user') body.textContent = m.content; else body.innerHTML = renderAssistant(m.content, false);
  if (m.error) body.insertAdjacentHTML('beforeend', `<div class="error">${esc(m.error)}</div>`);
  div.querySelector('.stats').textContent = statsLine(m.usage) + (m.finish && m.finish !== 'stop' ? ` · finish: ${m.finish}` : '');
  return div;
}

function chatParams() {
  const p = {};
  for (const k of ['temperature', 'top_p', 'top_k', 'min_p', 'max_tokens', 'repetition_penalty', 'seed']) {
    const v = $('#p-' + k).value;
    if (v !== '') p[k] = Number(v);
  }
  store.set('jllm.params', p);
  return p;
}

async function send(text) {
  if (!currentId) newConv();
  const c = current();
  const model = $('#chat-model').value;
  if (!model) { alert('Load a model first (Models tab).'); return; }
  c.messages.push({role: 'user', content: text});
  if (c.title === 'New chat') c.title = text.slice(0, 40);
  const reply = {role: 'assistant', content: '', model};
  c.messages.push(reply);
  saveConvs(); renderConvs(); renderMessages();

  const box = $('#messages');
  const el = box.lastElementChild;
  const body = el.querySelector('.body');
  body.innerHTML = renderAssistant('', true);
  const system = $('#p-system').value.trim();
  store.set('jllm.system', system);
  const messages = [...(system ? [{role: 'system', content: system}] : []),
    ...c.messages.slice(0, -1).filter(m => !m.error || m.content).map(m => ({role: m.role, content: m.content}))];
  const payload = {model, messages, stream: true, stream_options: {include_usage: true}, enable_thinking: $('#chat-think').checked, ...chatParams()};

  abort = new AbortController();
  $('#send').classList.add('hidden'); $('#stop').classList.remove('hidden');
  try {
    const res = await fetch('/v1/chat/completions', {method: 'POST', headers: {'Content-Type': 'application/json', ...authHeaders()},
      body: JSON.stringify(payload), signal: abort.signal});
    if (!res.ok) {
      const t = await res.text();
      let msg = t; try { msg = JSON.parse(t).error.message; } catch { /* plain text */ }
      throw new Error(msg);
    }
    const reader = res.body.getReader();
    const dec = new TextDecoder();
    let buf = '';
    let frame = 0;
    for (;;) {
      const {value, done} = await reader.read();
      if (done) break;
      buf += dec.decode(value, {stream: true});
      let i;
      while ((i = buf.indexOf('\n\n')) >= 0) {
        const ev = buf.slice(0, i); buf = buf.slice(i + 2);
        for (const line of ev.split('\n')) {
          if (!line.startsWith('data: ')) continue;
          const data = line.slice(6);
          if (data === '[DONE]') continue;
          const j = JSON.parse(data);
          if (j.error) throw new Error(j.error.message);
          const ch = j.choices?.[0];
          if (ch?.delta?.content) reply.content += ch.delta.content;
          if (ch?.finish_reason) reply.finish = ch.finish_reason;
          if (j.usage) reply.usage = j.usage;
        }
      }
      if (!frame) frame = requestAnimationFrame(() => {
        frame = 0;
        const stick = box.scrollHeight - box.scrollTop - box.clientHeight < 60;
        body.innerHTML = renderAssistant(reply.content, true);
        if (stick) box.scrollTop = box.scrollHeight;
      });
    }
  } catch (e) {
    if (e.name !== 'AbortError') reply.error = e.message; else reply.finish = 'stopped';
  } finally {
    abort = null;
    $('#send').classList.remove('hidden'); $('#stop').classList.add('hidden');
    saveConvs(); renderMessages();
  }
}

$('#composer').addEventListener('submit', ev => {
  ev.preventDefault();
  const t = $('#input').value.trim();
  if (!t || abort) return;
  $('#input').value = '';
  send(t);
});
$('#input').addEventListener('keydown', ev => {
  if (ev.key === 'Enter' && !ev.shiftKey && !ev.isComposing) { ev.preventDefault(); $('#composer').requestSubmit(); }
});
$('#stop').addEventListener('click', () => abort?.abort());
$('#new-chat').addEventListener('click', newConv);
$('#toggle-params').addEventListener('click', () => $('#params').classList.toggle('hidden'));
$('#chat-think').addEventListener('change', () => store.set('jllm.think', $('#chat-think').checked));
$('#chat-model').addEventListener('change', () => store.set('jllm.model', $('#chat-model').value));

async function refreshChatModels() {
  try { loadedModels = await api('/api/models/loaded'); } catch { return; }
  const sel = $('#chat-model');
  const want = sel.value || store.get('jllm.model', '');
  sel.innerHTML = loadedModels.length ? loadedModels.map(m => `<option value="${esc(m.id)}">${esc(m.id)}</option>`).join('') : '<option value="">(no model loaded)</option>';
  if (loadedModels.some(m => m.id === want)) sel.value = want;
  if (!current()?.messages.length) renderMessages();
}

// ------------------------------------------------------------------ models

function loadForm(opts) {
  // opts: {source, revision, gguf, ggufFiles, ggufFile, defaultId, local, download}
  const frag = $('#load-form-tpl').content.cloneNode(true);
  const form = frag.querySelector('form');
  form.querySelectorAll('.gguf-only').forEach(e => e.classList.toggle('hidden', !opts.gguf || !opts.ggufFiles?.length));
  form.querySelectorAll('.st-only').forEach(e => e.classList.toggle('hidden', !!opts.gguf));
  form.querySelectorAll('.dl-only').forEach(e => e.classList.toggle('hidden', !opts.download));
  form.id.placeholder = opts.defaultId || '(default)';
  if (opts.revision && opts.revision !== 'main') form.revision.value = opts.revision;
  if (opts.local) form.revision.closest('label').classList.add('hidden');
  if (opts.ggufFiles?.length) {
    form.ggufFile.innerHTML = opts.ggufFiles.map(f => `<option value="${esc(f.name)}">${esc(f.name)} (${fmtBytes(f.size)})</option>`).join('');
    const pick = opts.ggufFile || opts.ggufFiles.find(f => opts.quant && f.name.toUpperCase().includes(opts.quant.toUpperCase()))?.name
      || opts.ggufFiles.find(f => /Q4_K_M/i.test(f.name))?.name;
    if (pick) form.ggufFile.value = pick;
  }
  let action = 'load';
  form.querySelectorAll('button[type=submit]').forEach(b => b.addEventListener('click', () => { action = b.value; }));
  form.addEventListener('submit', async ev => {
    ev.preventDefault();
    const settings = {};
    for (const el of form.elements) {
      if (!el.name || !el.name.includes('.') || el.closest('.hidden')) continue;
      if (el.type === 'checkbox') { if (el.checked) settings[el.name] = 'true'; } else if (el.value !== '') settings[el.name] = el.value;
    }
    const req = {source: opts.source, revision: form.revision.value || null,
      ggufFile: opts.gguf && opts.ggufFiles?.length ? form.ggufFile.value : (opts.ggufFile || null), ggufQuant: null, settings};
    try {
      if (action === 'download') await api('/api/models/download', {method: 'POST', body: req});
      else await api('/api/models/load', {method: 'POST', body: {...req, id: form.id.value || null}});
      form.closest('.inline-form')?.remove();
      $('#inspect-result').innerHTML = `<p class="okmsg">${action === 'download' ? 'Download' : 'Load'} started: see Jobs below.</p>`;
      refreshModels(true);
    } catch (e) { alert(e.message); }
  });
  return frag;
}

const looksLocal = s => /\.gguf$/i.test(s) && /^(\/|~|\.\/|[A-Za-z]:[\\/])/.test(s);

$('#inspect-form').addEventListener('submit', async ev => {
  ev.preventDefault();
  const source = $('#hf-link').value.trim();
  const out = $('#inspect-result');
  if (looksLocal(source)) {
    out.innerHTML = `<p>Local file <code>${esc(source)}</code></p>`;
    out.appendChild(loadForm({source, gguf: true, local: true}));
    return;
  }
  out.innerHTML = '<p class="muted">Looking up the repository…</p>';
  try {
    const info = await api('/api/hub/inspect?source=' + encodeURIComponent(source));
    const total = info.files.reduce((a, f) => a + Math.max(0, f.size), 0);
    out.innerHTML = `<p><strong>${esc(info.repo)}</strong> @ ${esc(info.revision)} · ${info.gguf ? 'GGUF repository' : 'safetensors checkpoint'} · ${info.files.length} files, ${fmtBytes(total)}
      · <a href="https://huggingface.co/${esc(info.repo)}" target="_blank" rel="noopener">model card</a></p>
      ${info.gguf ? '' : '<p class="hint">Quantized once on first load into the model directory (Q4_0 by default), memory-mapped afterwards.</p>'}`;
    out.appendChild(loadForm({source, revision: info.revision, gguf: info.gguf, ggufFiles: info.ggufFiles, ggufFile: info.file,
      quant: info.quant, defaultId: info.defaultId, download: true}));
  } catch (e) { out.innerHTML = `<p class="error">${esc(e.message)}</p>`; }
});

function renderLoaded(list) {
  const box = $('#loaded-list');
  if (!list.length) { box.innerHTML = '<p class="empty">Nothing loaded.</p>'; return; }
  box.innerHTML = `<table class="table"><tr><th>id</th><th>model</th><th>backend</th><th>context</th><th>weights</th><th>requests</th><th>last speed</th><th></th></tr>
    ${list.map(m => `<tr>
      <td><strong>${esc(m.id)}</strong><div class="hint">${esc(m.source)}</div></td>
      <td><div class="hint" style="max-width:380px">${esc(m.description)}</div><div class="hint">template ${esc(m.template)} · loaded ${fmtTime(m.loadedAt)} in ${m.loadSeconds.toFixed(1)}s</div>
        <details><summary class="hint">settings</summary><pre class="hint">${esc(Object.entries(m.settings).map(([k, v]) => k + ' = ' + v).join('\n'))}</pre></details></td>
      <td>${esc(m.backend)}</td><td>${m.contextLength}</td><td>${esc(m.quant)}<div class="hint">${fmtBytes(m.weightBytes)}</div></td>
      <td>${m.requests}${m.active ? ` <span class="badge load">${m.active} active</span>` : ''}<div class="hint">${m.generatedTokens} tokens</div></td>
      <td>${m.lastTokensPerSecond ? m.lastTokensPerSecond + ' tok/s' : '–'}</td>
      <td><button class="small" data-chat="${esc(m.id)}">chat</button> <button class="small danger" data-unload="${esc(m.id)}">unload</button></td></tr>`).join('')}</table>`;
  $$('[data-unload]', box).forEach(b => b.addEventListener('click', async () => {
    if (!confirm('Unload ' + b.dataset.unload + '? Running requests on it are cancelled.')) return;
    b.disabled = true;
    try { await api('/api/models/loaded?id=' + encodeURIComponent(b.dataset.unload), {method: 'DELETE'}); } catch (e) { alert(e.message); }
    refreshModels(true); refreshChatModels();
  }));
  $$('[data-chat]', box).forEach(b => b.addEventListener('click', () => {
    store.set('jllm.model', b.dataset.chat); $('#chat-model').value = b.dataset.chat; showTab('chat');
  }));
}

function renderJobs(jobs) {
  const box = $('#job-list');
  if (!jobs.length) { box.innerHTML = '<p class="empty">No jobs.</p>'; return; }
  box.innerHTML = `<table class="table"><tr><th>job</th><th>model</th><th>state</th><th>progress</th><th></th></tr>
    ${jobs.map(j => {
      const pct = j.totalBytes > 0 ? Math.min(100, 100 * j.downloadedBytes / j.totalBytes) : null;
      const running = j.state === 'RUNNING' || j.state === 'QUEUED';
      return `<tr><td>${esc(j.kind.toLowerCase())}<div class="hint">${fmtTime(j.created)}</div></td>
        <td>${esc(j.model)}<div class="hint">${esc(j.source)}</div></td>
        <td class="state-${j.state}">${esc(j.phase)}${j.error ? `<div class="error">${esc(j.error)}</div>` : ''}</td>
        <td>${pct != null ? `${fmtBytes(j.downloadedBytes)} / ${fmtBytes(j.totalBytes)} (${pct.toFixed(0)}%)<div class="progress"><div style="width:${pct}%"></div></div>` : ''}
          ${running && j.phase === 'loading' ? '<div class="hint">reading / quantizing the weights (see Logs)</div>' : ''}</td>
        <td>${running ? `<button class="small" data-cancel="${esc(j.id)}">cancel</button>` : ''}</td></tr>`;
    }).join('')}</table>`;
  $$('[data-cancel]', box).forEach(b => b.addEventListener('click', async () => {
    try { await api(`/api/jobs/${b.dataset.cancel}/cancel`, {method: 'POST'}); } catch (e) { alert(e.message); }
    refreshModels(true);
  }));
}

function renderLocal(list) {
  const box = $('#local-list');
  if (!list.length) { box.innerHTML = '<p class="empty">No models downloaded yet.</p>'; return; }
  box.innerHTML = `<table class="table"><tr><th>model</th><th>format</th><th>size</th><th></th></tr>
    ${list.map((m, i) => `<tr data-row="${i}"><td><strong>${esc(m.name)}</strong><div class="hint">${esc(m.path)}</div>
        ${m.caches.length ? `<div class="hint">quantized: ${m.caches.map(esc).join(', ')}</div>` : ''}</td>
      <td>${esc(m.format)}</td><td>${fmtBytes(m.bytes)}</td>
      <td>${m.loadedAs.length ? `<span class="badge live">loaded</span> ` : ''}<button class="small" data-load="${i}">load…</button>
        <button class="small danger" data-delete="${i}">delete</button></td></tr>`).join('')}</table>`;
  $$('[data-load]', box).forEach(b => b.addEventListener('click', () => {
    const m = list[+b.dataset.load];
    const row = b.closest('tr');
    if (row.nextElementSibling?.classList.contains('inline-form')) { row.nextElementSibling.remove(); return; }
    const tr = document.createElement('tr');
    tr.className = 'inline-form';
    const td = document.createElement('td');
    td.colSpan = 4;
    const gguf = m.format === 'gguf';
    td.appendChild(loadForm({source: m.source, revision: m.revision, gguf, ggufFile: m.ggufFile, local: gguf && !m.ggufFile,
      defaultId: gguf && m.ggufFile ? m.source + ':' + m.ggufFile.replace(/\.gguf$/i, '').split('/').pop() : m.source}));
    tr.appendChild(td);
    row.after(tr);
  }));
  $$('[data-delete]', box).forEach(b => b.addEventListener('click', async () => {
    const m = list[+b.dataset.delete];
    if (!confirm(`Delete ${m.name} (${fmtBytes(m.bytes)}) from disk?`)) return;
    try { await api('/api/models/local?path=' + encodeURIComponent(m.path), {method: 'DELETE'}); } catch (e) { alert(e.message); }
    refreshModels(true);
  }));
}

let lastLocal = 0;
async function refreshModels(full) {
  try {
    const [loaded, jobs] = await Promise.all([api('/api/models/loaded'), api('/api/jobs')]);
    renderLoaded(loaded); renderJobs(jobs);
    const busy = jobs.some(j => j.state === 'RUNNING');
    // the disk listing is redrawn only when asked or while jobs change it, and never under an open load form
    if ((full || busy || Date.now() - lastLocal > 15000) && !$('#local-list .inline-form')) {
      lastLocal = Date.now();
      renderLocal(await api('/api/models/local'));
    }
    if (busy || full) refreshChatModels();
  } catch (e) { $('#loaded-list').innerHTML = `<p class="error">${esc(e.message)}</p>`; }
}
$('#clear-jobs').addEventListener('click', async () => { await api('/api/jobs', {method: 'DELETE'}); refreshModels(); });

// ------------------------------------------------------------------ settings

let settingsData = [];
async function loadSettings() {
  try { settingsData = await api('/api/settings'); renderSettings(); } catch (e) { $('#settings-msg').innerHTML = `<p class="error">${esc(e.message)}</p>`; }
  const dir = settingsData.find(s => s.key === 'model.dir');
  if (dir) $('#model-dir').textContent = dir.value;
}

function renderSettings() {
  const filter = $('#settings-filter').value.toLowerCase();
  const groups = {};
  for (const s of settingsData) {
    if (filter && !(s.key + ' ' + s.help + ' ' + s.value).toLowerCase().includes(filter)) continue;
    (groups[s.group] ||= []).push(s);
  }
  $('#settings-list').innerHTML = Object.entries(groups).map(([g, items]) => `<div class="set-group"><h3>${esc(g)}</h3>
    ${items.map(s => `<div class="set-row"><div><div class="key">${esc(s.key)}</div><div class="help">${esc(s.help)}</div></div>
      <div><input data-key="${esc(s.key)}" value="${esc(s.value)}" data-orig="${esc(s.value)}" style="width:100%">
        ${s.value !== s.defaultValue ? `<div class="hint">default: ${esc(s.defaultValue || '(empty)')}</div>` : ''}</div>
      <div class="meta"><span class="badge ${s.scope.toLowerCase()}">${s.scope.toLowerCase()}</span>
        ${s.source !== 'default' ? `<span class="badge web" title="${esc(s.source)}">${s.source === 'web' ? 'changed' : 'file'}</span>` : ''}
        ${s.source === 'web' ? `<button class="small ghost" data-reset="${esc(s.key)}">reset</button>` : ''}</div></div>`).join('')}</div>`).join('');
  $$('#settings-list input').forEach(i => i.addEventListener('input', () => i.classList.toggle('changed', i.value !== i.dataset.orig)));
  $$('#settings-list [data-reset]').forEach(b => b.addEventListener('click', () => saveSettings({}, [b.dataset.reset])));
}

async function saveSettings(changes, reset) {
  try {
    settingsData = await api('/api/settings', {method: 'PUT', body: {changes, reset}});
    $('#settings-msg').innerHTML = '<p class="okmsg">Saved. Live settings apply now; load settings to the next model load.</p>';
    renderSettings();
  } catch (e) { $('#settings-msg').innerHTML = `<p class="error">${esc(e.message)}</p>`; }
}
$('#settings-save').addEventListener('click', () => {
  const changes = {};
  $$('#settings-list input.changed').forEach(i => { changes[i.dataset.key] = i.value; });
  if (!Object.keys(changes).length) { $('#settings-msg').innerHTML = '<p class="muted">Nothing changed.</p>'; return; }
  saveSettings(changes, []);
});
$('#settings-filter').addEventListener('input', renderSettings);

// ------------------------------------------------------------------ logs

let logAfter = 0;
async function pollLogs() {
  try {
    const r = await api('/api/logs?after=' + logAfter);
    const pre = $('#log');
    if (r.last < logAfter) { pre.textContent = ''; } // server restarted
    for (const l of r.lines) {
      const span = document.createElement('span');
      span.className = l.stream;
      span.textContent = l.text + '\n';
      pre.appendChild(span);
    }
    while (pre.childNodes.length > 3000) pre.removeChild(pre.firstChild);
    logAfter = r.last;
    $('#log-pending').textContent = r.pending;
    if ($('#log-follow').checked && r.lines.length) pre.scrollTop = pre.scrollHeight;
  } catch { /* shown by the status line */ }
}

// ------------------------------------------------------------------ api tab

function renderApiTab() {
  const base = location.origin + '/v1';
  $('#base-url').textContent = base;
  const model = $('#chat-model').value || 'MODEL';
  $('#ex-curl').textContent = `curl ${base}/chat/completions -H 'Content-Type: application/json' \\
  -H 'Authorization: Bearer $JLLM_API_KEY' \\
  -d '{"model": "${model}", "messages": [{"role": "user", "content": "Why is the sky blue?"}], "stream": false}'`;
  $('#ex-python').textContent = `from openai import OpenAI

client = OpenAI(base_url="${base}", api_key="anything-or-your-key")
for chunk in client.chat.completions.create(
        model="${model}", stream=True,
        messages=[{"role": "user", "content": "Why is the sky blue?"}]):
    print(chunk.choices[0].delta.content or "", end="", flush=True) if chunk.choices else None`;
}
$('#api-key').value = store.get('jllm.apiKey', '');
$('#api-key-save').addEventListener('click', () => { store.set('jllm.apiKey', $('#api-key').value.trim()); refreshStatus(); refreshChatModels(); });

// ------------------------------------------------------------------ start

(function init() {
  const p = store.get('jllm.params', {});
  for (const [k, v] of Object.entries(p)) { const el = $('#p-' + k); if (el) el.value = v; }
  $('#p-system').value = store.get('jllm.system', '');
  $('#chat-think').checked = store.get('jllm.think', false);
  if (!convs.length) newConv(); else { if (!current()) currentId = convs[0].id; renderConvs(); renderMessages(); }
  refreshStatus(); refreshChatModels(); loadSettings();
  setInterval(refreshStatus, 4000);
  setInterval(() => { if (activeTab === 'models') refreshModels(false); }, 1500);
  setInterval(() => { if (activeTab === 'logs') pollLogs(); }, 1000);
  setInterval(() => { if (activeTab === 'chat' && !abort) refreshChatModels(); }, 5000);
  setInterval(() => { if (activeTab === 'api') renderApiTab(); }, 2000);
  renderApiTab();
  const tab = location.hash.slice(1);
  if (['chat', 'models', 'settings', 'logs', 'api'].includes(tab)) showTab(tab);
})();
