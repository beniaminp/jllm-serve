'use strict';
// Settings, logs and the API page; then start the console.

const Admin = (() => {
  let settingsData = [];
  let logAfter = 0;
  let logLines = [];

  // ------------------------------------------------------------------ settings

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
        <div><input data-key="${esc(s.key)}" value="${esc(s.value)}" data-orig="${esc(s.value)}">
          ${s.value !== s.defaultValue ? `<div class="hint">default: ${esc(s.defaultValue || '(empty)')}</div>` : ''}</div>
        <div class="meta"><span class="badge ${s.scope.toLowerCase()}">${s.scope.toLowerCase()}</span>
          ${s.source !== 'default' ? `<span class="badge warn" title="${esc(s.source)}">${s.source === 'web' ? 'changed' : 'file'}</span>` : ''}
          ${s.source === 'web' ? `<button class="small ghost" data-reset="${esc(s.key)}">reset</button>` : ''}</div></div>`).join('')}</div>`).join('')
      || '<p class="empty">No setting matches.</p>';
    $$('#settings-list input').forEach(i => i.addEventListener('input', () => { i.classList.toggle('changed', i.value !== i.dataset.orig); dirty(); }));
    $$('#settings-list [data-reset]').forEach(b => b.addEventListener('click', () => saveSettings({}, [b.dataset.reset])));
    dirty();
  }

  function dirty() {
    const n = $$('#settings-list input.changed').length;
    $('#settings-dirty').textContent = n ? `${n} unsaved change${n === 1 ? '' : 's'}` : '';
    $('#settings-save').disabled = !n;
    $('#settings-discard').disabled = !n;
  }

  async function saveSettings(changes, reset) {
    try {
      settingsData = await api('/api/settings', {method: 'PUT', body: {changes, reset}});
      $('#settings-msg').innerHTML = '';
      toast('Saved. Live settings apply now; load settings to the next model load.', 'ok');
      renderSettings();
    } catch (e) { $('#settings-msg').innerHTML = `<p class="error">${esc(e.message)}</p>`; }
  }

  // ------------------------------------------------------------------ logs

  async function pollLogs() {
    try {
      const r = await api('/api/logs?after=' + logAfter);
      if (r.last < logAfter) logLines = []; // the server restarted
      logLines.push(...r.lines);
      if (logLines.length > 4000) logLines = logLines.slice(-4000);
      logAfter = r.last;
      $('#log-pending').textContent = r.pending;
      if (r.lines.length || !$('#log').childNodes.length) renderLog();
    } catch { /* the status line shows it */ }
  }

  function renderLog() {
    const f = $('#log-filter').value.toLowerCase();
    const pre = $('#log');
    const shown = f ? logLines.filter(l => l.text.toLowerCase().includes(f)) : logLines;
    pre.innerHTML = shown.slice(-2500).map(l => {
      let t = esc(l.text);
      if (f) t = t.replace(new RegExp(f.replace(/[.*+?^${}()|[\]\\]/g, '\\$&').replace(/[&<>"']/g, c => esc(c)), 'gi'), m => `<mark>${m}</mark>`);
      return `<span class="${l.stream}">${t}\n</span>`;
    }).join('');
    if ($('#log-follow').checked) pre.scrollTop = pre.scrollHeight;
  }

  // ------------------------------------------------------------------ api page

  let ocFile = '';
  async function renderOpencode() {
    $('#oc-curl').textContent = `curl -s ${location.origin}/api/integrations/opencode?raw=true > opencode.json`;
    try {
      const r = await api('/api/integrations/opencode');
      ocFile = r.file;
      $('#oc-config').textContent = r.file;
      $('#oc-warnings').innerHTML = (r.warnings.length ? `<div class="warn-box"><b>Before you start</b><ul>${r.warnings.map(w => `<li>${esc(w)}</li>`).join('')}</ul></div>` : '')
        + (r.apiKeyRequired ? '<p class="hint">This server needs an API key: <code>export JLLM_API_KEY=…</code> before starting opencode.</p>' : '');
    } catch (e) {
      ocFile = '';
      $('#oc-config').textContent = e.message;
      $('#oc-warnings').innerHTML = '';
    }
  }

  // Continue / aider / Cline: one card, a tab per agent
  let agent = 'continue';
  const AGENT_HELP = {
    continue: `<a href="https://docs.continue.dev" target="_blank" rel="noopener">Continue ↗</a> (VS Code, JetBrains): save as
      <code>~/.continue/config.yaml</code> (or copy the <code>models</code> entries into yours), then pick the model in Continue's chat.
      Agent mode uses the models' tool calls; Qwen3 thinking is off through <code>extraBodyProperties</code>.`,
    aider: `<a href="https://aider.chat" target="_blank" rel="noopener">aider ↗</a>: save the three files in the root of your repository,
      then run <code>aider</code> there. Small models get <code>edit_format: whole</code> (rewriting a file is more reliable for them than
      search/replace blocks), larger ones <code>diff</code>.`,
    cline: `<a href="https://cline.bot" target="_blank" rel="noopener">Cline ↗</a> has no configuration file: in its settings (⚙️) choose the
      API provider <b>OpenAI Compatible</b> and enter these values. Its prompt is long (~12k tokens), so the first request of a task takes a while on a CPU.`,
  };

  function fileBlock(name, text, type) {
    const id = 'f' + Math.random().toString(36).slice(2);
    setTimeout(() => {
      $('#' + id + '-copy')?.addEventListener('click', e => copyText(text, e.target));
      $('#' + id + '-dl')?.addEventListener('click', () => download(name.replace(/^.*\//, ''), text, type));
    });
    return `<h3>${esc(name)}</h3><pre class="code">${esc(text)}</pre>
      <div class="row"><button id="${id}-copy" class="small">Copy</button><button id="${id}-dl" class="small">Download</button></div>`;
  }

  async function renderAgent() {
    $$('#agent-tabs [data-agent]').forEach(b => b.classList.toggle('primary', b.dataset.agent === agent));
    const body = $('#agent-body');
    try {
      const r = await api('/api/integrations/' + agent);
      $('#agent-warnings').innerHTML = r.warnings.length
        ? `<div class="warn-box"><b>Before you start</b><ul>${r.warnings.map(w => `<li>${esc(w)}</li>`).join('')}</ul></div>` : '';
      let html = `<p>${AGENT_HELP[agent]}</p>`;
      if (agent === 'continue') {
        html += fileBlock('~/.continue/config.yaml', r.file, 'application/yaml');
      } else if (agent === 'aider') {
        html += Object.entries(r.files).map(([n, t]) => fileBlock(n, t, n.endsWith('.json') ? 'application/json' : 'application/yaml')).join('');
        if (r.apiKeyRequired) html += '<p class="hint">This server needs an API key: <code>export OPENAI_API_KEY=…</code> (the console\'s key) before running aider.</p>';
      } else {
        html += `<table class="table"><tr><th>API Provider</th><td>${esc(r.provider)}</td></tr>
          <tr><th>Base URL</th><td><code>${esc(r.baseUrl)}</code></td></tr>
          <tr><th>API Key</th><td><code>${esc(r.apiKey)}</code></td></tr>
          ${r.models.map(m => `<tr><th>Model ID</th><td><code>${esc(m.modelId)}</code> · context window <b>${m.contextWindow}</b>
            · max output tokens <b>${m.maxOutputTokens}</b> · supports images: off · computer use: off · prices: 0</td></tr>`).join('')}</table>`;
      }
      body.innerHTML = html;
    } catch (e) {
      $('#agent-warnings').innerHTML = '';
      body.innerHTML = `<p class="muted">${esc(e.message)}</p>`;
    }
  }

  function renderApi() {
    renderOpencode();
    renderAgent();
    const base = location.origin + '/v1';
    $('#base-url').textContent = base;
    const model = $('#chat-model').value || 'MODEL';
    $('#ex-curl').textContent = `curl ${base}/chat/completions -H 'Content-Type: application/json' \\
  -H "Authorization: Bearer $JLLM_API_KEY" \\
  -d '{"model": "${model}", "messages": [{"role": "user", "content": "Why is the sky blue?"}]}'`;
    $('#ex-python').textContent = `from openai import OpenAI

client = OpenAI(base_url="${base}", api_key="anything-or-your-key")
stream = client.chat.completions.create(
    model="${model}", stream=True,
    messages=[{"role": "user", "content": "Why is the sky blue?"}])
for chunk in stream:
    if chunk.choices:
        print(chunk.choices[0].delta.content or "", end="", flush=True)`;
    $('#ex-tools').textContent = `tools = [{"type": "function", "function": {
    "name": "get_weather", "description": "Weather for a city",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
r = client.chat.completions.create(model="${model}", tools=tools,
        messages=[{"role": "user", "content": "Weather in Paris?"}])
call = r.choices[0].message.tool_calls[0]          # finish_reason == "tool_calls"
# run it, then send {"role": "tool", "tool_call_id": call.id, "content": "..."} back`;
  }

  function init() {
    $('#settings-filter').addEventListener('input', renderSettings);
    $('#settings-save').addEventListener('click', () => {
      const changes = {};
      $$('#settings-list input.changed').forEach(i => { changes[i.dataset.key] = i.value; });
      saveSettings(changes, []);
    });
    $('#settings-discard').addEventListener('click', renderSettings);
    $('#log-filter').addEventListener('input', renderLog);
    $('#log-clear').addEventListener('click', () => { logLines = []; renderLog(); });
    $('#api-key').value = store.get('jllm.apiKey', '');
    $('#api-key-save').addEventListener('click', () => { store.set('jllm.apiKey', $('#api-key').value.trim()); toast('Saved', 'ok'); refreshStatus(); refreshLoaded(); });
    onShow.settings = loadSettings;
    onShow.logs = pollLogs;
    onTick.logs = pollLogs;
    onShow.api = renderApi;
    $('#oc-copy').addEventListener('click', e => { if (ocFile) copyText(ocFile, e.target); });
    $$('#agent-tabs [data-agent]').forEach(b => b.addEventListener('click', () => { agent = b.dataset.agent; renderAgent(); }));
    $('#oc-download').addEventListener('click', () => { if (ocFile) download('opencode.json', ocFile, 'application/json'); });
    loadSettings();
  }

  return {init};
})();

Chat.init();
Models.init();
ToolsPage.init();
Admin.init();
startApp();
