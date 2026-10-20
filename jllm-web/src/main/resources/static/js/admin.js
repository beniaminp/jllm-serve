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

  function renderApi() {
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
    loadSettings();
  }

  return {init};
})();

Chat.init();
Models.init();
ToolsPage.init();
Admin.init();
startApp();
