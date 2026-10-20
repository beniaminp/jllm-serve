'use strict';
// Tools page: built-in tools and their settings, MCP servers, and a manual "try a tool" runner.

const ToolsPage = (() => {
  let data = null;
  let mcpType = 'stdio';

  async function load() {
    try { data = await api('/api/tools'); } catch (e) { toast(e.message, 'err'); return; }
    render();
  }

  function render() {
    const g = data.settings;
    $('#tools-file').textContent = data.file;
    $('#builtin-list').innerHTML = data.tools.filter(t => t.source === 'builtin').map(t => `<div class="tool-item">
        <label class="switch"><input type="checkbox" data-builtin="${esc(t.name)}" ${t.enabled ? 'checked' : ''}><span></span></label>
        <div><div class="tname">${esc(t.name)}</div><p>${esc(t.description)}</p></div></div>`).join('');
    $('#ts-provider').value = g.search.provider;
    $('#ts-searxng').value = g.search.searxngUrl;
    $('#ts-brave').value = g.search.braveApiKey;
    $('#ts-max').value = g.search.maxResults;
    $('#ts-fetch-max').value = g.fetch.maxChars;
    $('#ts-private').checked = g.fetch.allowPrivateAddresses;
    $('#ts-rounds').value = g.agent.maxRounds;
    $('#ts-result').value = g.agent.maxResultChars;
    $('#ts-timeout').value = g.agent.timeoutSeconds;
    $$('[data-builtin]').forEach(x => x.addEventListener('change', saveSettings));
    renderServers();
    const sel = $('#try-tool');
    const keep = sel.value;
    sel.innerHTML = data.tools.filter(t => t.enabled).map(t => `<option value="${esc(t.name)}">${esc(t.name)}${t.source === 'builtin' ? '' : ' (' + esc(t.source) + ')'}</option>`).join('');
    if (keep && data.tools.some(t => t.name === keep)) sel.value = keep; else fillExample();
  }

  function renderServers() {
    const box = $('#mcp-list');
    if (!data.servers.length) {
      box.innerHTML = `<p class="empty">No MCP servers yet. Add one below — e.g. command <code>npx</code> with arguments
        <code>-y @modelcontextprotocol/server-filesystem /some/folder</code>, or the URL of a remote server.</p>`;
      return;
    }
    box.innerHTML = data.servers.map(s => {
      const c = s.config;
      const where = s.type === 'stdio' ? [c.command, ...(c.args || [])].join(' ') : c.url;
      return `<div class="server"><div class="server-head"><span class="dot ${s.state}"></span><span class="sname">${esc(s.name)}</span>
          <span class="badge">${esc(s.type)}</span><span class="badge ${s.state === 'CONNECTED' ? 'ok' : s.state === 'FAILED' ? 'err' : ''}">${esc(s.state.toLowerCase())}</span>
          ${s.serverName ? `<span class="muted small">${esc(s.serverName)} ${esc(s.serverVersion)}</span>` : ''}
          <div class="spacer"></div>
          <button class="small" data-reconnect="${esc(s.name)}">reconnect</button>
          <button class="small" data-toggle="${esc(s.name)}" data-on="${s.state === 'DISABLED' ? 'true' : 'false'}">${s.state === 'DISABLED' ? 'enable' : 'disable'}</button>
          <button class="small" data-edit="${esc(s.name)}">edit</button>
          <button class="small danger" data-remove="${esc(s.name)}">remove</button></div>
        <div class="hint mono">${esc(where)}</div>
        ${s.error ? `<div class="error small">${esc(s.error)}</div>` : ''}
        ${s.instructions ? `<details><summary class="hint">server instructions</summary><p class="hint">${esc(s.instructions)}</p></details>` : ''}
        ${s.tools.length ? `<div class="tools-chips">${s.tools.map(t => `<span title="${esc(t.description)}">${esc(t.name)}</span>`).join('')}</div>` : ''}</div>`;
    }).join('');
    $$('[data-reconnect]', box).forEach(b => b.addEventListener('click', () => act(b, `/api/tools/mcp/${encodeURIComponent(b.dataset.reconnect)}/reconnect`, 'POST')));
    $$('[data-toggle]', box).forEach(b => b.addEventListener('click', () => act(b, `/api/tools/mcp/${encodeURIComponent(b.dataset.toggle)}/enabled`, 'POST', {enabled: b.dataset.on === 'true'})));
    $$('[data-remove]', box).forEach(b => b.addEventListener('click', () => {
      if (confirm(`Remove the MCP server ${b.dataset.remove}?`)) act(b, `/api/tools/mcp/${encodeURIComponent(b.dataset.remove)}`, 'DELETE');
    }));
    $$('[data-edit]', box).forEach(b => b.addEventListener('click', () => edit(data.servers.find(s => s.name === b.dataset.edit))));
  }

  async function act(button, url, method, body) {
    button.disabled = true;
    button.innerHTML = '<span class="spinner"></span>';
    try { await api(url, {method, body}); } catch (e) { toast(e.message, 'err', 8000); }
    await load();
    Chat.refreshTools();
  }

  function setType(t) {
    mcpType = t;
    $$('#mcp-type button').forEach(b => b.classList.toggle('active', b.dataset.type === t));
    const f = $('#mcp-form');
    $$('.t-stdio, .t-http, .t-sse, .t-json', f).forEach(el => el.classList.add('hidden'));
    $$('.t-' + t, f).forEach(el => el.classList.remove('hidden'));
    f.name.closest('label').classList.toggle('hidden', t === 'json');
    f.name.required = t !== 'json';
  }

  function edit(s) {
    const f = $('#mcp-form');
    $('#mcp-add').open = true;
    setType(s.type);
    const c = s.config;
    f.name.value = s.name;
    f.command.value = c.command || '';
    f.args.value = (c.args || []).join('\n');
    f.env.value = Object.entries(c.env || {}).map(([k, v]) => `${k}=${v}`).join('\n');
    f.url.value = c.url || '';
    f.headers.value = Object.entries(c.headers || {}).map(([k, v]) => `${k}: ${v}`).join('\n');
    $('#mcp-add').scrollIntoView({behavior: 'smooth'});
  }

  const lines = s => s.split('\n').map(x => x.trim()).filter(Boolean);
  const pairs = (s, sep) => Object.fromEntries(lines(s).map(l => { const i = l.indexOf(sep); return i < 0 ? [l, ''] : [l.slice(0, i).trim(), l.slice(i + 1).trim()]; }));

  async function addServer(ev) {
    ev.preventDefault();
    const f = ev.target;
    let body;
    if (mcpType === 'json') {
      try { body = JSON.parse(f.json.value); } catch (e) { toast('Not valid JSON: ' + e.message, 'err'); return; }
    } else {
      const entry = {type: mcpType};
      if (mcpType === 'stdio') {
        entry.command = f.command.value.trim();
        entry.args = lines(f.args.value);
        const env = pairs(f.env.value, '=');
        if (Object.keys(env).length) entry.env = env;
      } else {
        entry.url = f.url.value.trim();
        const h = pairs(f.headers.value, ':');
        if (Object.keys(h).length) entry.headers = h;
      }
      body = {[f.name.value.trim()]: entry};
    }
    $('#mcp-msg').innerHTML = '<span class="spinner"></span> connecting… (npx / uvx may download the server first)';
    try {
      const res = await api('/api/tools/mcp', {method: 'POST', body});
      const failed = res.filter(s => s.state === 'FAILED');
      $('#mcp-msg').innerHTML = failed.length ? `<span class="error">${esc(failed.map(s => s.name + ': ' + s.error).join('\n'))}</span>`
        : `<span class="okmsg">Connected: ${res.map(s => `${esc(s.name)} (${s.tools.length} tools)`).join(', ')}</span>`;
      if (!failed.length) { f.reset(); $('#mcp-add').open = false; }
    } catch (e) { $('#mcp-msg').innerHTML = `<span class="error">${esc(e.message)}</span>`; }
    await load();
    Chat.refreshTools();
  }

  async function saveSettings() {
    const num = id => Number($(id).value);
    const body = {
      search: {provider: $('#ts-provider').value, searxngUrl: $('#ts-searxng').value, braveApiKey: $('#ts-brave').value, maxResults: num('#ts-max')},
      fetch: {allowPrivateAddresses: $('#ts-private').checked, maxChars: num('#ts-fetch-max')},
      agent: {maxRounds: num('#ts-rounds'), maxResultChars: num('#ts-result'), timeoutSeconds: num('#ts-timeout')},
      disabledBuiltins: $$('[data-builtin]').filter(x => !x.checked).map(x => x.dataset.builtin)};
    try {
      data = await api('/api/tools/settings', {method: 'PUT', body});
      render();
      $('#ts-msg').innerHTML = '<span class="okmsg">Saved.</span>';
      setTimeout(() => { $('#ts-msg').innerHTML = ''; }, 2500);
      Chat.refreshTools();
    } catch (e) { $('#ts-msg').innerHTML = `<span class="error">${esc(e.message)}</span>`; }
  }

  // an example arguments object from the schema, so the runner is quick to try
  function fillExample() {
    const t = data?.tools.find(x => x.name === $('#try-tool').value);
    if (!t) return;
    const ex = {web_search: {query: 'Java Vector API'}, fetch_url: {url: 'https://example.com'}, calculator: {expression: '2^10 / 3'},
      current_time: {timezone: 'Europe/Bucharest'}}[t.name];
    if (ex) { $('#try-args').value = JSON.stringify(ex, null, 2); return; }
    const o = {};
    for (const [k, v] of Object.entries(t.parameters?.properties || {})) {
      if ((t.parameters.required || []).includes(k)) o[k] = v.type === 'number' || v.type === 'integer' ? 0 : v.type === 'boolean' ? false : v.type === 'array' ? [] : '';
    }
    $('#try-args').value = JSON.stringify(o, null, 2);
  }

  async function tryRun() {
    let args;
    try { args = JSON.parse($('#try-args').value || '{}'); } catch (e) { toast('Arguments are not valid JSON', 'err'); return; }
    const out = $('#try-out');
    out.classList.remove('hidden');
    out.textContent = 'running…';
    $('#try-meta').textContent = '';
    try {
      const r = await api('/api/tools/call', {method: 'POST', body: {name: $('#try-tool').value, arguments: args}});
      out.textContent = r.content;
      $('#try-meta').innerHTML = `${r.millis} ms${r.error ? ' · <span class="error">error</span>' : ''}`;
    } catch (e) { out.textContent = e.message; }
  }

  function init() {
    $$('#mcp-type button').forEach(b => b.addEventListener('click', () => setType(b.dataset.type)));
    setType('stdio');
    $('#mcp-form').addEventListener('submit', addServer);
    $('#ts-save').addEventListener('click', saveSettings);
    $('#try-tool').addEventListener('change', fillExample);
    $('#try-run').addEventListener('click', tryRun);
    onShow.tools = load;
    onTick.tools = async () => {
      // refresh while servers are connecting
      if (data?.servers.some(s => s.state === 'CONNECTING')) await load();
    };
  }

  return {init};
})();
