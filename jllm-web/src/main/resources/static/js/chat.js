'use strict';
// Chat: conversations in the browser, streaming through POST /api/chat, which runs the chosen tools server side.

const Chat = (() => {
  let convs = store.get('jllm.convs', []);
  let currentId = store.get('jllm.current', null);
  let availableTools = [];   // from /api/tools
  let running = null;        // {abort, conv, live, calls: {id: {status, result}}}
  let frame = 0;

  const PARAMS = ['temperature', 'top_p', 'top_k', 'min_p', 'max_tokens', 'repetition_penalty', 'presence_penalty', 'seed'];
  const SUGGESTIONS = [
    {title: 'Explain a concept', text: 'Explain how a KV cache speeds up transformer inference, with a small example.'},
    {title: 'Search the web', text: 'Search the web for the latest Java release and summarize what is new.', tools: ['web_search', 'fetch_url']},
    {title: 'Calculate', text: 'What is the monthly payment on a 250000 loan at 4.2% over 25 years? Use the calculator.', tools: ['calculator']},
    {title: 'Write code', text: 'Write a Java record for a 2D vector with add, dot and length methods.'},
  ];

  const save = () => { store.set('jllm.convs', convs); store.set('jllm.current', currentId); };
  const current = () => convs.find(c => c.id === currentId);

  function newConv(focus = true) {
    if (running) return;
    const c = {id: Date.now().toString(36) + Math.random().toString(36).slice(2, 6), title: 'New chat', created: Date.now(),
      updated: Date.now(), tools: store.get('jllm.tools', []), messages: []};
    convs.unshift(c);
    currentId = c.id;
    save(); renderConvs(); renderMessages(); renderToolsButton();
    if (focus) $('#input').focus();
    if (innerWidth <= 900) $('#convs').classList.add('collapsed');
  }

  // ------------------------------------------------------------------ sidebar

  function group(ts) {
    const d = new Date(ts), now = new Date();
    const day = 864e5, start = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime();
    if (ts >= start) return 'Today';
    if (ts >= start - day) return 'Yesterday';
    if (ts >= start - 7 * day) return 'Previous 7 days';
    return d.toLocaleDateString([], {month: 'long', year: 'numeric'});
  }

  function renderConvs() {
    const q = $('#conv-search').value.toLowerCase();
    const list = $('#conv-list');
    list.innerHTML = '';
    let last = null;
    for (const c of [...convs].sort((a, b) => (b.updated || 0) - (a.updated || 0))) {
      if (q && !(c.title.toLowerCase().includes(q) || c.messages.some(m => typeof m.content === 'string' && m.content.toLowerCase().includes(q)))) continue;
      const g = group(c.updated || c.created || 0);
      if (g !== last) { list.insertAdjacentHTML('beforeend', `<div class="conv-group">${esc(g)}</div>`); last = g; }
      const el = document.createElement('div');
      el.className = 'conv' + (c.id === currentId ? ' active' : '');
      el.innerHTML = `<span class="title" title="${esc(c.title)}">${esc(c.title)}</span><button data-act="rename" title="rename">✎</button><button data-act="delete" title="delete">✕</button>`;
      el.addEventListener('click', e => {
        const act = e.target.dataset.act;
        if (act === 'delete') {
          e.stopPropagation();
          if (running?.conv === c) return;
          if (c.messages.length && !confirm(`Delete "${c.title}"?`)) return;
          convs = convs.filter(x => x !== c);
          if (currentId === c.id) currentId = convs[0]?.id ?? null;
          if (!currentId) newConv(false); else { save(); renderConvs(); renderMessages(); renderToolsButton(); }
        } else if (act === 'rename') {
          e.stopPropagation();
          const span = el.querySelector('.title');
          span.innerHTML = `<input value="${esc(c.title)}">`;
          const inp = span.querySelector('input');
          inp.focus(); inp.select();
          const done = () => { c.title = inp.value.trim() || c.title; save(); renderConvs(); };
          inp.addEventListener('blur', done);
          inp.addEventListener('keydown', ev => { if (ev.key === 'Enter') inp.blur(); if (ev.key === 'Escape') { inp.value = c.title; inp.blur(); } });
          inp.addEventListener('click', ev => ev.stopPropagation());
        } else if (!running) {
          currentId = c.id; save(); renderConvs(); renderMessages(); renderToolsButton();
          if (innerWidth <= 900) $('#convs').classList.add('collapsed');
        }
      });
      list.appendChild(el);
    }
  }

  // ------------------------------------------------------------------ rendering

  // splits "<think>…</think>rest" (Qwen3, DeepSeek-R1); an unclosed block is still being thought
  function splitThink(text) {
    const t = text || '';
    const open = t.indexOf('<think>');
    if (open < 0 || t.slice(0, open).trim()) return {think: null, rest: t, open: false};
    const close = t.indexOf('</think>');
    return close < 0 ? {think: t.slice(open + 7), rest: '', open: true}
      : {think: t.slice(open + 7, close), rest: t.slice(close + 8), open: false};
  }

  function assistantHtml(text, streaming) {
    const {think, rest, open} = splitThink(text);
    let html = '';
    if (think != null && think.trim()) {
      html += `<details class="think"${open ? ' open' : ''}><summary>${open ? '<span class="spinner"></span> Thinking…' : 'Thought process'}</summary>`
        + `<div class="think-body">${esc(think.trim())}</div></details>`;
    }
    html += `<div class="md">${Markdown.render(rest.replace(/^\s+/, ''))}</div>`;
    return html + (streaming && !open ? '<span class="cursor"></span>' : '');
  }

  function toolCardHtml(call, result, status) {
    let args = call.function?.arguments ?? '{}';
    try { args = JSON.stringify(JSON.parse(args), null, 2); } catch { /* as given */ }
    const short = (() => { try { const o = JSON.parse(call.function?.arguments || '{}'); return Object.values(o).map(v => typeof v === 'string' ? v : JSON.stringify(v)).join(', '); } catch { return args; } })();
    const err = result?.error || (result?.content || '').startsWith('Error: ');
    const icon = status === 'running' ? '<span class="spinner"></span>' : err ? '<span class="badge err">error</span>' : result ? '<span class="badge ok">done</span>' : '<span class="badge">pending</span>';
    return `<details class="toolcard${err ? ' err' : ''}"><summary>🛠 <span class="tc-name">${esc(call.function?.name || '?')}</span>`
      + `<span class="tc-args">${esc(short)}</span>${result?.ms != null ? `<span class="muted small">${result.ms} ms</span>` : ''}${icon}</summary>`
      + `<div class="tc-body"><div class="muted small">arguments</div><pre>${esc(args)}</pre>`
      + (result ? `<div class="muted small">result</div><pre>${esc(result.content)}</pre>` : '') + '</div></details>';
  }

  function statsLine(usage, extra) {
    if (!usage) return extra || '';
    return `${fmtTok(usage.prompt_tokens)} prompt (${fmtTok(usage.cached_tokens)} cached) · ${usage.completion_tokens} generated · ${usage.tokens_per_second} tok/s`
      + (usage.drafted_tokens ? ` · ${usage.accepted_draft_tokens}/${usage.drafted_tokens} drafted accepted` : '') + (extra ? ' · ' + extra : '');
  }

  // turns: a user message, then everything the assistant did until the next user message
  function turns(messages) {
    const out = [];
    messages.forEach((m, i) => {
      if (m.role === 'user') out.push({role: 'user', index: i, msg: m});
      else if (m.role === 'assistant' || m.role === 'tool') {
        let t = out[out.length - 1];
        if (!t || t.role !== 'assistant') { t = {role: 'assistant', index: i, msgs: []}; out.push(t); }
        t.msgs.push(m);
      }
    });
    return out;
  }

  function assistantTurnHtml(msgs, live) {
    const results = {};
    for (const m of msgs) if (m.role === 'tool') results[m.tool_call_id] = {content: m.content, error: m.error, ms: m.ms};
    let html = '';
    for (const m of msgs) {
      if (m.role !== 'assistant') continue;
      if (m.content) html += assistantHtml(m.content, false);
      for (const c of m.tool_calls || []) html += toolCardHtml(c, results[c.id], live?.calls?.[c.id]);
      if (m.error) html += `<div class="error">${esc(m.error)}</div>`;
    }
    if (live) {
      if (live.text) html += assistantHtml(live.text, true);
      else if (!live.toolRunning) html += '<span class="cursor"></span>';
      if (live.error) html += `<div class="error">${esc(live.error)}</div>`;
    }
    return html;
  }

  function renderMessages() {
    const box = $('#messages');
    const c = current();
    box.innerHTML = '';
    if (!c || !c.messages.length) { box.appendChild(welcome()); updateCtx(); return; }
    const ts = turns(c.messages);
    ts.forEach((t, k) => box.appendChild(turnEl(c, t, k === ts.length - 1)));
    if (running?.conv === c && ts[ts.length - 1]?.role === 'user') box.appendChild(turnEl(c, {role: 'assistant', msgs: [], index: c.messages.length}, true));
    box.scrollTop = box.scrollHeight;
    updateCtx();
  }

  function turnEl(c, t, isLast) {
    const el = document.createElement('div');
    el.className = 'turn ' + t.role;
    if (t.role === 'user') {
      el.innerHTML = `<div class="bubble"></div><div class="actions"><button data-a="copy">copy</button><button data-a="edit">edit</button></div>`;
      el.querySelector('.bubble').textContent = t.msg.content;
      el.querySelector('[data-a=copy]').onclick = e => copyText(t.msg.content, e.target);
      el.querySelector('[data-a=edit]').onclick = () => editUser(c, t.index, el);
      return el;
    }
    const live = running?.conv === c && isLast ? running : null;
    const lastMsg = t.msgs[t.msgs.length - 1];
    const model = t.msgs.find(m => m.model)?.model || (live && $('#chat-model').value) || 'assistant';
    el.innerHTML = `<div class="who">${esc(model)}</div><div class="bubble">${assistantTurnHtml(t.msgs, live)}</div>`
      + `<div class="stats">${live ? '' : esc(statsLine(lastMsg?.usage, lastMsg?.finish && lastMsg.finish !== 'stop' ? 'finish: ' + lastMsg.finish : ''))}</div>`
      + (live ? '' : `<div class="actions"><button data-a="copy">copy</button>${isLast ? '<button data-a="regen">regenerate</button>' : ''}</div>`);
    if (!live) {
      const text = t.msgs.filter(m => m.role === 'assistant').map(m => splitThink(m.content).rest.trim()).filter(Boolean).join('\n\n');
      el.querySelector('[data-a=copy]').onclick = e => copyText(text, e.target);
      el.querySelector('[data-a=regen]')?.addEventListener('click', () => regenerate(c));
    }
    return el;
  }

  function welcome() {
    const div = document.createElement('div');
    div.className = 'welcome';
    if (!loadedModels.length) {
      div.innerHTML = `<h2>No model loaded</h2><p class="muted">Open <a href="#models">Models</a> to download one from Hugging Face and load it.</p>`;
      return div;
    }
    div.innerHTML = `<h2>What can I help with?</h2><p class="muted">${esc($('#chat-model').value)} · ${current()?.tools?.length ? current().tools.length + ' tools enabled' : 'no tools'}</p>`
      + `<div class="suggestions">${SUGGESTIONS.map((s, i) => `<button data-i="${i}"><b>${esc(s.title)}</b><small>${esc(s.text)}</small></button>`).join('')}</div>`;
    div.querySelectorAll('[data-i]').forEach(b => b.addEventListener('click', () => {
      const s = SUGGESTIONS[+b.dataset.i];
      if (s.tools) {
        const c = current();
        c.tools = [...new Set([...(c.tools || []), ...s.tools.filter(n => availableTools.some(t => t.name === n && t.enabled))])];
        save(); renderToolsButton();
      }
      $('#input').value = s.text;
      autoGrow();
      $('#input').focus();
    }));
    return div;
  }

  function renderLive() {
    if (frame) return;
    frame = requestAnimationFrame(() => {
      frame = 0;
      const box = $('#messages');
      const c = running?.conv;
      if (!c || c.id !== currentId) return;
      const last = box.lastElementChild;
      if (!last || !last.classList.contains('assistant')) { renderMessages(); return; }
      const ts = turns(c.messages);
      const t = ts[ts.length - 1]?.role === 'assistant' ? ts[ts.length - 1] : {msgs: []};
      const stick = box.scrollHeight - box.scrollTop - box.clientHeight < 80;
      const openCards = [...last.querySelectorAll('details')].map(d => d.open);
      last.querySelector('.bubble').innerHTML = assistantTurnHtml(t.msgs, running);
      last.querySelectorAll('details').forEach((d, i) => { if (openCards[i] && !d.classList.contains('think')) d.open = true; });
      if (stick) box.scrollTop = box.scrollHeight;
    });
  }

  function updateCtx() {
    const meter = $('#ctx-meter');
    const c = current();
    const m = loadedModels.find(x => x.id === $('#chat-model').value);
    const u = c && [...c.messages].reverse().find(x => x.usage)?.usage;
    if (!m || !u) { meter.classList.add('hidden'); return; }
    const used = u.prompt_tokens + u.completion_tokens;
    const pct = Math.min(100, 100 * used / m.contextLength);
    meter.classList.remove('hidden');
    meter.classList.toggle('high', pct > 80);
    meter.querySelector('.bar div').style.width = pct + '%';
    meter.querySelector('span').textContent = `${fmtTok(used)} / ${fmtTok(m.contextLength)}`;
  }

  // ------------------------------------------------------------------ sending

  function params() {
    const p = {};
    for (const k of PARAMS) { const v = $('#p-' + k).value; if (v !== '') p[k] = Number(v); }
    store.set('jllm.params', p);
    return p;
  }

  // what goes to the server: the OpenAI fields only
  function wire(messages) {
    return messages.filter(m => !(m.role === 'assistant' && !m.content && !m.tool_calls?.length)).map(m => {
      const o = {role: m.role, content: m.content ?? ''};
      if (m.tool_calls) o.tool_calls = m.tool_calls;
      if (m.tool_call_id) { o.tool_call_id = m.tool_call_id; o.name = m.name; }
      return o;
    });
  }

  async function run(c) {
    const model = $('#chat-model').value;
    if (!model) { toast('Load a model first (Models page).', 'err'); return; }
    const system = $('#p-system').value.trim();
    store.set('jllm.system', system);
    const body = {model, messages: [...(system ? [{role: 'system', content: system}] : []), ...wire(c.messages)],
      tools: (c.tools || []).filter(n => availableTools.some(t => t.name === n && t.enabled)),
      enable_thinking: $('#chat-think').checked, ...params()};
    running = {abort: new AbortController(), conv: c, text: '', calls: {}, usage: null, error: null, toolRunning: false};
    c.updated = Date.now();
    setBusy(true);
    renderMessages();
    let finish = null;
    let usageTotal = null;
    try {
      const res = await fetch('/api/chat', {method: 'POST', headers: {'Content-Type': 'application/json', ...authHeaders()},
        body: JSON.stringify(body), signal: running.abort.signal});
      if (!res.ok) {
        const t = await res.text();
        let msg = t; try { msg = JSON.parse(t).error.message; } catch { /* plain text */ }
        throw new Error(msg);
      }
      const reader = res.body.getReader();
      const dec = new TextDecoder();
      let buf = '';
      for (;;) {
        const {value, done} = await reader.read();
        if (done) break;
        buf += dec.decode(value, {stream: true});
        let i;
        while ((i = buf.indexOf('\n\n')) >= 0) {
          const ev = buf.slice(0, i); buf = buf.slice(i + 2);
          if (!ev.startsWith('data: ')) continue;
          const e = JSON.parse(ev.slice(6));
          switch (e.type) {
            case 'delta': running.text += e.content; break;
            case 'usage':
              usageTotal = e.usage;
              break;
            case 'message': {
              const m = {...e.message, model};
              if (m.role === 'assistant') running.text = '';
              if (m.role === 'tool') {
                const r = running.calls[m.tool_call_id + ':r'];
                if (r) { m.ms = r.ms; m.error = r.error; }
              }
              c.messages.push(m);
              break;
            }
            case 'tool_call': running.calls[e.id] = 'running'; running.toolRunning = true; break;
            case 'tool_result':
              running.calls[e.id] = 'done'; running.toolRunning = false;
              running.calls[e.id + ':r'] = {ms: e.ms, error: e.error};
              break;
            case 'error': running.error = e.message; break;
            case 'done': finish = e.finish_reason; break;
          }
          renderLive();
        }
      }
    } catch (e) {
      if (e.name === 'AbortError') finish = 'stopped'; else running.error = e.message;
    }
    // keep what was generated even when stopped half-way
    if (running.text) c.messages.push({role: 'assistant', content: running.text, model});
    const last = [...c.messages].reverse().find(m => m.role === 'assistant');
    if (running.error) {
      if (last && last === c.messages[c.messages.length - 1]) last.error = running.error;
      else c.messages.push({role: 'assistant', content: '', error: running.error, model});
    }
    if (last) { last.usage = usageTotal; last.finish = finish; }
    running = null;
    setBusy(false);
    save(); renderConvs(); renderMessages();
  }

  function setBusy(b) {
    $('#send').classList.toggle('hidden', b);
    $('#stop').classList.toggle('hidden', !b);
    $('#chat-model').disabled = b;
  }

  function send(text) {
    if (running) return;
    if (!current()) newConv(false);
    const c = current();
    c.messages.push({role: 'user', content: text});
    if (c.title === 'New chat') c.title = text.replace(/\s+/g, ' ').slice(0, 48);
    save(); renderConvs();
    run(c);
  }

  function regenerate(c) {
    if (running) return;
    let i = c.messages.length - 1;
    while (i >= 0 && c.messages[i].role !== 'user') i--;
    if (i < 0) return;
    c.messages.splice(i + 1);
    run(c);
  }

  function editUser(c, index, el) {
    if (running) return;
    const bubble = el.querySelector('.bubble');
    const old = c.messages[index].content;
    bubble.innerHTML = `<textarea rows="3" style="width:100%;min-width:320px"></textarea><div class="row" style="justify-content:flex-end">`
      + `<button class="ghost small" data-x>cancel</button><button class="primary small" data-ok>send</button></div>`;
    const ta = bubble.querySelector('textarea');
    ta.value = old; ta.focus();
    bubble.querySelector('[data-x]').onclick = () => renderMessages();
    bubble.querySelector('[data-ok]').onclick = () => {
      const t = ta.value.trim();
      if (!t) return;
      c.messages.splice(index);
      c.messages.push({role: 'user', content: t});
      save();
      run(c);
    };
  }

  function exportMarkdown() {
    const c = current();
    if (!c?.messages.length) return;
    let md = `# ${c.title}\n\n`;
    for (const m of c.messages) {
      if (m.role === 'user') md += `## You\n\n${m.content}\n\n`;
      else if (m.role === 'assistant') {
        const {rest} = splitThink(m.content);
        if (rest.trim()) md += `## ${m.model || 'Assistant'}\n\n${rest.trim()}\n\n`;
        for (const t of m.tool_calls || []) md += `> 🛠 \`${t.function.name}(${t.function.arguments})\`\n\n`;
      } else if (m.role === 'tool') md += '```\n' + String(m.content).slice(0, 2000) + '\n```\n\n';
    }
    download(c.title.replace(/[^\w -]+/g, '').slice(0, 60) + '.md', md);
  }

  // ------------------------------------------------------------------ tools popover

  async function refreshTools() {
    try { availableTools = (await api('/api/tools')).tools; } catch { /* keep */ }
    renderToolsButton();
  }

  function renderToolsButton() {
    const c = current();
    const on = (c?.tools || []).filter(n => availableTools.some(t => t.name === n && t.enabled));
    $('#tools-count').textContent = on.length || '';
    $('#tools-btn').classList.toggle('on', on.length > 0);
    $('#composer-hint').textContent = on.length ? `tools: ${on.join(', ')}` : '';
  }

  function renderToolsPopover() {
    const c = current();
    const pop = $('#tools-pop');
    const usable = availableTools.filter(t => t.enabled);
    if (!usable.length) { pop.innerHTML = '<p class="muted">No tools available. Configure them on the <a href="#tools">Tools</a> page.</p>'; return; }
    const groups = {};
    for (const t of usable) (groups[t.source] ||= []).push(t);
    pop.innerHTML = Object.entries(groups).map(([src, ts]) => `<h4>${src === 'builtin' ? 'Built-in' : 'MCP · ' + esc(src)}</h4>`
      + ts.map(t => `<label><input type="checkbox" value="${esc(t.name)}" ${c.tools?.includes(t.name) ? 'checked' : ''}><span>${esc(t.name)}`
        + `<small>${esc(t.description.length > 140 ? t.description.slice(0, 137) + '…' : t.description)}</small></span></label>`).join('')).join('')
      + `<div class="pop-actions"><button class="small" data-all>all</button><button class="small" data-none>none</button>`
      + `<div class="spacer"></div><a class="small" href="#tools">manage…</a></div>`;
    const sync = () => {
      c.tools = $$('input[type=checkbox]', pop).filter(x => x.checked).map(x => x.value);
      store.set('jllm.tools', c.tools);
      save(); renderToolsButton();
    };
    $$('input', pop).forEach(x => x.addEventListener('change', sync));
    pop.querySelector('[data-all]').onclick = () => { $$('input', pop).forEach(x => { x.checked = true; }); sync(); };
    pop.querySelector('[data-none]').onclick = () => { $$('input', pop).forEach(x => { x.checked = false; }); sync(); };
  }

  // ------------------------------------------------------------------ models

  function renderModelSelect(list) {
    const sel = $('#chat-model');
    const want = sel.value || store.get('jllm.model', '');
    const html = list.length ? list.map(m => `<option value="${esc(m.id)}">${esc(m.id)}</option>`).join('') : '<option value="">(no model loaded)</option>';
    if (sel.innerHTML !== html) sel.innerHTML = html;
    if (list.some(m => m.id === want)) sel.value = want;
    if (!current()?.messages.length && !running) renderMessages();
  }

  // ------------------------------------------------------------------ wiring

  function autoGrow() {
    const t = $('#input');
    t.style.height = 'auto';
    t.style.height = Math.min(260, t.scrollHeight) + 'px';
  }

  function init() {
    const p = store.get('jllm.params', {});
    for (const [k, v] of Object.entries(p)) { const el = $('#p-' + k); if (el) el.value = v; }
    $('#p-system').value = store.get('jllm.system', '');
    $('#chat-think').checked = store.get('jllm.think', false);
    $('#input').placeholder = 'Message…   (Enter to send, Shift+Enter for a new line)';
    if (!convs.length) newConv(false); else { if (!current()) currentId = convs[0].id; renderConvs(); renderMessages(); }
    if (store.get('jllm.convsCollapsed', innerWidth <= 900)) $('#convs').classList.add('collapsed');

    $('#composer').addEventListener('submit', ev => {
      ev.preventDefault();
      const t = $('#input').value.trim();
      if (!t || running) return;
      $('#input').value = '';
      autoGrow();
      send(t);
    });
    $('#input').addEventListener('keydown', ev => {
      if (ev.key === 'Enter' && !ev.shiftKey && !ev.isComposing) { ev.preventDefault(); $('#composer').requestSubmit(); }
    });
    $('#input').addEventListener('input', autoGrow);
    $('#stop').addEventListener('click', () => running?.abort.abort());
    document.addEventListener('keydown', ev => {
      if (ev.key === 'Escape' && running && activeTab === 'chat') running.abort.abort();
      if (ev.key === 'Escape') $('#tools-pop').classList.add('hidden');
    });
    $('#new-chat').addEventListener('click', () => newConv());
    $('#conv-search').addEventListener('input', renderConvs);
    $('#toggle-convs').addEventListener('click', () => {
      $('#convs').classList.toggle('collapsed');
      store.set('jllm.convsCollapsed', $('#convs').classList.contains('collapsed'));
    });
    $('#toggle-params').addEventListener('click', () => $('#params').classList.toggle('hidden'));
    $('#close-params').addEventListener('click', () => $('#params').classList.add('hidden'));
    $('#reset-params').addEventListener('click', () => { PARAMS.forEach(k => { $('#p-' + k).value = ''; }); params(); });
    $('#chat-think').addEventListener('change', () => store.set('jllm.think', $('#chat-think').checked));
    $('#chat-model').addEventListener('change', () => { store.set('jllm.model', $('#chat-model').value); updateCtx(); });
    $('#export-chat').addEventListener('click', exportMarkdown);
    $('#tools-btn').addEventListener('click', ev => {
      ev.stopPropagation();
      const pop = $('#tools-pop');
      if (pop.classList.toggle('hidden')) return;
      renderToolsPopover();
      refreshTools().then(() => { if (!pop.classList.contains('hidden')) renderToolsPopover(); });
    });
    document.addEventListener('click', ev => { if (!ev.target.closest('.popover-wrap')) $('#tools-pop').classList.add('hidden'); });
    modelListeners.push(renderModelSelect);
    refreshTools();
    onShow.chat = () => { refreshTools(); $('#input').focus(); };
  }

  return {init, refreshTools, current};
})();
