'use strict';
// Models: inspect Hugging Face links, download / load with per-model settings, jobs, unload, models on disk.

const Models = (() => {
  const QUICK = [
    ['Qwen3 0.6B · GGUF Q4_K_M', 'unsloth/Qwen3-0.6B-GGUF:Q4_K_M'],
    ['Qwen3 1.7B · GGUF Q4_K_M', 'unsloth/Qwen3-1.7B-GGUF:Q4_K_M'],
    ['Qwen3 4B · GGUF Q4_K_M', 'unsloth/Qwen3-4B-GGUF:Q4_K_M'],
    ['Llama 3.2 1B Instruct', 'unsloth/Llama-3.2-1B-Instruct'],
    ['Qwen2.5 1.5B Instruct', 'Qwen/Qwen2.5-1.5B-Instruct'],
  ];
  let lastLocal = 0;

  function loadForm(opts) {
    // opts: {source, revision, gguf, ggufFiles, ggufFile, quant, defaultId, local, download}
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
    form.querySelector('.coding-preset').addEventListener('click', () => {
      form['backend.ctx'].value = 32768;
      form['backend.kv-dtype'].value = 'f16';
      form.slots.value = 2;
      form['server.queue-timeout-ms'].value = 1800000; // a request may wait behind another one's long prefill
      toast('Context 32768, f16 KV cache, 2 slots. On the CPU backend the KV caches live on the Java heap: about 3.7 GB per slot for a 0.6B model, more for bigger ones (JLLM_XMX).', 'info', 9000);
    });
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
        else await api('/api/models/load', {method: 'POST', body: {...req, id: form.id.value || null, slots: form.slots.value ? Number(form.slots.value) : null}});
        form.closest('.inline-form')?.remove();
        $('#inspect-result').innerHTML = '';
        toast(`${action === 'download' ? 'Download' : 'Load'} started — progress under Jobs.`, 'ok');
        refresh(true);
      } catch (e) { toast(e.message, 'err', 8000); }
    });
    return frag;
  }

  const looksLocal = s => /\.gguf$/i.test(s) && /^(\/|~|\.\/|[A-Za-z]:[\\/])/.test(s);

  async function inspect(source) {
    const out = $('#inspect-result');
    if (looksLocal(source)) {
      out.innerHTML = `<p>Local file <code>${esc(source)}</code></p>`;
      out.appendChild(loadForm({source, gguf: true, local: true}));
      return;
    }
    out.innerHTML = '<p class="muted"><span class="spinner"></span> Looking up the repository…</p>';
    try {
      const info = await api('/api/hub/inspect?source=' + encodeURIComponent(source));
      const total = info.files.reduce((a, f) => a + Math.max(0, f.size), 0);
      out.innerHTML = `<p><strong>${esc(info.repo)}</strong> @ ${esc(info.revision)} · <span class="badge info">${info.gguf ? 'GGUF' : 'safetensors'}</span>
        ${info.files.length} files, ${fmtBytes(total)} · <a href="https://huggingface.co/${esc(info.repo)}" target="_blank" rel="noopener">model card ↗</a></p>
        ${info.gguf ? '' : '<p class="hint">Quantized once on first load into the model directory (Q4_0 by default), memory-mapped afterwards.</p>'}`;
      out.appendChild(loadForm({source, revision: info.revision, gguf: info.gguf, ggufFiles: info.ggufFiles, ggufFile: info.file,
        quant: info.quant, defaultId: info.defaultId, download: true}));
    } catch (e) { out.innerHTML = `<p class="error">${esc(e.message)}</p>`; }
  }

  function renderLoaded(list) {
    const box = $('#loaded-list');
    if (!list.length) { box.innerHTML = '<p class="empty">Nothing loaded yet.</p>'; return; }
    const open = new Set($$('details[data-id]', box).filter(d => d.open).map(d => d.dataset.id));
    box.innerHTML = `<table class="table"><tr><th>model</th><th>backend</th><th>context</th><th>weights</th><th>usage</th><th></th></tr>
      ${list.map(m => `<tr>
        <td><strong>${esc(m.id)}</strong><div class="hint">${esc(m.description)}</div><div class="hint">template ${esc(m.template)} · loaded ${fmtTime(m.loadedAt)} in ${m.loadSeconds.toFixed(1)} s</div>
          <details data-id="${esc(m.id)}" ${open.has(m.id) ? 'open' : ''}><summary class="hint">load settings</summary><pre class="hint">${esc(Object.entries(m.settings).map(([k, v]) => k + ' = ' + v).join('\n'))}</pre></details></td>
        <td class="small">${esc(m.backend)}</td><td>${fmtTok(m.contextLength)}${m.slots > 1 ? `<div class="hint">${m.slots} slots${m.busy ? ', ' + m.busy + ' busy' : ''}</div>` : ''}</td><td>${esc(m.quant)}<div class="hint">${fmtBytes(m.weightBytes)}</div></td>
        <td>${m.requests} req${m.active ? ` <span class="badge load">${m.active} active</span>` : ''}<div class="hint">${m.generatedTokens} tokens${m.lastTokensPerSecond ? ' · ' + m.lastTokensPerSecond + ' tok/s' : ''}</div></td>
        <td style="white-space:nowrap"><button class="small" data-chat="${esc(m.id)}">chat</button> <button class="small danger" data-unload="${esc(m.id)}">unload</button></td></tr>`).join('')}</table>`;
    $$('[data-unload]', box).forEach(b => b.addEventListener('click', async () => {
      if (!confirm('Unload ' + b.dataset.unload + '? Running requests on it are cancelled.')) return;
      b.disabled = true;
      try { await api('/api/models/loaded?id=' + encodeURIComponent(b.dataset.unload), {method: 'DELETE'}); toast('Unloaded ' + b.dataset.unload, 'ok'); }
      catch (e) { toast(e.message, 'err'); }
      refresh(true);
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
        const active = j.state === 'RUNNING' || j.state === 'QUEUED';
        return `<tr><td>${esc(j.kind.toLowerCase())}<div class="hint">${fmtTime(j.created)}</div></td>
          <td>${esc(j.model)}<div class="hint">${esc(j.source)}</div></td>
          <td class="state-${j.state}">${active && j.state === 'RUNNING' ? '<span class="spinner"></span> ' : ''}${esc(j.phase)}${j.error ? `<div class="error">${esc(j.error)}</div>` : ''}</td>
          <td>${pct != null ? `${fmtBytes(j.downloadedBytes)} / ${fmtBytes(j.totalBytes)} (${pct.toFixed(0)}%)<div class="progress"><div style="width:${pct}%"></div></div>` : ''}
            ${active && j.phase === 'loading' ? '<div class="hint">reading / quantizing the weights (see Logs)</div>' : ''}</td>
          <td>${active ? `<button class="small" data-cancel="${esc(j.id)}">cancel</button>` : ''}</td></tr>`;
      }).join('')}</table>`;
    $$('[data-cancel]', box).forEach(b => b.addEventListener('click', async () => {
      try { await api(`/api/jobs/${b.dataset.cancel}/cancel`, {method: 'POST'}); } catch (e) { toast(e.message, 'err'); }
      refresh(true);
    }));
  }

  function renderLocal(list) {
    const box = $('#local-list');
    if (!list.length) { box.innerHTML = '<p class="empty">No models downloaded yet.</p>'; return; }
    box.innerHTML = `<table class="table"><tr><th>model</th><th>format</th><th>size</th><th></th></tr>
      ${list.map((m, i) => `<tr><td><strong>${esc(m.name)}</strong><div class="hint">${esc(m.path)}</div>
          ${m.caches.length ? `<div class="hint">quantized: ${m.caches.map(esc).join(', ')}</div>` : ''}</td>
        <td><span class="badge">${esc(m.format)}</span></td><td>${fmtBytes(m.bytes)}</td>
        <td style="white-space:nowrap">${m.loadedAs.length ? '<span class="badge ok">loaded</span> ' : ''}<button class="small" data-load="${i}">load…</button>
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
      try { await api('/api/models/local?path=' + encodeURIComponent(m.path), {method: 'DELETE'}); toast('Deleted ' + m.name, 'ok'); }
      catch (e) { toast(e.message, 'err'); }
      refresh(true);
    }));
  }

  async function refresh(full) {
    try {
      const [loaded, jobs] = await Promise.all([refreshLoaded(), api('/api/jobs')]);
      renderLoaded(loaded); renderJobs(jobs);
      const busy = jobs.some(j => j.state === 'RUNNING');
      // the disk listing is redrawn when asked or while jobs change it, never under an open load form
      if ((full || busy || Date.now() - lastLocal > 15000) && !$('#local-list .inline-form')) {
        lastLocal = Date.now();
        renderLocal(await api('/api/models/local'));
      }
    } catch (e) { $('#loaded-list').innerHTML = `<p class="error">${esc(e.message)}</p>`; }
  }

  function init() {
    $('#quick-picks').innerHTML = '<span class="muted small">Popular:</span>' + QUICK.map(([label, src]) =>
      `<button type="button" data-src="${esc(src)}">${esc(label)}</button>`).join('');
    $$('#quick-picks [data-src]').forEach(b => b.addEventListener('click', () => { $('#hf-link').value = b.dataset.src; inspect(b.dataset.src); }));
    $('#inspect-form').addEventListener('submit', ev => { ev.preventDefault(); inspect($('#hf-link').value.trim()); });
    $('#clear-jobs').addEventListener('click', async () => { await api('/api/jobs', {method: 'DELETE'}); refresh(); });
    onShow.models = () => refresh(true);
    onTick.models = () => refresh(false);
  }

  return {init};
})();
