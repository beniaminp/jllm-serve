'use strict';
// Models: inspect Hugging Face links, download / load with per-model settings, jobs, unload, models on disk.

const Models = (() => {
  const QUICK = [
    ['Qwen3 0.6B · GGUF Q4_K_M', 'unsloth/Qwen3-0.6B-GGUF:Q4_K_M'],
    ['Qwen3 1.7B · GGUF Q4_K_M', 'unsloth/Qwen3-1.7B-GGUF:Q4_K_M'],
    ['Qwen3 4B · GGUF Q4_K_M', 'unsloth/Qwen3-4B-GGUF:Q4_K_M'],
    ['Llama 3.2 1B Instruct', 'unsloth/Llama-3.2-1B-Instruct'],
    ['Qwen2.5 1.5B Instruct', 'Qwen/Qwen2.5-1.5B-Instruct'],
    // images: begin
    ['SD-Turbo · image (2.6 GB)', 'stabilityai/sd-turbo'],
    ['SDXL-Turbo · image (7 GB)', 'stabilityai/sdxl-turbo'],
    // images: end
    // video: begin
    ['AnimateDiff · video (1.8 GB + SD 1.5, 2.6 GB)', 'guoyww/animatediff-motion-adapter-v1-5-2'],
    // video: end
  ];
  let lastLocal = 0;

  function loadForm(opts) {
    // opts: {source, revision, gguf, ggufFiles, ggufFile, quant, defaultId, local, download, modelDir}
    const frag = $('#load-form-tpl').content.cloneNode(true);
    const form = frag.querySelector('form');
    form.querySelectorAll('.gguf-only').forEach(e => e.classList.toggle('hidden', !opts.gguf || !opts.ggufFiles?.length));
    form.querySelectorAll('.st-only').forEach(e => e.classList.toggle('hidden', !!opts.gguf));
    form.querySelectorAll('.dl-only').forEach(e => e.classList.toggle('hidden', !opts.download));
    // images: begin
    if (opts.image) {   // an image model: backend / precision of the image engine instead of the text-only fields
      form.querySelectorAll('.img-only').forEach(e => e.classList.remove('hidden'));
      for (const n of ['backend', 'backend.ctx', 'backend.kv-dtype', 'model.quant', 'backend.batch', 'backend.threads', 'engine.speculative', 'engine.draft-model', 'slots', 'model.low-disk', 'ggufFile'])
        form.elements[n]?.closest('label')?.classList.add('hidden');
      form.querySelector('.coding-preset')?.classList.add('hidden');
    }
    // images: end
    // video: begin
    if (opts.video) {   // a video model (AnimateDiff motion adapter): the video engine's device and its base model
      form.querySelectorAll('.vid-only').forEach(e => e.classList.remove('hidden'));
      for (const n of ['backend', 'backend.ctx', 'backend.kv-dtype', 'model.quant', 'backend.batch', 'backend.threads', 'engine.speculative', 'engine.draft-model', 'slots', 'model.low-disk', 'ggufFile'])
        form.elements[n]?.closest('label')?.classList.add('hidden');
      form.querySelector('.coding-preset')?.classList.add('hidden');
    }
    // video: end
    // vision: the multimodal projector of a GGUF vision-language model
    form.querySelectorAll('.mm-only').forEach(e => e.classList.toggle('hidden', !opts.gguf));
    form.mmprojChoice.innerHTML = '<option value="">auto (mmproj next to the model)</option><option value="__none__">none (text only)</option>'
      + (opts.mmprojs || []).map(f => `<option value="${esc(f)}">${esc(f)}</option>`).join('');
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
      toast('Context 32768, f16 KV cache, 2 slots. On the CPU backend the two slots share one paged KV pool outside the Java heap: up to about 3.7 GB per slot for a 0.6B model, less when they share a prompt, more for bigger models (run.sh lets it use the physical memory; JLLM_DIRECT_MEMORY bounds it). On the CPU a q8 cache decodes faster than f16 (docs/28).', 'info', 9000);
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
      if (!form.mmprojChoice.closest('.hidden')) { // vision: the chosen projector
        const mm = form.mmprojChoice.value;
        if (mm === '__none__') settings['vision.enabled'] = 'false'; else if (mm) settings['model.mmproj'] = mm;
      }
      // a model listed from a models folder is read from that folder (not downloaded into the default one)
      if (opts.modelDir && !settings['model.dir']) settings['model.dir'] = opts.modelDir;
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
      const shown = (info.diffusers || info.video) && info.imageFiles ? info.imageFiles : info.files;   // images / video: only the files such a model downloads
      const total = shown.reduce((a, f) => a + Math.max(0, f.size), 0);
      out.innerHTML = `<p><strong>${esc(info.repo)}</strong> @ ${esc(info.revision)} · <span class="badge info">${info.video ? 'video model' : info.diffusers ? 'image model' : info.gguf ? 'GGUF' : 'safetensors'}</span>
        ${shown.length} files, ${fmtBytes(total)} · <a href="https://huggingface.co/${esc(info.repo)}" target="_blank" rel="noopener">model card ↗</a></p>
        ${info.video ? '<p class="hint">An AnimateDiff motion adapter: loading it also downloads the Stable Diffusion 1.5 model it works with (<code>video.base-model</code>, fp16 files, about 2.1 GB, once). It makes short clips on the <a href="#video">Video</a> tab and on <code>/v1/videos/generations</code>.</p>' : ''}
        ${info.diffusers ? '<p class="hint">A diffusers repository: the fp16 files are downloaded and run as text-to-image / image-to-image / inpainting on the <a href="#images">Images</a> tab.</p>' : info.gguf ? '' : '<p class="hint">Quantized once on first load into the model directory (Q4_0 by default), memory-mapped afterwards.</p>'}`;
      out.appendChild(loadForm({source, image: !!info.diffusers, video: !!info.video /* video */, revision: info.revision, gguf: info.gguf, ggufFiles: info.ggufFiles, ggufFile: info.file,
        quant: info.quant, defaultId: info.defaultId, download: true,
        mmprojs: info.files.map(f => f.name).filter(n => /(^|\/)mmproj[^/]*\.gguf$/i.test(n))})); // vision
    } catch (e) { out.innerHTML = `<p class="error">${esc(e.message)}</p>`; }
  }

  function renderLoaded(list) {
    const box = $('#loaded-list');
    if (!list.length) { box.innerHTML = '<p class="empty">Nothing loaded yet.</p>'; return; }
    const open = new Set($$('details[data-id]', box).filter(d => d.open).map(d => d.dataset.id));
    box.innerHTML = `<table class="table"><tr><th>model</th><th>backend</th><th>context</th><th>weights</th><th>usage</th><th></th></tr>
      ${list.map(m => `<tr>
        <td><strong>${esc(m.id)}</strong><div class="hint">${esc(m.description)}</div><div class="hint">template ${esc(m.template)} · loaded ${fmtTime(m.loadedAt)} in ${m.loadSeconds.toFixed(1)} s</div>
          ${m.vision ? `<div class="hint"><span class="badge info">vision</span> ${esc(m.projector || '')}</div>` : ''}
          <details data-id="${esc(m.id)}" ${open.has(m.id) ? 'open' : ''}><summary class="hint">load settings</summary><pre class="hint">${esc(Object.entries(m.settings).map(([k, v]) => k + ' = ' + v).join('\n'))}</pre></details></td>
        <td class="small">${esc(m.backend)}</td><td>${fmtTok(m.contextLength)}${m.slots > 1 ? `<div class="hint">${m.slots} slots${m.busy ? ', ' + m.busy + ' busy' : ''}</div>` : ''}</td><td>${esc(m.quant)}<div class="hint">${fmtBytes(m.weightBytes)}</div></td>
        <td>${m.requests} req${m.active ? ` <span class="badge load">${m.active} active</span>` : ''}<div class="hint">${m.generatedTokens} tokens${m.lastTokensPerSecond ? ' · ' + m.lastTokensPerSecond + ' tok/s' : ''}</div></td>
        <td style="white-space:nowrap"><button class="small" data-chat="${esc(m.id)}">chat</button> <button class="small danger" data-unload="${esc(m.id)}">unload</button></td></tr>`).join('')}</table>`;
    // images: begin
    list.forEach((m, i) => {   // an image model: badge, defaults, memory and an "images" button instead of "chat"
      if (m.kind !== 'image') return;
      const tr = $$('tr', box)[i + 1];
      if (!tr) return;
      const d = m.defaults || {};
      tr.cells[0].querySelector('strong').insertAdjacentHTML('afterend', ' <span class="badge info">image</span>');
      const lh = tr.cells[0].querySelectorAll(':scope > .hint')[1];
      if (lh) lh.textContent = `loaded ${fmtTime(m.loadedAt)} in ${m.loadSeconds.toFixed(1)} s`;
      tr.cells[2].innerHTML = `${d.steps ?? '?'} step${d.steps === 1 ? '' : 's'} · ${d.size ?? '?'}×${d.size ?? '?'} · ${esc(d.scheduler || '')}<div class="hint">sizes in multiples of ${m.sizeMultiple}${d.distilled ? ' · distilled' : ''}</div>`;
      tr.cells[3].innerHTML = `${esc(m.quant)}<div class="hint">${fmtBytes(m.weightBytes)} weights</div><div class="hint">${fmtBytes(m.memoryBytes)} in use</div>`;
      tr.cells[4].innerHTML = `${m.requests} req${m.active ? ` <span class="badge load">${m.active} active</span>` : ''}<div class="hint">${m.generatedTokens} images</div>`;
      const c = tr.querySelector('[data-chat]');
      if (c) { c.textContent = 'images'; c.title = 'Open the Images tab with this model'; c.removeAttribute('data-chat'); c.dataset.images = m.id; }
    });
    $$('[data-images]', box).forEach(b => b.addEventListener('click', () => Images.selectModel(b.dataset.images)));
    // images: end
    // video: begin
    list.forEach((m, i) => {   // a video model: badge, defaults, memory and a "video" button instead of "chat"
      if (m.kind !== 'video') return;
      const tr = $$('tr', box)[i + 1];
      if (!tr) return;
      const d = m.defaults || {};
      tr.cells[0].querySelector('strong').insertAdjacentHTML('afterend', ' <span class="badge info">video</span>');
      const lh = tr.cells[0].querySelectorAll(':scope > .hint')[1];
      if (lh) lh.textContent = `loaded ${fmtTime(m.loadedAt)} in ${m.loadSeconds.toFixed(1)} s`;
      tr.cells[2].innerHTML = `${d.frames ?? '?'} frames at ${d.fps ?? '?'} fps · ${d.steps ?? '?'} steps · ${d.size ?? '?'}×${d.size ?? '?'} · ${esc(d.scheduler || '')}<div class="hint">at most ${d.max_frames ?? '?'} frames · sizes in multiples of ${m.sizeMultiple}</div>`;
      tr.cells[3].innerHTML = `${esc(m.quant)}<div class="hint">${fmtBytes(m.weightBytes)} weights</div><div class="hint">${fmtBytes(m.memoryBytes)} in use</div>`;
      tr.cells[4].innerHTML = `${m.requests} req${m.active ? ` <span class="badge load">${m.active} active</span>` : ''}<div class="hint">${m.generatedTokens} clips</div>`;
      const c = tr.querySelector('[data-chat]');
      if (c) { c.textContent = 'video'; c.title = 'Open the Video tab with this model'; c.removeAttribute('data-chat'); c.dataset.video = m.id; }
    });
    $$('[data-video]', box).forEach(b => b.addEventListener('click', () => Video.selectModel(b.dataset.video)));
    // video: end
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

  // The Memory card (docs/26): per loaded model the GPU budget as a bar of its components, the experts on the CPU, the KV pool by tier
  // (hot = GPU blocks, warm = host memory, cold = SSD files), streaming counters and the planner's latest decisions.
  function renderMemory(list) {
    const box = $('#memory-list');
    const rows = (list || []).filter(m => m.report);
    if (!rows.length) { box.innerHTML = '<p class="empty">No loaded model reports its memory (the CPU backend with a flat cache does not).</p>'; return; }
    const COLORS = ['#4c78a8', '#f58518', '#54a24b', '#b279a2', '#9d755d'];
    box.innerHTML = rows.map(m => {
      const r = m.report, dev = r.device, kv = r.kv || {}, ex = r.experts, plan = r.plan;
      let html = `<div class="mem-model"><strong>${esc(m.model)}</strong> <span class="muted small">${esc(r.backend || '')}${m.slots > 1 ? ' · ' + m.slots + ' slots' : ''} · KV ${esc(kv.mode || '?')}</span>`;
      if (dev) {
        const comps = dev.components.filter(c => c.bytes > 0);
        const total = Math.max(dev.budgetBytes, dev.usedBytes);
        html += `<div class="mem-bar" title="${fmtBytes(dev.usedBytes)} of ${fmtBytes(dev.budgetBytes)}" style="display:flex;height:18px;background:var(--bg2,#eee);border-radius:4px;overflow:hidden;margin:6px 0">`
          + comps.map((c, i) => `<div title="${esc(c.name)}: ${fmtBytes(c.bytes)}" style="width:${(100 * c.bytes / total).toFixed(2)}%;background:${COLORS[i % COLORS.length]}"></div>`).join('')
          + '</div><div class="hint">' + comps.map((c, i) => `<span style="color:${COLORS[i % COLORS.length]}">&#9632;</span> ${esc(c.name)} ${fmtBytes(c.bytes)}`).join(' &nbsp; ')
          + `<br>GPU budget ${fmtBytes(dev.budgetBytes)} · used ${fmtBytes(dev.usedBytes)} · ${dev.fits ? '<span class="badge ok">fits</span>' : '<span class="badge err">over budget</span>'}</div>`;
      }
      if (ex && ex.totalBytes > 0) {
        html += `<div class="hint">experts: ${fmtBytes(ex.totalBytes - ex.onCpuBytes)} on the GPU, ${fmtBytes(ex.onCpuBytes)} on the CPU (${ex.onCpuExperts} experts in ${ex.cpuLayers} layers, cpu-moe ${esc(ex.mode)})</div>`;
      }
      if (kv.hot) {
        html += `<div class="hint">KV blocks of ${kv.blockTokens} positions (${fmtBytes(kv.blockBytes)}): hot ${kv.hot.usedBlocks} in use + ${kv.hot.cachedBlocks} cached, committed ${kv.hot.committedBlocks} of cap ${kv.hot.capBlocks} (${fmtBytes(kv.hot.capBytes)})`
          + ` · warm ${kv.warm?.blocks ?? 0} blocks ${fmtBytes(kv.warm?.bytes ?? 0)}${kv.warm?.budgetBytes ? ' of ' + fmtBytes(kv.warm.budgetBytes) : ''}`
          + ` · cold ${kv.cold?.files ?? 0} files ${fmtBytes(kv.cold?.bytes ?? 0)}`
          + `<br>traffic: ${kv.traffic.evictions} evicted, ${kv.traffic.reloads} reloaded, ${kv.warm?.prefetchHits ?? 0}/${kv.traffic.prefetchHints} prefetched, ${kv.traffic.dedups} shared</div>`;
      } else if (kv.mode === 'streamed') {
        html += `<div class="hint">streamed KV: ${fmtBytes(kv.hostBytes)} on the host (${kv.mappedLayers} layers memory-mapped), windows ${fmtBytes(kv.windowBytes)} on the GPU · ${kv.passes} passes, ${fmtBytes(kv.stagedBytes)} staged at ${Number(kv.stageGBPerSecond).toFixed(1)} GB/s</div>`;
      }
      if (plan && (plan.enabled || (plan.decisions || []).length)) {
        html += `<details class="sub"><summary class="hint">memory plan${plan.enabled ? '' : ' (planner off)'}</summary><pre class="hint">`
          + esc((plan.summary || []).join('\n') + '\n' + (plan.decisions || []).slice(-8).map(d => new Date(d.atMillis).toLocaleTimeString() + '  ' + d.kind + ': ' + d.detail).join('\n')) + '</pre></details>';
      }
      return html + '</div>';
    }).join('<hr>');
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
    if (!list.length) { box.innerHTML = '<p class="empty">No models in the models folders yet.</p>'; return; }
    const folders = [...new Set(list.map(m => m.modelDir))];
    $('#model-dir').textContent = folders.join(' · ');
    box.innerHTML = `<table class="table"><tr><th>model</th><th>format</th><th>size</th><th></th></tr>
      ${list.map((m, i) => `${folders.length > 1 && (i === 0 || list[i - 1].modelDir !== m.modelDir)
          ? `<tr class="folder-row"><td colspan="4"><span class="muted small">folder</span> <code>${esc(m.modelDir)}</code></td></tr>` : ''}
        <tr><td><strong>${esc(m.name)}</strong><div class="hint">${esc(m.path)}</div>
          ${m.caches.length ? `<div class="hint">quantized: ${m.caches.map(esc).join(', ')}</div>` : ''}
          ${m.mmprojs?.length ? `<div class="hint"><span class="badge info">vision</span> projector: ${m.mmprojs.map(esc).join(', ')}</div>` : ''}</td>
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
      td.appendChild(loadForm({source: m.source, image: m.format === 'diffusers' /* images */, video: m.format === 'video-adapter' /* video */, revision: m.revision, gguf, ggufFile: m.ggufFile, local: gguf && !m.ggufFile,
        modelDir: gguf && !m.ggufFile ? null : m.modelDir, mmprojs: m.mmprojs,
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

  function renderKv(kv) {
    const box = $('#kv-list');
    const head = `<p class="hint">${kv.enabled ? `Saved after a prefill of ${Math.round(kv.saveAfterMs / 1000)} s or more and on unload, when at least
      ${kv.minTokens} tokens are cached; restored when a prompt starts with a saved conversation. ${fmtBytes(kv.bytes)} of ${fmtBytes(kv.maxBytes)}
      in <code>${esc(kv.dir)}</code>.` : 'Disabled (<code>jllm.web.kv-cache=false</code>).'}</p>`;
    box.innerHTML = head + (kv.saved.length ? `<table class="table"><tr><th>model</th><th>tokens</th><th>size</th><th>last used</th></tr>
      ${kv.saved.map(e => `<tr><td>${esc(e.model)}</td><td>${e.tokens}</td><td>${fmtBytes(e.bytes)}</td>
        <td>${new Date(e.usedAt).toLocaleString()}</td></tr>`).join('')}</table>` : '<p class="empty">Nothing saved yet.</p>');
  }

  async function refresh(full) {
    try {
      const [loaded, jobs] = await Promise.all([refreshLoaded(), api('/api/jobs')]);
      renderLoaded(loaded); renderJobs(jobs);
      api('/api/memory').then(renderMemory).catch(() => { $('#memory-list').innerHTML = ''; });
      // audio:begin
      window.AudioUI?.renderModelsRows();
      // audio:end
      const busy = jobs.some(j => j.state === 'RUNNING');
      // the disk listing is redrawn when asked or while jobs change it, never under an open load form
      if ((full || busy || Date.now() - lastLocal > 15000) && !$('#local-list .inline-form')) {
        lastLocal = Date.now();
        renderLocal(await api('/api/models/local'));
        renderKv(await api('/api/kv-cache'));
      }
    } catch (e) { $('#loaded-list').innerHTML = `<p class="error">${esc(e.message)}</p>`; }
  }

  function init() {
    $('#quick-picks').innerHTML = '<span class="muted small">Popular:</span>' + QUICK.map(([label, src]) =>
      `<button type="button" data-src="${esc(src)}">${esc(label)}</button>`).join('');
    $$('#quick-picks [data-src]').forEach(b => b.addEventListener('click', () => { $('#hf-link').value = b.dataset.src; inspect(b.dataset.src); }));
    $('#inspect-form').addEventListener('submit', ev => { ev.preventDefault(); inspect($('#hf-link').value.trim()); });
    $('#clear-jobs').addEventListener('click', async () => { await api('/api/jobs', {method: 'DELETE'}); refresh(); });
    $('#kv-save').addEventListener('click', async () => {
      try { renderKv(await api('/api/kv-cache/save', {method: 'POST'})); toast('Saved the cached conversations.', 'ok'); }
      catch (e) { toast(e.message, 'err'); }
    });
    $('#kv-clear').addEventListener('click', async () => {
      if (!confirm('Delete every saved conversation cache?')) return;
      try { const r = await api('/api/kv-cache', {method: 'DELETE'}); toast(`Deleted ${r.deleted} files.`, 'ok'); renderKv(await api('/api/kv-cache')); }
      catch (e) { toast(e.message, 'err'); }
    });
    onShow.models = () => refresh(true);
    onTick.models = () => refresh(false);
  }

  return {init};
})();
