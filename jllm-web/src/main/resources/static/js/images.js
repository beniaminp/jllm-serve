'use strict';
// Images tab: Create (text to image), the Gallery (IndexedDB, in this browser) and the Editor (images-editor.js).
// Public API for the other tabs: Images.useAsInput(urlOrBlob), Images.selectModel(id), Images.inlineHtml(text).

const Images = (() => {
  const SUGGEST = 'stabilityai/sd-turbo';
  let root, view = store.get('jllm.img.view', 'create');
  let slots = [];
  let createRun = false;
  let loadJob = null;          // id of the sd-turbo load started from the empty state
  let jobsSeen = null;
  let galleryShown = 60;
  let lightboxId = null;
  let lightboxUrl = null;
  let lastFocus = null;
  const thumbUrls = new Map(); // gallery id -> object URL of the thumbnail
  let renderTimer = 0;

  const q = sel => $(sel, root);

  // ------------------------------------------------------------------ skeleton

  function build() {
    root = $('#tab-images');
    root.classList.add('im-page');
    root.innerHTML = `
    <header class="page-head im-head">
      <div class="row" style="margin:0">
        <h1>Images</h1>
        <div class="im-seg" role="tablist" aria-label="Images view">
          <button type="button" role="tab" data-view="create" id="im-tab-create" aria-controls="im-view-create">Create</button>
          <button type="button" role="tab" data-view="editor" id="im-tab-editor" aria-controls="im-view-editor">Editor</button>
        </div>
      </div>
      <p class="muted">Generate pictures with a loaded image model, edit them, inpaint and outpaint. The gallery lives in this browser.</p>
    </header>

    <div id="im-empty" class="card im-emptycard hidden">
      <h2>No image model is loaded</h2>
      <p class="muted">Image models (Stable Diffusion, SDXL, their Turbo variants) load from the <a href="#models">Models</a> page like text models.</p>
      <div class="row">
        <button type="button" class="primary" id="im-load-turbo">Load ${SUGGEST} (2.6 GB)</button>
        <button type="button" id="im-open-models">Open Models</button>
      </div>
      <div id="im-job" class="hidden"><div class="progress"><div></div></div><div class="hint" id="im-job-text"></div></div>
    </div>

    <div id="im-view-create" class="im-create" role="tabpanel" aria-labelledby="im-tab-create">
      <form class="card im-form" id="im-form" autocomplete="off">
        <label>Model <select data-p="model" aria-label="Image model"></select></label>
        <p class="hint" id="im-model-hint"></p>
        <label>Prompt <textarea data-p="prompt" rows="4" placeholder="A lighthouse on a cliff at dawn, oil painting" aria-label="Prompt"></textarea></label>
        <label>Negative prompt <textarea data-p="negative" rows="2" placeholder="(optional) what to keep out" aria-label="Negative prompt"></textarea></label>
        <p class="hint hidden" id="im-neg-hint">The negative prompt has no effect at guidance 1 or below.</p>
        <div class="im-sizebox">
          <label>Size <select data-p="preset" id="im-preset" aria-label="Size preset"></select></label>
          <div class="im-custom hidden" id="im-custom">
            <label>Width <input type="number" data-p="w" id="im-w" min="64" max="2048" step="8" aria-label="Width"></label>
            <button type="button" class="icon-btn" id="im-swap" title="Swap width and height" aria-label="Swap width and height">⇄</button>
            <label>Height <input type="number" data-p="h" id="im-h" min="64" max="2048" step="8" aria-label="Height"></label>
            <label class="check" title="Keep the aspect ratio when one side changes"><input type="checkbox" data-p="lock" aria-label="Lock aspect ratio"> lock</label>
          </div>
          <p class="hint" id="im-size-note"></p>
        </div>
        <div class="im-grid2">
          <label>Steps <input type="number" data-p="steps" min="1" max="150" aria-label="Steps"></label>
          <label>Guidance (CFG) <input type="number" data-p="guidance" min="0" max="30" step="0.1" aria-label="Guidance"></label>
          <label>Scheduler <select data-p="scheduler" aria-label="Scheduler"></select></label>
          <label>Images <select data-p="n" aria-label="Number of images"><option>1</option><option>2</option><option>3</option><option>4</option></select></label>
        </div>
        <label>Seed
          <span class="im-seedrow"><input type="text" inputmode="numeric" data-p="seed" id="im-seed" placeholder="random" aria-label="Seed (empty = random)">
            <button type="button" id="im-dice" title="Random seed" aria-label="Random seed">🎲</button></span></label>
        <p class="hint" id="im-seed-used"></p>
        <p class="hint hidden" id="im-distilled"></p>
        <div class="row">
          <button type="submit" class="primary" id="im-generate">Generate</button>
          <button type="button" class="danger hidden" id="im-stop">■ Stop</button>
          <span class="hint" id="im-gen-hint">Ctrl/Cmd+Enter in the prompt</span>
        </div>
      </form>
      <div class="card im-result">
        <h2>Result <span class="muted small" id="im-timing"></span></h2>
        <div class="im-run hidden" id="im-run"><div class="progress"><div></div></div><div class="hint" id="im-run-text"></div></div>
        <div class="im-slots" id="im-slots" aria-live="polite"><p class="empty">Your pictures appear here, and in the gallery below.</p></div>
      </div>
    </div>

    <div id="im-view-editor" class="im-editorwrap hidden" role="tabpanel" aria-labelledby="im-tab-editor"></div>

    <div class="card im-gallery" id="im-gallery">
      <h2>Gallery <span class="muted small" id="im-gal-info"></span><span class="spacer"></span>
        <button type="button" class="small danger" id="im-gal-clear">Clear gallery</button></h2>
      <p class="warn-box hidden" id="im-gal-warn"></p>
      <div class="im-tiles" id="im-tiles"></div>
      <div class="row"><button type="button" class="small hidden" id="im-gal-more">Show more</button></div>
    </div>`;

    $('#im-preset', root).innerHTML = ImgParams.PRESETS.map(p => `<option value="${p}">${p.replace('x', ' × ')}</option>`).join('')
      + '<option value="custom">custom…</option>';
    ImgParams.bind(q('#im-form'));
    ImgEditor.mount(q('#im-view-editor'));
    buildLightbox();
  }

  // ------------------------------------------------------------------ views

  function setView(v) {
    view = v === 'editor' ? 'editor' : 'create';
    store.set('jllm.img.view', view);
    $$('[data-view]', root).forEach(b => { b.setAttribute('aria-selected', String(b.dataset.view === view)); b.tabIndex = b.dataset.view === view ? 0 : -1; });
    q('#im-view-create').classList.toggle('hidden', view !== 'create');
    q('#im-view-editor').classList.toggle('hidden', view !== 'editor');
    if (view === 'editor') ImgEditor.onVisible();
  }

  // ------------------------------------------------------------------ models: empty state, hints

  function updateHints() {
    const m = ImgParams.model();
    const P = ImgParams.state;
    const none = !ImgParams.models().length;
    q('#im-empty').classList.toggle('hidden', !none);
    q('#im-model-hint').textContent = m ? `${m.description || ''}${m.backend ? ' · ' + m.backend : ''}` : '';
    const d = $('#im-distilled', root);
    d.classList.toggle('hidden', !(m && m.defaults?.distilled));
    d.textContent = 'A few-step (distilled) model: 1–4 steps and guidance 0 are right; the negative prompt is ignored at guidance 1 or below.';
    q('#im-neg-hint').classList.toggle('hidden', !(P.negative.trim() && ImgParams.num('guidance', 7) <= 1));
    const mult = m?.sizeMultiple || 64;
    const sz = ImgParams.size();
    q('#im-size-note').textContent = m ? `Width and height are multiples of ${mult}.${(sz.w % mult || sz.h % mult) ? ' They are rounded when you generate.' : ''}` : '';
    q('#im-custom').classList.toggle('hidden', P.preset !== 'custom');
    q('#im-w').step = q('#im-h').step = mult;
    refreshButtons();
  }

  function refreshButtons() {
    const busy = ImgRun.busy();
    const none = !ImgParams.models().length;
    const gen = q('#im-generate');
    gen.disabled = busy || none;
    gen.classList.toggle('hidden', busy && createRun);
    q('#im-stop').classList.toggle('hidden', !(busy && createRun));
    gen.title = none ? 'Load an image model first' : busy ? 'Another generation is running' : '';
  }

  async function startTurboLoad() {
    try {
      const job = await api('/api/models/load', {method: 'POST', body: {source: SUGGEST, settings: {}}});
      loadJob = job.id;
      toast('Loading ' + SUGGEST + ': the download can take a few minutes.', 'ok');
      pollJobs();
    } catch (e) { toast(e.message, 'err', 8000); }
  }

  async function pollJobs() {
    const box = q('#im-job');
    if (ImgParams.models().length) { box.classList.add('hidden'); loadJob = null; return; }
    let jobs;
    try { jobs = await api('/api/jobs'); } catch { return; }
    jobsSeen = jobs;
    const job = jobs.find(j => j.id === loadJob) || jobs.find(j => j.state === 'RUNNING' || j.state === 'QUEUED');
    if (!job) { box.classList.add('hidden'); return; }
    box.classList.remove('hidden');
    const pct = job.totalBytes > 0 ? Math.min(100, 100 * job.downloadedBytes / job.totalBytes) : null;
    box.querySelector('.progress div').style.width = (pct ?? (job.state === 'RUNNING' ? 5 : 0)) + '%';
    $('#im-job-text', root).textContent = `${job.model} · ${job.phase}${pct != null ? ` · ${fmtBytes(job.downloadedBytes)} / ${fmtBytes(job.totalBytes)} (${pct.toFixed(0)}%)` : ''}${job.error ? ' · ' + job.error : ''}`;
    if (job.state === 'DONE' || job.state === 'FAILED') {
      if (job.state === 'FAILED') toast(job.error || 'The model could not be loaded.', 'err', 8000);
      if (job.id === loadJob) loadJob = null;
      refreshLoaded();
    }
  }

  // ------------------------------------------------------------------ create

  function sizeForRequest() {
    const mult = ImgParams.mult();
    const s = ImgParams.size();
    const snap = v => Math.max(mult, Math.min(2048, Math.round(v / mult) * mult));
    const w = snap(s.w), h = snap(s.h);
    if (w !== s.w || h !== s.h) {
      toast(`Size rounded to ${w}×${h} (multiples of ${mult}).`, 'info');
      if (ImgParams.state.preset === 'custom') ImgParams.setMany({w, h});
    }
    return {w, h};
  }

  async function generate() {
    if (ImgRun.busy()) return;
    const P = ImgParams.state;
    if (!ImgParams.model()) { toast('Load an image model first.', 'err'); return; }
    if (!P.prompt.trim()) { toast('Write a prompt first.', 'err'); q('[data-p=prompt]').focus(); return; }
    const {w, h} = sizeForRequest();
    const body = ImgParams.requestBody();
    body.size = `${w}x${h}`;
    slots.forEach(s => s.dispose());
    slots = [];
    const box = q('#im-slots');
    box.replaceChildren();
    for (let i = 0; i < body.n; i++) { const s = new ImgSlot(w, h); slots.push(s); box.appendChild(s.el); }
    q('#im-timing').textContent = '';
    if (window.innerWidth <= 900) q('.im-result').scrollIntoView({block: 'nearest', behavior: 'smooth'});   // one column: show where the pictures appear
    const run = q('#im-run'), txt = q('#im-run-text');
    run.classList.remove('hidden');
    createRun = true;
    refreshButtons();
    const seeds = [];
    await ImgRun.start('/v1/images/generations', body, {
      slots,
      progress: (t, f) => { txt.textContent = t; run.querySelector('.progress div').style.width = Math.round(f * 100) + '%'; },
      image: async (i, blob, ev) => {
        const s = slots[i];
        seeds[i] = ev.seed;
        s.final(blob, `seed ${ev.seed} · ${ev.width || w}×${ev.height || h}`);
        const meta = {prompt: P.prompt, negative: P.negative, model: body.model,
          params: {size: `${ev.width || w}x${ev.height || h}`, width: ev.width || w, height: ev.height || h, steps: body.steps, guidance: body.guidance,
            seed: ev.seed, scheduler: body.scheduler, mode: 'txt2img'}};
        const rec = await saveBlob(blob, meta);
        s.addAction('Open', 'Open with details', () => openLightbox(rec.id));
        s.addAction('Edit', 'Open in the editor', () => useAsInput(rec.id));
        s.addAction('Download', 'Download the PNG', () => ImgApi.saveBlob(blob, ImgApi.fileName(P.prompt, ev.seed)));
        s.img.addEventListener('click', () => openLightbox(rec.id));
        s.img.style.cursor = 'zoom-in';
      },
      end: (err, done) => {
        createRun = false;
        run.classList.add('hidden');
        if (err === 'stopped') toast('Stopped.', 'info');
        else if (err) { toast(err.message, 'err', 9000); }
        const t = done?.timing?.seconds;
        if (t != null) q('#im-timing').textContent = `${slots.filter(s => s.blob).length} image${body.n === 1 ? '' : 's'} in ${Number(t).toFixed(1)} s`;
        const used = seeds.filter(s => s != null);
        q('#im-seed-used').innerHTML = used.length ? `Seed${used.length > 1 ? 's' : ''} used: <code>${used.join(', ')}</code> <button type="button" class="small" id="im-seed-reuse">use for the next picture</button>` : '';
        $('#im-seed-reuse', root)?.addEventListener('click', () => ImgParams.set('seed', String(used[0])));
        refreshButtons();
        refreshLoaded();
      },
    });
  }

  function wireCreate() {
    q('#im-form').addEventListener('submit', ev => { ev.preventDefault(); generate(); });
    q('[data-p=prompt]', q('#im-form')).addEventListener('keydown', e => { if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) { e.preventDefault(); generate(); } });
    q('#im-stop').addEventListener('click', () => ImgRun.stop());
    q('#im-dice').addEventListener('click', () => { ImgParams.set('seed', ''); q('#im-seed').focus(); });
    q('#im-load-turbo').addEventListener('click', startTurboLoad);
    q('#im-open-models').addEventListener('click', () => showTab('models'));
    q('#im-swap').addEventListener('click', () => { const s = ImgParams.state; ImgParams.setMany({w: s.h, h: s.w}); });

    // size: presets fill width / height; custom values snap to the model's multiple and may keep the aspect ratio
    let last = {w: ImgParams.state.w, h: ImgParams.state.h};
    ImgParams.on((name, value) => {
      if (name === 'preset' && value !== 'custom') {
        const [w, h] = value.split('x').map(Number);
        ImgParams.setMany({w, h});
      }
      if (['model', 'model-list', 'defaults', 'preset', 'guidance', 'negative', 'w', 'h'].includes(name)) updateHints();
    });
    const snapField = which => () => {
      const mult = ImgParams.mult(), s = ImgParams.state;
      const snap = v => Math.max(mult, Math.min(2048, Math.round((parseFloat(v) || 512) / mult) * mult));
      let w = snap(s.w), h = snap(s.h);
      if (s.lock && last.w && last.h) {
        if (which === 'w') h = snap(w * last.h / last.w); else w = snap(h * last.w / last.h);
      }
      ImgParams.setMany({w, h});
      q('#im-w').value = w; q('#im-h').value = h;
      last = {w, h};
      updateHints();
    };
    q('#im-w').addEventListener('change', snapField('w'));
    q('#im-h').addEventListener('change', snapField('h'));
    ImgParams.on(name => { if (name === 'preset' || name === 'model' || name === 'defaults') last = {w: ImgParams.state.w, h: ImgParams.state.h}; });
    ImgRun.onBusy(() => { refreshButtons(); });
  }

  // ------------------------------------------------------------------ gallery

  async function makeThumb(blob) {
    try {
      const bmp = await createImageBitmap(blob);
      const s = Math.min(1, 256 / Math.max(bmp.width, bmp.height));
      const c = ImgCanvas.make(bmp.width * s, bmp.height * s);
      c.getContext('2d').drawImage(bmp, 0, 0, c.width, c.height);
      bmp.close?.();
      return await ImgCanvas.toBlob(c, 'image/jpeg');
    } catch { return blob; }
  }

  async function saveBlob(blob, meta) {
    const rec = {blob, thumb: await makeThumb(blob), prompt: meta.prompt || '', negative: meta.negative || '', model: meta.model || '', params: meta.params || {}};
    await ImgStore.add(rec);
    scheduleGallery();
    return rec;
  }

  function scheduleGallery() {
    clearTimeout(renderTimer);
    renderTimer = setTimeout(renderGallery, 30);
  }

  function thumbUrl(rec) {
    let u = thumbUrls.get(rec.id);
    if (!u) { u = URL.createObjectURL(rec.thumb || rec.blob); thumbUrls.set(rec.id, u); }
    return u;
  }

  function renderGallery() {
    const list = ImgStore.list();
    // free the thumbnails of deleted records
    const ids = new Set(list.map(r => r.id));
    for (const [id, u] of thumbUrls) if (!ids.has(id)) { URL.revokeObjectURL(u); thumbUrls.delete(id); }
    const bytes = ImgStore.bytes();
    q('#im-gal-info').textContent = list.length ? `${list.length} image${list.length === 1 ? '' : 's'} · about ${fmtBytes(bytes)}` : '';
    q('#im-gal-clear').classList.toggle('hidden', !list.length);
    const warn = q('#im-gal-warn');
    warn.classList.toggle('hidden', ImgStore.isPersistent());
    if (!ImgStore.isPersistent()) warn.textContent = `This browser does not allow a saved gallery${ImgStore.reason() ? ' (' + ImgStore.reason() + ')' : ''}: pictures stay in memory until the page is closed. Download the ones you want to keep.`;
    const shown = list.slice(0, galleryShown);
    const tiles = q('#im-tiles');
    if (!list.length) { tiles.innerHTML = '<p class="empty">Nothing yet: every generated picture is kept here, in this browser only.</p>'; }
    else {
      tiles.innerHTML = shown.map(r => `<button type="button" class="im-tile" data-id="${esc(r.id)}" title="${esc(r.prompt)}" aria-label="Open: ${esc(r.prompt || 'picture')}">`
        + `<img src="${thumbUrl(r)}" alt="" loading="lazy">${r.params?.mode && r.params.mode !== 'txt2img' ? `<span class="im-tile-badge">${esc(r.params.mode)}</span>` : ''}</button>`).join('');
    }
    q('#im-gal-more').classList.toggle('hidden', list.length <= galleryShown);
    q('#im-gal-more').textContent = `Show more (${list.length - galleryShown} left)`;
    if (lightboxId && !ImgStore.get(lightboxId)) closeLightbox();
  }

  function wireGallery() {
    q('#im-tiles').addEventListener('click', e => { const t = e.target.closest('.im-tile'); if (t) openLightbox(t.dataset.id); });
    q('#im-gal-more').addEventListener('click', () => { galleryShown += 60; renderGallery(); });
    q('#im-gal-clear').addEventListener('click', async () => {
      const n = ImgStore.list().length;
      if (!n || !confirm(`Delete all ${n} pictures of the gallery from this browser? This cannot be undone.`)) return;
      await ImgStore.clear();
      closeLightbox();
      renderGallery();
      toast('Gallery cleared.', 'ok');
    });
  }

  // ------------------------------------------------------------------ lightbox and picker

  let lb;
  function buildLightbox() {
    lb = document.createElement('div');
    lb.className = 'im-lb hidden';
    lb.innerHTML = `<div class="im-lb-back" data-close></div>
      <div class="im-lb-panel" role="dialog" aria-modal="true" aria-label="Picture details">
        <div class="im-lb-img"><img alt=""><button type="button" class="im-lb-nav prev" aria-label="Previous picture">‹</button><button type="button" class="im-lb-nav next" aria-label="Next picture">›</button></div>
        <div class="im-lb-side">
          <div class="im-lb-top"><h3>Details</h3><button type="button" class="icon-btn" data-close aria-label="Close">✕</button></div>
          <p class="im-lb-prompt" id="lb-prompt"></p>
          <p class="im-lb-neg muted small" id="lb-neg"></p>
          <dl class="im-lb-params" id="lb-params"></dl>
          <div class="im-btns">
            <button type="button" class="primary" id="lb-download">Download</button>
            <button type="button" id="lb-input">Use as input</button>
            <button type="button" id="lb-rerun">Re-run</button>
            <button type="button" id="lb-copy">Copy prompt</button>
            <button type="button" class="danger" id="lb-delete">Delete</button>
          </div>
        </div>
      </div>`;
    document.body.appendChild(lb);
    $$('[data-close]', lb).forEach(b => b.addEventListener('click', closeLightbox));
    $('.prev', lb).addEventListener('click', () => stepLightbox(-1));
    $('.next', lb).addEventListener('click', () => stepLightbox(1));
    $('#lb-download', lb).addEventListener('click', () => { const r = ImgStore.get(lightboxId); if (r) ImgApi.saveBlob(r.blob, ImgApi.fileName(r.prompt, r.params?.seed)); });
    $('#lb-input', lb).addEventListener('click', () => { const id = lightboxId; closeLightbox(); useAsInput(id); });
    $('#lb-rerun', lb).addEventListener('click', () => { const r = ImgStore.get(lightboxId); if (r) { closeLightbox(); rerun(r); } });
    $('#lb-copy', lb).addEventListener('click', e => { const r = ImgStore.get(lightboxId); if (r) copyText(r.prompt, e.target); });
    $('#lb-delete', lb).addEventListener('click', async () => {
      const r = ImgStore.get(lightboxId);
      if (!r || !confirm('Delete this picture from the gallery?')) return;
      const list = ImgStore.list();
      const i = list.findIndex(x => x.id === r.id);
      const next = list[i + 1] || list[i - 1];
      await ImgStore.remove(r.id);
      renderGallery();
      if (next) openLightbox(next.id); else closeLightbox();
    });
    lb.addEventListener('keydown', e => {
      if (e.key === 'Escape') { e.stopPropagation(); closeLightbox(); }
      else if (e.key === 'ArrowLeft') stepLightbox(-1);
      else if (e.key === 'ArrowRight') stepLightbox(1);
      else if (e.key === 'Tab') {   // keep the focus inside the dialog
        const f = $$('button:not([disabled])', lb).filter(b => b.offsetParent !== null);
        if (!f.length) return;
        if (e.shiftKey && document.activeElement === f[0]) { e.preventDefault(); f[f.length - 1].focus(); }
        else if (!e.shiftKey && document.activeElement === f[f.length - 1]) { e.preventDefault(); f[0].focus(); }
      }
    });
  }

  function openLightbox(id) {
    const r = ImgStore.get(id);
    if (!r) return;
    if (lb.classList.contains('hidden')) lastFocus = document.activeElement;
    lightboxId = id;
    if (lightboxUrl) URL.revokeObjectURL(lightboxUrl);
    lightboxUrl = URL.createObjectURL(r.blob);
    const img = $('.im-lb-img img', lb);
    img.src = lightboxUrl;
    img.alt = r.prompt || 'picture';
    $('#lb-prompt', lb).textContent = r.prompt || '(no prompt)';
    $('#lb-neg', lb).textContent = r.negative ? 'Negative: ' + r.negative : '';
    const p = r.params || {};
    const rows = [['Model', r.model], ['Mode', p.mode], ['Size', p.size || (p.width ? `${p.width}x${p.height}` : '')], ['Steps', p.steps], ['Guidance', p.guidance],
      ['Seed', p.seed], ['Scheduler', p.scheduler], ['Strength', p.strength], ['Created', new Date(r.created).toLocaleString()], ['File', fmtBytes(r.blob.size)]];
    $('#lb-params', lb).innerHTML = rows.filter(([, v]) => v != null && v !== '').map(([k, v]) => `<dt>${k}</dt><dd>${esc(v)}</dd>`).join('');
    const list = ImgStore.list();
    const i = list.findIndex(x => x.id === id);
    $('.prev', lb).disabled = i <= 0;
    $('.next', lb).disabled = i < 0 || i >= list.length - 1;
    lb.classList.remove('hidden');
    document.body.classList.add('im-modal');
    if (!lb.contains(document.activeElement)) $('#lb-download', lb).focus();
  }

  function stepLightbox(d) {
    const list = ImgStore.list();
    const i = list.findIndex(x => x.id === lightboxId);
    const n = list[i + d];
    if (n) openLightbox(n.id);
  }

  function closeLightbox() {
    if (lb.classList.contains('hidden')) return;
    lb.classList.add('hidden');
    document.body.classList.remove('im-modal');
    lightboxId = null;
    const img = $('.im-lb-img img', lb);
    img.removeAttribute('src');
    if (lightboxUrl) { URL.revokeObjectURL(lightboxUrl); lightboxUrl = null; }
    try { lastFocus?.focus(); } catch { /* gone */ }
  }

  // a modal gallery grid: resolves with the chosen picture's Blob (or null)
  function pick() {
    return new Promise(resolve => {
      const list = ImgStore.list();
      const d = document.createElement('div');
      d.className = 'im-lb im-pick';
      const urls = [];
      d.innerHTML = `<div class="im-lb-back" data-close></div><div class="im-lb-panel im-pick-panel" role="dialog" aria-modal="true" aria-label="Pick a picture from the gallery">
        <div class="im-lb-top"><h3>Pick a picture</h3><button type="button" class="icon-btn" data-close aria-label="Close">✕</button></div>
        ${list.length ? '<div class="im-tiles"></div>' : '<p class="empty">The gallery is empty: generate or save a picture first.</p>'}</div>`;
      const done = v => { urls.forEach(u => URL.revokeObjectURL(u)); d.remove(); document.removeEventListener('keydown', onKey, true); try { prevFocus?.focus(); } catch { /* gone */ } resolve(v); };
      const prevFocus = document.activeElement;
      const onKey = e => { if (e.key === 'Escape') { e.stopPropagation(); done(null); } };
      document.addEventListener('keydown', onKey, true);
      $$('[data-close]', d).forEach(b => b.addEventListener('click', () => done(null)));
      const tiles = $('.im-tiles', d);
      if (tiles) {
        tiles.innerHTML = list.slice(0, 120).map(r => { const u = URL.createObjectURL(r.thumb || r.blob); urls.push(u);
          return `<button type="button" class="im-tile" data-id="${esc(r.id)}" title="${esc(r.prompt)}" aria-label="Pick: ${esc(r.prompt || 'picture')}"><img src="${u}" alt=""></button>`; }).join('');
        tiles.addEventListener('click', e => { const t = e.target.closest('.im-tile'); if (t) done(ImgStore.get(t.dataset.id)?.blob || null); });
      }
      document.body.appendChild(d);
      (tiles?.querySelector('button') || $('[data-close]', d)).focus();
    });
  }

  // ------------------------------------------------------------------ actions between views

  function rerun(rec) {
    const p = rec.params || {};
    const patch = {prompt: rec.prompt || '', negative: rec.negative || '', seed: p.seed != null ? String(p.seed) : ''};
    if (rec.model && ImgParams.models().some(m => m.id === rec.model)) patch.model = rec.model;
    if (p.steps != null) patch.steps = p.steps;
    if (p.guidance != null) patch.guidance = p.guidance;
    if (p.scheduler) patch.scheduler = p.scheduler;
    const m = /^(\d+)x(\d+)$/.exec(p.size || '');
    if (m) { patch.preset = ImgParams.PRESETS.includes(p.size) ? p.size : 'custom'; patch.w = Number(m[1]); patch.h = Number(m[2]); }
    ImgParams.setMany(patch);
    showTab('images');
    setView('create');
    q('[data-p=prompt]', q('#im-form')).focus();
    toast('The settings of this picture are in the form.' + (['img2img', 'inpaint', 'outpaint', 'edit'].includes(p.mode) ? ' (It was an edit: the input picture is not restored.)' : ''), 'ok', 5000);
  }

  // src: a gallery id, a URL (a picture served by this server), or a Blob
  async function useAsInput(src) {
    try {
      let blob = src;
      if (typeof src === 'string') {
        const rec = ImgStore.get(src);
        if (rec) blob = rec.blob;
        else {
          const res = await fetch(src, {headers: authHeaders()});
          if (!res.ok) throw new Error('could not fetch the picture (HTTP ' + res.status + ')');
          blob = await res.blob();
        }
      }
      showTab('images');
      setView('editor');
      await ImgEditor.loadBlob(blob);
    } catch (e) { toast(e.message, 'err'); }
  }

  function selectModel(id) {
    showTab('images');
    ImgParams.selectModel(id);
    setView('create');
  }

  // pictures inside a tool result of the chat: links to /v1/images/files/<id>.png shown inline
  const FILE_RE = /\/v1\/images\/files\/[0-9a-f]{8,64}\.png/g;
  function inlineHtml(text) {
    const urls = [...new Set(String(text || '').match(FILE_RE) || [])];
    if (!urls.length) return '';
    return '<div class="tc-images">' + urls.map(u => `<figure class="tc-img"><a href="${esc(u)}" target="_blank" rel="noopener"><img src="${esc(u)}" alt="generated image" loading="lazy"></a>`
      + `<button type="button" class="small" data-img-edit="${esc(u)}" title="Open this picture in the image editor">Open in editor</button></figure>`).join('') + '</div>';
  }

  // ------------------------------------------------------------------ init

  function init() {
    build();
    wireCreate();
    wireGallery();
    ImgParams.populateModels();
    $$('[data-view]', root).forEach(b => b.addEventListener('click', () => setView(b.dataset.view)));
    root.querySelector('[role=tablist]').addEventListener('keydown', e => {
      if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') { const v = view === 'create' ? 'editor' : 'create'; setView(v); q(`[data-view=${v}]`).focus(); }
    });
    document.addEventListener('click', e => { const b = e.target.closest('[data-img-edit]'); if (b) useAsInput(b.dataset.imgEdit); });
    modelListeners.push(() => { ImgParams.populateModels(); updateHints(); ImgEditor.refresh(); });
    setView(view);
    updateHints();
    ImgStore.init().then(renderGallery);
    onShow.images = () => {
      refreshLoaded().then(() => { ImgParams.populateModels(); updateHints(); });
      renderGallery();
      pollJobs();
      if (view === 'editor') ImgEditor.onVisible();
    };
    onTick.images = () => { if (!ImgParams.models().length) pollJobs(); };
  }

  return {init, saveBlob, pick, useAsInput, selectModel, inlineHtml, rerun, generationEnded: () => { refreshButtons(); refreshLoaded(); }};
})();

Images.init();
