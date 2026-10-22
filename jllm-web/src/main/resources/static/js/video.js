'use strict';
// Video tab: Create (text to video with a loaded AnimateDiff model), the Clip editor (play, frame strip, trim, reverse,
// ping-pong, speed, export as MP4 / WebP / GIF) and the Gallery (IndexedDB, in this browser).
// Public API for the other tabs: Video.selectModel(id), Video.inlineHtml(text).
//
// The server sends every frame of a finished clip as a JPEG (include_frames), so the editor works on frames in the browser and only asks
// the server (POST /v1/videos/encode) for a file when you export.

const VidApi = (() => {
  const b64ToBlob = (b64, type) => {
    const bin = atob(b64);
    const u8 = new Uint8Array(bin.length);
    for (let i = 0; i < u8.length; i++) u8[i] = bin.charCodeAt(i);
    return new Blob([u8], {type});
  };

  const blobToB64 = blob => new Promise((resolve, reject) => {
    const fr = new FileReader();
    fr.onload = () => resolve(String(fr.result).slice(String(fr.result).indexOf(',') + 1));
    fr.onerror = () => reject(fr.error || new Error('could not read the frame'));
    fr.readAsDataURL(blob);
  });

  async function errorOf(res) {
    let msg = res.statusText || ('HTTP ' + res.status);
    try { const d = await res.json(); msg = d?.error?.message || (typeof d?.error === 'string' ? d.error : null) || d?.message || msg; } catch { /* not json */ }
    if (res.status === 401) {
      showTab('api');
      const note = $('#api-key-note');
      if (note) note.innerHTML = '<span class="error">This server requires an API key: enter it below.</span>';
    }
    return new Error(msg);
  }

  // POST /v1/videos/generations with "stream": true and dispatch the server-sent events to
  // h: {progress(ev), partial(ev, blob), decoding(ev), completed(ev), done(ev)}. Aborting `signal` cancels the generation on the server.
  async function stream(body, h, signal) {
    const res = await fetch('/v1/videos/generations', {method: 'POST', signal, body: JSON.stringify({...body, stream: true}),
      headers: {...authHeaders(), 'Content-Type': 'application/json'}});
    if (!res.ok) throw await errorOf(res);
    let done = null;
    const handle = ev => {
      switch (ev.type) {
        case 'video_generation.progress': h.progress?.(ev); break;
        case 'video_generation.partial_image': h.partial?.(ev, ev.b64_json ? b64ToBlob(ev.b64_json, 'image/png') : null); break;
        case 'video_generation.decoding': h.decoding?.(ev); break;
        case 'video_generation.completed': h.completed?.(ev); break;
        case 'video_generation.done': done = ev; h.done?.(ev); break;
        case 'error': throw new Error(ev.error?.message || 'video generation failed');
        default: break;
      }
    };
    if ((res.headers.get('content-type') || '').includes('json')) {
      const d = await res.json();
      (d.data || []).forEach((it, i) => handle({type: 'video_generation.completed', video: i, ...it}));
      handle({type: 'video_generation.done', usage: d.usage, timing: d.timing, model: d.model});
      return done;
    }
    const reader = res.body.getReader();
    const dec = new TextDecoder();
    let buf = '';
    for (;;) {
      const {done: end, value} = await reader.read();
      if (end) break;
      buf += dec.decode(value, {stream: true}).replace(/\r\n/g, '\n');
      let i;
      while ((i = buf.indexOf('\n\n')) >= 0) {
        const block = buf.slice(0, i);
        buf = buf.slice(i + 2);
        for (const line of block.split('\n')) {
          if (!line.startsWith('data:')) continue;
          const data = line.slice(5).trim();
          if (data === '[DONE]') return done;
          let ev;
          try { ev = JSON.parse(data); } catch { continue; }
          handle(ev);
        }
      }
    }
    return done;
  }

  // frames: Blobs (JPEG / PNG), all one size. Resolves {blob, ext, mime, item}.
  async function encode(frames, fps, format) {
    const body = {frames: await Promise.all(frames.map(blobToB64)), fps, output_format: format, response_format: 'b64_json'};
    const res = await fetch('/v1/videos/encode', {method: 'POST', body: JSON.stringify(body), headers: {...authHeaders(), 'Content-Type': 'application/json'}});
    if (!res.ok) throw await errorOf(res);
    const d = await res.json();
    const it = d.data[0];
    return {blob: b64ToBlob(it.b64_json, it.mime_type), ext: it.format, mime: it.mime_type, item: it};
  }

  const slug = s => String(s || '').toLowerCase().normalize('NFKD').replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '').slice(0, 48);
  const fileName = (prompt, seed, ext) => (slug(prompt) || 'video') + (seed != null && seed !== '' ? '-' + seed : '') + '.' + ext;

  function saveBlob(blob, name) {
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = name;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(a.href), 4000);
  }

  return {b64ToBlob, blobToB64, stream, encode, slug, fileName, saveBlob};
})();

const Video = (() => {
  const SUGGEST = 'guoyww/animatediff-motion-adapter-v1-5-2';
  const KEY = 'jllm.vid.params';
  const PRESETS = ['256x256', '384x384', '512x512', '512x768', '768x512'];
  const SCHEDULERS = [['ddim', 'DDIM'], ['euler', 'Euler'], ['euler_a', 'Euler ancestral'], ['dpmpp_2m', 'DPM++ 2M'], ['dpmpp_2m_karras', 'DPM++ 2M Karras']];
  const P = Object.assign({model: '', prompt: '', negative: '', preset: '512x512', steps: 25, guidance: 7.5, frames: 16, fps: 8, scheduler: 'ddim', seed: '',
    format: 'auto'}, store.get(KEY, {}));
  let root, run = null, loadJob = null, galleryShown = 40;
  const thumbUrls = new Map();     // gallery id -> object URL of the thumbnail
  let cur = null;                  // the clip in the editor: {rec, order, fps, bitmaps, urls, pos, playing, sel}
  let raf = 0, lastTick = 0;
  const q = sel => $(sel, root);

  const models = () => (typeof loadedModels !== 'undefined' ? loadedModels : []).filter(m => m.kind === 'video');
  const model = () => models().find(m => m.id === P.model) || null;
  const mult = () => model()?.sizeMultiple || 64;
  const maxFrames = () => model()?.defaults?.max_frames || 32;
  const save = () => store.set(KEY, P);
  const num = (v, d) => { const x = parseFloat(v); return Number.isFinite(x) ? x : d; };

  // ------------------------------------------------------------------ skeleton

  function build() {
    root = $('#tab-video');
    root.classList.add('vd-page');
    root.innerHTML = `
    <header class="page-head">
      <h1>Video</h1>
      <p class="muted">Short clips from a sentence with a loaded video model (AnimateDiff on Stable Diffusion 1.5), and a small editor: trim, reverse, change the speed, export as MP4, WebP or GIF. The gallery lives in this browser.</p>
    </header>

    <div id="vd-empty" class="card hidden">
      <h2>No video model is loaded</h2>
      <p class="muted">A video model is an AnimateDiff motion adapter. Loading it also downloads the Stable Diffusion 1.5 model it works with (about 2.6 GB, once).
        A 16-frame 512 × 512 clip takes a few minutes on a Mac GPU; 256 × 256 is about four times faster.</p>
      <div class="row">
        <button type="button" class="primary" id="vd-load-suggest">Load ${SUGGEST} (1.8 GB + 2.6 GB)</button>
        <button type="button" id="vd-open-models">Open Models</button>
      </div>
      <div id="vd-job" class="hidden"><div class="progress"><div></div></div><div class="hint" id="vd-job-text"></div></div>
    </div>

    <div class="vd-create">
      <form class="card vd-form" id="vd-form" autocomplete="off">
        <label>Model <select data-p="model" aria-label="Video model"></select></label>
        <p class="hint" id="vd-model-hint"></p>
        <label>Prompt <textarea data-p="prompt" rows="4" placeholder="A red balloon floating over a green hill, white clouds moving, photo" aria-label="Prompt"></textarea></label>
        <label>Negative prompt <textarea data-p="negative" rows="2" placeholder="(optional) blurry, low quality" aria-label="Negative prompt"></textarea></label>
        <div class="vd-grid2">
          <label>Size <select data-p="preset" aria-label="Size"></select></label>
          <label>Frames <input type="number" data-p="frames" min="1" max="32" step="1" aria-label="Frames"></label>
          <label>Frames per second <input type="number" data-p="fps" min="1" max="30" step="1" aria-label="Frames per second"></label>
          <label>Steps <input type="number" data-p="steps" min="1" max="100" aria-label="Steps"></label>
          <label>Guidance (CFG) <input type="number" data-p="guidance" min="0" max="30" step="0.1" aria-label="Guidance"></label>
          <label>Scheduler <select data-p="scheduler" aria-label="Scheduler"></select></label>
          <label>File <select data-p="format" aria-label="File format"><option value="auto">auto</option><option value="mp4">MP4</option><option value="webp">WebP</option><option value="gif">GIF</option></select></label>
          <label>Seed <span class="vd-seedrow"><input type="text" inputmode="numeric" data-p="seed" id="vd-seed" placeholder="random" aria-label="Seed (empty = random)">
            <button type="button" id="vd-dice" title="Random seed" aria-label="Random seed">🎲</button></span></label>
        </div>
        <p class="hint" id="vd-cost"></p>
        <div class="row">
          <button type="submit" class="primary" id="vd-generate">Generate</button>
          <button type="button" class="danger hidden" id="vd-stop">■ Stop</button>
          <span class="hint">Ctrl/Cmd+Enter in the prompt</span>
        </div>
      </form>
      <div class="card vd-result">
        <h2>Result <span class="muted small" id="vd-timing"></span></h2>
        <div class="vd-run hidden" id="vd-run"><div class="progress"><div></div></div><div class="hint" id="vd-run-text"></div>
          <img id="vd-preview" class="vd-preview hidden" alt="preview of the first frame"></div>
        <div id="vd-clip"><p class="empty">Your clip appears here, ready to play and edit, and in the gallery below.</p></div>
      </div>
    </div>

    <div class="card vd-gallery">
      <h2>Gallery <span class="muted small" id="vd-gal-info"></span><span class="spacer"></span>
        <button type="button" class="small danger" id="vd-gal-clear">Clear gallery</button></h2>
      <p class="warn-box hidden" id="vd-gal-warn"></p>
      <div class="vd-tiles" id="vd-tiles"></div>
      <div class="row"><button type="button" class="small hidden" id="vd-gal-more">Show more</button></div>
    </div>`;
    q('[data-p=preset]').innerHTML = PRESETS.map(p => `<option value="${p}">${p.replace('x', ' × ')}</option>`).join('');
    q('[data-p=scheduler]').innerHTML = SCHEDULERS.map(([v, l]) => `<option value="${v}">${l}</option>`).join('');
    for (const el of $$('[data-p]', root)) {
      el.value = P[el.dataset.p] ?? '';
      el.addEventListener(el.tagName === 'SELECT' ? 'change' : 'input', () => { P[el.dataset.p] = el.value; save(); hints(); });
    }
  }

  // ------------------------------------------------------------------ models and hints

  function populateModels() {
    const list = models();
    const sel = q('[data-p=model]');
    const html = list.length ? list.map(m => `<option value="${esc(m.id)}">${esc(m.id)}</option>`).join('') : '<option value="">(no video model loaded)</option>';
    if (!list.some(m => m.id === P.model)) {
      P.model = list[0]?.id || '';
      if (P.model) applyDefaults(list[0]);
      save();
    }
    if (sel.innerHTML !== html) sel.innerHTML = html;
    sel.value = P.model;
    sel.disabled = !list.length;
    hints();
  }

  function applyDefaults(m) {
    const d = m?.defaults;
    if (!d) return;
    const size = Number(d.size) || 512;
    const k = size + 'x' + size;
    Object.assign(P, {steps: d.steps ?? P.steps, guidance: d.guidance ?? P.guidance, frames: d.frames ?? P.frames, fps: d.fps ?? P.fps,
      scheduler: d.scheduler || P.scheduler, preset: PRESETS.includes(k) ? k : P.preset});
    for (const el of $$('[data-p]', root)) if (el.dataset.p !== 'model') el.value = P[el.dataset.p] ?? '';
  }

  function hints() {
    const m = model();
    q('#vd-empty').classList.toggle('hidden', models().length > 0);
    q('#vd-model-hint').textContent = m ? `${m.description || ''}${m.backend ? ' · ' + m.backend : ''}` : '';
    q('#vd-form').querySelector('[data-p=frames]').max = maxFrames();
    const [w, h] = P.preset.split('x').map(Number);
    const frames = Math.max(1, Math.round(num(P.frames, 16)));
    const fps = Math.max(1, num(P.fps, 8));
    q('#vd-cost').textContent = m ? `${frames} frames at ${fps} fps = ${(frames / fps).toFixed(1)} s of video, ${w} × ${h}. Time grows with frames × pixels × steps; a 512 × 512 clip of 16 frames needs a few GB of GPU memory more than an image.` : '';
    buttons();
  }

  function buttons() {
    const gen = q('#vd-generate');
    gen.disabled = !!run || !models().length;
    gen.classList.toggle('hidden', !!run);
    q('#vd-stop').classList.toggle('hidden', !run);
    gen.title = !models().length ? 'Load a video model first' : '';
  }

  async function startSuggestLoad() {
    try {
      const job = await api('/api/models/load', {method: 'POST', body: {source: SUGGEST, settings: {}}});
      loadJob = job.id;
      toast('Loading ' + SUGGEST + ' and its Stable Diffusion base: the download can take a few minutes.', 'ok');
      pollJobs();
    } catch (e) { toast(e.message, 'err', 8000); }
  }

  async function pollJobs() {
    const box = q('#vd-job');
    if (models().length) { box.classList.add('hidden'); loadJob = null; return; }
    let jobs;
    try { jobs = await api('/api/jobs'); } catch { return; }
    const job = jobs.find(j => j.id === loadJob) || jobs.find(j => j.state === 'RUNNING' || j.state === 'QUEUED');
    if (!job) { box.classList.add('hidden'); return; }
    box.classList.remove('hidden');
    const pct = job.totalBytes > 0 ? Math.min(100, 100 * job.downloadedBytes / job.totalBytes) : null;
    box.querySelector('.progress div').style.width = (pct ?? (job.state === 'RUNNING' ? 5 : 0)) + '%';
    $('#vd-job-text', root).textContent = `${job.model} · ${job.phase}${pct != null ? ` · ${fmtBytes(job.downloadedBytes)} / ${fmtBytes(job.totalBytes)} (${pct.toFixed(0)}%)` : ''}${job.error ? ' · ' + job.error : ''}`;
    if (job.state === 'DONE' || job.state === 'FAILED') {
      if (job.state === 'FAILED') toast(job.error || 'The model could not be loaded.', 'err', 8000);
      if (job.id === loadJob) loadJob = null;
      refreshLoaded();
    }
  }

  // ------------------------------------------------------------------ generate

  function snap(v) { const m = mult(); return Math.max(m, Math.round(v / m) * m); }

  async function generate() {
    if (run) return;
    if (!model()) { toast('Load a video model first.', 'err'); return; }
    if (!P.prompt.trim()) { toast('Write a prompt first.', 'err'); q('[data-p=prompt]').focus(); return; }
    const [w0, h0] = P.preset.split('x').map(Number);
    const w = snap(w0), h = snap(h0);
    const frames = Math.max(1, Math.min(maxFrames(), Math.round(num(P.frames, 16))));
    const fps = Math.max(1, Math.min(60, Math.round(num(P.fps, 8))));
    const body = {model: P.model, prompt: P.prompt.trim(), size: `${w}x${h}`, num_frames: frames, fps, steps: Math.max(1, Math.round(num(P.steps, 25))),
      guidance: num(P.guidance, 7.5), scheduler: P.scheduler, n: 1, preview: true, include_frames: 'jpeg', output_format: P.format};
    if (P.negative.trim()) body.negative_prompt = P.negative.trim();
    if (/^-?\d+$/.test(String(P.seed).trim())) body.seed = parseInt(P.seed, 10);
    const ctrl = new AbortController();
    run = {ctrl};
    const box = q('#vd-run'), txt = q('#vd-run-text'), bar = box.querySelector('.progress div'), pv = q('#vd-preview');
    box.classList.remove('hidden');
    pv.classList.add('hidden');
    q('#vd-timing').textContent = '';
    closeClip();
    q('#vd-clip').innerHTML = '<p class="empty">Generating…</p>';
    if (window.innerWidth <= 900) q('.vd-result').scrollIntoView({block: 'nearest', behavior: 'smooth'});
    buttons();
    let rec = null, recP = null;
    const t0 = Date.now();
    const set = (text, f) => { txt.textContent = text; bar.style.width = Math.round(100 * Math.min(1, f)) + '%'; };
    set('starting…', 0);
    try {
      const done = await VidApi.stream(body, {
        progress: ev => set(`step ${ev.step}/${ev.steps}`, 0.9 * ev.step / ev.steps),
        partial: (ev, blob) => { if (!blob) return; const old = pv.src; pv.src = URL.createObjectURL(blob); pv.classList.remove('hidden'); if (old.startsWith('blob:')) URL.revokeObjectURL(old); },
        decoding: ev => set(`decoding frame ${ev.frame}/${ev.frames}`, 0.9 + 0.1 * ev.frame / ev.frames),
        completed: ev => { recP = recordOf(ev, body); },
      }, ctrl.signal);
      rec = recP ? await recP : null;
      if (rec) {
        await VidStore.add(rec);
        scheduleGallery();
        await openClip(rec);
        const secs = done?.timing?.seconds ?? (Date.now() - t0) / 1000;
        q('#vd-timing').textContent = `${rec.frames.length} frames in ${Number(secs).toFixed(1)} s`;
      } else { q('#vd-clip').innerHTML = '<p class="empty">Stopped before the clip was finished.</p>'; }
    } catch (e) {
      const aborted = e.name === 'AbortError';
      q('#vd-clip').innerHTML = `<p class="empty">${aborted ? 'Stopped.' : 'Failed: ' + esc(e.message)}</p>`;
      if (!aborted) toast(e.message, 'err', 9000);
    } finally {
      run = null;
      box.classList.add('hidden');
      if (pv.src.startsWith('blob:')) URL.revokeObjectURL(pv.src);
      pv.removeAttribute('src');
      buttons();
      refreshLoaded();
    }
  }

  async function recordOf(ev, body) {
    const frames = (ev.frame_images || []).map(b => VidApi.b64ToBlob(b, 'image/jpeg'));
    if (!frames.length) throw new Error('the server returned no frames');
    const file = ev.b64_json ? VidApi.b64ToBlob(ev.b64_json, ev.mime_type || 'application/octet-stream') : null;
    return {frames, thumb: await thumbOf(frames[0]), file: file && file.size <= 12e6 ? file : null, fileExt: ev.format, fps: ev.fps || body.fps,
      prompt: body.prompt, negative: body.negative_prompt || '', model: body.model,
      params: {size: `${ev.width}x${ev.height}`, width: ev.width, height: ev.height, frames: frames.length, steps: body.steps, guidance: body.guidance,
        seed: ev.seed, scheduler: body.scheduler, format: ev.format}};
  }

  async function thumbOf(blob) {
    try {
      const bmp = await createImageBitmap(blob);
      const s = Math.min(1, 256 / Math.max(bmp.width, bmp.height));
      const c = document.createElement('canvas');
      c.width = Math.max(1, Math.round(bmp.width * s));
      c.height = Math.max(1, Math.round(bmp.height * s));
      c.getContext('2d').drawImage(bmp, 0, 0, c.width, c.height);
      bmp.close?.();
      return await new Promise(r => c.toBlob(r, 'image/jpeg', 0.85));
    } catch { return blob; }
  }

  // ------------------------------------------------------------------ the clip: player and editor

  function closeClip() {
    cancelAnimationFrame(raf);
    if (cur) { cur.playing = false; cur.bitmaps?.forEach(b => b?.close?.()); cur.urls?.forEach(u => URL.revokeObjectURL(u)); }
    cur = null;
  }

  async function openClip(rec) {
    closeClip();
    const box = q('#vd-clip');
    box.innerHTML = '<p class="empty"><span class="spinner"></span> Decoding the frames…</p>';
    const bitmaps = await Promise.all(rec.frames.map(b => createImageBitmap(b)));
    const urls = rec.frames.map(b => URL.createObjectURL(b));
    cur = {rec, order: rec.frames.map((_, i) => i), fps: rec.fps || 8, bitmaps, urls, pos: 0, playing: true, sel: 0, mode: 'loop', dirty: false};
    renderClip();
    drawFrame();
    lastTick = 0;
    raf = requestAnimationFrame(tick);
    box.scrollIntoView({block: 'nearest', behavior: 'smooth'});
  }

  const frameCount = () => cur.order.length;
  const seconds = () => (frameCount() / cur.fps).toFixed(2);

  function renderClip() {
    const r = cur.rec;
    const box = q('#vd-clip');
    const p = r.params || {};
    box.innerHTML = `
      <div class="vd-player"><canvas id="vd-canvas" width="${cur.bitmaps[0].width}" height="${cur.bitmaps[0].height}" aria-label="Video preview"></canvas></div>
      <div class="vd-bar">
        <button type="button" id="vd-play" class="primary" aria-label="Play or pause">⏸</button>
        <input type="range" id="vd-scrub" min="0" max="${frameCount() - 1}" value="0" aria-label="Frame">
        <span class="muted small" id="vd-pos"></span>
      </div>
      <div class="vd-tools">
        <label>Speed <input type="range" id="vd-fps" min="1" max="30" step="1" value="${Math.round(cur.fps)}" aria-label="Frames per second"> <b id="vd-fps-l"></b></label>
        <label>Playback <select id="vd-mode" aria-label="Playback mode"><option value="loop">loop</option><option value="pingpong">ping-pong</option><option value="once">once</option></select></label>
      </div>
      <div class="vd-strip" id="vd-strip" role="listbox" aria-label="Frames"></div>
      <div class="vd-edit">
        <button type="button" data-op="start" title="Drop the frames before the selected one">Trim start here</button>
        <button type="button" data-op="end" title="Drop the frames after the selected one">Trim end here</button>
        <button type="button" data-op="drop" title="Drop the selected frame">Delete frame</button>
        <button type="button" data-op="dup" title="Show the selected frame twice (a short hold)">Hold frame</button>
        <button type="button" data-op="reverse" title="Play backwards">Reverse</button>
        <button type="button" data-op="pingpong" title="Forward, then back: appends the reversed clip">Ping-pong</button>
        <button type="button" data-op="reset" title="Undo all edits">Reset</button>
      </div>
      <div class="vd-export">
        <label>Export as <select id="vd-fmt" aria-label="Export format"><option value="mp4">MP4 (smallest)</option><option value="webp">WebP (lossless, animated)</option><option value="gif">GIF (256 colours)</option></select></label>
        <button type="button" class="primary" id="vd-export">Export</button>
        <button type="button" id="vd-orig" title="The file the server wrote, unedited">Download original (${esc(r.fileExt || p.format || '')})</button>
        <button type="button" id="vd-savecopy" title="Keep the edited clip as a new gallery entry">Save edits as a new clip</button>
        <button type="button" class="danger" id="vd-delete">Delete from gallery</button>
        <span class="muted small" id="vd-exp-msg"></span>
      </div>
      <details class="vd-meta"><summary>Details</summary>
        <p class="vd-prompt">${esc(r.prompt || '')}</p>${r.negative ? `<p class="muted small">Negative: ${esc(r.negative)}</p>` : ''}
        <dl>${[['Model', r.model], ['Size', p.size], ['Frames', p.frames], ['Steps', p.steps], ['Guidance', p.guidance], ['Seed', p.seed], ['Scheduler', p.scheduler],
          ['Created', new Date(r.created || Date.now()).toLocaleString()]].filter(([, v]) => v != null && v !== '').map(([k, v]) => `<dt>${k}</dt><dd>${esc(v)}</dd>`).join('')}</dl>
        <div class="row"><button type="button" class="small" id="vd-rerun">Use these settings</button><button type="button" class="small" id="vd-copy">Copy prompt</button></div>
      </details>`;
    const strip = $('#vd-strip', box);
    strip.innerHTML = cur.order.map((idx, i) => `<button type="button" role="option" class="vd-th" data-i="${i}" aria-label="Frame ${i + 1}"><img src="${cur.urls[idx]}" alt=""><span>${i + 1}</span></button>`).join('');
    $('#vd-fmt', box).value = store.get('jllm.vid.fmt', 'mp4');
    $('#vd-mode', box).value = cur.mode;
    wireClip(box);
    syncClip();
  }

  function wireClip(box) {
    $('#vd-play', box).addEventListener('click', () => setPlaying(!cur.playing));
    $('#vd-scrub', box).addEventListener('input', e => { setPlaying(false); cur.pos = +e.target.value; cur.sel = cur.pos; drawFrame(); syncClip(); });
    $('#vd-fps', box).addEventListener('input', e => { cur.fps = +e.target.value; cur.dirty = true; syncClip(); });
    $('#vd-mode', box).addEventListener('change', e => { cur.mode = e.target.value; if (cur.mode !== 'once') setPlaying(true); });
    $('#vd-strip', box).addEventListener('click', e => { const b = e.target.closest('.vd-th'); if (b) { setPlaying(false); cur.pos = cur.sel = +b.dataset.i; drawFrame(); syncClip(); } });
    $$('[data-op]', box).forEach(b => b.addEventListener('click', () => edit(b.dataset.op)));
    $('#vd-fmt', box).addEventListener('change', e => store.set('jllm.vid.fmt', e.target.value));
    $('#vd-export', box).addEventListener('click', exportClip);
    $('#vd-orig', box).addEventListener('click', () => {
      const r = cur.rec;
      if (r.file) VidApi.saveBlob(r.file, VidApi.fileName(r.prompt, r.params?.seed, r.fileExt || 'bin'));
      else toast('The original file was too large to keep: use Export.', 'info');
    });
    $('#vd-savecopy', box).addEventListener('click', saveCopy);
    $('#vd-delete', box).addEventListener('click', async () => {
      if (!cur.rec.id || !confirm('Delete this clip from the gallery?')) return;
      await VidStore.remove(cur.rec.id);
      closeClip();
      q('#vd-clip').innerHTML = '<p class="empty">Deleted.</p>';
      renderGallery();
    });
    $('#vd-rerun', box).addEventListener('click', () => rerun(cur.rec));
    $('#vd-copy', box).addEventListener('click', e => copyText(cur.rec.prompt || '', e.target));
  }

  function edit(op) {
    const o = cur.order;
    const i = cur.sel;
    if (op === 'start') cur.order = o.slice(i);
    else if (op === 'end') cur.order = o.slice(0, i + 1);
    else if (op === 'drop') { if (o.length <= 1) { toast('A clip needs at least one frame.', 'info'); return; } cur.order = o.filter((_, k) => k !== i); }
    else if (op === 'dup') cur.order = [...o.slice(0, i + 1), o[i], ...o.slice(i + 1)];
    else if (op === 'reverse') cur.order = [...o].reverse();
    else if (op === 'pingpong') cur.order = [...o, ...o.slice(1, -1).reverse()];
    else if (op === 'reset') { cur.order = cur.rec.frames.map((_, k) => k); cur.fps = cur.rec.fps || 8; cur.dirty = false; }
    if (op !== 'reset') cur.dirty = true;
    cur.pos = cur.sel = Math.min(op === 'start' ? 0 : i, cur.order.length - 1);
    const fps = $('#vd-fps', q('#vd-clip'));
    if (fps) fps.value = Math.round(cur.fps);
    rebuildStrip();
    drawFrame();
    syncClip();
  }

  function rebuildStrip() {
    const box = q('#vd-clip');
    $('#vd-strip', box).innerHTML = cur.order.map((idx, i) => `<button type="button" role="option" class="vd-th" data-i="${i}" aria-label="Frame ${i + 1}"><img src="${cur.urls[idx]}" alt=""><span>${i + 1}</span></button>`).join('');
    $('#vd-scrub', box).max = frameCount() - 1;
  }

  function syncClip() {
    const box = q('#vd-clip');
    if (!cur || !box.querySelector('#vd-pos')) return;
    $('#vd-pos', box).textContent = `frame ${cur.pos + 1}/${frameCount()} · ${seconds()} s${cur.dirty ? ' · edited' : ''}`;
    $('#vd-fps-l', box).textContent = Math.round(cur.fps) + ' fps';
    $('#vd-scrub', box).value = cur.pos;
    $('#vd-play', box).textContent = cur.playing ? '⏸' : '▶';
    $$('.vd-th', box).forEach((b, i) => { b.setAttribute('aria-selected', String(i === cur.sel)); b.classList.toggle('on', i === cur.sel); b.classList.toggle('now', i === cur.pos); });
    const sel = $('.vd-th.on', box);
    if (sel && !cur.playing) sel.scrollIntoView({block: 'nearest', inline: 'nearest'});
    $('#vd-orig', box).classList.toggle('hidden', cur.dirty || !cur.rec.fileExt);
    $('#vd-savecopy', box).classList.toggle('hidden', !cur.dirty);
  }

  function drawFrame() {
    const c = $('#vd-canvas', q('#vd-clip'));
    if (!c || !cur) return;
    const bmp = cur.bitmaps[cur.order[Math.min(cur.pos, cur.order.length - 1)]];
    c.getContext('2d').drawImage(bmp, 0, 0, c.width, c.height);
  }

  function setPlaying(on) {
    if (!cur) return;
    cur.playing = on;
    cancelAnimationFrame(raf);
    if (on) { lastTick = 0; if (cur.mode === 'once' && cur.pos >= frameCount() - 1) cur.pos = 0; raf = requestAnimationFrame(tick); }
    syncClip();
  }

  let dir = 1;
  function tick(ts) {
    if (!cur || !cur.playing) return;
    if (!root.classList.contains('active') || !q('#vd-canvas')) return;   // another tab is shown: the loop restarts when this one is (onShow)
    const dt = 1000 / cur.fps;
    if (!lastTick) lastTick = ts;
    if (ts - lastTick >= dt) {
      lastTick = ts - ((ts - lastTick) % dt);
      const n = frameCount();
      if (cur.mode === 'pingpong') {
        if (n > 1) { if (cur.pos + dir >= n || cur.pos + dir < 0) dir = -dir; cur.pos += dir; }
      } else if (cur.mode === 'once') {
        if (cur.pos + 1 >= n) { cur.playing = false; syncClip(); return; }
        cur.pos++;
      } else cur.pos = (cur.pos + 1) % n;
      cur.sel = cur.pos;
      drawFrame();
      syncClip();
    }
    raf = requestAnimationFrame(tick);
  }

  async function exportClip() {
    const btn = $('#vd-export', q('#vd-clip')), msg = $('#vd-exp-msg', q('#vd-clip'));
    const fmt = $('#vd-fmt', q('#vd-clip')).value;
    const r = cur.rec;
    btn.disabled = true;
    msg.textContent = 'encoding…';
    try {
      // an unedited clip in the format the server already wrote needs no new file
      if (!cur.dirty && r.file && r.fileExt === fmt) { VidApi.saveBlob(r.file, VidApi.fileName(r.prompt, r.params?.seed, fmt)); msg.textContent = 'saved'; return; }
      const out = await VidApi.encode(cur.order.map(i => r.frames[i]), cur.fps, fmt);
      VidApi.saveBlob(out.blob, VidApi.fileName(r.prompt, r.params?.seed, out.ext));
      msg.textContent = `${out.ext.toUpperCase()}, ${fmtBytes(out.blob.size)}`;
    } catch (e) { msg.textContent = ''; toast(e.message, 'err', 8000); }
    finally { btn.disabled = false; }
  }

  async function saveCopy() {
    const r = cur.rec;
    const frames = cur.order.map(i => r.frames[i]);
    const rec = {frames, thumb: await thumbOf(frames[0]), file: null, fileExt: '', fps: cur.fps, prompt: r.prompt, negative: r.negative, model: r.model,
      params: {...r.params, frames: frames.length, edited: true}};
    await VidStore.add(rec);
    scheduleGallery();
    toast('Saved as a new clip in the gallery.', 'ok');
    await openClip(rec);
  }

  function rerun(r) {
    const p = r.params || {};
    const k = `${p.width}x${p.height}`;
    Object.assign(P, {prompt: r.prompt || '', negative: r.negative || '', seed: p.seed != null ? String(p.seed) : '', steps: p.steps ?? P.steps,
      guidance: p.guidance ?? P.guidance, frames: p.frames ?? P.frames, fps: Math.round(r.fps || P.fps), scheduler: p.scheduler || P.scheduler,
      preset: PRESETS.includes(k) ? k : P.preset});
    if (r.model && models().some(m => m.id === r.model)) P.model = r.model;
    save();
    for (const el of $$('[data-p]', root)) el.value = P[el.dataset.p] ?? '';
    populateModels();
    root.scrollIntoView({block: 'start'});
    q('[data-p=prompt]').focus();
    toast('The settings of this clip are in the form.', 'ok', 4000);
  }

  // ------------------------------------------------------------------ gallery

  let renderTimer = 0;
  const scheduleGallery = () => { clearTimeout(renderTimer); renderTimer = setTimeout(renderGallery, 30); };

  function thumbUrl(rec) {
    let u = thumbUrls.get(rec.id);
    if (!u) { u = URL.createObjectURL(rec.thumb || rec.frames[0]); thumbUrls.set(rec.id, u); }
    return u;
  }

  function renderGallery() {
    const list = VidStore.list();
    const ids = new Set(list.map(r => r.id));
    for (const [id, u] of thumbUrls) if (!ids.has(id)) { URL.revokeObjectURL(u); thumbUrls.delete(id); }
    q('#vd-gal-info').textContent = list.length ? `${list.length} clip${list.length === 1 ? '' : 's'} · about ${fmtBytes(VidStore.bytes())}` : '';
    q('#vd-gal-clear').classList.toggle('hidden', !list.length);
    const warn = q('#vd-gal-warn');
    warn.classList.toggle('hidden', VidStore.isPersistent());
    if (!VidStore.isPersistent()) warn.textContent = `This browser does not allow a saved gallery${VidStore.reason() ? ' (' + VidStore.reason() + ')' : ''}: clips stay in memory until the page is closed. Export the ones you want to keep.`;
    const tiles = q('#vd-tiles');
    tiles.innerHTML = list.length ? list.slice(0, galleryShown).map(r => `<button type="button" class="vd-tile${cur?.rec.id === r.id ? ' on' : ''}" data-id="${esc(r.id)}" title="${esc(r.prompt)}" aria-label="Open: ${esc(r.prompt || 'clip')}">`
      + `<img src="${thumbUrl(r)}" alt=""><span class="vd-tile-badge">${r.frames.length} fr · ${(r.frames.length / (r.fps || 8)).toFixed(1)} s${r.params?.edited ? ' · edited' : ''}</span></button>`).join('')
      : '<p class="empty">Nothing yet: every generated clip is kept here, in this browser only.</p>';
    q('#vd-gal-more').classList.toggle('hidden', list.length <= galleryShown);
    q('#vd-gal-more').textContent = `Show more (${list.length - galleryShown} left)`;
  }

  function wireGallery() {
    q('#vd-tiles').addEventListener('click', e => { const t = e.target.closest('.vd-tile'); const r = t && VidStore.get(t.dataset.id); if (r) { openClip(r).then(renderGallery); } });
    q('#vd-gal-more').addEventListener('click', () => { galleryShown += 40; renderGallery(); });
    q('#vd-gal-clear').addEventListener('click', async () => {
      const n = VidStore.list().length;
      if (!n || !confirm(`Delete all ${n} clips of the gallery from this browser? This cannot be undone.`)) return;
      await VidStore.clear();
      closeClip();
      q('#vd-clip').innerHTML = '<p class="empty">Gallery cleared.</p>';
      renderGallery();
    });
  }

  // ------------------------------------------------------------------ chat: clips inside a tool result

  const FILE_RE = /\/v1\/videos\/files\/[0-9a-f]{8,64}\.(mp4|webp|gif)/g;
  function mediaHtml(u, alt) {
    return u.endsWith('.mp4') ? `<video class="tc-vid" src="${esc(u)}" controls loop muted playsinline preload="metadata" aria-label="${esc(alt || 'generated video')}"></video>`
      : `<img class="tc-vid" src="${esc(u)}" alt="${esc(alt || 'generated video')}" loading="lazy">`;
  }
  function inlineHtml(text) {
    const urls = [...new Set(String(text || '').match(FILE_RE) || [])];
    if (!urls.length) return '';
    return '<div class="tc-images">' + urls.map(u => `<figure class="tc-img">${mediaHtml(u)}<a class="small" href="${esc(u)}" target="_blank" rel="noopener">open the file</a></figure>`).join('') + '</div>';
  }

  function selectModel(id) {
    showTab('video');
    if (models().some(m => m.id === id)) { P.model = id; applyDefaults(model()); save(); populateModels(); }
  }

  // ------------------------------------------------------------------ init

  function init() {
    build();
    populateModels();
    wireGallery();
    q('#vd-form').addEventListener('submit', ev => { ev.preventDefault(); generate(); });
    q('[data-p=prompt]').addEventListener('keydown', e => { if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) { e.preventDefault(); generate(); } });
    q('#vd-stop').addEventListener('click', () => run?.ctrl.abort());
    q('#vd-dice').addEventListener('click', () => { P.seed = ''; q('#vd-seed').value = ''; save(); q('#vd-seed').focus(); });
    q('#vd-load-suggest').addEventListener('click', startSuggestLoad);
    q('#vd-open-models').addEventListener('click', () => showTab('models'));
    q('[data-p=model]').addEventListener('change', e => { P.model = e.target.value; applyDefaults(model()); save(); hints(); });
    modelListeners.push(() => populateModels());
    VidStore.init().then(renderGallery);
    onShow.video = () => { refreshLoaded().then(populateModels); renderGallery(); pollJobs(); if (cur?.playing) { lastTick = 0; cancelAnimationFrame(raf); raf = requestAnimationFrame(tick); } };
    onTick.video = () => { if (!models().length) pollJobs(); };
  }

  return {init, selectModel, inlineHtml, mediaHtml};
})();

Video.init();
