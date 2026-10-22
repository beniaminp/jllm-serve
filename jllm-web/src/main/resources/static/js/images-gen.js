'use strict';
// Images: API client (streaming generations / edits), the form state shared by the Create and Editor views,
// and the result "slot" widget. Plain JavaScript, no build step.

const ImgApi = (() => {
  function b64ToBlob(b64, type = 'image/png') {
    const bin = atob(b64);
    const u8 = new Uint8Array(bin.length);
    for (let i = 0; i < u8.length; i++) u8[i] = bin.charCodeAt(i);
    return new Blob([u8], {type});
  }

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

  // POST a JSON body to /v1/images/generations or /v1/images/edits with "stream": true and dispatch the events.
  // handlers: {progress(ev), partial(ev, blob), image(ev, blob), done(ev)}. Resolves with the "done" event.
  // Aborting `signal` cancels the generation on the server. Throws Error(message) on HTTP and stream errors.
  async function stream(path, body, h, signal) {
    const res = await fetch(path, {method: 'POST', signal, body: JSON.stringify(body),
      headers: {...authHeaders(), 'Content-Type': 'application/json'}});
    if (!res.ok) throw await errorOf(res);
    let done = null;
    const handle = ev => {
      switch (ev.type) {
        case 'image_generation.progress': h.progress?.(ev); break;
        case 'image_generation.partial_image': h.partial?.(ev, ev.b64_json ? b64ToBlob(ev.b64_json) : null); break;
        case 'image_generation.completed': h.image?.(ev, ev.b64_json ? b64ToBlob(ev.b64_json) : null); break;
        case 'image_generation.done': done = ev; h.done?.(ev); break;
        case 'error': throw new Error(ev.error?.message || 'image generation failed');
        default: break;
      }
    };
    if ((res.headers.get('content-type') || '').includes('json')) {
      // the server answered without a stream: a plain response
      const d = await res.json();
      (d.data || []).forEach((it, i) => handle({type: 'image_generation.completed', image: i, b64_json: it.b64_json, seed: it.seed,
        width: it.width, height: it.height, revised_prompt: it.revised_prompt}));
      handle({type: 'image_generation.done', usage: d.usage, timing: d.timing, model: d.model});
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

  const slug = s => String(s || '').toLowerCase().normalize('NFKD').replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '').slice(0, 48);
  const fileName = (prompt, seed) => (slug(prompt) || 'image') + (seed != null && seed !== '' ? '-' + seed : '') + '.png';

  function saveBlob(blob, name) {
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = name;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(a.href), 4000);
  }

  return {b64ToBlob, stream, slug, fileName, saveBlob};
})();

// ------------------------------------------------------------------ shared form state

const ImgParams = (() => {
  const KEY = 'jllm.img.params';
  const PRESETS = ['512x512', '512x768', '768x512', '768x768', '1024x1024'];
  const SCHEDULERS = [['euler', 'Euler'], ['euler_a', 'Euler ancestral'], ['dpmpp_2m', 'DPM++ 2M'], ['dpmpp_2m_karras', 'DPM++ 2M Karras']];
  const state = Object.assign({model: '', prompt: '', negative: '', preset: '512x512', w: 512, h: 512, lock: true, steps: 20, guidance: 7.5,
    seed: '', scheduler: 'euler', n: 1, mode: 'img2img', strength: 0.8, maxSide: 1024, grow: 4, blur: 6}, store.get(KEY, {}));
  const bound = [];
  const listeners = [];

  const models = () => (typeof loadedModels !== 'undefined' ? loadedModels : []).filter(m => m.kind === 'image');
  const model = () => models().find(m => m.id === state.model) || null;
  const mult = () => model()?.sizeMultiple || 64;

  function save() { store.set(KEY, state); }

  function setField(el, v) {
    if (el.type === 'checkbox') el.checked = !!v; else if (el.value !== String(v)) el.value = v;
  }

  function set(name, value, src) {
    if (state[name] === value) return;
    state[name] = value;
    save();
    for (const el of bound) if (el !== src && el.dataset.p === name) setField(el, value);
    listeners.forEach(f => f(name, value));
    if (name === 'model') { const m = models().find(x => x.id === value); if (m) applyDefaults(m); }
  }

  function setMany(obj) {
    if ('model' in obj) set('model', obj.model);   // first: choosing a model resets steps / size / ... to its defaults
    for (const [k, v] of Object.entries(obj)) if (k !== 'model') set(k, v);
  }

  function bind(root) {
    root.querySelectorAll('[data-p]').forEach(el => {
      bound.push(el);
      if (el.tagName === 'SELECT' && el.dataset.p === 'scheduler') {
        el.innerHTML = SCHEDULERS.map(([v, l]) => `<option value="${v}">${l}</option>`).join('');
      }
      setField(el, state[el.dataset.p]);
      const type = el.type === 'checkbox' ? 'change' : 'input';
      el.addEventListener(type, () => set(el.dataset.p, el.type === 'checkbox' ? el.checked : el.value, el));
      if (el.tagName === 'SELECT') el.addEventListener('change', () => set(el.dataset.p, el.value, el));
    });
  }

  // fill every bound model <select> with the loaded image models and keep the selection valid
  function populateModels() {
    const list = models();
    const html = list.length ? list.map(m => `<option value="${esc(m.id)}">${esc(m.id)}</option>`).join('') : '<option value="">(no image model loaded)</option>';
    const hadModel = state.model;
    if (!list.some(m => m.id === state.model)) {
      state.model = list[0]?.id || '';
      if (state.model && state.model !== hadModel) applyDefaults(list[0], true);
      save();
    }
    for (const el of bound) {
      if (el.dataset.p !== 'model') continue;
      if (el.innerHTML !== html) el.innerHTML = html;
      el.value = state.model;
      el.disabled = !list.length;
    }
    listeners.forEach(f => f('model-list', list));
  }

  // steps / guidance / size / scheduler of the model's recommended settings
  function applyDefaults(m, silent) {
    const d = m?.defaults;
    if (!d) return;
    const size = Number(d.size) || 512;
    const k = size + 'x' + size;
    const patch = {steps: d.steps ?? state.steps, guidance: d.guidance ?? state.guidance, scheduler: d.scheduler || state.scheduler};
    if (PRESETS.includes(k)) { patch.preset = k; patch.w = size; patch.h = size; } else { patch.preset = 'custom'; patch.w = size; patch.h = size; }
    // a scheduler the select does not offer must still be selectable
    if (!SCHEDULERS.some(([v]) => v === patch.scheduler)) SCHEDULERS.push([patch.scheduler, patch.scheduler]);
    bound.forEach(el => { if (el.dataset.p === 'scheduler' && !el.querySelector(`option[value="${patch.scheduler}"]`)) el.insertAdjacentHTML('beforeend', `<option value="${esc(patch.scheduler)}">${esc(patch.scheduler)}</option>`); });
    setMany(patch);
    if (!silent) listeners.forEach(f => f('defaults', m));
  }

  function selectModel(id) {
    const m = models().find(x => x.id === id);
    if (!m) return;
    if (state.model !== id) set('model', id);
    populateModels();
  }

  function num(name, fallback) {
    const v = parseFloat(state[name]);
    return Number.isFinite(v) ? v : fallback;
  }

  function size() {
    if (state.preset !== 'custom' && /^\d+x\d+$/.test(state.preset)) { const [w, h] = state.preset.split('x').map(Number); return {w, h}; }
    return {w: Math.round(num('w', 512)), h: Math.round(num('h', 512))};
  }

  // fields every generation request shares
  function requestBody() {
    const body = {model: state.model, prompt: state.prompt.trim(), n: Math.max(1, Math.min(4, Math.round(num('n', 1)))), stream: true, preview: true,
      steps: Math.max(1, Math.round(num('steps', 20))), guidance: num('guidance', 7.5), scheduler: state.scheduler};
    if (state.negative.trim()) body.negative_prompt = state.negative.trim();
    const seed = String(state.seed).trim();
    if (/^-?\d+$/.test(seed)) body.seed = parseInt(seed, 10);
    return body;
  }

  return {state, models, model, mult, set, setMany, bind, populateModels, applyDefaults, selectModel, num, size, requestBody,
    on: f => listeners.push(f), PRESETS, SCHEDULERS};
})();

// ------------------------------------------------------------------ a result slot: preview while generating, the picture afterwards

class ImgSlot {
  constructor(w, h) {
    this.urls = [];
    this.el = document.createElement('figure');
    this.el.className = 'im-slot';
    this.el.style.setProperty('--ar', `${w || 1} / ${h || 1}`);
    this.el.innerHTML = '<img alt="" class="hidden"><div class="im-slot-ph"><span class="spinner"></span><span class="im-slot-ph-text">waiting…</span></div>'
      + '<div class="im-slot-bar hidden"><div></div></div><figcaption class="im-slot-cap hidden"></figcaption><div class="im-slot-actions hidden"></div>';
    this.img = this.el.querySelector('img');
    this.ph = this.el.querySelector('.im-slot-ph');
    this.bar = this.el.querySelector('.im-slot-bar');
    this.cap = this.el.querySelector('.im-slot-cap');
    this.actions = this.el.querySelector('.im-slot-actions');
    this.blob = null;
    this.finalUrl = null;
  }

  _show(blob) {
    const url = URL.createObjectURL(blob);
    this.urls.push(url);
    this.img.src = url;
    this.img.classList.remove('hidden');
    this.ph.classList.add('hidden');
    // keep only the picture on screen alive
    while (this.urls.length > 1) URL.revokeObjectURL(this.urls.shift());
    return url;
  }

  text(t) { this.ph.querySelector('.im-slot-ph-text').textContent = t; }

  preview(blob) {
    if (!blob || this.blob) return;
    this._show(blob);
    this.el.classList.add('preview');
  }

  progress(frac) {
    this.bar.classList.toggle('hidden', frac == null || frac >= 1);
    this.bar.firstElementChild.style.width = Math.round(100 * (frac || 0)) + '%';
  }

  final(blob, caption) {
    this.blob = blob;
    this.finalUrl = this._show(blob);
    this.el.classList.remove('preview');
    this.progress(null);
    this.cap.textContent = caption || '';
    this.cap.classList.toggle('hidden', !caption);
    this.actions.classList.remove('hidden');
    return this.finalUrl;
  }

  fail(msg) {
    this.el.classList.add('failed');
    if (!this.blob) { this.ph.classList.remove('hidden'); this.ph.innerHTML = ''; this.ph.textContent = msg; }
    this.progress(null);
  }

  addAction(label, title, fn) {
    const b = document.createElement('button');
    b.type = 'button';
    b.className = 'small';
    b.textContent = label;
    b.title = title;
    b.setAttribute('aria-label', title);
    b.addEventListener('click', ev => { ev.stopPropagation(); fn(ev); });
    this.actions.appendChild(b);
    return b;
  }

  dispose() {
    this.urls.forEach(u => URL.revokeObjectURL(u));
    this.urls = [];
  }
}
