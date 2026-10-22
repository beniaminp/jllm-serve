'use strict';
// Images: the canvas editor (crop / rotate / flip / resize / adjust, inpainting mask brush, outpainting, undo, zoom)
// and image-to-image / inpaint / outpaint generation from it. Everything but the generation call is client side.
//
// The editor state is {pic, mask}: two canvases of the same size that are never modified once they are part of the
// undo history (an operation makes new canvases; a brush stroke starts by cloning the mask). The mask canvas holds
// a colour with the mask coverage in its alpha channel; it is sent as white (repaint) on black (keep).

const ImgCanvas = (() => {
  const make = (w, h) => { const c = document.createElement('canvas'); c.width = Math.max(1, Math.round(w)); c.height = Math.max(1, Math.round(h)); return c; };
  const clone = c => { const o = make(c.width, c.height); o.getContext('2d').drawImage(c, 0, 0); return o; };
  const filterOk = (() => { try { const g = make(1, 1).getContext('2d'); g.filter = 'blur(1px)'; return g.filter === 'blur(1px)'; } catch { return false; } })();

  // scaling in steps of at most one half keeps downscaled pictures sharp and free of aliasing
  function resize(src, w, h) {
    w = Math.max(1, Math.round(w)); h = Math.max(1, Math.round(h));
    let cur = src;
    while (cur.width > w * 2 || cur.height > h * 2) {
      const nw = cur.width > w * 2 ? Math.ceil(cur.width / 2) : w;
      const nh = cur.height > h * 2 ? Math.ceil(cur.height / 2) : h;
      const t = make(nw, nh);
      const g = t.getContext('2d');
      g.imageSmoothingEnabled = true; g.imageSmoothingQuality = 'high';
      g.drawImage(cur, 0, 0, nw, nh);
      cur = t;
    }
    const out = make(w, h);
    const g = out.getContext('2d');
    g.imageSmoothingEnabled = true; g.imageSmoothingQuality = 'high';
    g.drawImage(cur, 0, 0, w, h);
    return out;
  }

  // gaussian blur through ctx.filter where it exists, else a down / up scale (smooth enough for context fills)
  function blur(src, r) {
    const out = make(src.width, src.height);
    const g = out.getContext('2d');
    if (r < 0.5) { g.drawImage(src, 0, 0); return out; }
    if (filterOk) { g.filter = `blur(${r}px)`; g.drawImage(src, 0, 0); g.filter = 'none'; return out; }
    const f = Math.max(1, r);
    const small = resize(src, Math.max(1, Math.ceil(src.width / f)), Math.max(1, Math.ceil(src.height / f)));
    g.imageSmoothingEnabled = true; g.imageSmoothingQuality = 'high';
    g.drawImage(small, 0, 0, out.width, out.height);
    return out;
  }

  // the union of copies shifted by up to r pixels in every direction
  function dilate(src, r) {
    const out = make(src.width, src.height);
    const g = out.getContext('2d');
    g.drawImage(src, 0, 0);
    if (r < 1) return out;
    for (const rr of [r, r * 0.5]) {
      const steps = Math.max(8, Math.min(40, Math.ceil(rr * 2)));
      for (let k = 0; k < steps; k++) {
        const a = 2 * Math.PI * k / steps;
        g.drawImage(src, Math.cos(a) * rr, Math.sin(a) * rr);
      }
    }
    return out;
  }

  const toBlob = (c, type = 'image/png') => new Promise((res, rej) => c.toBlob(b => b ? res(b) : rej(new Error('could not encode the picture')), type));

  async function fromBlob(blob) {
    let src, release = () => {};
    try { src = await createImageBitmap(blob); release = () => src.close?.(); }
    catch {
      src = await new Promise((res, rej) => {
        const img = new Image();
        const url = URL.createObjectURL(blob);
        img.onload = () => { URL.revokeObjectURL(url); res(img); };
        img.onerror = () => { URL.revokeObjectURL(url); rej(new Error('this file is not a picture the browser can read')); };
        img.src = url;
      });
    }
    const w = src.width, h = src.height;
    const c = make(w, h);
    c.getContext('2d').drawImage(src, 0, 0);
    release();
    return c;
  }

  return {make, clone, resize, blur, dilate, toBlob, fromBlob, filterOk};
})();

const ImgEditor = (() => {
  const HIST = 30;
  const MASK_RGB = '255, 0, 190';
  const MAX_DIM = 4096;      // the largest canvas the editor accepts (bigger pictures are scaled on load)
  const LOAD_MAX = 2048;
  const ASPECTS = {free: 0, '1:1': 1, '4:3': 4 / 3, '3:4': 3 / 4, '3:2': 3 / 2, '2:3': 2 / 3, '16:9': 16 / 9, '9:16': 9 / 16};

  let root, stage, world, cursorEl, statusEl, cropEl, compareEl, compareImg, compareRange, emptyEl, fileInput, stripEl;
  let st = null;                 // {pic, mask}
  let undoStack = [], redoStack = [];
  let view = {z: 1, x: 0, y: 0};
  let fitMode = true;
  let tool = 'brush';
  let crop = null;               // {x, y, w, h} in picture pixels
  let drag = null;
  let spaceDown = false;
  let overStage = false;
  const pointers = new Map();
  const brush = Object.assign({size: 48, hard: 60}, store.get('jllm.img.brush', {}));
  let stampCache = null;
  let results = [];              // [{slot, blob, seed, w, h}]
  let compareIndex = -1;
  let adjust = {b: 100, c: 100, s: 100};
  let sendInfo = null;

  const q = sel => $(sel, root);
  const visible = () => !!root && root.offsetParent !== null;

  // ------------------------------------------------------------------ markup

  function mount(container) {
    root = container;
    root.innerHTML = `
    <div class="im-ed">
      <div class="im-ed-main">
        <div class="im-toolbar" role="toolbar" aria-label="Editor">
          <div class="im-seg" role="group" aria-label="Tool">
            <button type="button" data-tool="pan" title="Move the view (H, or hold Space)" aria-pressed="false">Pan</button>
            <button type="button" data-tool="crop" title="Crop (C)" aria-pressed="false">Crop</button>
            <button type="button" data-tool="brush" title="Paint the inpainting mask (B)" aria-pressed="true">Brush</button>
            <button type="button" data-tool="eraser" title="Erase mask (E)" aria-pressed="false">Eraser</button>
            <button type="button" data-tool="rect" title="Mask a rectangle (R)" aria-pressed="false">Rect</button>
          </div>
          <span class="im-sep"></span>
          <button type="button" id="ed-undo" title="Undo (Ctrl/Cmd+Z)" aria-label="Undo">↶ Undo</button>
          <button type="button" id="ed-redo" title="Redo (Shift+Ctrl/Cmd+Z)" aria-label="Redo">↷ Redo</button>
          <span class="im-sep"></span>
          <button type="button" id="ed-zout" title="Zoom out (-)" aria-label="Zoom out">−</button>
          <button type="button" id="ed-z100" title="Actual size" aria-label="Zoom to 100%">100%</button>
          <button type="button" id="ed-zin" title="Zoom in (+)" aria-label="Zoom in">＋</button>
          <button type="button" id="ed-fit" title="Fit to screen (0)" aria-label="Fit to screen">Fit</button>
          <div class="spacer"></div>
          <button type="button" id="ed-open" title="Open a picture file">Open…</button>
          <button type="button" id="ed-gallery" title="Pick a picture from the gallery">From gallery</button>
          <button type="button" id="ed-export" title="Download the current picture as PNG">Export PNG</button>
          <button type="button" id="ed-save" title="Keep the current picture in the gallery">Save to gallery</button>
          <input type="file" id="ed-file" accept="image/*" class="hidden" aria-label="Picture file">
        </div>
        <div class="im-stage" id="ed-stage" tabindex="0" aria-label="Picture canvas">
          <div class="im-world" id="ed-world"></div>
          <div class="im-brush-cursor hidden" id="ed-cursor"></div>
          <div class="im-stage-empty" id="ed-empty">
            <div><strong>Drop a picture here</strong></div>
            <div class="muted">or paste one (Ctrl/Cmd+V), open a file, pick one from the gallery</div>
            <div class="row" style="justify-content:center"><button type="button" class="primary" id="ed-open2">Open a picture…</button><button type="button" id="ed-gallery2">From gallery</button></div>
          </div>
          <div class="im-compare-bar hidden" id="ed-cmpbar">
            <span>original</span><input type="range" min="0" max="100" value="50" id="ed-cmprange" aria-label="Compare the result with the original"><span>result</span>
            <button type="button" class="small" id="ed-cmpclose">Close</button>
          </div>
        </div>
        <div class="im-ed-status muted small" id="ed-status" aria-live="polite"></div>
        <div class="im-strip-wrap"><div class="im-strip" id="ed-strip" aria-label="Results"></div></div>
      </div>

      <aside class="im-ed-side">
        <details class="im-sec" open id="ed-sec-tool"><summary>Tool</summary>
          <div id="ed-opt-pan" class="im-opt hidden"><p class="hint">Drag to move the picture, scroll or pinch to zoom. Space + drag works in every tool.</p></div>
          <div id="ed-opt-crop" class="im-opt hidden">
            <label>Aspect <select id="ed-aspect" aria-label="Crop aspect ratio">
              <option value="free">free</option><option value="orig">as the picture</option><option>1:1</option><option>4:3</option><option>3:4</option><option>3:2</option><option>2:3</option><option>16:9</option><option>9:16</option></select></label>
            <div class="row"><button type="button" class="primary small" id="ed-crop-apply">Apply crop</button><button type="button" class="small" id="ed-crop-cancel">Cancel</button><button type="button" class="small" id="ed-crop-all">All</button></div>
            <p class="hint" id="ed-crop-info">Drag a rectangle on the picture. Enter applies, Esc cancels.</p>
          </div>
          <div id="ed-opt-brush" class="im-opt">
            <label>Size <span class="muted" id="ed-bsize-v"></span><input type="range" id="ed-bsize" min="2" max="300" aria-label="Brush size"></label>
            <label>Hardness <span class="muted" id="ed-bhard-v"></span><input type="range" id="ed-bhard" min="0" max="100" aria-label="Brush hardness"></label>
            <p class="hint">The painted area (magenta) is what the model redraws in Inpaint / Outpaint mode. [ and ] change the size.</p>
          </div>
          <div id="ed-opt-rect" class="im-opt hidden">
            <label class="check"><input type="checkbox" id="ed-rect-erase"> erase instead of fill</label>
            <p class="hint">Drag a rectangle to add it to the mask.</p>
          </div>
        </details>

        <details class="im-sec" open id="ed-gen"><summary>Generate</summary>
          <div id="ed-nomodel" class="warn-box hidden">No image model is loaded. Load one on the <a href="#models">Models</a> page or from the Create view.</div>
          <label>Mode <select data-p="mode" id="ed-mode" aria-label="Mode">
            <option value="img2img">Image to image</option><option value="inpaint">Inpaint (painted mask)</option><option value="outpaint">Outpaint (after extending)</option></select></label>
          <p class="hint" id="ed-mode-hint"></p>
          <label>Model <select data-p="model" aria-label="Image model"></select></label>
          <label>Prompt <textarea data-p="prompt" rows="3" placeholder="What the result should show" aria-label="Prompt"></textarea></label>
          <label>Negative prompt <textarea data-p="negative" rows="2" placeholder="(optional)" aria-label="Negative prompt"></textarea></label>
          <label>Strength <span class="muted" id="ed-str-v"></span><input type="range" data-p="strength" min="0" max="1" step="0.01" aria-label="Strength: how much is redrawn"></label>
          <div class="im-grid2">
            <label>Steps <input type="number" data-p="steps" min="1" max="150" aria-label="Steps"></label>
            <label>Guidance <input type="number" data-p="guidance" min="0" max="30" step="0.1" aria-label="Guidance"></label>
            <label>Seed <input type="text" inputmode="numeric" data-p="seed" placeholder="random" aria-label="Seed"></label>
            <label>Images <select data-p="n" aria-label="Number of images"><option>1</option><option>2</option><option>3</option><option>4</option></select></label>
          </div>
          <label>Scheduler <select data-p="scheduler" aria-label="Scheduler"></select></label>
          <div class="im-grid2" id="ed-maskopts">
            <label title="Grow the mask by this many pixels (at the size sent to the model)">Mask grow (px) <input type="number" data-p="grow" min="0" max="64" aria-label="Mask grow in pixels"></label>
            <label title="Soften the mask edge by this many pixels">Mask blur (px) <input type="number" data-p="blur" min="0" max="64" aria-label="Mask blur in pixels"></label>
          </div>
          <label>Send at most <select data-p="maxSide" aria-label="Largest side sent to the model">
            <option value="512">512 px</option><option value="768">768 px</option><option value="1024">1024 px</option><option value="1536">1536 px</option><option value="0">original size</option></select></label>
          <p class="hint" id="ed-send-note"></p>
          <div class="row">
            <button type="button" class="primary" id="ed-generate">Generate</button>
            <button type="button" class="danger hidden" id="ed-stop">■ Stop</button>
          </div>
          <div class="im-run hidden" id="ed-run"><div class="progress"><div></div></div><div class="hint" id="ed-run-text"></div></div>
        </details>

        <details class="im-sec" open><summary>Picture</summary>
          <div class="im-btns">
            <button type="button" id="ed-rotl" title="Rotate 90° counter-clockwise" aria-label="Rotate left">⟲ 90°</button>
            <button type="button" id="ed-rotr" title="Rotate 90° clockwise" aria-label="Rotate right">⟳ 90°</button>
            <button type="button" id="ed-fliph" title="Flip horizontally" aria-label="Flip horizontally">⇋ Flip H</button>
            <button type="button" id="ed-flipv" title="Flip vertically" aria-label="Flip vertically">⇅ Flip V</button>
          </div>
          <div class="im-resize">
            <label>Width <input type="number" id="ed-rw" min="1" max="${MAX_DIM}"></label>
            <label>Height <input type="number" id="ed-rh" min="1" max="${MAX_DIM}"></label>
            <label class="check"><input type="checkbox" id="ed-rlock" checked> lock aspect</label>
            <button type="button" id="ed-resize">Resize</button>
          </div>
          <div class="im-btns" id="ed-fit-btns" aria-label="Fit to the model size"></div>
          <p class="hint" id="ed-fit-note"></p>
        </details>

        <details class="im-sec"><summary>Adjust</summary>
          <label>Brightness <span class="muted" id="ed-adj-b-v">100</span><input type="range" id="ed-adj-b" min="0" max="200" value="100" aria-label="Brightness"></label>
          <label>Contrast <span class="muted" id="ed-adj-c-v">100</span><input type="range" id="ed-adj-c" min="0" max="200" value="100" aria-label="Contrast"></label>
          <label>Saturation <span class="muted" id="ed-adj-s-v">100</span><input type="range" id="ed-adj-s" min="0" max="200" value="100" aria-label="Saturation"></label>
          <div class="row"><button type="button" class="primary small" id="ed-adj-apply">Apply</button><button type="button" class="small" id="ed-adj-reset">Reset</button></div>
        </details>

        <details class="im-sec" open><summary>Mask</summary>
          <div class="im-btns">
            <button type="button" id="ed-minv" title="Swap painted and unpainted areas">Invert</button>
            <button type="button" id="ed-mclear" title="Remove the whole mask">Clear</button>
            <button type="button" id="ed-mall" title="Mask the whole picture">Fill all</button>
          </div>
          <p class="hint" id="ed-mask-info"></p>
        </details>

        <details class="im-sec"><summary>Outpaint: extend the canvas</summary>
          <div class="im-ext">
            <label>Extend by <select id="ed-ext-amt" aria-label="Extension in pixels"><option>64</option><option selected>128</option><option>256</option><option>512</option></select> px</label>
            <div class="im-btns">
              <button type="button" data-ext="l" aria-label="Extend left">← left</button><button type="button" data-ext="r" aria-label="Extend right">right →</button>
              <button type="button" data-ext="t" aria-label="Extend up">↑ up</button><button type="button" data-ext="b" aria-label="Extend down">down ↓</button>
              <button type="button" data-ext="a" aria-label="Extend on all sides">all sides</button>
            </div>
            <div class="im-ext4">
              <label>left <input type="number" id="ed-el" min="0" value="0"></label><label>top <input type="number" id="ed-et" min="0" value="0"></label>
              <label>right <input type="number" id="ed-er" min="0" value="0"></label><label>bottom <input type="number" id="ed-eb" min="0" value="0"></label>
            </div>
            <div class="row"><button type="button" id="ed-ext-go">Extend by these values</button>
              <label class="im-inline">overlap <input type="number" id="ed-ext-ov" min="0" max="128" value="16"> px</label></div>
            <p class="hint">The new border is filled with the blurred edge pixels and masked together with a soft overlap into the picture; then use the mode Outpaint under Generate.</p>
          </div>
        </details>

      </aside>
    </div>`;

    stage = q('#ed-stage'); world = q('#ed-world'); cursorEl = q('#ed-cursor'); statusEl = q('#ed-status'); emptyEl = q('#ed-empty');
    fileInput = q('#ed-file'); stripEl = q('#ed-strip');
    cropEl = document.createElement('div'); cropEl.className = 'im-crop hidden';
    cropEl.innerHTML = ['nw', 'n', 'ne', 'e', 'se', 's', 'sw', 'w'].map(h => `<i class="h-${h}"></i>`).join('');
    compareEl = document.createElement('div'); compareEl.className = 'im-compare hidden';
    compareImg = document.createElement('img'); compareImg.alt = 'Generated result'; compareEl.appendChild(compareImg);
    compareRange = q('#ed-cmprange');
    ImgParams.bind(root);
    wire();
    setTool(tool);
    renderState();
  }

  // ------------------------------------------------------------------ state, history, display

  function renderState() {
    world.replaceChildren();
    emptyEl.classList.toggle('hidden', !!st);
    if (st) {
      for (const c of [st.pic, st.mask]) { c.style.width = c.width + 'px'; c.style.height = c.height + 'px'; }
      st.pic.className = 'im-pic'; st.mask.className = 'im-maskc';
      world.style.width = st.pic.width + 'px'; world.style.height = st.pic.height + 'px';
      world.append(st.pic, st.mask, compareEl, cropEl);
      q('#ed-rw').value = st.pic.width; q('#ed-rh').value = st.pic.height;
      adjustPreview();
    }
    updateButtons();
    applyView();
    updateInfo();
  }

  function commit(next) {
    const dimsChanged = !st || next.pic.width !== st.pic.width || next.pic.height !== st.pic.height;
    if (st) { undoStack.push(st); if (undoStack.length > HIST) undoStack.shift(); }
    redoStack = [];
    st = next;
    crop = null;
    closeCompare();
    renderState();
    if (dimsChanged) fit();
  }

  function undo() {
    if (!undoStack.length) return;
    redoStack.push(st);
    const dimsChanged = undoStack[undoStack.length - 1].pic.width !== st.pic.width || undoStack[undoStack.length - 1].pic.height !== st.pic.height;
    st = undoStack.pop();
    crop = null; closeCompare();
    renderState();
    if (dimsChanged) fit();
  }

  function redo() {
    if (!redoStack.length) return;
    undoStack.push(st);
    const next = redoStack.pop();
    const dimsChanged = next.pic.width !== st.pic.width || next.pic.height !== st.pic.height;
    st = next;
    crop = null; closeCompare();
    renderState();
    if (dimsChanged) fit();
  }

  function updateButtons() {
    const has = !!st;
    q('#ed-undo').disabled = !undoStack.length;
    q('#ed-redo').disabled = !redoStack.length;
    for (const id of ['ed-export', 'ed-save', 'ed-rotl', 'ed-rotr', 'ed-fliph', 'ed-flipv', 'ed-resize', 'ed-adj-apply', 'ed-minv', 'ed-mclear', 'ed-mall', 'ed-ext-go', 'ed-zin', 'ed-zout', 'ed-z100', 'ed-fit']) q('#' + id).disabled = !has;
    $$('[data-ext]', root).forEach(b => { b.disabled = !has; });
    refreshGenerateState();
  }

  function applyView() {
    world.style.transform = `translate(${view.x}px, ${view.y}px) scale(${view.z})`;
    world.style.setProperty('--z', view.z);
    stage.classList.toggle('im-pixel', view.z >= 3);
    updateStatus();
    updateCropUi();
  }

  function updateStatus() {
    if (!st) { statusEl.textContent = ''; return; }
    statusEl.textContent = `${st.pic.width} × ${st.pic.height} px · zoom ${Math.round(view.z * 100)}%`
      + (undoStack.length ? ` · ${undoStack.length} undo step${undoStack.length === 1 ? '' : 's'}` : '');
  }

  function updateInfo() {
    if (!st) { q('#ed-mask-info').textContent = ''; return; }
    q('#ed-mask-info').textContent = maskEmpty(st.mask) ? 'The mask is empty.' : 'Magenta = repainted by the model; everything else is kept.';
    updateFitButtons();
    updateSendNote();
  }

  function fit() {
    fitMode = true;
    if (!st) { view = {z: 1, x: 0, y: 0}; applyView(); return; }
    const r = stage.getBoundingClientRect();
    if (r.width < 10 || r.height < 10) return;
    const pad = 16;
    const z = Math.max(0.02, Math.min((r.width - 2 * pad) / st.pic.width, (r.height - 2 * pad) / st.pic.height, 8));
    view = {z, x: (r.width - st.pic.width * z) / 2, y: (r.height - st.pic.height * z) / 2};
    applyView();
  }

  function zoomAt(factor, cx, cy) {
    if (!st) return;
    const r = stage.getBoundingClientRect();
    if (cx == null) { cx = r.width / 2; cy = r.height / 2; }
    const z = Math.min(32, Math.max(0.02, view.z * factor));
    view.x = cx - (cx - view.x) * z / view.z;
    view.y = cy - (cy - view.y) * z / view.z;
    view.z = z;
    fitMode = false;
    applyView();
  }

  function zoomTo(z) {
    const r = stage.getBoundingClientRect();
    zoomAt(z / view.z, r.width / 2, r.height / 2);
  }

  const toPic = e => {
    const r = stage.getBoundingClientRect();
    return {x: (e.clientX - r.left - view.x) / view.z, y: (e.clientY - r.top - view.y) / view.z};
  };

  // ------------------------------------------------------------------ loading pictures

  async function loadBlob(blob, label) {
    try {
      let c = await ImgCanvas.fromBlob(blob);
      let note = '';
      if (Math.max(c.width, c.height) > LOAD_MAX) {
        const s = LOAD_MAX / Math.max(c.width, c.height);
        note = `The picture (${c.width}×${c.height}) was scaled down to fit the editor. `;
        c = ImgCanvas.resize(c, c.width * s, c.height * s);
      }
      commit({pic: c, mask: ImgCanvas.make(c.width, c.height)});
      if (note) toast(note, 'info', 6000);
      return true;
    } catch (e) { toast('Could not open the picture: ' + e.message, 'err'); return false; }
  }

  // ------------------------------------------------------------------ operations

  function rotate(dir) {
    if (!st) return;
    const turn = c => {
      const o = ImgCanvas.make(c.height, c.width);
      const g = o.getContext('2d');
      if (dir > 0) { g.translate(c.height, 0); g.rotate(Math.PI / 2); } else { g.translate(0, c.width); g.rotate(-Math.PI / 2); }
      g.drawImage(c, 0, 0);
      return o;
    };
    commit({pic: turn(st.pic), mask: turn(st.mask)});
  }

  function flip(horizontal) {
    if (!st) return;
    const f = c => {
      const o = ImgCanvas.make(c.width, c.height);
      const g = o.getContext('2d');
      if (horizontal) { g.translate(c.width, 0); g.scale(-1, 1); } else { g.translate(0, c.height); g.scale(1, -1); }
      g.drawImage(c, 0, 0);
      return o;
    };
    commit({pic: f(st.pic), mask: f(st.mask)});
  }

  function resizeTo(w, h) {
    if (!st) return;
    w = Math.round(w); h = Math.round(h);
    if (!(w >= 1 && h >= 1) || w > MAX_DIM || h > MAX_DIM) { toast(`Sizes must be between 1 and ${MAX_DIM} px.`, 'err'); return; }
    if (w === st.pic.width && h === st.pic.height) return;
    commit({pic: ImgCanvas.resize(st.pic, w, h), mask: ImgCanvas.resize(st.mask, w, h)});
  }

  function applyCrop() {
    if (!st || !crop) return;
    const x = Math.max(0, Math.round(crop.x)), y = Math.max(0, Math.round(crop.y));
    const w = Math.min(st.pic.width - x, Math.round(crop.w)), h = Math.min(st.pic.height - y, Math.round(crop.h));
    if (w < 1 || h < 1) { toast('Drag a rectangle first.', 'err'); return; }
    const cut = c => { const o = ImgCanvas.make(w, h); o.getContext('2d').drawImage(c, x, y, w, h, 0, 0, w, h); return o; };
    commit({pic: cut(st.pic), mask: cut(st.mask)});
  }

  function adjustPixels() {
    const {b, c, s} = adjust;
    const o = ImgCanvas.clone(st.pic);
    const g = o.getContext('2d');
    const img = g.getImageData(0, 0, o.width, o.height);
    const d = img.data, bf = b / 100, cf = c / 100, sf = s / 100;
    for (let i = 0; i < d.length; i += 4) {
      let r = d[i] * bf, gg = d[i + 1] * bf, bb = d[i + 2] * bf;
      r = (r - 127.5) * cf + 127.5; gg = (gg - 127.5) * cf + 127.5; bb = (bb - 127.5) * cf + 127.5;
      const y = 0.2126 * r + 0.7152 * gg + 0.0722 * bb;
      r = y + (r - y) * sf; gg = y + (gg - y) * sf; bb = y + (bb - y) * sf;
      d[i] = r < 0 ? 0 : r > 255 ? 255 : r; d[i + 1] = gg < 0 ? 0 : gg > 255 ? 255 : gg; d[i + 2] = bb < 0 ? 0 : bb > 255 ? 255 : bb;
    }
    g.putImageData(img, 0, 0);
    return o;
  }

  function adjustPreview() {
    if (!st) return;
    const {b, c, s} = adjust;
    st.pic.style.filter = (b === 100 && c === 100 && s === 100) ? '' : `brightness(${b / 100}) contrast(${c / 100}) saturate(${s / 100})`;
    for (const k of ['b', 'c', 's']) q(`#ed-adj-${k}-v`).textContent = adjust[k];
  }

  function adjustReset() {
    adjust = {b: 100, c: 100, s: 100};
    for (const k of ['b', 'c', 's']) q(`#ed-adj-${k}`).value = 100;
    adjustPreview();
  }

  function adjustApply() {
    if (!st) return;
    if (adjust.b === 100 && adjust.c === 100 && adjust.s === 100) return;
    const pic = adjustPixels();
    adjustReset();
    commit({pic, mask: st.mask});
  }

  const maskColor = a => `rgba(${MASK_RGB}, ${a})`;

  function maskEmpty(m) {
    const f = Math.min(1, 160 / Math.max(m.width, m.height));
    const small = f < 1 ? ImgCanvas.resize(m, m.width * f, m.height * f) : m;
    const s = ImgCanvas.make(small.width, small.height);   // a scratch canvas made for pixel reads
    const sg = s.getContext('2d', {willReadFrequently: true});
    sg.drawImage(small, 0, 0);
    const d = sg.getImageData(0, 0, s.width, s.height).data;
    for (let i = 3; i < d.length; i += 4) if (d[i] > 3) return false;
    return true;
  }

  function maskOp(kind) {
    if (!st) return;
    const o = ImgCanvas.make(st.mask.width, st.mask.height);
    const g = o.getContext('2d');
    if (kind === 'all' || kind === 'invert') { g.fillStyle = maskColor(1); g.fillRect(0, 0, o.width, o.height); }
    if (kind === 'invert') { g.globalCompositeOperation = 'destination-out'; g.drawImage(st.mask, 0, 0); }
    commit({pic: st.pic, mask: o});
  }

  function extend(l, t, r, b, ov) {
    if (!st) return;
    l = Math.max(0, Math.round(l)); t = Math.max(0, Math.round(t)); r = Math.max(0, Math.round(r)); b = Math.max(0, Math.round(b));
    if (!(l || t || r || b)) { toast('Choose how far to extend the canvas.', 'err'); return; }
    const W = st.pic.width, H = st.pic.height, NW = W + l + r, NH = H + t + b;
    if (NW > MAX_DIM || NH > MAX_DIM) { toast(`The extended canvas would exceed ${MAX_DIM} px.`, 'err'); return; }
    // edge pixels replicated into the border, blurred, then the sharp picture on top
    const bg = ImgCanvas.make(NW, NH);
    const g = bg.getContext('2d');
    g.imageSmoothingEnabled = false;
    const src = st.pic;
    g.drawImage(src, 0, 0, W, H, l, t, W, H);
    if (l) g.drawImage(src, 0, 0, 1, H, 0, t, l, H);
    if (r) g.drawImage(src, W - 1, 0, 1, H, l + W, t, r, H);
    if (t) g.drawImage(src, 0, 0, W, 1, l, 0, W, t);
    if (b) g.drawImage(src, 0, H - 1, W, 1, l, t + H, W, b);
    if (l && t) g.drawImage(src, 0, 0, 1, 1, 0, 0, l, t);
    if (r && t) g.drawImage(src, W - 1, 0, 1, 1, l + W, 0, r, t);
    if (l && b) g.drawImage(src, 0, H - 1, 1, 1, 0, t + H, l, b);
    if (r && b) g.drawImage(src, W - 1, H - 1, 1, 1, l + W, t + H, r, b);
    const radius = Math.max(8, Math.min(64, Math.max(l, r, t, b) / 4));
    const pic = ImgCanvas.blur(bg, radius);
    pic.getContext('2d').drawImage(src, l, t);

    const mask = ImgCanvas.make(NW, NH);
    const m = mask.getContext('2d');
    m.drawImage(st.mask, l, t);
    m.fillStyle = maskColor(1);
    if (l) m.fillRect(0, 0, l, NH);
    if (r) m.fillRect(l + W, 0, r, NH);
    if (t) m.fillRect(0, 0, NW, t);
    if (b) m.fillRect(0, t + H, NW, b);
    ov = Math.max(0, Math.min(Math.round(ov), Math.floor(Math.min(W, H) / 2)));
    if (ov > 0) {
      const strip = (x0, y0, x1, y1, fx, fy, fw, fh) => {
        const gr = m.createLinearGradient(x0, y0, x1, y1);
        gr.addColorStop(0, maskColor(1)); gr.addColorStop(1, maskColor(0));
        m.fillStyle = gr; m.fillRect(fx, fy, fw, fh);
      };
      if (l) strip(l, 0, l + ov, 0, l, 0, ov, NH);
      if (r) strip(l + W, 0, l + W - ov, 0, l + W - ov, 0, ov, NH);
      if (t) strip(0, t, 0, t + ov, 0, t, NW, ov);
      if (b) strip(0, t + H, 0, t + H - ov, 0, t + H - ov, NW, ov);
    }
    commit({pic, mask});
    ImgParams.setMany({mode: 'outpaint', strength: 1});
    toast('Canvas extended and the new border masked: describe the whole picture and press Generate (mode Outpaint).', 'ok', 6000);
  }

  // ------------------------------------------------------------------ brush, rectangle mask, crop (pointer handling)

  function stamp() {
    const d = Math.max(1, Math.round(brush.size));
    if (stampCache && stampCache.d === d && stampCache.h === brush.hard) return stampCache.c;
    const c = ImgCanvas.make(d, d);
    const g = c.getContext('2d');
    const r = d / 2;
    if (brush.hard >= 99) {
      g.fillStyle = maskColor(1);
      g.beginPath(); g.arc(r, r, r, 0, Math.PI * 2); g.fill();
    } else {
      const h = brush.hard / 100;
      const gr = g.createRadialGradient(r, r, 0, r, r, r);
      gr.addColorStop(0, maskColor(1));
      gr.addColorStop(Math.min(0.98, h), maskColor(h > 0.02 ? 1 : 0.9));
      gr.addColorStop(1, maskColor(0));
      g.fillStyle = gr; g.fillRect(0, 0, d, d);
    }
    stampCache = {d, h: brush.hard, c};
    return c;
  }

  function dab(x, y) {
    const s = stamp();
    const g = st.mask.getContext('2d');
    g.globalCompositeOperation = tool === 'eraser' ? 'destination-out' : 'source-over';
    g.globalAlpha = brush.hard >= 99 ? 1 : 0.5;
    g.drawImage(s, x - s.width / 2, y - s.height / 2);
    g.globalAlpha = 1;
    g.globalCompositeOperation = 'source-over';
  }

  function beginStroke(p) {
    commit({pic: st.pic, mask: ImgCanvas.clone(st.mask)});
    drag = {kind: 'stroke', last: p, rest: 0};
    dab(p.x, p.y);
  }

  function moveStroke(p) {
    const dx = p.x - drag.last.x, dy = p.y - drag.last.y;
    const dist = Math.hypot(dx, dy);
    const step = Math.max(1, brush.size * 0.12);
    let t = step - drag.rest;
    while (t <= dist) {
      dab(drag.last.x + dx * t / dist, drag.last.y + dy * t / dist);
      t += step;
    }
    drag.rest = dist - (t - step);
    drag.last = p;
  }

  const clampPt = p => ({x: Math.min(st.pic.width, Math.max(0, p.x)), y: Math.min(st.pic.height, Math.max(0, p.y))});

  function aspectRatio() {
    const v = q('#ed-aspect').value;
    if (v === 'orig') return st ? st.pic.width / st.pic.height : 0;
    return ASPECTS[v] || 0;
  }

  function handleAt(p) {
    if (!crop) return null;
    const tol = 12 / view.z;
    const {x, y, w, h} = crop;
    const near = (a, b) => Math.abs(a - b) <= tol;
    const L = near(p.x, x), R = near(p.x, x + w), T = near(p.y, y), B = near(p.y, y + h);
    const inX = p.x > x - tol && p.x < x + w + tol, inY = p.y > y - tol && p.y < y + h + tol;
    if (T && L) return 'nw'; if (T && R) return 'ne'; if (B && L) return 'sw'; if (B && R) return 'se';
    if (T && inX) return 'n'; if (B && inX) return 's'; if (L && inY) return 'w'; if (R && inY) return 'e';
    if (p.x > x && p.x < x + w && p.y > y && p.y < y + h) return 'move';
    return null;
  }

  function beginCrop(p) {
    const hit = handleAt(p);
    if (hit) drag = {kind: 'crop', mode: hit, start: p, orig: {...crop}};
    else {
      const s = clampPt(p);
      crop = {x: s.x, y: s.y, w: 0, h: 0};
      drag = {kind: 'crop', mode: 'new', start: s, orig: {...crop}};
    }
  }

  function moveCrop(p) {
    const o = drag.orig, W = st.pic.width, H = st.pic.height, ratio = aspectRatio();
    const c = clampPt(p);
    if (drag.mode === 'move') {
      const dx = p.x - drag.start.x, dy = p.y - drag.start.y;
      crop = {x: Math.min(W - o.w, Math.max(0, o.x + dx)), y: Math.min(H - o.h, Math.max(0, o.y + dy)), w: o.w, h: o.h};
    } else if (drag.mode === 'new' || drag.mode.length === 2) {
      // a corner is dragged against the opposite one
      let ax, ay;
      if (drag.mode === 'new') { ax = o.x; ay = o.y; } else {
        ax = drag.mode.includes('w') ? o.x + o.w : o.x; ay = drag.mode.includes('n') ? o.y + o.h : o.y;
      }
      let w = Math.abs(c.x - ax), h = Math.abs(c.y - ay);
      if (ratio) { if (w / Math.max(h, 1e-6) > ratio) w = h * ratio; else h = w / ratio; }
      const sx = c.x < ax ? -1 : 1, sy = c.y < ay ? -1 : 1;
      crop = {x: sx > 0 ? ax : ax - w, y: sy > 0 ? ay : ay - h, w, h};
    } else {
      let x0 = o.x, y0 = o.y, x1 = o.x + o.w, y1 = o.y + o.h;
      if (drag.mode === 'w') x0 = c.x; if (drag.mode === 'e') x1 = c.x; if (drag.mode === 'n') y0 = c.y; if (drag.mode === 's') y1 = c.y;
      let r = {x: Math.min(x0, x1), y: Math.min(y0, y1), w: Math.abs(x1 - x0), h: Math.abs(y1 - y0)};
      if (ratio) {
        if (drag.mode === 'e' || drag.mode === 'w') {
          const cy = o.y + o.h / 2, maxH = 2 * Math.min(cy, H - cy);
          const h = Math.min(r.w / ratio, maxH); r.w = h * ratio; r.h = h; r.y = cy - h / 2;
          if (drag.mode === 'w') r.x = (o.x + o.w) - r.w; else r.x = o.x;
        } else {
          const cx = o.x + o.w / 2, maxW = 2 * Math.min(cx, W - cx);
          const w = Math.min(r.h * ratio, maxW); r.h = w / ratio; r.w = w; r.x = cx - w / 2;
          if (drag.mode === 'n') r.y = (o.y + o.h) - r.h; else r.y = o.y;
        }
      }
      crop = r;
    }
    updateCropUi();
  }

  function updateCropUi() {
    const show = !!crop && tool === 'crop' && crop.w > 0 && crop.h > 0;
    const isRect = drag?.kind === 'rect';
    cropEl.classList.toggle('hidden', !(show || isRect));
    cropEl.classList.toggle('rect', isRect);
    const r = isRect ? drag.rect : crop;
    if (!r) return;
    cropEl.style.left = r.x + 'px'; cropEl.style.top = r.y + 'px'; cropEl.style.width = r.w + 'px'; cropEl.style.height = r.h + 'px';
    const info = q('#ed-crop-info');
    if (info && crop && tool === 'crop') info.textContent = crop.w > 0 ? `${Math.round(crop.w)} × ${Math.round(crop.h)} px at ${Math.round(crop.x)}, ${Math.round(crop.y)}. Enter applies, Esc cancels.`
      : 'Drag a rectangle on the picture. Enter applies, Esc cancels.';
  }

  function endRect() {
    const r = drag.rect;
    if (r.w >= 1 && r.h >= 1) {
      const o = ImgCanvas.clone(st.mask);
      const g = o.getContext('2d');
      if (q('#ed-rect-erase').checked) { g.globalCompositeOperation = 'destination-out'; g.fillStyle = '#000'; } else g.fillStyle = maskColor(1);
      g.fillRect(Math.round(r.x), Math.round(r.y), Math.round(r.w), Math.round(r.h));
      commit({pic: st.pic, mask: o});
    }
  }

  function setTool(t) {
    tool = t;
    $$('[data-tool]', root).forEach(b => b.setAttribute('aria-pressed', String(b.dataset.tool === t)));
    for (const k of ['pan', 'crop', 'brush', 'rect']) q('#ed-opt-' + k).classList.toggle('hidden', !(k === t || (k === 'brush' && t === 'eraser')));
    if (t !== 'crop') crop = null;
    stage.dataset.tool = t;
    stage.style.cursor = '';
    updateCropUi();
    updateCursor();
    if (t === 'brush' || t === 'eraser' || t === 'rect') {
      if (st && ImgParams.state.mode === 'img2img') hintMask();
    }
  }

  let hinted = false;
  function hintMask() {
    if (hinted) return;
    hinted = true;
    toast('Tip: switch the mode below to Inpaint so the painted mask is used.', 'info', 5000);
  }

  function updateCursor() {
    const show = overStage && st && (tool === 'brush' || tool === 'eraser') && !spaceDown;
    cursorEl.classList.toggle('hidden', !show);
    cursorEl.style.width = cursorEl.style.height = Math.max(4, brush.size * view.z) + 'px';
    cursorEl.classList.toggle('erase', tool === 'eraser');
    stage.classList.toggle('im-nocursor', !!show);
  }

  function wireStage() {
    stage.addEventListener('pointerdown', e => {
      if (e.target.closest('.im-compare-bar, .im-stage-empty')) return;   // buttons laid over the canvas
      stage.focus({preventScroll: true});
      if (!st) return;
      if (e.pointerType === 'mouse' && e.button !== 0 && e.button !== 1) return;
      try { stage.setPointerCapture(e.pointerId); } catch { /* ok */ }
      pointers.set(e.pointerId, {x: e.clientX, y: e.clientY});
      if (pointers.size === 2) {
        const [a, b] = [...pointers.values()];
        drag = {kind: 'pinch', d: Math.hypot(a.x - b.x, a.y - b.y) || 1, cx: (a.x + b.x) / 2, cy: (a.y + b.y) / 2, view: {...view}};
        return;
      }
      if (pointers.size > 2) return;
      if (e.button === 1 || tool === 'pan' || spaceDown) { drag = {kind: 'pan', sx: e.clientX, sy: e.clientY, vx: view.x, vy: view.y}; e.preventDefault(); return; }
      const p = toPic(e);
      if (tool === 'brush' || tool === 'eraser') beginStroke(p);
      else if (tool === 'crop') beginCrop(p);
      else if (tool === 'rect') { const s = clampPt(p); drag = {kind: 'rect', start: s, rect: {x: s.x, y: s.y, w: 0, h: 0}}; }
      e.preventDefault();
    });

    stage.addEventListener('pointermove', e => {
      const r = stage.getBoundingClientRect();
      cursorEl.style.left = (e.clientX - r.left) + 'px'; cursorEl.style.top = (e.clientY - r.top) + 'px';
      if (e.pointerType !== 'touch' && !overStage) { overStage = true; updateCursor(); }
      if (pointers.has(e.pointerId)) pointers.set(e.pointerId, {x: e.clientX, y: e.clientY});
      if (!drag) {
        if (st && tool === 'crop') stage.style.cursor = cropCursor(handleAt(toPic(e)));
        return;
      }
      if (drag.kind === 'pinch' && pointers.size >= 2) {
        const [a, b] = [...pointers.values()];
        const d = Math.hypot(a.x - b.x, a.y - b.y) || 1;
        const z = Math.min(32, Math.max(0.02, drag.view.z * d / drag.d));
        const cx = (a.x + b.x) / 2 - r.left, cy = (a.y + b.y) / 2 - r.top;
        const ox = drag.cx - r.left, oy = drag.cy - r.top;
        view.x = cx - (ox - drag.view.x) * z / drag.view.z;
        view.y = cy - (oy - drag.view.y) * z / drag.view.z;
        view.z = z; fitMode = false;
        applyView();
        return;
      }
      if (drag.kind === 'pan') { view.x = drag.vx + e.clientX - drag.sx; view.y = drag.vy + e.clientY - drag.sy; fitMode = false; applyView(); }
      else if (drag.kind === 'stroke') { for (const ev of (e.getCoalescedEvents ? e.getCoalescedEvents() : [e])) moveStroke(toPic(ev)); }
      else if (drag.kind === 'crop') moveCrop(toPic(e));
      else if (drag.kind === 'rect') {
        const c = clampPt(toPic(e));
        drag.rect = {x: Math.min(drag.start.x, c.x), y: Math.min(drag.start.y, c.y), w: Math.abs(c.x - drag.start.x), h: Math.abs(c.y - drag.start.y)};
        updateCropUi();
      }
    });

    const up = e => {
      pointers.delete(e.pointerId);
      try { stage.releasePointerCapture(e.pointerId); } catch { /* ok */ }
      if (!drag) return;
      if (drag.kind === 'pinch') { if (pointers.size < 2) drag = null; return; }
      const d = drag;
      drag = null;
      if (d.kind === 'stroke') updateInfo();
      else if (d.kind === 'rect') { drag = d; endRect(); drag = null; updateCropUi(); }
      else if (d.kind === 'crop') { if (crop && (crop.w < 2 || crop.h < 2)) crop = null; updateCropUi(); }
    };
    stage.addEventListener('pointerup', up);
    stage.addEventListener('pointercancel', up);
    stage.addEventListener('pointerleave', () => { overStage = false; updateCursor(); });
    stage.addEventListener('wheel', e => {
      if (!st) return;
      e.preventDefault();
      const r = stage.getBoundingClientRect();
      zoomAt(Math.exp(-e.deltaY * (e.ctrlKey ? 0.01 : 0.0015)), e.clientX - r.left, e.clientY - r.top);
      updateCursor();
    }, {passive: false});
    stage.addEventListener('dragover', e => { if (e.dataTransfer?.types?.includes('Files')) { e.preventDefault(); stage.classList.add('drop'); } });
    stage.addEventListener('dragleave', () => stage.classList.remove('drop'));
    stage.addEventListener('drop', async e => {
      stage.classList.remove('drop');
      const f = [...(e.dataTransfer?.files || [])].find(x => x.type.startsWith('image/'));
      if (!f) return;
      e.preventDefault();
      await loadBlob(f, f.name);
    });
    new ResizeObserver(() => { if (fitMode && st) fit(); }).observe(stage);
  }

  function cropCursor(h) {
    return {nw: 'nwse-resize', se: 'nwse-resize', ne: 'nesw-resize', sw: 'nesw-resize', n: 'ns-resize', s: 'ns-resize', e: 'ew-resize', w: 'ew-resize', move: 'move'}[h] || 'crosshair';
  }

  // ------------------------------------------------------------------ fitting and the generation size

  function snap(v, mult, max) {
    let r = Math.max(mult, Math.round(v / mult) * mult);
    if (max && r > max) r = Math.max(mult, Math.floor(max / mult) * mult);
    return r;
  }

  // size of the picture as sent to the model: both sides a multiple of the model's, the long side at most maxSide
  function sendSize() {
    const mult = ImgParams.mult();
    const W = st.pic.width, H = st.pic.height;
    const max = Number(ImgParams.state.maxSide) || 0;
    const s = max ? Math.min(1, max / Math.max(W, H)) : 1;
    return {w: snap(W * s, mult, max || 0), h: snap(H * s, mult, max || 0), mult};
  }

  function updateSendNote() {
    const note = q('#ed-send-note');
    if (!st) { note.textContent = ''; return; }
    const {w, h, mult} = sendSize();
    const W = st.pic.width, H = st.pic.height;
    sendInfo = {w, h};
    note.textContent = (w === W && h === H) ? `Sent as ${w}×${h} px (multiples of ${mult}).`
      : `The ${W}×${H} picture is sent as ${w}×${h} px (multiples of ${mult}, long side at most ${Number(ImgParams.state.maxSide) || 'any'}), and the result has that size.`;
  }

  function updateFitButtons() {
    const mult = ImgParams.mult();
    const wrap = q('#ed-fit-btns');
    const sizes = [...new Set([512, 768, 1024].map(s => snap(s, mult)))];
    if (!wrap.dataset.mult || wrap.dataset.mult !== String(mult)) {
      wrap.dataset.mult = mult;
      wrap.innerHTML = sizes.map(s => `<button type="button" data-fit="${s}" title="Scale so the long side is ${s} px and both sides are multiples of ${mult}">Fit ${s}</button>`).join('')
        + `<button type="button" data-fit="snap" title="Resize to the nearest multiples of ${mult}">Snap to ×${mult}</button>`;
      $$('[data-fit]', wrap).forEach(b => b.addEventListener('click', () => {
        if (!st) return;
        const W = st.pic.width, H = st.pic.height;
        if (b.dataset.fit === 'snap') { resizeTo(snap(W, mult), snap(H, mult)); return; }
        const L = Number(b.dataset.fit), s = L / Math.max(W, H);
        resizeTo(snap(W * s, mult), snap(H * s, mult));
      }));
    }
    $$('[data-fit]', wrap).forEach(b => { b.disabled = !st; });
    q('#ed-fit-note').textContent = `The model needs sides that are multiples of ${mult}.`;
  }

  // ------------------------------------------------------------------ generation

  function refreshGenerateState() {
    const busy = ImgRun.busy();
    const noModel = !ImgParams.models().length;
    q('#ed-nomodel')?.classList.toggle('hidden', !noModel);
    const gen = q('#ed-generate');
    if (!gen) return;
    gen.disabled = busy || noModel || !st;
    gen.title = !st ? 'Open a picture first' : noModel ? 'Load an image model first' : '';
    q('#ed-stop').classList.toggle('hidden', !busy || !editorRun);
    gen.classList.toggle('hidden', busy && editorRun);
    const mode = ImgParams.state.mode;
    q('#ed-maskopts').classList.toggle('hidden', mode === 'img2img');
    q('#ed-str-v').textContent = Number(ImgParams.state.strength).toFixed(2);
    q('#ed-mode-hint').textContent = mode === 'img2img'
      ? 'The whole picture is redrawn; strength is how much (0 keeps it, 1 ignores it). The mask is not used.'
      : mode === 'inpaint' ? 'Only the painted (magenta) area is redrawn, the rest is kept exactly. Strength 1 repaints it completely.'
        : 'Extend the canvas first (Outpaint section): the new border and a soft overlap are masked and redrawn.';
  }

  function buildMask(w, h) {
    const P = ImgParams.state;
    let a = ImgCanvas.resize(st.mask, w, h);
    const grow = Math.max(0, Number(P.grow) || 0), blurPx = Math.max(0, Number(P.blur) || 0);
    if (grow > 0) a = ImgCanvas.dilate(a, grow);
    if (blurPx > 0) a = ImgCanvas.blur(a, blurPx);
    const white = ImgCanvas.make(w, h);
    const wg = white.getContext('2d');
    wg.drawImage(a, 0, 0);
    wg.globalCompositeOperation = 'source-in';
    wg.fillStyle = '#fff'; wg.fillRect(0, 0, w, h);
    const out = ImgCanvas.make(w, h);
    const g = out.getContext('2d');
    g.fillStyle = '#000'; g.fillRect(0, 0, w, h);
    g.drawImage(white, 0, 0);
    return out;
  }

  let editorRun = false;

  function clearResults() {
    results.forEach(r => r.slot.dispose());
    results = [];
    stripEl.replaceChildren();
    closeCompare();
  }

  async function generate() {
    if (ImgRun.busy()) return;
    if (!st) { toast('Open a picture first.', 'err'); return; }
    const P = ImgParams.state;
    if (!ImgParams.model()) { toast('Load an image model first.', 'err'); return; }
    if (!P.prompt.trim()) { toast('Write a prompt first.', 'err'); q('[data-p=prompt]').focus(); return; }
    const mode = P.mode;
    if (mode !== 'img2img' && maskEmpty(st.mask)) {
      toast(mode === 'outpaint' ? 'Extend the canvas first (Outpaint section), or use Inpaint with a painted mask.' : 'Paint the area to redraw with the Brush first.', 'err', 6000);
      return;
    }
    const {w, h} = sendSize();
    const body = ImgParams.requestBody();
    body.size = `${w}x${h}`;
    let strength = Number(P.strength);
    if (!(strength >= 0 && strength <= 1)) strength = mode === 'img2img' ? 0.8 : 1;
    body.strength = strength;
    try {
      body.image = ImgCanvas.resize(st.pic, w, h).toDataURL('image/png');
      if (mode !== 'img2img') body.mask = buildMask(w, h).toDataURL('image/png');
    } catch (e) { toast('Could not prepare the picture: ' + e.message, 'err'); return; }

    clearResults();
    const n = body.n;
    const slots = [];
    for (let i = 0; i < n; i++) { const s = new ImgSlot(w, h); slots.push(s); stripEl.appendChild(s.el); }
    const bar = q('#ed-run'), txt = q('#ed-run-text');
    bar.classList.remove('hidden');
    editorRun = true;
    const params = () => ({size: `${w}x${h}`, steps: body.steps, guidance: body.guidance, scheduler: body.scheduler, mode, strength});
    await ImgRun.start('/v1/images/edits', body, {
      slots,
      progress: (t, f) => { txt.textContent = t; bar.querySelector('.progress div').style.width = Math.round(f * 100) + '%'; },
      image: (i, blob, ev) => {
        const s = slots[i];
        s.final(blob, `seed ${ev.seed} · ${ev.width || w}×${ev.height || h}`);
        const rec = {slot: s, blob, seed: ev.seed, w: ev.width || w, h: ev.height || h, saved: false, params: params()};
        results[i] = rec;
        s.el.tabIndex = 0;
        s.img.addEventListener('click', () => compare(i));
        s.el.addEventListener('keydown', e => { if (e.key === 'Enter') compare(i); });
        s.addAction('Compare', 'Compare with the original', () => compare(i));
        s.addAction('Use as input', 'Use this result as the editor picture', () => useResult(i));
        const sv = s.addAction('Save', 'Save to the gallery', async () => { if (rec.saved) return; rec.saved = true; sv.textContent = 'Saved ✓'; await saveResult(rec); });
      },
      end: (err, done) => {
        editorRun = false;
        bar.classList.add('hidden');
        if (err === 'stopped') toast('Stopped.', 'info');
        else if (err) toast(err.message, 'err', 9000);
        else if (done?.timing) txt.textContent = `${done.timing.seconds?.toFixed?.(1) ?? done.timing.seconds} s`;
        if (!err && n === 1 && results[0]) compare(0);
        refreshGenerateState();
        Images.generationEnded?.();
      },
    });
  }

  async function saveResult(rec) {
    await Images.saveBlob(rec.blob, {prompt: ImgParams.state.prompt, negative: ImgParams.state.negative, model: ImgParams.state.model,
      params: {...rec.params, seed: rec.seed, width: rec.w, height: rec.h}});
  }

  function compare(i) {
    const r = results[i];
    if (!r || !st) return;
    if (compareIndex === i && !compareEl.classList.contains('hidden')) { closeCompare(); return; }
    compareIndex = i;
    compareImg.src = r.slot.finalUrl;
    compareImg.style.width = st.pic.width + 'px'; compareImg.style.height = st.pic.height + 'px';
    compareEl.style.width = st.pic.width + 'px'; compareEl.style.height = st.pic.height + 'px';
    compareEl.classList.remove('hidden');
    q('#ed-cmpbar').classList.remove('hidden');
    setCompare(compareRange.value);
    results.forEach((x, k) => x?.slot.el.classList.toggle('selected', k === i));
  }

  function setCompare(v) {
    compareEl.style.clipPath = `inset(0 0 0 ${v}%)`;   // the result shows to the right of the divider
  }

  function closeCompare() {
    compareIndex = -1;
    compareEl?.classList.add('hidden');
    q('#ed-cmpbar')?.classList.add('hidden');
    results.forEach(x => x?.slot.el.classList.remove('selected'));
  }

  async function useResult(i) {
    const r = results[i];
    if (!r) return;
    if (await loadBlob(r.blob, 'result')) toast('The result is now the editor picture (Undo goes back).', 'ok');
  }

  // ------------------------------------------------------------------ wiring

  function wire() {
    wireStage();
    $$('[data-tool]', root).forEach(b => b.addEventListener('click', () => setTool(b.dataset.tool)));
    q('#ed-undo').addEventListener('click', undo);
    q('#ed-redo').addEventListener('click', redo);
    q('#ed-zin').addEventListener('click', () => { zoomAt(1.25); updateCursor(); });
    q('#ed-zout').addEventListener('click', () => { zoomAt(0.8); updateCursor(); });
    q('#ed-z100').addEventListener('click', () => { zoomTo(1); updateCursor(); });
    q('#ed-fit').addEventListener('click', () => { fit(); updateCursor(); });
    const open = () => fileInput.click();
    q('#ed-open').addEventListener('click', open);
    q('#ed-open2').addEventListener('click', open);
    fileInput.addEventListener('change', async () => { const f = fileInput.files[0]; fileInput.value = ''; if (f) await loadBlob(f, f.name); });
    const gal = async () => { const b = await Images.pick(); if (b) await loadBlob(b, 'gallery'); };
    q('#ed-gallery').addEventListener('click', gal);
    q('#ed-gallery2').addEventListener('click', gal);
    q('#ed-export').addEventListener('click', async () => {
      if (!st) return;
      try { ImgApi.saveBlob(await ImgCanvas.toBlob(st.pic), ImgApi.fileName(ImgParams.state.prompt || 'edited', 'edit')); }
      catch (e) { toast(e.message, 'err'); }
    });
    q('#ed-save').addEventListener('click', async () => {
      if (!st) return;
      try {
        await Images.saveBlob(await ImgCanvas.toBlob(st.pic), {prompt: ImgParams.state.prompt || '(edited picture)', negative: '', model: '',
          params: {size: `${st.pic.width}x${st.pic.height}`, width: st.pic.width, height: st.pic.height, mode: 'edit'}});
        toast('Saved to the gallery.', 'ok');
      } catch (e) { toast(e.message, 'err'); }
    });

    q('#ed-rotl').addEventListener('click', () => rotate(-1));
    q('#ed-rotr').addEventListener('click', () => rotate(1));
    q('#ed-fliph').addEventListener('click', () => flip(true));
    q('#ed-flipv').addEventListener('click', () => flip(false));
    const rw = q('#ed-rw'), rh = q('#ed-rh'), lock = q('#ed-rlock');
    rw.addEventListener('input', () => { if (lock.checked && st && rw.value) rh.value = Math.max(1, Math.round(rw.value * st.pic.height / st.pic.width)); });
    rh.addEventListener('input', () => { if (lock.checked && st && rh.value) rw.value = Math.max(1, Math.round(rh.value * st.pic.width / st.pic.height)); });
    q('#ed-resize').addEventListener('click', () => resizeTo(Number(rw.value), Number(rh.value)));

    q('#ed-crop-apply').addEventListener('click', applyCrop);
    q('#ed-crop-cancel').addEventListener('click', () => { crop = null; updateCropUi(); });
    q('#ed-crop-all').addEventListener('click', () => { if (st) { crop = {x: 0, y: 0, w: st.pic.width, h: st.pic.height}; updateCropUi(); } });
    q('#ed-aspect').addEventListener('change', () => {
      const r = aspectRatio();
      if (!r || !crop || !st) return;
      let w = crop.w, h = crop.w / r;
      if (h > crop.h) { h = crop.h; w = h * r; }
      crop = {x: crop.x, y: crop.y, w, h};
      updateCropUi();
    });

    const bs = q('#ed-bsize'), bh = q('#ed-bhard');
    bs.value = brush.size; bh.value = brush.hard;
    const bsync = () => {
      brush.size = Number(bs.value); brush.hard = Number(bh.value);
      q('#ed-bsize-v').textContent = brush.size + ' px'; q('#ed-bhard-v').textContent = brush.hard + '%';
      store.set('jllm.img.brush', brush); updateCursor();
    };
    bs.addEventListener('input', bsync); bh.addEventListener('input', bsync); bsync();

    for (const k of ['b', 'c', 's']) q(`#ed-adj-${k}`).addEventListener('input', e => { adjust[k] = Number(e.target.value); adjustPreview(); });
    q('#ed-adj-apply').addEventListener('click', adjustApply);
    q('#ed-adj-reset').addEventListener('click', adjustReset);

    q('#ed-minv').addEventListener('click', () => maskOp('invert'));
    q('#ed-mclear').addEventListener('click', () => maskOp('clear'));
    q('#ed-mall').addEventListener('click', () => maskOp('all'));

    $$('[data-ext]', root).forEach(b => b.addEventListener('click', () => {
      const a = Number(q('#ed-ext-amt').value), k = b.dataset.ext, ov = Number(q('#ed-ext-ov').value) || 0;
      extend(k === 'l' || k === 'a' ? a : 0, k === 't' || k === 'a' ? a : 0, k === 'r' || k === 'a' ? a : 0, k === 'b' || k === 'a' ? a : 0, ov);
    }));
    q('#ed-ext-go').addEventListener('click', () => extend(Number(q('#ed-el').value), Number(q('#ed-et').value), Number(q('#ed-er').value), Number(q('#ed-eb').value), Number(q('#ed-ext-ov').value) || 0));

    q('#ed-generate').addEventListener('click', generate);
    q('#ed-stop').addEventListener('click', () => ImgRun.stop());
    compareRange.addEventListener('input', () => setCompare(compareRange.value));
    q('#ed-cmpclose').addEventListener('click', closeCompare);

    ImgParams.on(name => {
      if (name === 'mode') {
        // each mode has its own useful strength
        ImgParams.set('strength', ImgParams.state.mode === 'img2img' ? 0.8 : 1);
        if (ImgParams.state.mode === 'inpaint' && tool !== 'brush' && tool !== 'eraser' && tool !== 'rect' && visible()) setTool('brush');
      }
      if (['mode', 'strength', 'model', 'model-list', 'defaults'].includes(name)) refreshGenerateState();
      if (['maxSide', 'model', 'model-list', 'defaults'].includes(name)) { updateSendNote(); updateFitButtons(); }
    });
    ImgRun.onBusy(() => refreshGenerateState());

    document.addEventListener('keydown', onKey);
    document.addEventListener('keyup', e => { if (e.code === 'Space') { spaceDown = false; stage.classList.remove('im-hand'); updateCursor(); } });
    document.addEventListener('paste', async e => {
      if (!visible()) return;
      const f = [...(e.clipboardData?.files || [])].find(x => x.type.startsWith('image/'));
      if (!f) return;
      e.preventDefault();
      await loadBlob(f, 'pasted');
    });
  }

  function onKey(e) {
    if (!visible()) return;
    const el = document.activeElement;
    const typing = el && (el.tagName === 'TEXTAREA' || el.tagName === 'SELECT' || (el.tagName === 'INPUT' && !['range', 'checkbox', 'button'].includes(el.type)));
    const mod = e.ctrlKey || e.metaKey;
    if (mod && !typing && (e.key === 'z' || e.key === 'Z')) { e.preventDefault(); e.shiftKey ? redo() : undo(); return; }
    if (mod && !typing && (e.key === 'y' || e.key === 'Y')) { e.preventDefault(); redo(); return; }
    if (typing || mod || e.altKey) return;
    if (e.code === 'Space' && !e.repeat && (el === stage || el === document.body)) { spaceDown = true; stage.classList.add('im-hand'); updateCursor(); if (el === stage || el === document.body) e.preventDefault(); return; }
    if (e.key === 'Enter' && tool === 'crop' && crop && el === stage) { e.preventDefault(); applyCrop(); return; }
    if (e.key === 'Escape') { if (crop) { crop = null; updateCropUi(); } else if (compareIndex >= 0) closeCompare(); return; }
    const k = e.key.toLowerCase();
    if (k === 'b') setTool('brush'); else if (k === 'e') setTool('eraser'); else if (k === 'c') setTool('crop');
    else if (k === 'h') setTool('pan'); else if (k === 'r') setTool('rect');
    else if (k === '[' || k === ']') {
      const bs = q('#ed-bsize'); bs.value = Math.max(2, Math.min(300, Number(bs.value) * (k === ']' ? 1.2 : 1 / 1.2) + (k === ']' ? 1 : -1))); bs.dispatchEvent(new Event('input'));
    } else if (k === '+' || k === '=') zoomAt(1.25);
    else if (k === '-') zoomAt(0.8);
    else if (k === '0') fit();
  }

  return {
    mount, loadBlob, fit, refresh: () => { refreshGenerateState(); updateSendNote(); updateFitButtons(); },
    onVisible: () => { if (fitMode) fit(); updateCursor(); refreshGenerateState(); },
    hasPicture: () => !!st,
  };
})();
