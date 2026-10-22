'use strict';
// Vision: pictures in the chat. Attach (button, paste, drag and drop), downscale in the browser, keep the pictures in
// IndexedDB (a message stores only their ids, so conversations in localStorage stay small), show thumbnails with a
// lightbox, and turn a message's pictures into OpenAI image_url content parts when a request is built.

const Vision = (() => {
  const MAX_SIDE = 1568;          // longest side sent to the server
  const KEEP_BYTES = 1.5 * 1024 * 1024; // a PNG / JPEG within the size limit and the side limit is sent as it is
  const DB = 'jllm-images', STORE = 'images';
  let pending = [];               // [{id, url}] attached to the composer, not yet sent
  const mem = new Map();          // id -> Blob: the fallback when IndexedDB is unavailable, and a cache
  const urls = new Map();         // id -> object URL for <img>
  const dataUrls = new Map();     // id -> data: URL, filled by prepare() for the request being built
  let dbp = null;

  // ------------------------------------------------------------------ storage

  function db() {
    if (!dbp) {
      dbp = new Promise(resolve => {
        try {
          const rq = indexedDB.open(DB, 1);
          rq.onupgradeneeded = () => rq.result.createObjectStore(STORE);
          rq.onsuccess = () => resolve(rq.result);
          rq.onerror = () => resolve(null);
          rq.onblocked = () => resolve(null);
        } catch { resolve(null); }
      });
    }
    return dbp;
  }

  async function idb(mode, fn) {
    try {
      const d = await db();
      if (!d) return undefined;
      return await new Promise(resolve => {
        try {
          const tx = d.transaction(STORE, mode);
          const rq = fn(tx.objectStore(STORE));
          tx.oncomplete = () => resolve(rq ? rq.result : undefined);
          tx.onerror = tx.onabort = () => resolve(undefined);
        } catch { resolve(undefined); }
      });
    } catch { return undefined; }
  }

  async function putBlob(id, blob) {
    mem.set(id, blob);
    await idb('readwrite', s => s.put(blob, id));
  }

  async function getBlob(id) {
    if (mem.has(id)) return mem.get(id);
    const b = await idb('readonly', s => s.get(id));
    if (b) mem.set(id, b);
    return b || null;
  }

  function objectUrl(id, blob) {
    if (!urls.has(id)) urls.set(id, URL.createObjectURL(blob));
    return urls.get(id);
  }

  const newId = () => 'img-' + Date.now().toString(36) + Math.random().toString(36).slice(2, 8);

  // ------------------------------------------------------------------ downscaling

  const canvasBlob = (cv, type, q) => new Promise(r => cv.toBlob(r, type, q));

  async function shrink(file) {
    const keepable = /^image\/(png|jpeg)$/.test(file.type) && file.size <= KEEP_BYTES;
    let bmp;
    try { bmp = await createImageBitmap(file); } catch { throw new Error(file.name + ' is not a picture this browser can read'); }
    const scale = Math.min(1, MAX_SIDE / Math.max(bmp.width, bmp.height));
    if (scale === 1 && keepable) { bmp.close?.(); return file; }
    const w = Math.max(1, Math.round(bmp.width * scale)), h = Math.max(1, Math.round(bmp.height * scale));
    const cv = document.createElement('canvas');
    cv.width = w; cv.height = h;
    const g = cv.getContext('2d', {willReadFrequently: true});
    g.drawImage(bmp, 0, 0, w, h);
    bmp.close?.();
    let alpha = false;
    if (file.type !== 'image/jpeg') {
      const px = g.getImageData(0, 0, w, h).data;
      for (let i = 3; i < px.length; i += 4) if (px[i] < 255) { alpha = true; break; }
    }
    if (alpha) return await canvasBlob(cv, 'image/png');
    return await canvasBlob(cv, 'image/jpeg', 0.9);
  }

  const toDataUrl = blob => new Promise((resolve, reject) => {
    const r = new FileReader();
    r.onload = () => resolve(r.result);
    r.onerror = () => reject(r.error);
    r.readAsDataURL(blob);
  });

  // ------------------------------------------------------------------ the model

  const model = () => loadedModels.find(m => m.id === $('#chat-model').value);
  const canSee = () => !!model()?.vision;

  function noVision() {
    toast('This model cannot see images: load it with its mmproj (Models tab).', 'err', 7000);
  }

  // ------------------------------------------------------------------ composer

  async function addFiles(files) {
    const pics = [...files].filter(f => f.type.startsWith('image/'));
    if (!pics.length) return;
    if (!canSee()) { noVision(); return; }
    for (const f of pics) {
      try {
        const blob = await shrink(f);
        const id = newId();
        await putBlob(id, blob);
        pending.push({id, url: objectUrl(id, blob)});
      } catch (e) { toast(e.message, 'err'); }
    }
    renderPending();
    $('#input').focus();
  }

  function renderPending() {
    const box = $('#attachments');
    box.classList.toggle('hidden', !pending.length);
    box.innerHTML = pending.map(p => `<span class="vthumb"><img src="${p.url}" alt="attached picture" data-view="${p.id}">`
      + `<button type="button" class="vx" data-rm="${p.id}" title="remove">✕</button></span>`).join('');
    $$('[data-rm]', box).forEach(b => b.addEventListener('click', () => {
      pending = pending.filter(p => p.id !== b.dataset.rm);
      renderPending();
    }));
    $$('[data-view]', box).forEach(i => i.addEventListener('click', () => lightbox(i.src)));
  }

  function updateHint() {
    const btn = $('#attach');
    if (!btn) return;
    const see = canSee();
    btn.classList.toggle('off', !see);
    btn.title = see ? 'Attach pictures (or paste, or drop them here)'
      : 'This model cannot see images: load it with its mmproj (Models tab)';
  }

  function lightbox(src) {
    const o = document.createElement('div');
    o.className = 'vlightbox';
    o.innerHTML = `<img src="${src}" alt="picture">`;
    const close = () => { o.remove(); document.removeEventListener('keydown', onKey, true); };
    const onKey = ev => { if (ev.key === 'Escape') { ev.stopPropagation(); close(); } };
    o.addEventListener('click', close);
    document.addEventListener('keydown', onKey, true);
    document.body.appendChild(o);
  }

  // ------------------------------------------------------------------ messages

  /** Takes the pictures attached to the composer: their ids, in order. */
  function take() {
    const ids = pending.map(p => p.id);
    pending = [];
    renderPending();
    return ids;
  }

  const hasPending = () => pending.length > 0;

  /** Adds thumbnails to a rendered user bubble. */
  function decorate(el, msg) {
    if (!msg.images?.length) return;
    const bubble = el.querySelector('.bubble');
    const strip = document.createElement('div');
    strip.className = 'vimgs';
    bubble.prepend(strip);
    for (const id of msg.images) {
      const img = document.createElement('img');
      img.className = 'vimg';
      img.alt = 'picture';
      strip.appendChild(img);
      getBlob(id).then(b => {
        if (!b) { img.replaceWith(Object.assign(document.createElement('span'), {className: 'vmissing', textContent: 'picture no longer stored'})); return; }
        img.src = objectUrl(id, b);
        img.addEventListener('click', () => lightbox(img.src));
      });
    }
    if (!bubble.textContent.trim() && strip.nextSibling == null) bubble.classList.add('imgs-only');
  }

  /** Loads the data: URLs of every picture in the messages, so that content() can stay synchronous. */
  async function prepare(messages) {
    for (const m of messages) {
      for (const id of m.images || []) {
        if (dataUrls.has(id)) continue;
        const b = await getBlob(id);
        if (b) dataUrls.set(id, await toDataUrl(b));
      }
    }
  }

  /** The message's content for the wire: a string, or OpenAI parts (pictures first, then the text). */
  function content(m) {
    const ids = (m.images || []).filter(id => dataUrls.has(id));
    if (!ids.length) return m.content ?? '';
    const parts = ids.map(id => ({type: 'image_url', image_url: {url: dataUrls.get(id)}}));
    if (m.content) parts.push({type: 'text', text: m.content});
    return parts;
  }

  /** Forgets the pictures of a deleted conversation. */
  function release(messages) {
    for (const m of messages || []) {
      for (const id of m.images || []) {
        mem.delete(id); dataUrls.delete(id);
        if (urls.has(id)) { URL.revokeObjectURL(urls.get(id)); urls.delete(id); }
        idb('readwrite', s => s.delete(id));
      }
    }
  }

  // ------------------------------------------------------------------ wiring

  function init() {
    const box = $('.composer-box');
    const strip = document.createElement('div');
    strip.id = 'attachments';
    strip.className = 'attachments hidden';
    box.prepend(strip);
    const file = document.createElement('input');
    file.type = 'file'; file.accept = 'image/*'; file.multiple = true; file.hidden = true;
    const btn = document.createElement('button');
    btn.type = 'button'; btn.id = 'attach'; btn.className = 'ghost small attach-btn'; btn.textContent = '📎';
    $('.composer-row', box).prepend(btn, file);
    btn.addEventListener('click', () => { if (canSee()) file.click(); else noVision(); });
    file.addEventListener('change', () => { addFiles(file.files); file.value = ''; });

    $('#input').addEventListener('paste', ev => {
      const files = [...(ev.clipboardData?.files || [])].filter(f => f.type.startsWith('image/'));
      if (!files.length) return;
      ev.preventDefault();
      addFiles(files);
    });

    const target = $('#tab-chat .chat');
    let depth = 0;
    const hasFiles = ev => [...(ev.dataTransfer?.types || [])].includes('Files');
    target.addEventListener('dragenter', ev => { if (hasFiles(ev)) { depth++; target.classList.add('drop'); } });
    target.addEventListener('dragleave', ev => { if (hasFiles(ev) && --depth <= 0) { depth = 0; target.classList.remove('drop'); } });
    target.addEventListener('dragover', ev => { if (hasFiles(ev)) ev.preventDefault(); });
    target.addEventListener('drop', ev => {
      if (!hasFiles(ev)) return;
      ev.preventDefault();
      depth = 0; target.classList.remove('drop');
      addFiles(ev.dataTransfer.files);
    });

    modelListeners.push(updateHint);
    $('#chat-model').addEventListener('change', updateHint);
    updateHint();
  }

  init();
  return {take, hasPending, decorate, prepare, content, release, addFiles, canSee};
})();
