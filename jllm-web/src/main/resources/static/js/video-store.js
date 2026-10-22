'use strict';
// Video: the browser-side gallery. IndexedDB (database "jllm-videos", object store "videos") with an in-memory
// fallback when IndexedDB is unavailable (private window, blocked site data). The UI reads the cached list, every
// change is written through to the database.
//
// A record: {id, created, frames (JPEG Blobs, one per frame), thumb (small JPEG Blob of the first frame), file (the clip as the
//            server wrote it, a Blob, or absent), fps, prompt, negative, model,
//            params: {size, width, height, frames, steps, guidance, seed, scheduler, format}}

const VidStore = (() => {
  const DB = 'jllm-videos', STORE = 'videos';
  let db = null;
  let persistent = false;
  let reason = '';
  let cache = [];      // newest first
  let ready = null;
  let warned = false;

  const uid = () => (typeof crypto !== 'undefined' && crypto.randomUUID) ? crypto.randomUUID()
    : Date.now().toString(36) + Math.random().toString(36).slice(2, 10);

  function warnOnce(e) {
    reason = e?.message || String(e || 'unknown error');
    if (!warned) { warned = true; toast('The gallery could not be saved in this browser (' + reason + '); it stays in memory until the page is closed.', 'err', 9000); }
  }

  function openDb() {
    return new Promise(resolve => {
      try {
        if (typeof indexedDB === 'undefined' || !indexedDB) throw new Error('IndexedDB is not available');
        const req = indexedDB.open(DB, 1);
        req.onupgradeneeded = () => {
          const s = req.result.createObjectStore(STORE, {keyPath: 'id'});
          s.createIndex('created', 'created');
        };
        req.onsuccess = () => {
          db = req.result;
          db.onversionchange = () => { db.close(); db = null; persistent = false; reason = 'database was upgraded elsewhere'; };
          persistent = true;
          resolve();
        };
        req.onerror = () => { reason = req.error?.message || 'could not open the database'; resolve(); };
        req.onblocked = () => { reason = 'database is blocked by another tab'; resolve(); };
      } catch (e) { reason = e.message; resolve(); }
    });
  }

  function request(mode, fn) {
    return new Promise((resolve, reject) => {
      try {
        const tx = db.transaction(STORE, mode);
        const out = fn(tx.objectStore(STORE));
        tx.oncomplete = () => resolve(out?.result);
        tx.onerror = () => reject(tx.error || new Error('transaction failed'));
        tx.onabort = () => reject(tx.error || new Error('transaction aborted'));
      } catch (e) { reject(e); }
    });
  }

  function init() {
    if (ready) return ready;
    ready = (async () => {
      await openDb();
      if (persistent) {
        try { cache = (await request('readonly', s => s.getAll())) || []; }
        catch (e) { persistent = false; reason = e.message; cache = []; }
      }
      cache.sort((a, b) => b.created - a.created);
    })();
    return ready;
  }

  async function add(rec) {
    await init();
    rec.id = rec.id || uid();
    rec.created = rec.created || Date.now();
    cache.unshift(rec);
    if (persistent) {
      try { await request('readwrite', s => s.put(rec)); }
      catch (e) { warnOnce(e); }
    }
    return rec;
  }

  async function remove(id) {
    await init();
    cache = cache.filter(r => r.id !== id);
    if (persistent) { try { await request('readwrite', s => s.delete(id)); } catch (e) { warnOnce(e); } }
  }

  async function clear() {
    await init();
    cache = [];
    if (persistent) { try { await request('readwrite', s => s.clear()); } catch (e) { warnOnce(e); } }
  }

  return {
    init, add, remove, clear,
    list: () => cache,
    get: id => cache.find(r => r.id === id),
    isPersistent: () => persistent,
    reason: () => reason,
    bytes: () => cache.reduce((a, r) => a + (r.file?.size || 0) + (r.thumb?.size || 0) + (r.frames || []).reduce((b, f) => b + f.size, 0), 0),
  };
})();
