'use strict';
// The waveform editor of the Audio tab: canvas waveform, selection, playback, edits with undo/redo, and the AI edits that
// use the loaded speech models (transcribe a selection, replace it with speech, edit the audio by editing its transcript).
// The DSP is audio-edit.js; the microphone, playback and server calls are audio-io.js.

const AudioEditor = (() => {
  const E = AudioEdit;
  const RULER = 24, MIN_SPP = 1 / 32, UNDO_MAX = 60, UNDO_BYTES = 600e6;

  const S = {
    clip: null, name: '', sel: null, cursor: 0, head: null, hint: null,
    start: 0, spp: 1,                    // view: first visible frame, frames per CSS pixel
    undo: [], redo: [], clipboard: null, second: null, peaks: null, version: 0, w: 0, h: 0, dpr: 1, colors: null,
  };
  let root, cv, g, pl, raf = 0, drag = null, playFrom = 0, playing = false, recCtl = null, edit = null, rep = null, aiAbort = null;
  const el = id => root.querySelector('#' + id);

  // ------------------------------------------------------------------ markup

  const HTML = `
  <div class="card au-ed">
    <div class="au-toolbar" role="toolbar" aria-label="Transport and view">
      <button id="ed-open">Open…</button><input type="file" id="ed-file" accept="audio/*,video/*,.wav,.mp3,.m4a,.flac,.ogg,.aiff,.aif,.au" class="hidden">
      <button id="ed-rec" aria-pressed="false" title="Record from the microphone: inserted at the cursor, over the selection, or as a new clip">● Record</button>
      <div id="ed-meter" class="au-meter hidden" aria-hidden="true"><div></div></div><span id="ed-rtime" class="au-time hidden"></span>
      <span class="au-sep"></span>
      <button id="ed-play" aria-label="Play" title="Play / pause (Space)">▶</button>
      <button id="ed-stop" aria-label="Stop" title="Stop (Esc)">■</button>
      <label class="check" title="Loop the selection (or from the cursor to the end)"><input type="checkbox" id="ed-loop"> loop</label>
      <span class="au-sep"></span>
      <button id="ed-zout" aria-label="Zoom out" title="Zoom out (-)">－</button><button id="ed-zin" aria-label="Zoom in" title="Zoom in (+)">＋</button>
      <button id="ed-zfit" title="Show the whole clip (0)">Fit</button><button id="ed-zsel" title="Zoom to the selection">Selection</button>
      <span class="au-sep"></span>
      <button id="ed-undo" title="Undo (Ctrl/Cmd+Z)">↶ Undo</button><button id="ed-redo" title="Redo (Shift+Ctrl/Cmd+Z)">↷ Redo</button>
      <div class="spacer"></div>
      <select id="ed-fmt" aria-label="Export format"><option value="16">WAV 16-bit</option><option value="32">WAV 32-bit float</option></select>
      <label class="check"><input type="checkbox" id="ed-expsel"> selection only</label>
      <button id="ed-export" class="primary">Export WAV</button>
    </div>
    <div id="ed-wave" class="au-wave" tabindex="0" aria-label="Waveform. Drag to select, click to place the cursor, Space plays, plus and minus zoom.">
      <canvas id="ed-canvas"></canvas>
      <div id="ed-empty" class="au-wave-empty">Open an audio file, drop one here, record, or send a clip from Speech to text / Text to speech.</div>
    </div>
    <div id="ed-scroll" class="au-scroll" aria-hidden="true"><div id="ed-thumb"></div></div>
    <div class="au-status"><span id="ed-pos" title="cursor / playhead">0:00.000</span><span id="ed-selinfo"></span><span class="spacer"></span><span id="ed-info"></span></div>
  </div>

  <div class="card">
    <h2>Edit <span class="muted small">on the selection, or the whole clip when nothing is selected</span></h2>
    <div class="au-ops">
      <div class="au-op"><h4>Clipboard</h4><div class="row">
        <button id="ed-cut">Cut</button><button id="ed-copy">Copy</button><button id="ed-paste">Paste</button>
        <button id="ed-del">Delete</button><button id="ed-trim" title="Keep only the selection">Trim to selection</button><button id="ed-all">Select all</button></div></div>
      <div class="au-op"><h4>Level</h4><div class="row">
        <input id="ed-gain" type="number" step="0.5" value="3" aria-label="Gain in dB" style="width:70px"><span class="muted small">dB</span><button id="ed-gain-go">Gain</button>
        <button id="ed-fin">Fade in</button><button id="ed-fout">Fade out</button></div>
        <div class="row"><select id="ed-nmode" aria-label="Normalize to"><option value="peak">peak</option><option value="rms">RMS</option></select>
        <input id="ed-ntarget" type="number" step="0.5" value="-1" aria-label="Target in dBFS" style="width:70px"><span class="muted small">dBFS</span><button id="ed-norm">Normalize</button></div></div>
      <div class="au-op"><h4>Time</h4><div class="row">
        <button id="ed-rev">Reverse</button>
        <input id="ed-sil" type="number" step="0.1" min="0" value="1" aria-label="Seconds of silence" style="width:60px"><span class="muted small">s</span><button id="ed-sil-go" title="At the cursor, or at the start of the selection">Insert silence</button></div>
        <div class="row"><input id="ed-speed" type="number" step="0.05" min="0.25" max="4" value="1.25" aria-label="Speed factor" style="width:70px"><span class="muted small">×</span>
        <label class="check" title="WSOLA: changes the duration, keeps the pitch"><input type="checkbox" id="ed-keeppitch" checked> keep pitch</label><button id="ed-speed-go">Change speed</button></div></div>
      <div class="au-op"><h4>Remove silence</h4><div class="row">
        <input id="ed-sthr" type="number" step="1" value="-40" aria-label="Threshold in dB" style="width:56px"><span class="muted small">dB</span>
        <input id="ed-smin" type="number" step="0.1" min="0.05" value="0.5" aria-label="Minimum silence in seconds" style="width:52px"><span class="muted small">min s</span>
        <input id="ed-skeep" type="number" step="0.05" min="0" value="0.1" aria-label="Seconds to keep" style="width:52px"><span class="muted small">keep s</span></div>
        <div class="row"><button id="ed-sprev">Preview</button><button id="ed-sgo">Remove</button><span id="ed-smsg" class="muted small"></span></div></div>
      <div class="au-op"><h4>Mix another clip in at the cursor</h4><div class="row">
        <button id="ed-mfile">Load…</button><input type="file" id="ed-mfile-in" accept="audio/*,.wav,.mp3,.m4a,.flac,.ogg" class="hidden">
        <button id="ed-mspeech" title="The last clip made in Text to speech">Generated speech</button>
        <button id="ed-mstt" title="The clip loaded in Speech to text">Transcribed clip</button></div>
        <div class="row"><span id="ed-mname" class="muted small">no second clip</span></div>
        <div class="row"><label class="au-range">gain <input type="range" id="ed-mgain" min="-36" max="12" step="1" value="-6" aria-label="Gain of the second clip"><output id="ed-mgain-v">-6 dB</output></label>
        <button id="ed-mix">Mix in</button></div></div>
    </div>
  </div>

  <div class="card">
    <h2>With the speech models <span id="ed-ai-note" class="muted small"></span></h2>
    <div class="au-ops">
      <div class="au-op"><h4>Transcribe the selection</h4>
        <div class="row"><button id="ed-ai-tr">Transcribe</button><button id="ed-ai-cancel" class="danger hidden">Cancel</button><span id="ed-ai-busy" class="muted small"></span></div>
        <textarea id="ed-ai-text" rows="3" readonly placeholder="The transcript of the selection appears here."></textarea>
        <div class="row"><button id="ed-ai-copy" class="small">Copy</button><button id="ed-ai-use" class="small" title="Use it as the text of 'Replace with speech'">Use as replacement text</button></div></div>
      <div class="au-op"><h4>Replace the selection with speech</h4>
        <textarea id="ed-rep-text" rows="3" placeholder="What the voice should say instead. Filled with the selection's transcript when a speech-to-text model is loaded."></textarea>
        <div class="row"><label>Voice <select id="ed-rep-voice"></select></label>
          <label>Speed <input id="ed-rep-speed" type="number" min="0.25" max="4" step="0.05" value="1" style="width:70px"></label>
          <label>Length <select id="ed-rep-fit"><option value="none">as spoken</option><option value="stretch">fit the selection (keep pitch)</option><option value="resample">fit the selection (resample)</option></select></label></div>
        <div class="row"><button id="ed-rep-go">Synthesize</button><button id="ed-rep-cancel" class="danger hidden">Cancel</button><span id="ed-rep-busy" class="muted small"></span></div>
        <div id="ed-rep-result" class="hidden"><audio id="ed-rep-audio" controls></audio>
          <div class="row"><button id="ed-rep-apply" class="primary">Replace selection</button><button id="ed-rep-discard" class="ghost">Discard</button></div></div></div>
      <div class="au-op au-wide"><h4>Edit by text</h4>
        <p class="hint">Transcribes the selection (or the whole clip) with word times. Change, delete or add words in the transcript; the new words are spoken by the
          text-to-speech model and spliced in place of the old ones, the rest of the audio stays as it is. The voice will differ from the original speaker.</p>
        <div class="row"><button id="ed-et-go">Transcribe for editing</button><button id="ed-et-cancel" class="danger hidden">Cancel</button><span id="ed-et-busy" class="muted small"></span></div>
        <div id="ed-et" class="hidden">
          <div id="ed-et-words" class="au-words" spellcheck="false" aria-label="Editable transcript"></div>
          <div class="hint">Hover a word to see its place in the waveform; Alt+click plays it. Punctuation and capital letters do not count as changes.</div>
          <div id="ed-et-changes" class="au-changes"></div>
          <div class="row"><label>Voice <select id="ed-et-voice"></select></label><label>Speed <input id="ed-et-speed" type="number" min="0.25" max="4" step="0.05" value="1" style="width:70px"></label>
            <label class="check"><input type="checkbox" id="ed-et-fit"> match the length of replaced words</label>
            <button id="ed-et-prev" class="primary">Preview</button><span id="ed-et-pbusy" class="muted small"></span></div>
          <div id="ed-et-result" class="hidden"><div class="row"><button id="ed-et-play">▶ Play the edited part</button><button id="ed-et-stop">■</button>
            <button id="ed-et-apply" class="primary">Apply (one undo step)</button><button id="ed-et-discard" class="ghost">Discard</button></div></div>
        </div></div>
    </div>
  </div>`;

  // ------------------------------------------------------------------ small helpers

  const sr = () => S.clip.sampleRate;
  const nFrames = () => (S.clip ? E.length(S.clip) : 0);
  const secOf = f => f / sr();
  const target = () => (S.sel ? [S.sel.a, S.sel.b] : [0, nFrames()]);
  const hasSel = () => !!(S.sel && S.sel.b > S.sel.a);
  const bytesOf = c => E.length(c) * c.channels.length * 4;
  const fmtPos = f => E.clock(secOf(f), 3);
  const db = v => (v > 1e-9 ? (20 * Math.log10(v)).toFixed(1) + ' dBFS' : '-∞ dBFS');

  function requireClip() {
    if (!S.clip) { toast('Open or record some audio first.', 'err'); return false; }
    return true;
  }

  // ------------------------------------------------------------------ clip state, undo / redo

  function viewW() { return Math.max(1, S.w); }
  function clampView() {
    const n = nFrames(), span = viewW() * S.spp;
    S.start = n <= span ? 0 : Math.min(Math.max(0, S.start), n - span);
  }
  function fit() {
    const n = nFrames();
    S.needFit = !S.w;      // the pane is hidden (no size yet): fit again when it gets one
    S.spp = Math.max(MIN_SPP, n / viewW());
    S.start = 0;
  }

  function setClip(clip, name) {
    stopPlay(false);
    S.clip = clip; S.peaks = null; S.version++;
    if (name != null) S.name = name;
    updateInfo();
  }

  /** Makes `clip` the current clip as one undo step. */
  function commit(clip, opts = {}) {
    if (S.clip) { S.undo.push(S.clip); S.redo = []; trimUndo(); }
    const oldN = nFrames();
    setClip(clip);
    if (opts.cursor != null) S.cursor = opts.cursor;
    if ('sel' in opts) S.sel = opts.sel;
    const n = nFrames();
    S.cursor = Math.min(S.cursor, n);
    if (S.sel) { S.sel.a = Math.min(S.sel.a, n); S.sel.b = Math.min(S.sel.b, n); if (S.sel.b <= S.sel.a) S.sel = null; }
    if (oldN === 0) fit();
    clampView();
    invalidateEdit();
    refresh();
  }

  function trimUndo() {
    let bytes = S.undo.reduce((s, c) => s + bytesOf(c), 0);
    while (S.undo.length > UNDO_MAX || (bytes > UNDO_BYTES && S.undo.length > 1)) bytes -= bytesOf(S.undo.shift());
  }

  function undo() {
    if (!S.undo.length) return;
    S.redo.push(S.clip);
    setClip(S.undo.pop());
    afterHistory();
  }
  function redo() {
    if (!S.redo.length) return;
    S.undo.push(S.clip);
    setClip(S.redo.pop());
    afterHistory();
  }
  function afterHistory() {
    const n = nFrames();
    S.cursor = Math.min(S.cursor, n);
    if (S.sel) { S.sel.b = Math.min(S.sel.b, n); if (S.sel.b <= S.sel.a) S.sel = null; }
    clampView(); invalidateEdit(); refresh();
  }

  function load(clip, name) {
    if (S.clip && S.undo.length && root && !confirm('Replace the clip in the editor? Your edits so far will be lost (Undo is cleared).')) return false;
    stopPlay(false);
    S.undo = []; S.redo = []; S.sel = null; S.cursor = 0; S.hint = null; S.start = 0;
    S.clip = null;
    setClip(clip, name);
    fit();
    invalidateEdit();
    refresh();
    return true;
  }

  // ------------------------------------------------------------------ drawing

  function readColors() {
    const cs = getComputedStyle(document.documentElement);
    const v = n => cs.getPropertyValue(n).trim();
    S.colors = {bg: v('--panel-2') || '#f0f2f5', wave: v('--accent') || '#5b6cf6', text: v('--text') || '#000', muted: v('--muted') || '#888',
      border: v('--border') || '#ddd', ok: v('--ok') || '#2a2', danger: v('--danger') || '#d44', warn: v('--warn') || '#c80'};
  }

  function resize() {
    if (!cv) return;
    const wrap = el('ed-wave');
    const w = wrap.clientWidth, h = wrap.clientHeight;
    if (!w || !h) return;
    const dpr = window.devicePixelRatio || 1;
    if (w !== S.w || h !== S.h || dpr !== S.dpr || cv.width !== Math.round(w * dpr)) {
      S.w = w; S.h = h; S.dpr = dpr;
      cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr);
    }
    readColors();
    if (S.clip && S.needFit) fit();
    clampView();
    draw();
  }

  const STEPS = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 1800, 3600];
  const xOf = f => (f - S.start) / S.spp;
  const fOf = x => S.start + x * S.spp;

  function draw() {
    if (!g || !S.w) return;
    if (!S.colors) readColors();
    const C = S.colors, w = S.w, h = S.h;
    g.setTransform(S.dpr, 0, 0, S.dpr, 0, 0);
    g.clearRect(0, 0, w, h);
    g.fillStyle = C.bg; g.fillRect(0, 0, w, h);
    const clip = S.clip;
    // ruler
    g.fillStyle = C.border; g.globalAlpha = 0.35; g.fillRect(0, 0, w, RULER); g.globalAlpha = 1;
    g.font = '11px system-ui, sans-serif'; g.textBaseline = 'middle';
    if (clip) {
      const pxPerSec = sr() / S.spp;
      const step = STEPS.find(s => s * pxPerSec >= 90) || 3600;
      const dec = step < 0.01 ? 3 : step < 0.1 ? 2 : step < 1 ? 1 : 0;
      const t0 = Math.floor(secOf(S.start) / step) * step;
      g.strokeStyle = C.muted; g.fillStyle = C.muted; g.lineWidth = 1;
      g.beginPath();
      for (let t = t0; t * pxPerSec - S.start / S.spp <= w + 1; t += step) {
        const x = Math.round(t * sr() / S.spp - S.start / S.spp) + 0.5;
        if (x < -60) continue;
        g.moveTo(x, RULER - 7); g.lineTo(x, RULER);
        g.fillText(E.clock(t, dec), x + 3, RULER / 2 - 2);
        const sub = step / 5;
        for (let k = 1; k < 5; k++) { const xs = Math.round((t + k * sub) * sr() / S.spp - S.start / S.spp) + 0.5; g.moveTo(xs, RULER - 3); g.lineTo(xs, RULER); }
      }
      g.stroke();
    }
    // waveform lanes
    if (clip) {
      const lanes = Math.min(2, clip.channels.length), laneH = (h - RULER) / lanes;
      if (!S.peaks) S.peaks = E.buildPeaks(clip);
      g.strokeStyle = C.wave; g.fillStyle = C.wave;
      for (let k = 0; k < lanes; k++) {
        const mid = RULER + laneH * (k + 0.5), amp = laneH * 0.46;
        g.strokeStyle = C.border; g.lineWidth = 1;
        g.beginPath(); g.moveTo(0, Math.round(mid) + 0.5); g.lineTo(w, Math.round(mid) + 0.5); g.stroke();
        if (lanes > 1 && k > 0) { g.beginPath(); g.moveTo(0, RULER + laneH * k + 0.5); g.lineTo(w, RULER + laneH * k + 0.5); g.stroke(); }
        g.strokeStyle = C.wave; g.fillStyle = C.wave;
        const ch = clip.channels[k];
        if (S.spp < 1) {                            // more than one pixel per sample: a line through the samples
          const i0 = Math.max(0, Math.floor(S.start) - 1), i1 = Math.min(ch.length - 1, Math.ceil(S.start + w * S.spp) + 1);
          g.lineWidth = 1.2; g.beginPath();
          for (let i = i0; i <= i1; i++) {
            const x = xOf(i), y = mid - ch[i] * amp;
            if (i === i0) g.moveTo(x, y); else g.lineTo(x, y);
          }
          g.stroke();
          if (S.spp < 0.25) for (let i = i0; i <= i1; i++) g.fillRect(xOf(i) - 1.5, mid - ch[i] * amp - 1.5, 3, 3);
        } else {
          const p = E.peaksFor(clip, S.peaks, k, S.start, S.start + w * S.spp, w);
          g.beginPath();
          for (let x = 0; x < w; x++) {
            if (S.start + x * S.spp >= ch.length) break;
            const y0 = mid - p.max[x] * amp, y1 = mid - p.min[x] * amp;
            g.rect(x, y0, 1, Math.max(1, y1 - y0));
          }
          g.fill();
        }
      }
      // clipping marks (the lane's top and bottom edge turn red where samples reach full scale)
    }
    // hint (hovered word), selection, cursor, playhead
    if (clip && S.hint) {
      const x0 = xOf(S.hint[0]), x1 = xOf(S.hint[1]);
      g.fillStyle = C.warn; g.globalAlpha = 0.28; g.fillRect(x0, RULER, Math.max(2, x1 - x0), h - RULER); g.globalAlpha = 1;
    }
    if (clip && hasSel()) {
      const x0 = xOf(S.sel.a), x1 = xOf(S.sel.b);
      g.fillStyle = C.wave; g.globalAlpha = 0.22; g.fillRect(x0, 0, Math.max(1, x1 - x0), h); g.globalAlpha = 1;
      g.fillStyle = C.wave; g.fillRect(Math.round(x0), 0, 1.5, h); g.fillRect(Math.round(x1) - 1, 0, 1.5, h);
    } else if (clip) {
      const x = Math.round(xOf(S.cursor)) + 0.5;
      g.strokeStyle = C.text; g.lineWidth = 1; g.beginPath(); g.moveTo(x, 0); g.lineTo(x, h); g.stroke();
    }
    if (clip && S.head != null) {
      const x = Math.round(xOf(S.head)) + 0.5;
      g.strokeStyle = C.danger; g.lineWidth = 2; g.beginPath(); g.moveTo(x, 0); g.lineTo(x, h); g.stroke();
    }
    updateScrollbar();
  }

  function updateScrollbar() {
    const n = nFrames(), thumb = el('ed-thumb'), track = el('ed-scroll');
    if (!n) { thumb.style.width = '0'; return; }
    const span = viewW() * S.spp, frac = Math.min(1, span / n);
    track.classList.toggle('full', frac >= 1);
    thumb.style.width = (frac * 100) + '%';
    thumb.style.left = (n > span ? (S.start / n) * 100 : 0) + '%';
  }

  function updateStatus() {
    const f = S.head != null ? S.head : S.cursor;
    el('ed-pos').textContent = S.clip ? fmtPos(f) : '0:00.000';
    el('ed-selinfo').textContent = hasSel() ? `selection ${fmtPos(S.sel.a)} – ${fmtPos(S.sel.b)} (${E.clock(secOf(S.sel.b - S.sel.a), 3)})` : '';
  }

  let infoPeak = 0;
  function updateInfo() {
    if (!S.clip) { el('ed-info').textContent = ''; return; }
    infoPeak = E.peak(S.clip);
    const c = S.clip;
    el('ed-info').textContent = `${S.name ? S.name + ' · ' : ''}${E.clock(E.duration(c), 3)} · ${c.sampleRate} Hz · ${c.channels.length === 1 ? 'mono' : c.channels.length === 2 ? 'stereo' : c.channels.length + ' channels'} · peak ${db(infoPeak)}${infoPeak > 1 ? ' (clips!)' : ''}`;
  }

  // ------------------------------------------------------------------ enable / disable

  function refresh() {
    const has = !!S.clip, sel = hasSel();
    el('ed-empty').classList.toggle('hidden', has);
    const set = (id, on, why) => { const b = el(id); if (b) { b.disabled = !on; if (why !== undefined) b.title = on ? (b.dataset.title || '') : why; } };
    for (const id of ['ed-play', 'ed-stop', 'ed-zin', 'ed-zout', 'ed-zfit', 'ed-export', 'ed-all', 'ed-gain-go', 'ed-fin', 'ed-fout', 'ed-norm', 'ed-rev',
      'ed-sil-go', 'ed-speed-go', 'ed-sprev', 'ed-sgo', 'ed-paste']) set(id, has);
    set('ed-zsel', sel); set('ed-cut', sel); set('ed-copy', sel); set('ed-del', sel); set('ed-trim', sel);
    set('ed-paste', has && !!S.clipboard);
    set('ed-undo', S.undo.length > 0); set('ed-redo', S.redo.length > 0);
    set('ed-mix', has && !!S.second);
    set('ed-mspeech', !!AudioTab.shared.speechClip); set('ed-mstt', !!AudioTab.shared.sttClip);
    modelsChanged();
    updateStatus();
    resize();
  }

  function modelsChanged() {
    if (!root) return;
    const stt = AudioState.hasStt(), tts = AudioState.hasTts(), has = !!S.clip, sel = hasSel();
    const no = (id, on, why) => { const b = el(id); b.disabled = !on; b.title = on ? '' : why; };
    no('ed-ai-tr', has && stt && !aiAbort, !stt ? 'Load a speech-to-text model (Whisper) on the Models page' : 'Open some audio first');
    no('ed-rep-go', has && tts && !rep?.busy, !tts ? 'Load a text-to-speech model on the Models page' : 'Open some audio first');
    no('ed-et-go', has && stt && tts, !stt || !tts ? 'Needs a speech-to-text and a text-to-speech model (Models page)' : 'Open some audio first');
    el('ed-ai-note').textContent = [!stt && 'no speech-to-text model loaded', !tts && 'no text-to-speech model loaded'].filter(Boolean).join(' · ');
    el('ed-rep-apply').textContent = sel ? 'Replace selection' : 'Insert at cursor';
    fillVoices('ed-rep-voice'); fillVoices('ed-et-voice');
  }

  function fillVoices(id) {
    const sel = el(id), m = AudioState.tts()[0], voices = m?.voices || [];
    const html = '<option value="">(default)</option>' + voices.map(v => `<option value="${esc(v)}">${esc(v)}</option>`).join('');
    if (sel.innerHTML !== html) { const keep = sel.value || store.get('jllm.audio', {}).ttsVoice || ''; sel.innerHTML = html; sel.value = voices.includes(keep) ? keep : ''; }
  }

  // ------------------------------------------------------------------ view

  function zoom(factor, anchorFrame, anchorX) {
    if (!S.clip) return;
    const n = nFrames();
    const old = S.spp;
    S.spp = Math.min(Math.max(MIN_SPP, old * factor), Math.max(MIN_SPP, n / viewW()));
    if (anchorFrame == null) {
      anchorFrame = S.head != null ? S.head : hasSel() ? (S.sel.a + S.sel.b) / 2 : S.cursor;
      anchorX = xOf(anchorFrame);
      if (anchorX < 0 || anchorX > viewW()) { anchorX = viewW() / 2; }
    }
    S.start = anchorFrame - anchorX * S.spp;
    clampView(); draw();
  }
  function zoomSel() {
    if (!hasSel()) return;
    const len = S.sel.b - S.sel.a;
    S.spp = Math.max(MIN_SPP, len * 1.1 / viewW());
    S.start = S.sel.a - len * 0.05;
    clampView(); draw();
  }
  function pan(frames) { S.start += frames; clampView(); draw(); }
  function reveal(frame) {
    const x = xOf(frame);
    if (x < 0 || x > viewW()) { S.start = frame - viewW() * S.spp * 0.25; clampView(); }
  }

  // ------------------------------------------------------------------ pointer interaction

  function frameAtEvent(ev) {
    const r = cv.getBoundingClientRect();
    return Math.min(nFrames(), Math.max(0, Math.round(fOf(ev.clientX - r.left))));
  }

  function onDown(ev) {
    if (!S.clip || ev.button !== 0) return;
    el('ed-wave').focus({preventScroll: true});
    cv.setPointerCapture(ev.pointerId);
    const r = cv.getBoundingClientRect(), x = ev.clientX - r.left, f = frameAtEvent(ev);
    drag = {x0: x, mode: 'new', anchor: f, moved: false};
    if (hasSel()) {
      if (Math.abs(x - xOf(S.sel.a)) < 6) { drag.mode = 'edge'; drag.anchor = S.sel.b; }
      else if (Math.abs(x - xOf(S.sel.b)) < 6) { drag.mode = 'edge'; drag.anchor = S.sel.a; }
    }
    if (drag.mode === 'new' && ev.shiftKey) { drag.anchor = hasSel() ? (Math.abs(f - S.sel.a) < Math.abs(f - S.sel.b) ? S.sel.b : S.sel.a) : S.cursor; drag.moved = true; setSelFrom(drag.anchor, f); }
  }
  function setSelFrom(a, b) { S.sel = a === b ? null : {a: Math.min(a, b), b: Math.max(a, b)}; }
  function onMove(ev) {
    const r = cv.getBoundingClientRect(), x = ev.clientX - r.left;
    if (!drag) {
      if (S.clip) cv.style.cursor = hasSel() && (Math.abs(x - xOf(S.sel.a)) < 6 || Math.abs(x - xOf(S.sel.b)) < 6) ? 'ew-resize' : 'text';
      return;
    }
    if (!drag.moved && Math.abs(x - drag.x0) < 4 && drag.mode === 'new') return;
    drag.moved = true;
    if (x < 0) pan(x * S.spp * 0.3); else if (x > viewW()) pan((x - viewW()) * S.spp * 0.3);
    setSelFrom(drag.anchor, frameAtEvent(ev));
    updateStatus(); draw();
  }
  function onUp(ev) {
    if (!drag) return;
    const d = drag; drag = null;
    if (!d.moved && d.mode === 'edge') return;
    if (!d.moved) {
      S.cursor = frameAtEvent(ev); S.sel = null;
      if (playing) startPlay(S.cursor);
    } else if (hasSel() && (S.sel.b - S.sel.a) < 2) S.sel = null;
    refresh();
  }

  function onWheel(ev) {
    if (!S.clip) return;
    const r = cv.getBoundingClientRect(), x = ev.clientX - r.left;
    if (ev.ctrlKey || ev.metaKey || ev.altKey) {
      ev.preventDefault();
      zoom(Math.exp(ev.deltaY * 0.01), fOf(x), x);
    } else if (Math.abs(ev.deltaX) > Math.abs(ev.deltaY) || ev.shiftKey) {
      ev.preventDefault();
      pan((ev.deltaX || ev.deltaY) * S.spp);
    }
  }

  function initScrollbar() {
    const track = el('ed-scroll'), thumb = el('ed-thumb');
    let d = null;
    thumb.addEventListener('pointerdown', ev => { ev.preventDefault(); thumb.setPointerCapture(ev.pointerId); d = {x: ev.clientX, start: S.start}; });
    thumb.addEventListener('pointermove', ev => {
      if (!d) return;
      S.start = d.start + (ev.clientX - d.x) / track.clientWidth * nFrames();
      clampView(); draw();
    });
    thumb.addEventListener('pointerup', () => { d = null; });
    track.addEventListener('pointerdown', ev => {
      if (ev.target !== track || !S.clip) return;
      const f = (ev.clientX - track.getBoundingClientRect().left) / track.clientWidth * nFrames();
      S.start = f - viewW() * S.spp / 2; clampView(); draw();
    });
  }

  // ------------------------------------------------------------------ playback

  function playbackSpan() {
    const n = nFrames();
    if (hasSel()) return [S.sel.a, S.sel.b, el('ed-loop').checked];
    return [Math.min(S.cursor, n), n, el('ed-loop').checked];
  }
  function startPlay(fromFrame) {
    if (!requireClip()) return;
    let [a, b, loop] = playbackSpan();
    if (fromFrame != null && !hasSel()) a = fromFrame;
    if (fromFrame != null && hasSel() && fromFrame >= S.sel.a && fromFrame < S.sel.b) a = fromFrame;
    if (a >= b - 1) a = 0;
    playFrom = hasSel() ? S.sel.a : a;
    try {
      if (!pl.play(S.clip, secOf(a), secOf(b), loop)) { toast('Nothing to play there.'); return; }
    } catch (e) { toast(e.message, 'err'); return; }
    playing = true;
    el('ed-play').textContent = '⏸'; el('ed-play').setAttribute('aria-label', 'Pause');
    cancelAnimationFrame(raf);
    raf = requestAnimationFrame(tick);
  }
  function tick() {
    const p = pl.position();
    if (p == null) { raf = 0; return; }
    S.head = p * sr();
    const x = xOf(S.head);
    if (x > viewW() * 0.95 || x < 0) { S.start = S.head - viewW() * S.spp * 0.1; clampView(); }
    updateStatus(); draw();
    raf = requestAnimationFrame(tick);
  }
  function stopPlay(returnCursor) {
    const pos = pl ? pl.stop() : null;
    cancelAnimationFrame(raf); raf = 0;
    if (playing) { el('ed-play').textContent = '▶'; el('ed-play').setAttribute('aria-label', 'Play'); }
    playing = false;
    S.head = null;
    if (pos != null && S.clip) S.cursor = returnCursor ? playFrom : Math.min(nFrames(), Math.round(pos * sr()));
    if (root) { updateStatus(); draw(); }
  }
  function togglePlay() {
    if (!requireClip()) return;
    if (playing) {
      const pos = pl.position();
      stopPlay(false);
      if (pos != null && !hasSel()) S.cursor = Math.round(pos * sr());
      refresh();
    } else startPlay();
  }

  // ------------------------------------------------------------------ operations

  function op(label, fn) {
    if (!requireClip()) return;
    try { fn(); } catch (e) { console.error(e); toast(label + ': ' + e.message, 'err'); }
  }

  function doCut() {
    if (!hasSel()) return;
    const [a, b] = target();
    S.clipboard = E.slice(S.clip, a, b);
    commit(E.deleteRange(S.clip, a, b), {cursor: a, sel: null});
    toast(`Cut ${E.clock(secOf(b - a), 2)}`, 'ok', 1500);
  }
  function doCopy() {
    if (!hasSel()) return;
    S.clipboard = E.slice(S.clip, S.sel.a, S.sel.b);
    refresh();
    toast(`Copied ${E.clock(E.duration(S.clipboard), 2)}`, 'ok', 1500);
  }
  function doPaste() {
    if (!S.clipboard) return;
    const piece = E.conform(S.clipboard, S.clip);
    if (hasSel()) commit(E.replace(S.clip, S.sel.a, S.sel.b, piece), {cursor: S.sel.a + E.length(piece), sel: null});
    else commit(E.insert(S.clip, S.cursor, piece), {cursor: S.cursor + E.length(piece), sel: null});
  }
  function doDelete() {
    if (!hasSel()) return;
    const [a, b] = target();
    commit(E.deleteRange(S.clip, a, b), {cursor: a, sel: null});
  }
  function doTrim() {
    if (!hasSel()) return;
    const [a, b] = target();
    commit(E.slice(S.clip, a, b), {cursor: 0, sel: null});
    fit(); draw();
  }
  const numVal = (id, def) => { const v = parseFloat(el(id).value); return Number.isFinite(v) ? v : def; };

  function bindOps() {
    const on = (id, fn) => el(id).addEventListener('click', fn);
    on('ed-open', () => el('ed-file').click());
    el('ed-file').addEventListener('change', async ev => { const f = ev.target.files[0]; ev.target.value = ''; if (f) await openFile(f); });
    on('ed-play', togglePlay);
    on('ed-stop', () => { stopPlay(true); refresh(); });
    on('ed-zin', () => zoom(0.5)); on('ed-zout', () => zoom(2)); on('ed-zfit', () => { fit(); draw(); }); on('ed-zsel', zoomSel);
    on('ed-undo', undo); on('ed-redo', redo);
    on('ed-cut', doCut); on('ed-copy', doCopy); on('ed-paste', doPaste); on('ed-del', doDelete); on('ed-trim', doTrim);
    on('ed-all', () => { if (S.clip) { S.sel = {a: 0, b: nFrames()}; refresh(); } });
    on('ed-gain-go', () => op('Gain', () => { const [a, b] = target(); commit(E.gain(S.clip, a, b, numVal('ed-gain', 0))); }));
    on('ed-fin', () => op('Fade in', () => { const [a, b] = target(); commit(E.fade(S.clip, a, b, 'in')); }));
    on('ed-fout', () => op('Fade out', () => { const [a, b] = target(); commit(E.fade(S.clip, a, b, 'out')); }));
    on('ed-norm', () => op('Normalize', () => {
      const [a, b] = target();
      const mode = el('ed-nmode').value, tgt = numVal('ed-ntarget', -1);
      if (E.peak(S.clip, a, b) < 1e-9) { toast('That part is silent.', 'err'); return; }
      commit(E.normalize(S.clip, a, b, mode, tgt));
      if (mode === 'rms' && E.peak(S.clip) > 1) toast('Normalized to RMS: the peaks now exceed 0 dBFS and will be clamped on export. Lower the target or use a fade/limiter.', 'info', 7000);
    }));
    on('ed-rev', () => op('Reverse', () => { const [a, b] = target(); commit(E.reverse(S.clip, a, b)); }));
    on('ed-sil-go', () => op('Silence', () => {
      const n = Math.round(Math.max(0, numVal('ed-sil', 1)) * sr());
      if (!n) return;
      const at = hasSel() ? S.sel.a : S.cursor;
      commit(E.insertSilence(S.clip, at, n), {cursor: at + n, sel: null});
    }));
    on('ed-speed-go', () => op('Speed', () => {
      const f = numVal('ed-speed', 1);
      if (!(f >= 0.25 && f <= 4)) { toast('Speed must be between 0.25 and 4.', 'err'); return; }
      const [a, b] = target();
      const out = el('ed-keeppitch').checked ? E.timeStretch(S.clip, a, b, f) : E.changeSpeed(S.clip, a, b, f);
      const newB = a + Math.round((b - a) / f);
      commit(out, {sel: hasSel() ? {a, b: newB} : null});
    }));
    const silParams = () => [numVal('ed-sthr', -40), Math.max(0.01, numVal('ed-smin', 0.5)), Math.max(0, numVal('ed-skeep', 0.1))];
    on('ed-sprev', () => op('Silence', () => {
      const [a, b] = target(), [thr, min, keep] = silParams();
      const r = E.removeSilence(S.clip, a, b, thr, min, keep);
      el('ed-smsg').textContent = r.count ? `${r.count} silence${r.count === 1 ? '' : 's'}, ${E.clock(secOf(r.removedFrames), 2)} would be removed` : 'no silence found with these settings';
    }));
    on('ed-sgo', () => op('Remove silence', () => {
      const [a, b] = target(), [thr, min, keep] = silParams();
      const r = E.removeSilence(S.clip, a, b, thr, min, keep);
      if (!r.count) { el('ed-smsg').textContent = 'no silence found with these settings'; return; }
      commit(r.clip, {cursor: Math.min(S.cursor, E.length(r.clip)), sel: null});
      el('ed-smsg').textContent = `removed ${E.clock(secOf(r.removedFrames), 2)}`;
    }));
    // second clip and mixing
    const setSecond = (clip, name) => { S.second = {clip, name}; el('ed-mname').textContent = `${name} · ${E.clock(E.duration(clip), 2)}`; refresh(); };
    on('ed-mfile', () => el('ed-mfile-in').click());
    el('ed-mfile-in').addEventListener('change', async ev => {
      const f = ev.target.files[0]; ev.target.value = '';
      if (!f) return;
      try { setSecond(await AudioIO.decode(f), f.name); } catch (e) { toast(e.message, 'err'); }
    });
    on('ed-mspeech', () => { if (AudioTab.shared.speechClip) setSecond(AudioTab.shared.speechClip, 'generated speech'); });
    on('ed-mstt', () => { if (AudioTab.shared.sttClip) setSecond(AudioTab.shared.sttClip, AudioTab.shared.sttName || 'transcribed clip'); });
    el('ed-mgain').addEventListener('input', () => { el('ed-mgain-v').textContent = el('ed-mgain').value + ' dB'; });
    on('ed-mix', () => op('Mix', () => {
      if (!S.second) return;
      commit(E.mix(S.clip, S.second.clip, S.cursor, +el('ed-mgain').value), {cursor: S.cursor});
      if (infoPeak > 1) toast('The mix is louder than 0 dBFS; lower its gain or normalize.', 'info', 6000);
    }));
    on('ed-export', exportWav);
  }

  async function openFile(file) {
    try {
      toast('Decoding ' + file.name + '…', 'info', 1500);
      const clip = await AudioIO.decode(file);
      load(clip, AudioTab.baseName(file.name));
    } catch (e) { toast(e.message, 'err', 7000); }
  }

  function exportWav() {
    if (!requireClip()) return;
    const bits = +el('ed-fmt').value;
    let clip = S.clip;
    if (el('ed-expsel').checked && hasSel()) clip = E.slice(clip, S.sel.a, S.sel.b);
    if (bits === 16 && E.peak(clip) > 1) toast('The audio exceeds 0 dBFS and will be clamped (distorted). Normalize first.', 'info', 6000);
    AudioIO.saveBlob((S.name || 'edited') + (bits === 32 ? '-f32' : '') + '.wav', new Blob([E.encodeWav(clip, bits)], {type: 'audio/wav'}));
  }

  // ------------------------------------------------------------------ keyboard

  function onKey(ev) {
    if (activeTab !== 'audio' || !root || root.classList.contains('hidden')) return;
    const t = ev.target;
    const typing = t && (t.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(t.tagName) && !(t.type === 'checkbox' || t.type === 'range'));
    const interactive = t && /^(BUTTON|A|SUMMARY|INPUT)$/.test(t.tagName);
    const mod = ev.metaKey || ev.ctrlKey;
    if (typing) return;
    const key = ev.key.toLowerCase();
    let handled = true;
    if (mod && key === 'z') { ev.shiftKey ? redo() : undo(); }
    else if (mod && key === 'y') redo();
    else if (mod && key === 'x') doCut();
    else if (mod && key === 'c') doCopy();
    else if (mod && key === 'v') doPaste();
    else if (mod && key === 'a') { if (S.clip) { S.sel = {a: 0, b: nFrames()}; refresh(); } }
    else if (mod || ev.altKey) handled = false;
    else if (key === ' ' && !interactive) togglePlay();
    else if (key === 'delete' || key === 'backspace') doDelete();
    else if (key === '+' || key === '=') zoom(0.5);
    else if (key === '-' || key === '_') zoom(2);
    else if (key === '0') { fit(); draw(); }
    else if (key === 'home') { S.cursor = 0; S.sel = null; reveal(0); refresh(); }
    else if (key === 'end') { S.cursor = nFrames(); S.sel = null; reveal(S.cursor); refresh(); }
    else if (key === 'escape') { if (playing) stopPlay(true); else S.sel = null; refresh(); }
    else if ((key === 'arrowleft' || key === 'arrowright') && S.clip && !interactive) {
      const d = (key === 'arrowleft' ? -1 : 1) * Math.max(1, Math.round(S.spp * 10));
      const nc = Math.min(nFrames(), Math.max(0, (hasSel() && !ev.shiftKey ? (d < 0 ? S.sel.a : S.sel.b) : S.cursor) + d));
      if (ev.shiftKey) setSelFrom(hasSel() ? (S.cursor === S.sel.a ? S.sel.b : S.sel.a) : S.cursor, nc);
      S.cursor = nc; if (!ev.shiftKey) S.sel = null;
      reveal(nc); refresh();
    }
    else handled = false;
    if (handled) ev.preventDefault();
  }

  // ------------------------------------------------------------------ recording into the editor

  function initRecording() {
    recCtl = AudioIO.bindRecorder({button: el('ed-rec'), meter: el('ed-meter'), time: el('ed-rtime'), raw: true, labelIdle: '● Record',
      onClip: clip => {
        const name = 'recording';
        if (!S.clip) { load(clip, name); return; }
        const piece = E.conform(clip, S.clip);
        if (hasSel()) commit(E.replace(S.clip, S.sel.a, S.sel.b, piece), {cursor: S.sel.a + E.length(piece), sel: null});
        else commit(E.insert(S.clip, S.cursor, piece), {cursor: S.cursor + E.length(piece)});
      }});
  }

  // ------------------------------------------------------------------ AI edits

  const voiceOf = id => el(id).value;
  function wav16k(a, b) { return AudioIO.toWavBlob(E.slice(S.clip, a, b), 16000); }
  function busyText(id, text) { el(id).innerHTML = text ? `<span class="spinner"></span> ${esc(text)}` : ''; }
  function sttModelId() { return AudioState.stt()[0]?.id; }
  function ttsModelId() { return AudioState.tts()[0]?.id; }

  // (a) transcribe the selection
  async function aiTranscribe() {
    if (!requireClip() || aiAbort) return;
    const [a, b] = target();
    aiAbort = new AbortController();
    el('ed-ai-cancel').classList.remove('hidden'); modelsChanged();
    busyText('ed-ai-busy', 'transcribing…');
    try {
      const r = await AudioIO.transcribe(wav16k(a, b), {model: sttModelId(), signal: aiAbort.signal});
      el('ed-ai-text').value = r.text || '(no speech detected)';
    } catch (e) {
      if (e.name !== 'AbortError') { toast(e.message, 'err', 7000); el('ed-ai-text').value = ''; }
    } finally {
      aiAbort = null; busyText('ed-ai-busy', ''); el('ed-ai-cancel').classList.add('hidden'); modelsChanged();
    }
  }

  // (b) replace the selection with speech
  function showRepResult(show) { el('ed-rep-result').classList.toggle('hidden', !show); }
  let repUrl = null;
  async function repPrefill() {
    if (!AudioState.hasStt() || !hasSel() || el('ed-rep-text').value.trim() || nFrames() === 0) return;
    const ver = S.version, a = S.sel.a, b = S.sel.b;
    if (b - a > sr() * 120) return;
    try {
      const r = await AudioIO.transcribe(wav16k(a, b), {model: sttModelId()});
      if (!el('ed-rep-text').value.trim() && ver === S.version) el('ed-rep-text').value = r.text;
    } catch { /* optional convenience */ }
  }
  async function repSynthesize() {
    if (!requireClip() || rep?.busy) return;
    const text = el('ed-rep-text').value.trim();
    if (!text) { toast('Type the text to speak.', 'err'); el('ed-rep-text').focus(); return; }
    rep = {busy: true, abort: new AbortController(), version: S.version};
    el('ed-rep-cancel').classList.remove('hidden'); modelsChanged(); showRepResult(false);
    busyText('ed-rep-busy', 'synthesizing…');
    try {
      const r = await AudioIO.speak({input: text, model: ttsModelId(), voice: voiceOf('ed-rep-voice'), speed: numVal('ed-rep-speed', 1), signal: rep.abort.signal});
      let piece = E.trimSilence(E.conform(r.clip, S.clip), -45, 0.02);
      const fitMode = el('ed-rep-fit').value;
      if (fitMode !== 'none' && hasSel()) {
        const want = S.sel.b - S.sel.a, ratio = E.length(piece) / want;
        const clamped = Math.min(2, Math.max(0.5, ratio));
        piece = fitMode === 'stretch' ? E.timeStretch(piece, 0, E.length(piece), clamped) : E.changeSpeed(piece, 0, E.length(piece), clamped);
        if (clamped !== ratio) toast('The speech is much longer/shorter than the selection; it was only fitted as far as sounds natural (0.5×–2×).', 'info', 6000);
      }
      rep.piece = piece;
      if (repUrl) URL.revokeObjectURL(repUrl);
      repUrl = URL.createObjectURL(new Blob([E.encodeWav(piece, 16)], {type: 'audio/wav'}));
      el('ed-rep-audio').src = repUrl;
      showRepResult(true);
    } catch (e) {
      if (e.name !== 'AbortError') toast(e.message, 'err', 8000);
    } finally {
      if (rep) rep.busy = false;
      busyText('ed-rep-busy', ''); el('ed-rep-cancel').classList.add('hidden'); modelsChanged();
    }
  }
  function repApply() {
    if (!rep?.piece || !requireClip()) return;
    if (rep.version !== S.version && !confirm('The clip changed since the speech was made. Apply it anyway?')) return;
    const xf = Math.round(0.006 * sr());
    const piece = rep.piece;
    if (hasSel()) {
      const a = S.sel.a, b = S.sel.b;
      const res = E.concatCrossfade([E.slice(S.clip, 0, a), E.conform(piece, S.clip), E.slice(S.clip, b, nFrames())], [xf, xf]);
      commit(res, {cursor: a + E.length(piece), sel: {a, b: a + E.length(piece)}});
    } else {
      const res = E.concatCrossfade([E.slice(S.clip, 0, S.cursor), E.conform(piece, S.clip), E.slice(S.clip, S.cursor, nFrames())], [xf, xf]);
      commit(res, {cursor: S.cursor + E.length(piece)});
    }
    showRepResult(false);
    rep.piece = null;
  }

  // (c) edit by text
  function invalidateEdit() {
    if (edit && edit.version !== S.version && !edit.applying) {
      edit = null;
      el('ed-et').classList.add('hidden');
      el('ed-et-result').classList.add('hidden');
    }
  }
  const wordTexts = () => edit.words.map(w => w.text);
  function etTokens() { return E.tokenize(el('ed-et-words').innerText || ''); }
  function etDiff() {
    const nw = etTokens();
    return {nw, ops: E.diffWords(wordTexts(), nw)};
  }
  function etRenderChanges() {
    const box = el('ed-et-changes');
    if (!edit) { box.innerHTML = ''; return; }
    const {nw, ops} = etDiff();
    edit.result = null; el('ed-et-result').classList.add('hidden');
    const old = wordTexts();
    box.innerHTML = ops.length ? ops.map(o => {
      const was = old.slice(o.a0, o.a1).join(' '), now = nw.slice(o.b0, o.b1).join(' ');
      return `<div class="au-chg ${o.type}">` + (o.type === 'delete' ? `delete <del>${esc(was)}</del>` : o.type === 'insert' ? `insert <ins>${esc(now)}</ins> before “${esc(old[o.a0] || '(end)')}”`
        : `<del>${esc(was)}</del> → <ins>${esc(now)}</ins>`) + '</div>';
    }).join('') : '<span class="muted small">No changes yet.</span>';
    el('ed-et-prev').disabled = !ops.length || !AudioState.hasTts();
  }
  async function etTranscribe() {
    if (!requireClip() || aiAbort) return;
    const [a, b] = target();
    aiAbort = new AbortController();
    el('ed-et-cancel').classList.remove('hidden'); el('ed-et-go').disabled = true;
    busyText('ed-et-busy', 'transcribing with word timestamps…');
    const version = S.version;
    try {
      const r = await AudioIO.transcribe(wav16k(a, b), {model: sttModelId(), words: true, signal: aiAbort.signal});
      const words = r.words.map(w => ({text: (w.word || '').trim(), s: a + Math.round(w.start * sr()), e: a + Math.round(w.end * sr())}))
        .filter(w => w.text).map(w => ({...w, s: Math.min(b, Math.max(a, w.s)), e: Math.min(b, Math.max(a, w.e))}));
      for (let i = 1; i < words.length; i++) if (words[i].s < words[i - 1].e) words[i].s = words[i - 1].e;   // keep them ordered
      if (!words.length) { toast('No speech was detected in that part.', 'err'); return; }
      edit = {words, a, b, version, approx: r.wordsApprox, result: null};
      if (r.wordsApprox) toast('The server sent no word timestamps: word times are estimated from the sentence times, so cuts may be a little off.', 'info', 8000);
      const box = el('ed-et-words');
      box.innerHTML = words.map((w, i) => `<span class="au-w" data-i="${i}" title="${AudioEdit.clock(secOf(w.s), 2)} – ${AudioEdit.clock(secOf(w.e), 2)}">${esc(w.text)}</span>`).join(' ');
      el('ed-et').classList.remove('hidden');
      etRenderChanges();
    } catch (e) {
      if (e.name !== 'AbortError') toast(e.message, 'err', 8000);
    } finally {
      aiAbort = null; busyText('ed-et-busy', ''); el('ed-et-cancel').classList.add('hidden'); modelsChanged();
    }
  }

  const synthCache = new Map();
  async function etPreview() {
    if (!edit || edit.busy) return;
    if (edit.version !== S.version) { toast('The clip changed: transcribe again.', 'err'); return; }
    const {nw, ops} = etDiff();
    if (!ops.length) return;
    edit.busy = true; el('ed-et-prev').disabled = true;
    const ed = edit;
    const maxPause = Math.round(0.25 * sr());
    const plan = E.planTextEdit(edit.words, ops, nw, maxPause);
    const model = ttsModelId(), voice = voiceOf('ed-et-voice'), speed = numVal('ed-et-speed', 1);
    const abort = aiAbort = new AbortController();
    el('ed-et-cancel').classList.remove('hidden');
    try {
      const pieces = [];
      const todo = plan.filter(r => r.text).length;
      let done = 0;
      for (const r of plan) {
        if (!r.text) { pieces.push(null); continue; }
        busyText('ed-et-pbusy', `synthesizing ${++done}/${todo}…`);
        const key = [model, voice, speed, r.text].join('\u0001');
        let clip = synthCache.get(key);
        if (!clip) { clip = (await AudioIO.speak({input: r.text, model, voice, speed, signal: abort.signal})).clip; synthCache.set(key, clip); }
        let piece = E.trimSilence(E.conform(clip, S.clip), -45, 0.012);
        if (r.type === 'replace' && el('ed-et-fit').checked) {
          const want = r.cutB - r.cutA, ratio = E.length(piece) / want;
          piece = E.timeStretch(piece, 0, E.length(piece), Math.min(1.5, Math.max(0.67, ratio)));
        }
        if (r.type === 'insert') piece = E.concat([piece, E.silent(sr(), S.clip.channels.length, Math.round(0.05 * sr()))]);
        pieces.push(piece);
      }
      const res = E.applyTextEdit(S.clip, plan, pieces, Math.round(0.008 * sr()));
      if (edit !== ed) return;                // the clip changed meanwhile
      ed.result = res;
      ed.plan = plan;
      el('ed-et-result').classList.remove('hidden');
    } catch (e) {
      if (e.name !== 'AbortError') toast(e.message, 'err', 8000);
    } finally {
      if (edit) edit.busy = false;
      aiAbort = null; busyText('ed-et-pbusy', ''); el('ed-et-prev').disabled = false; el('ed-et-cancel').classList.add('hidden');
    }
  }
  const etPlayer = AudioIO.player();
  function etPlay() {
    if (!edit?.result) return;
    const {clip, spans} = edit.result;
    const a = Math.max(0, spans[0][0] - Math.round(sr() * 0.8)), b = Math.min(E.length(clip), spans[spans.length - 1][1] + Math.round(sr() * 0.8));
    stopPlay(false);
    etPlayer.play(clip, a / clip.sampleRate, b / clip.sampleRate);
  }
  function etApply() {
    if (!edit?.result) return;
    if (edit.version !== S.version) { toast('The clip changed: transcribe again.', 'err'); return; }
    etPlayer.stop();
    const {clip, spans} = edit.result;
    edit.applying = true;
    commit(clip, {cursor: spans[0][0], sel: {a: spans[0][0], b: Math.max(spans[0][0] + 1, spans[spans.length - 1][1])}});
    edit = null;
    el('ed-et').classList.add('hidden'); el('ed-et-result').classList.add('hidden');
    toast('Edit applied. Undo (Ctrl/Cmd+Z) restores the original audio.', 'ok', 4000);
  }

  function initEditByText() {
    const box = el('ed-et-words');
    box.contentEditable = 'plaintext-only';
    if (box.contentEditable !== 'plaintext-only') box.contentEditable = 'true';
    box.addEventListener('input', etRenderChanges);
    box.addEventListener('paste', ev => { ev.preventDefault(); document.execCommand('insertText', false, (ev.clipboardData || window.clipboardData).getData('text/plain')); });
    box.addEventListener('mouseover', ev => {
      const w = ev.target.closest('.au-w');
      if (!w || !edit) return;
      const word = edit.words[+w.dataset.i];
      if (word) { S.hint = [word.s, word.e]; draw(); }
    });
    box.addEventListener('mouseleave', () => { S.hint = null; draw(); });
    box.addEventListener('click', ev => {
      const w = ev.target.closest('.au-w');
      if (!w || !edit || !ev.altKey) return;
      const word = edit.words[+w.dataset.i];
      stopPlay(false);
      pl.play(S.clip, secOf(word.s) - 0.05, secOf(word.e) + 0.05);
    });
  }

  // ------------------------------------------------------------------ init

  function init(host) {
    root = host;
    root.innerHTML = HTML;
    cv = el('ed-canvas');
    g = cv.getContext('2d');
    pl = AudioIO.player();
    pl.onEnd = () => { stopPlay(true); refresh(); };
    bindOps();
    initScrollbar();
    initRecording();
    initEditByText();
    cv.addEventListener('pointerdown', onDown);
    cv.addEventListener('pointermove', onMove);
    cv.addEventListener('pointerup', onUp);
    cv.addEventListener('pointercancel', () => { drag = null; });
    cv.addEventListener('dblclick', () => { if (S.clip) { S.sel = {a: 0, b: nFrames()}; refresh(); } });
    cv.addEventListener('wheel', onWheel, {passive: false});
    document.addEventListener('keydown', onKey);
    // dropping a file on the waveform opens it
    const wave = el('ed-wave');
    ['dragenter', 'dragover'].forEach(t => wave.addEventListener(t, ev => { ev.preventDefault(); wave.classList.add('over'); }));
    ['dragleave', 'drop'].forEach(t => wave.addEventListener(t, ev => { ev.preventDefault(); wave.classList.remove('over'); }));
    wave.addEventListener('drop', ev => { const f = ev.dataTransfer?.files?.[0]; if (f) openFile(f); });
    // the canvas follows its box and the theme
    new ResizeObserver(() => resize()).observe(wave);
    new MutationObserver(() => { readColors(); draw(); }).observe(document.documentElement, {attributes: true, attributeFilter: ['data-theme']});
    matchMedia('(prefers-color-scheme: dark)').addEventListener?.('change', () => { readColors(); draw(); });
    el('ed-ai-tr').addEventListener('click', aiTranscribe);
    el('ed-ai-cancel').addEventListener('click', () => aiAbort?.abort());
    el('ed-ai-copy').addEventListener('click', e => copyText(el('ed-ai-text').value, e.target));
    el('ed-ai-use').addEventListener('click', () => { el('ed-rep-text').value = el('ed-ai-text').value; el('ed-rep-text').focus(); });
    el('ed-rep-go').addEventListener('click', repSynthesize);
    el('ed-rep-cancel').addEventListener('click', () => rep?.abort?.abort());
    el('ed-rep-apply').addEventListener('click', repApply);
    el('ed-rep-discard').addEventListener('click', () => { showRepResult(false); if (rep) rep.piece = null; });
    el('ed-rep-text').addEventListener('focus', repPrefill);
    el('ed-et-go').addEventListener('click', etTranscribe);
    el('ed-et-cancel').addEventListener('click', () => aiAbort?.abort());
    el('ed-et-prev').addEventListener('click', etPreview);
    el('ed-et-play').addEventListener('click', etPlay);
    el('ed-et-stop').addEventListener('click', () => etPlayer.stop());
    el('ed-et-apply').addEventListener('click', etApply);
    el('ed-et-discard').addEventListener('click', () => { etPlayer.stop(); if (edit) edit.result = null; el('ed-et-result').classList.add('hidden'); });
    refresh();
  }

  /** The editor became visible: its canvas has a size now. */
  function shown() { if (root && !root.classList.contains('hidden')) { requestAnimationFrame(() => { resize(); }); } }

  return {init, load, shown, modelsChanged, hasClip: () => !!S.clip};
})();
