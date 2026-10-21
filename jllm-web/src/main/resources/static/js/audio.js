'use strict';
// The Audio tab: loaded audio models, speech to text, text to speech. The waveform editor is audio-editor.js.
// Needs core.js, audio-edit.js, audio-io.js, audio-editor.js.

TABS.push('audio');

const AudioTab = (() => {
  const E = AudioEdit;
  const prefs = store.get('jllm.audio', {});
  const savePref = (k, v) => { prefs[k] = v; store.set('jllm.audio', prefs); };
  const baseName = n => (n || 'audio').replace(/\.[A-Za-z0-9]{1,5}$/, '').replace(/[^\w.-]+/g, '_').slice(0, 60) || 'audio';
  const shared = {sttClip: null, sttName: '', speechClip: null, speechName: ''};   // what the other panes can pick up

  // ------------------------------------------------------------------ shell

  const SHELL = `
    <header class="page-head"><h1>Audio</h1>
      <p class="muted">Speech to text with Whisper, text to speech, and a waveform editor. Everything runs on this machine; the models are loaded on the <a href="#models">Models</a> page.</p>
      <div id="au-models" class="au-models"></div></header>
    <div class="tabs-inline" id="au-sub" role="tablist">
      <button data-sub="stt" role="tab">Speech to text</button><button data-sub="tts" role="tab">Text to speech</button><button data-sub="edit" role="tab">Editor</button>
    </div>
    <div id="au-stt" class="au-pane"></div>
    <div id="au-tts" class="au-pane hidden"></div>
    <div id="au-edit" class="au-pane hidden"></div>`;

  function showSub(name) {
    if (!['stt', 'tts', 'edit'].includes(name)) name = 'stt';
    savePref('sub', name);
    $$('#au-sub button').forEach(b => { b.classList.toggle('active', b.dataset.sub === name); b.setAttribute('aria-selected', String(b.dataset.sub === name)); });
    $$('.au-pane').forEach(p => p.classList.toggle('hidden', p.id !== 'au-' + name));
    if (name === 'edit') AudioEditor.shown();
  }

  function renderModels() {
    const box = $('#au-models');
    if (!box) return;
    const ms = AudioState.models;
    const badge = m => `<span class="badge ${m.kind === 'speech-to-text' ? 'info' : 'ok'}" title="${esc(m.description || '')}">${esc(m.id)} · ${esc(m.kind)}</span>`;
    const miss = [];
    if (!AudioState.hasStt()) miss.push('a speech-to-text model such as <code>openai/whisper-small</code>');
    if (!AudioState.hasTts()) miss.push('a text-to-speech model such as <code>OuteAI/OuteTTS-0.3-500M</code>');
    box.innerHTML = (ms.length ? ms.map(badge).join(' ') : '<span class="muted">No audio model loaded.</span>')
      + (miss.length ? `<div class="hint">Load ${miss.join(' and ')} on the <a href="#models">Models</a> page to enable ${AudioState.hasStt() || AudioState.hasTts() ? 'the rest' : 'transcription and speech'}. The editor works without a model.</div>` : '');
  }

  const modelOptions = (list, sel) => list.map(m => `<option value="${esc(m.id)}" ${m.id === sel ? 'selected' : ''}>${esc(m.id)}</option>`).join('');

  // ------------------------------------------------------------------ speech to text

  const stt = {clip: null, name: '', url: null, result: null, raw: null, abort: null, busy: false, showWords: false, timer: 0, stopAt: null};

  const STT_HTML = `
    <div class="card">
      <h2>Audio <span class="muted small">record, choose or drop a file</span></h2>
      <div id="stt-drop" class="au-drop" tabindex="0" aria-label="Drop an audio file here or press Enter to choose one">
        <div class="row">
          <button id="stt-rec" class="primary" aria-pressed="false">● Record</button>
          <button id="stt-choose">Choose a file…</button>
          <input type="file" id="stt-file" accept="audio/*,video/*,.wav,.mp3,.m4a,.flac,.ogg,.aiff,.aif,.au" class="hidden">
          <div id="stt-meter" class="au-meter hidden" aria-hidden="true"><div></div></div>
          <span id="stt-time" class="au-time hidden" aria-live="off"></span>
          <span class="muted small">or drop an audio file here (WAV, MP3, M4A, FLAC, Ogg, anything your browser can play)</span>
        </div>
      </div>
      <div id="stt-clip" class="au-clipbox hidden">
        <div class="row"><strong id="stt-name"></strong><span id="stt-info" class="muted small"></span><div class="spacer"></div>
          <button id="stt-clear" class="ghost small">remove</button></div>
        <audio id="stt-audio" controls preload="metadata"></audio>
      </div>
    </div>
    <div class="card">
      <h2>Transcribe</h2>
      <div class="grid">
        <label>Model <select id="stt-model"></select></label>
        <label>Language <select id="stt-lang"></select></label>
        <label class="wide2">Prompt (optional: names, jargon, style) <input id="stt-prompt" placeholder="e.g. Glossary: Kubernetes, jllm"></label>
      </div>
      <div class="row">
        <label class="check"><input type="checkbox" id="stt-translate"> translate to English</label>
        <label class="check"><input type="checkbox" id="stt-words"> word timestamps</label>
        <div class="spacer"></div>
        <span id="stt-busy" class="muted small"></span>
        <button id="stt-go" class="primary">Transcribe</button>
        <button id="stt-cancel" class="danger hidden">Cancel</button>
      </div>
      <div id="stt-msg"></div>
    </div>
    <div class="card hidden" id="stt-result">
      <h2>Transcript <span id="stt-meta" class="muted small"></span></h2>
      <div class="row au-actions">
        <button id="stt-copy" class="small">Copy text</button>
        <button class="small" data-exp="txt">TXT</button><button class="small" data-exp="srt">SRT</button>
        <button class="small" data-exp="vtt">VTT</button><button class="small" data-exp="json">JSON</button>
        <span class="au-sep"></span>
        <button id="stt-toedit" class="small">Send to editor</button>
        <button id="stt-tochat" class="small">Send to chat input</button>
        <div class="spacer"></div>
        <label class="check"><input type="checkbox" id="stt-showwords"> words</label>
      </div>
      <div id="stt-segs" class="au-segs" aria-label="Transcript segments"></div>
      <div class="hint">Click a time or a sentence to play it from the loaded audio.</div>
    </div>`;

  function sttModel() { return AudioState.stt().find(m => m.id === $('#stt-model').value) || AudioState.stt()[0]; }

  function renderSttModels() {
    const list = AudioState.stt();
    const sel = $('#stt-model');
    const keep = sel.value;
    const html = list.length ? modelOptions(list, keep) : '<option value="">(no speech-to-text model loaded)</option>';
    if (sel.innerHTML !== html) sel.innerHTML = html;
    if (list.some(m => m.id === keep)) sel.value = keep;
    renderLangs();
    updateSttButtons();
  }

  function renderLangs() {
    const m = sttModel();
    const sel = $('#stt-lang');
    const langs = m?.languages?.length ? [...m.languages] : [];
    langs.sort((a, b) => AudioIO.langName(a).localeCompare(AudioIO.langName(b)));
    const multi = !m || m.multilingual !== false;
    const html = `<option value="">Auto detect</option>` + langs.map(l => `<option value="${esc(l)}">${esc(AudioIO.langName(l))}</option>`).join('');
    const keep = sel.value || prefs.sttLang || '';
    if (sel.innerHTML !== html) sel.innerHTML = html;
    sel.value = langs.includes(keep) ? keep : '';
    sel.disabled = !multi;
    sel.title = multi ? '' : 'This model only understands English';
    $('#stt-translate').disabled = !multi;
    if (!multi) $('#stt-translate').checked = false;
  }

  function updateSttButtons() {
    const can = AudioState.hasStt() && stt.clip && !stt.busy;
    $('#stt-go').disabled = !can;
    $('#stt-go').title = !AudioState.hasStt() ? 'Load a speech-to-text model (Whisper) on the Models page' : !stt.clip ? 'Record or choose some audio first' : '';
  }

  function setSttClip(clip, name) {
    stt.clip = clip; stt.name = name;
    shared.sttClip = clip; shared.sttName = name;
    if (stt.url) URL.revokeObjectURL(stt.url);
    stt.url = URL.createObjectURL(AudioIO.toWavBlob(clip, 0, false));
    $('#stt-audio').src = stt.url;
    $('#stt-name').textContent = name;
    $('#stt-info').textContent = `${E.clock(E.duration(clip))} · ${clip.sampleRate} Hz · ${clip.channels.length === 1 ? 'mono' : clip.channels.length + ' channels'}`;
    $('#stt-clip').classList.remove('hidden');
    stt.result = null;
    $('#stt-result').classList.add('hidden');
    $('#stt-msg').innerHTML = '';
    updateSttButtons();
  }

  function clearSttClip() {
    stt.clip = null; shared.sttClip = null;
    if (stt.url) { URL.revokeObjectURL(stt.url); stt.url = null; }
    $('#stt-audio').removeAttribute('src');
    $('#stt-clip').classList.add('hidden');
    $('#stt-result').classList.add('hidden');
    stt.result = null;
    updateSttButtons();
  }

  async function loadFile(file) {
    try {
      $('#stt-msg').innerHTML = '<span class="spinner"></span> decoding…';
      const clip = await AudioIO.decode(file);
      setSttClip(clip, file.name);
      $('#stt-msg').innerHTML = '';
    } catch (e) { $('#stt-msg').innerHTML = `<span class="error">${esc(e.message)}</span>`; }
  }

  async function runTranscribe() {
    if (stt.busy || !stt.clip) return;
    const model = sttModel();
    if (!model) { toast('Load a speech-to-text model on the Models page.', 'err'); return; }
    stt.busy = true;
    stt.abort = new AbortController();
    $('#stt-go').classList.add('hidden'); $('#stt-cancel').classList.remove('hidden');
    $('#stt-msg').innerHTML = '';
    const t0 = performance.now();
    $('#stt-busy').innerHTML = '<span class="spinner"></span> transcribing… 0 s';
    stt.timer = setInterval(() => { $('#stt-busy').innerHTML = `<span class="spinner"></span> transcribing… ${Math.round((performance.now() - t0) / 1000)} s`; }, 500);
    savePref('sttLang', $('#stt-lang').value);
    savePref('sttWords', $('#stt-words').checked);
    try {
      const wav = AudioIO.toWavBlob(stt.clip, 16000);
      const r = await AudioIO.transcribe(wav, {model: model.id, language: $('#stt-lang').value, translate: $('#stt-translate').checked,
        prompt: $('#stt-prompt').value.trim(), words: $('#stt-words').checked, signal: stt.abort.signal});
      stt.result = r;
      stt.took = (performance.now() - t0) / 1000;
      renderResult();
    } catch (e) {
      if (e.name === 'AbortError') $('#stt-msg').innerHTML = '<span class="muted">Cancelled.</span>';
      else $('#stt-msg').innerHTML = `<span class="error">${esc(e.message)}</span>`;
    } finally {
      clearInterval(stt.timer);
      stt.busy = false; stt.abort = null;
      $('#stt-busy').textContent = '';
      $('#stt-go').classList.remove('hidden'); $('#stt-cancel').classList.add('hidden');
      updateSttButtons();
    }
  }

  const stamp = s => E.clock(s, 1);

  function renderResult() {
    const r = stt.result;
    const box = $('#stt-result');
    box.classList.remove('hidden');
    $('#stt-meta').textContent = [r.language && 'language: ' + r.language, r.duration && E.clock(r.duration) + ' of audio',
      stt.took != null && `took ${stt.took.toFixed(1)} s`, r.wordsApprox && $('#stt-words').checked ? 'word times estimated' : ''].filter(Boolean).join(' · ');
    const hasWords = r.words.length > 0;
    $('#stt-showwords').disabled = !hasWords;
    const segs = r.segments.length ? r.segments : (r.text ? [{start: 0, end: r.duration || E.duration(stt.clip), text: r.text}] : []);
    if (!segs.length) { $('#stt-segs').innerHTML = '<p class="empty">No speech was detected.</p>'; return; }
    const words = r.words;
    $('#stt-segs').innerHTML = segs.map((s, i) => {
      let body;
      if (stt.showWords && hasWords) {
        const ws = words.filter(w => (w.start + w.end) / 2 >= s.start - 0.01 && (w.start + w.end) / 2 < s.end + 0.01);
        body = ws.length ? ws.map(w => `<span class="au-w" data-s="${w.start}" data-e="${w.end}" title="${stamp(w.start)} – ${stamp(w.end)}">${esc(w.word.trim())}</span>`).join(' ') : esc(s.text.trim());
      } else body = esc(s.text.trim());
      return `<div class="au-seg" data-i="${i}" data-s="${s.start}" data-e="${s.end}"><button class="au-ts" aria-label="Play from ${stamp(s.start)}">${stamp(s.start)}</button><span class="au-seg-text">${body}</span></div>`;
    }).join('');
    $$('#stt-segs .au-seg').forEach(el => el.addEventListener('click', ev => {
      const w = ev.target.closest('.au-w');
      const src = w || el;
      playSpan(+src.dataset.s, +src.dataset.e);
    }));
  }

  function playSpan(start, end) {
    const a = $('#stt-audio');
    if (!stt.url) return;
    stt.stopAt = end;
    a.currentTime = start;
    a.play().catch(() => {});
  }

  function highlightPlaying() {
    const a = $('#stt-audio');
    const t = a.currentTime;
    if (stt.stopAt != null && t >= stt.stopAt) { a.pause(); stt.stopAt = null; }
    $$('#stt-segs .au-seg').forEach(el => el.classList.toggle('on', !a.paused && t >= +el.dataset.s && t < +el.dataset.e));
    $$('#stt-segs .au-w').forEach(el => el.classList.toggle('on', !a.paused && t >= +el.dataset.s && t < +el.dataset.e));
  }

  function exportResult(kind) {
    const r = stt.result;
    if (!r) return;
    const segs = r.segments.length ? r.segments : [{start: 0, end: r.duration || E.duration(stt.clip), text: r.text}];
    const base = baseName(stt.name);
    if (kind === 'txt') AudioIO.saveBlob(base + '.txt', new Blob([r.text + '\n'], {type: 'text/plain'}));
    else if (kind === 'srt') AudioIO.saveBlob(base + '.srt', new Blob([E.toSrt(segs)], {type: 'application/x-subrip'}));
    else if (kind === 'vtt') AudioIO.saveBlob(base + '.vtt', new Blob([E.toVtt(segs)], {type: 'text/vtt'}));
    else AudioIO.saveBlob(base + '.json', new Blob([JSON.stringify({text: r.text, language: r.language, duration: r.duration,
      segments: r.segments.map(s => ({start: s.start, end: s.end, text: s.text})), words: r.wordsApprox ? undefined : r.words}, null, 2)], {type: 'application/json'}));
  }

  function initStt() {
    const root = $('#au-stt');
    root.innerHTML = STT_HTML;
    $('#stt-words').checked = !!prefs.sttWords;
    AudioIO.bindRecorder({button: $('#stt-rec'), meter: $('#stt-meter'), time: $('#stt-time'),
      onClip: clip => setSttClip(clip, 'recording ' + new Date().toLocaleTimeString([], {hour: '2-digit', minute: '2-digit', second: '2-digit'}))});
    $('#stt-choose').addEventListener('click', () => $('#stt-file').click());
    $('#stt-file').addEventListener('change', ev => { const f = ev.target.files[0]; ev.target.value = ''; if (f) loadFile(f); });
    const drop = $('#stt-drop');
    ['dragenter', 'dragover'].forEach(t => drop.addEventListener(t, ev => { ev.preventDefault(); drop.classList.add('over'); }));
    ['dragleave', 'drop'].forEach(t => drop.addEventListener(t, ev => { ev.preventDefault(); drop.classList.remove('over'); }));
    drop.addEventListener('drop', ev => { const f = ev.dataTransfer?.files?.[0]; if (f) loadFile(f); });
    drop.addEventListener('keydown', ev => { if (ev.target === drop && (ev.key === 'Enter' || ev.key === ' ')) { ev.preventDefault(); $('#stt-file').click(); } });
    $('#stt-clear').addEventListener('click', clearSttClip);
    $('#stt-go').addEventListener('click', runTranscribe);
    $('#stt-cancel').addEventListener('click', () => stt.abort?.abort());
    $('#stt-model').addEventListener('change', renderLangs);
    $('#stt-showwords').addEventListener('change', ev => { stt.showWords = ev.target.checked; if (stt.result) renderResult(); });
    $('#stt-words').addEventListener('change', () => savePref('sttWords', $('#stt-words').checked));
    const a = $('#stt-audio');
    ['timeupdate', 'pause', 'play', 'ended'].forEach(t => a.addEventListener(t, highlightPlaying));
    a.addEventListener('seeking', () => { /* a manual seek ends a "play this span" */ });
    a.addEventListener('pause', () => { stt.stopAt = null; });
    $('#stt-copy').addEventListener('click', e => stt.result && copyText(stt.result.text, e.target));
    $$('[data-exp]', root).forEach(b => b.addEventListener('click', () => exportResult(b.dataset.exp)));
    $('#stt-toedit').addEventListener('click', () => {
      if (!stt.clip) return;
      AudioEditor.load(stt.clip, baseName(stt.name));
      showSub('edit');
    });
    $('#stt-tochat').addEventListener('click', () => {
      if (!stt.result) return;
      sendToChat(stt.result.text);
    });
    renderSttModels();
  }

  /** Appends text to the chat input and switches to the chat. */
  function sendToChat(text) {
    const inp = $('#input');
    inp.value = (inp.value && !/\s$/.test(inp.value) ? inp.value + ' ' : inp.value) + text;
    inp.dispatchEvent(new Event('input', {bubbles: true}));
    showTab('chat');
    inp.focus();
  }

  // ------------------------------------------------------------------ text to speech

  const tts = {abort: null, busy: false, clip: null, url: null, name: ''};

  const TTS_HTML = `
    <div class="card">
      <h2>Speak</h2>
      <label>Text <textarea id="tts-text" rows="6" placeholder="Type or paste what the voice should say…"></textarea></label>
      <div class="hint"><span id="tts-count">0</span> characters</div>
      <div class="grid">
        <label>Model <select id="tts-model"></select></label>
        <label>Voice <select id="tts-voice"></select></label>
        <label>Speed <span class="au-range"><input type="range" id="tts-speed" min="0.25" max="4" step="0.05" value="1" aria-label="Speed"><output id="tts-speed-v">1.00×</output></span></label>
      </div>
      <div class="row">
        <button id="tts-go" class="primary">Speak</button>
        <button id="tts-cancel" class="danger hidden">Cancel</button>
        <span id="tts-busy" class="muted small"></span>
      </div>
      <div id="tts-msg"></div>
    </div>
    <div class="card hidden" id="tts-result">
      <h2>Result <span id="tts-meta" class="muted small"></span></h2>
      <audio id="tts-audio" controls></audio>
      <div class="row au-actions">
        <button id="tts-dl" class="small">Download WAV</button>
        <button id="tts-toedit" class="small">Send to editor</button>
      </div>
    </div>`;

  function ttsModel() { return AudioState.tts().find(m => m.id === $('#tts-model').value) || AudioState.tts()[0]; }

  function renderTtsModels() {
    const list = AudioState.tts();
    const sel = $('#tts-model');
    const keep = sel.value || prefs.ttsModel;
    const html = list.length ? modelOptions(list, keep) : '<option value="">(no text-to-speech model loaded)</option>';
    if (sel.innerHTML !== html) sel.innerHTML = html;
    if (list.some(m => m.id === keep)) sel.value = keep;
    renderVoices();
    updateTtsButtons();
  }

  function renderVoices() {
    const m = ttsModel();
    const sel = $('#tts-voice');
    const voices = m?.voices || [];
    const keep = sel.value || prefs.ttsVoice || '';
    const html = '<option value="">(default voice)</option>' + voices.map(v => `<option value="${esc(v)}">${esc(v)}</option>`).join('');
    if (sel.innerHTML !== html) sel.innerHTML = html;
    sel.value = voices.includes(keep) ? keep : '';
  }

  function updateTtsButtons() {
    const can = AudioState.hasTts() && !tts.busy && $('#tts-text').value.trim();
    $('#tts-go').disabled = !can;
    $('#tts-go').title = !AudioState.hasTts() ? 'Load a text-to-speech model on the Models page' : !$('#tts-text').value.trim() ? 'Type some text first' : '';
  }

  async function runSpeak() {
    if (tts.busy) return;
    const text = $('#tts-text').value.trim();
    const model = ttsModel();
    if (!text) return;
    if (!model) { toast('Load a text-to-speech model on the Models page.', 'err'); return; }
    tts.busy = true; tts.abort = new AbortController();
    $('#tts-go').classList.add('hidden'); $('#tts-cancel').classList.remove('hidden');
    $('#tts-msg').innerHTML = '';
    const t0 = performance.now();
    $('#tts-busy').innerHTML = '<span class="spinner"></span> synthesizing… 0 s';
    const timer = setInterval(() => { $('#tts-busy').innerHTML = `<span class="spinner"></span> synthesizing… ${Math.round((performance.now() - t0) / 1000)} s`; }, 500);
    savePref('ttsText', text); savePref('ttsModel', model.id); savePref('ttsVoice', $('#tts-voice').value); savePref('ttsSpeed', +$('#tts-speed').value);
    try {
      const r = await AudioIO.speak({input: text, model: model.id, voice: $('#tts-voice').value, speed: +$('#tts-speed').value, signal: tts.abort.signal});
      tts.clip = r.clip; tts.name = 'speech';
      shared.speechClip = r.clip; shared.speechName = 'speech';
      if (tts.url) URL.revokeObjectURL(tts.url);
      tts.url = URL.createObjectURL(new Blob([r.wav], {type: 'audio/wav'}));
      const a = $('#tts-audio');
      a.src = tts.url;
      $('#tts-result').classList.remove('hidden');
      $('#tts-meta').textContent = `${E.clock(E.duration(r.clip))} · ${r.clip.sampleRate} Hz · took ${((performance.now() - t0) / 1000).toFixed(1)} s`;
      a.play().catch(() => { /* autoplay may be blocked: the controls are there */ });
    } catch (e) {
      if (e.name === 'AbortError') $('#tts-msg').innerHTML = '<span class="muted">Cancelled.</span>';
      else { $('#tts-msg').innerHTML = `<span class="error">${esc(e.message)}</span>`; toast(e.message, 'err', 7000); }
    } finally {
      clearInterval(timer);
      tts.busy = false; tts.abort = null;
      $('#tts-busy').textContent = '';
      $('#tts-go').classList.remove('hidden'); $('#tts-cancel').classList.add('hidden');
      updateTtsButtons();
    }
  }

  function initTts() {
    $('#au-tts').innerHTML = TTS_HTML;
    $('#tts-text').value = prefs.ttsText || '';
    $('#tts-speed').value = prefs.ttsSpeed || 1;
    const speedLabel = () => { $('#tts-speed-v').textContent = (+$('#tts-speed').value).toFixed(2) + '×'; };
    speedLabel();
    const count = () => { $('#tts-count').textContent = $('#tts-text').value.length; updateTtsButtons(); };
    count();
    $('#tts-text').addEventListener('input', count);
    $('#tts-text').addEventListener('keydown', ev => { if (ev.key === 'Enter' && (ev.metaKey || ev.ctrlKey)) { ev.preventDefault(); runSpeak(); } });
    $('#tts-speed').addEventListener('input', speedLabel);
    $('#tts-model').addEventListener('change', renderVoices);
    $('#tts-go').addEventListener('click', runSpeak);
    $('#tts-cancel').addEventListener('click', () => tts.abort?.abort());
    $('#tts-dl').addEventListener('click', () => tts.clip && AudioIO.saveBlob('speech.wav', AudioIO.toWavBlob(tts.clip, 0, false)));
    $('#tts-toedit').addEventListener('click', () => { if (tts.clip) { AudioEditor.load(tts.clip, 'speech'); showSub('edit'); } });
    renderTtsModels();
  }

  // ------------------------------------------------------------------ models on the Models page

  let lastModelsHtml = '';
  /** Called by models.js after it renders the loaded list: adds the loaded audio models below it. */
  function renderModelsRows() {
    const host = $('#loaded-list');
    if (!host) return;
    let box = $('#audio-loaded-list');
    if (!box) { box = document.createElement('div'); box.id = 'audio-loaded-list'; host.after(box); lastModelsHtml = ''; }
    const ms = AudioState.models;
    if (ms.length) host.querySelector('.empty')?.remove();   // "Nothing loaded yet." is wrong when only audio models are loaded
    const html = ms.length ? `<table class="table"><tr><th>audio model</th><th>kind</th><th>backend</th><th></th></tr>${ms.map(m => `<tr>
        <td><strong>${esc(m.id)}</strong><div class="hint">${esc(m.description || '')}</div>
          <div class="hint">${m.loadedAt ? 'loaded ' + fmtTime(m.loadedAt) : ''}${m.loadSeconds != null ? ' in ' + (+m.loadSeconds).toFixed(1) + ' s' : ''}${m.sampleRate ? ' · ' + m.sampleRate + ' Hz output' : ''}${m.voices?.length ? ' · ' + m.voices.length + ' voices' : ''}${m.kind === 'speech-to-text' && m.languages ? ' · ' + (m.multilingual ? m.languages.length + ' languages' : 'English only') : ''}</div></td>
        <td><span class="badge ${m.kind === 'speech-to-text' ? 'info' : 'ok'}">${esc(m.kind)}</span></td>
        <td class="small">${esc(m.backend || '')}</td>
        <td style="white-space:nowrap"><button class="small" data-au-open="${m.kind === 'speech-to-text' ? 'stt' : 'tts'}">open in Audio</button>
          <button class="small danger" data-au-unload="${esc(m.id)}">unload</button></td></tr>`).join('')}</table>` : '';
    if (html === lastModelsHtml) return;
    lastModelsHtml = html;
    box.innerHTML = html;
    $$('[data-au-open]', box).forEach(b => b.addEventListener('click', () => { savePref('sub', b.dataset.auOpen); showTab('audio'); showSub(b.dataset.auOpen); }));
    $$('[data-au-unload]', box).forEach(b => b.addEventListener('click', async () => {
      if (!confirm('Unload ' + b.dataset.auUnload + '?')) return;
      b.disabled = true;
      try { await api('/api/models/loaded?id=' + encodeURIComponent(b.dataset.auUnload), {method: 'DELETE'}); toast('Unloaded ' + b.dataset.auUnload, 'ok'); }
      catch (e) { toast(e.message, 'err'); }
      await AudioState.refresh();
      refreshLoaded();
    }));
  }

  // ------------------------------------------------------------------ wiring

  function init() {
    const sec = $('#tab-audio');
    if (!sec) return;
    sec.innerHTML = SHELL;
    $$('#au-sub button').forEach(b => b.addEventListener('click', () => showSub(b.dataset.sub)));
    initStt();
    initTts();
    AudioEditor.init($('#au-edit'));
    AudioState.onChange(() => { renderModels(); renderSttModels(); renderTtsModels(); renderModelsRows(); AudioEditor.modelsChanged(); });
    renderModels();
    showSub(prefs.sub || 'stt');
    onShow.audio = () => AudioState.refresh().then(() => { renderModels(); AudioEditor.shown(); });
    onTick.audio = () => AudioState.refresh();
    modelListeners.push(() => AudioState.refresh());
    setInterval(() => { if (activeTab !== 'audio') AudioState.refresh(); }, 6000);
    AudioState.refresh();
  }

  return {init, showSub, shared, sendToChat, renderModelsRows, baseName};
})();

// the Models page asks for the audio rows after it has drawn the loaded list
window.AudioUI = {renderModelsRows: () => AudioTab.renderModelsRows()};
AudioTab.init();
