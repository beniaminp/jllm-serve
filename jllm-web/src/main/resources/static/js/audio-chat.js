'use strict';
// Audio in the chat, without touching chat.js: dictation and "attach audio" buttons in the composer, a "read aloud" action
// on assistant messages, and inline players for [audio:ID] in messages and tool results. The controls are added to the
// DOM that chat.js renders (a MutationObserver re-applies them after every render, including restored conversations).
// Visibility follows the loaded audio models through classes on <body> (see audio.css).

const AudioChat = (() => {
  const E = AudioEdit;
  const TOKEN = /\[audio:([A-Za-z0-9_-]{1,64})\]/g;
  const READ_MAX = 1500;
  const ICON = {
    mic: '<svg viewBox="0 0 24 24" aria-hidden="true"><rect x="9" y="3" width="6" height="11" rx="3"/><path d="M5 11a7 7 0 0 0 14 0M12 18v3"/></svg>',
    clip: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M20 11.5 12 19.5a5 5 0 0 1-7-7l8-8a3.3 3.3 0 0 1 4.7 4.7l-8 8a1.7 1.7 0 0 1-2.4-2.4l7-7"/></svg>',
  };

  // ------------------------------------------------------------------ body classes follow the loaded models

  function classes() {
    document.body.classList.toggle('au-has-stt', AudioState.hasStt());
    document.body.classList.toggle('au-has-tts', AudioState.hasTts());
    document.body.classList.toggle('au-avail', AudioState.available);
  }

  // ------------------------------------------------------------------ composer: dictation and attach

  let micBtn, attachBtn, menu, timeEl, fileIn;
  const dict = {rec: null, busy: false};
  const att = {rec: null, busy: false};

  function appendToInput(text) {
    const inp = $('#input');
    const sep = inp.value && !/\s$/.test(inp.value) ? ' ' : '';
    inp.value += sep + text;
    inp.dispatchEvent(new Event('input', {bubbles: true}));
    inp.focus();
  }

  function setRecUi(btn, on, label) {
    btn.classList.toggle('recording', on);
    btn.setAttribute('aria-pressed', String(on));
    btn.setAttribute('aria-label', label);
    btn.title = label;
  }
  function setBusyUi(btn, on) { btn.classList.toggle('busy', on); btn.disabled = on; }

  async function startRec(state, btn, labelOn, labelOff) {
    const rec = new AudioIO.Recorder({onTime: t => { timeEl.textContent = E.clock(t, 0); }});
    rec.onLimit = () => { if (state.rec === rec) stopRec(state, btn, labelOff); };
    btn.disabled = true;
    try { await rec.start(); } catch (e) { toast(e.message, 'err', 9000); btn.disabled = false; return false; }
    btn.disabled = false;
    state.rec = rec;
    timeEl.classList.remove('hidden'); timeEl.textContent = '0:00';
    setRecUi(btn, true, labelOn);
    return true;
  }
  async function stopRec(state, btn, labelOff) {
    const rec = state.rec;
    state.rec = null;
    timeEl.classList.add('hidden');
    setRecUi(btn, false, labelOff);
    return rec ? rec.stop() : null;
  }

  const MIC_OFF = 'Dictate: click to record, click again to stop and transcribe';
  const MIC_ON = 'Stop recording and transcribe';
  const ATT_OFF = 'Attach audio';
  const ATT_ON = 'Stop recording and attach';

  async function micClick() {
    if (dict.busy) return;
    if (!dict.rec) {
      if (att.rec) { toast('Finish the audio recording first.', 'err'); return; }
      await startRec(dict, micBtn, MIC_ON, MIC_OFF);
      return;
    }
    const clip = await stopRec(dict, micBtn, MIC_OFF);
    if (!clip) { toast('Nothing was recorded.', 'err'); return; }
    dict.busy = true; setBusyUi(micBtn, true);
    try {
      const model = AudioState.stt()[0];
      if (!model) throw new Error('No speech-to-text model is loaded (Models page).');
      const r = await AudioIO.transcribe(AudioIO.toWavBlob(clip, 16000), {model: model.id, language: store.get('jllm.audio', {}).sttLang || ''});
      if (r.text) appendToInput(r.text); else toast('No speech was detected.', 'info');
    } catch (e) { toast(e.message, 'err', 8000); } finally { dict.busy = false; setBusyUi(micBtn, false); }
  }

  async function attachUpload(clip, label) {
    att.busy = true; setBusyUi(attachBtn, true);
    try {
      const mono = E.toMono(clip);
      const rate = Math.min(mono.sampleRate, 22050);
      const info = await AudioIO.uploadClip(AudioIO.toWavBlob(mono, rate), label);
      appendToInput(`[audio:${info.id}] `);
      toast(`Attached ${E.clock(info.seconds, 1)} of audio. Ask the model to transcribe it (it has the transcribe_audio tool; enable it under Tools).`, 'ok', 6000);
    } catch (e) { toast(e.message, 'err', 8000); } finally { att.busy = false; setBusyUi(attachBtn, false); }
  }

  function toggleMenu(show) {
    menu.classList.toggle('hidden', !show);
    attachBtn.setAttribute('aria-expanded', String(show));
  }

  async function attachClick(ev) {
    ev.stopPropagation();
    if (att.busy) return;
    if (att.rec) {
      const clip = await stopRec(att, attachBtn, ATT_OFF);
      if (clip) attachUpload(clip, 'recording'); else toast('Nothing was recorded.', 'err');
      return;
    }
    toggleMenu(menu.classList.contains('hidden'));
  }

  function injectComposer() {
    const row = $('.composer-row');
    if (!row || $('.au-comp')) return;
    const wrap = document.createElement('span');
    wrap.className = 'au-comp';
    wrap.innerHTML = `<button type="button" class="icon-btn au-mic" aria-label="${MIC_OFF}" title="${MIC_OFF}" aria-pressed="false">${ICON.mic}</button>`
      + `<span class="au-comp-time hidden" aria-live="off"></span>`
      + `<span class="au-attach-wrap"><button type="button" class="icon-btn au-attach" aria-label="${ATT_OFF}" title="${ATT_OFF}" aria-haspopup="menu" aria-expanded="false" aria-pressed="false">${ICON.clip}</button>`
      + `<div class="au-menu hidden" role="menu"><button type="button" role="menuitem" data-m="file">Choose an audio file…</button>`
      + `<button type="button" role="menuitem" data-m="rec">Record a clip</button></div></span>`
      + `<input type="file" class="hidden" accept="audio/*,video/*,.wav,.mp3,.m4a,.flac,.ogg,.aiff,.aif,.au">`;
    row.prepend(wrap);
    micBtn = $('.au-mic', wrap); attachBtn = $('.au-attach', wrap); menu = $('.au-menu', wrap); timeEl = $('.au-comp-time', wrap); fileIn = $('input[type=file]', wrap);
    micBtn.addEventListener('click', micClick);
    attachBtn.addEventListener('click', attachClick);
    $('[data-m=file]', menu).addEventListener('click', () => { toggleMenu(false); fileIn.click(); });
    $('[data-m=rec]', menu).addEventListener('click', async () => {
      toggleMenu(false);
      if (dict.rec) { toast('Finish the dictation first.', 'err'); return; }
      await startRec(att, attachBtn, ATT_ON, ATT_OFF);
    });
    fileIn.addEventListener('change', async () => {
      const f = fileIn.files[0]; fileIn.value = '';
      if (!f) return;
      try { attachUpload(await AudioIO.decode(f), f.name); } catch (e) { toast(e.message, 'err', 8000); }
    });
    document.addEventListener('click', ev => { if (!ev.target.closest('.au-attach-wrap')) toggleMenu(false); });
    document.addEventListener('keydown', ev => {
      if (ev.key !== 'Escape') return;
      toggleMenu(false);
      if (dict.rec) { dict.rec.cancel(); dict.rec = null; timeEl.classList.add('hidden'); setRecUi(micBtn, false, MIC_OFF); toast('Dictation cancelled.'); }
      else if (att.rec) { att.rec.cancel(); att.rec = null; timeEl.classList.add('hidden'); setRecUi(attachBtn, false, ATT_OFF); toast('Recording cancelled.'); }
    });
  }

  // ------------------------------------------------------------------ [audio:ID] players

  function playerEl(id) {
    const wrap = document.createElement('span');
    wrap.className = 'au-clip';
    wrap.innerHTML = '<audio controls preload="metadata"></audio><a class="au-dl" aria-label="Download audio" title="Download">⤓</a>';
    const a = wrap.querySelector('audio'), dl = wrap.querySelector('.au-dl');
    dl.download = 'audio-' + id + '.wav';
    AudioIO.clipUrl(id).then(url => { a.src = url; dl.href = url; })
      .catch(() => { wrap.className = 'au-clip gone'; wrap.textContent = `audio ${id} is no longer available (the server keeps clips for a few hours)`; });
    return wrap;
  }

  /** Replaces [audio:ID] inside the text nodes of `root` (not in code) by a small label and a player. */
  function playersInText(root) {
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
      acceptNode: n => (n.nodeValue.includes('[audio:') && !n.parentElement.closest('pre, code, .au-clip, textarea') ? NodeFilter.FILTER_ACCEPT : NodeFilter.FILTER_REJECT)});
    const nodes = [];
    while (walker.nextNode()) nodes.push(walker.currentNode);
    for (const n of nodes) {
      const text = n.nodeValue, frag = document.createDocumentFragment();
      let last = 0, m;
      TOKEN.lastIndex = 0;
      while ((m = TOKEN.exec(text))) {
        if (m.index > last) frag.append(text.slice(last, m.index));
        frag.append(playerEl(m[1]));
        last = m.index + m[0].length;
      }
      if (last === 0) continue;
      if (last < text.length) frag.append(text.slice(last));
      n.replaceWith(frag);
    }
  }

  function playersForToolCards(box) {
    for (const card of $$('details.toolcard', box)) {
      if (card.dataset.auDone === card.textContent.length + '') continue;
      const ids = [...card.querySelectorAll('.tc-body pre')].flatMap(pre => [...pre.textContent.matchAll(TOKEN)].map(m => m[1]));
      card.dataset.auDone = card.textContent.length + '';
      if (!ids.length || (card.nextElementSibling?.classList.contains('au-clips'))) continue;
      const row = document.createElement('div');
      row.className = 'au-clips';
      [...new Set(ids)].forEach(id => row.append(playerEl(id)));
      card.after(row);     // outside the (usually collapsed) card, so the player is visible
    }
  }

  // ------------------------------------------------------------------ read aloud

  const reader = {btn: null, audio: null, abort: null, url: null};

  function readableText(turn) {
    const bubble = turn.querySelector('.bubble');
    if (!bubble) return '';
    const c = bubble.cloneNode(true);
    c.querySelectorAll('.toolcard, details.think, .codeblock, pre, .cursor, .au-clip, .au-clips, script, style').forEach(e => e.remove());
    c.querySelectorAll('p, li, h1, h2, h3, h4, tr, blockquote, br').forEach(e => e.append('\n'));
    return c.textContent
      .replace(TOKEN, ' ')
      .replace(/https?:\/\/\S+/g, ' link ')
      .replace(/[*_`#>|~]+/g, ' ')
      .replace(/[ \t]+/g, ' ').replace(/\s*\n\s*/g, '\n').trim();
  }

  function truncate(text) {
    if (text.length <= READ_MAX) return {text, cut: false};
    const head = text.slice(0, READ_MAX);
    const k = Math.max(head.lastIndexOf('. '), head.lastIndexOf('! '), head.lastIndexOf('? '), head.lastIndexOf('\n'));
    return {text: k > READ_MAX * 0.5 ? head.slice(0, k + 1) : head, cut: true};
  }

  function resetReader() {
    reader.abort?.abort();
    if (reader.audio) { reader.audio.pause(); reader.audio.src = ''; }
    if (reader.url) URL.revokeObjectURL(reader.url);
    if (reader.btn) { reader.btn.textContent = '🔊 read aloud'; reader.btn.classList.remove('on'); reader.btn.disabled = false; }
    Object.assign(reader, {btn: null, audio: null, abort: null, url: null});
  }

  async function readAloud(btn, turn) {
    if (reader.btn === btn) { resetReader(); return; }
    resetReader();
    const model = AudioState.tts()[0];
    if (!model) { toast('Load a text-to-speech model on the Models page.', 'err'); return; }
    const {text, cut} = truncate(readableText(turn));
    if (!text) { toast('There is nothing to read in this message.', 'info'); return; }
    const prefs = store.get('jllm.audio', {});
    const abort = new AbortController();
    Object.assign(reader, {btn, abort});
    btn.textContent = '… preparing'; btn.classList.add('on');
    try {
      const r = await AudioIO.speak({input: text, model: model.id, voice: prefs.ttsVoice || '', speed: prefs.ttsSpeed || 1, signal: abort.signal});
      if (reader.btn !== btn) return;
      if (cut) toast(`Reading the first ${text.length} characters.`, 'info', 3000);
      reader.url = URL.createObjectURL(new Blob([r.wav], {type: 'audio/wav'}));
      const a = reader.audio = document.createElement('audio');
      a.src = reader.url;
      a.addEventListener('ended', () => { if (reader.btn === btn) resetReader(); });
      a.addEventListener('error', () => { if (reader.btn === btn) resetReader(); });
      btn.textContent = '⏹ stop reading';
      await a.play();
    } catch (e) {
      if (e.name !== 'AbortError') toast(e.message, 'err', 8000);
      if (reader.btn === btn) resetReader();
    }
  }

  function addReadButtons() {
    for (const turn of $$('#messages .turn.assistant')) {
      const actions = turn.querySelector('.actions');
      if (!actions || actions.querySelector('.au-speak')) continue;
      const b = document.createElement('button');
      b.type = 'button'; b.className = 'au-speak'; b.dataset.a = 'speak'; b.textContent = '🔊 read aloud';
      b.setAttribute('aria-label', 'Read this answer aloud');
      b.addEventListener('click', () => readAloud(b, turn));
      actions.append(b);
    }
  }

  // ------------------------------------------------------------------ observing the message list

  let scheduled = false;
  function process() {
    scheduled = false;
    const box = $('#messages');
    if (!box) return;
    mo.disconnect();     // our own changes must not trigger another pass
    try {
      for (const b of $$('.turn .bubble', box)) {
        if (b.textContent.includes('[audio:')) playersInText(b);
      }
      playersForToolCards(box);
      addReadButtons();
    } catch (e) { console.error(e); }
    mo.observe(box, {childList: true, subtree: true, characterData: true});
  }
  const mo = new MutationObserver(() => { if (!scheduled) { scheduled = true; requestAnimationFrame(process); } });

  function init() {
    classes();
    AudioState.onChange(classes);
    injectComposer();
    const box = $('#messages');
    if (box) { mo.observe(box, {childList: true, subtree: true, characterData: true}); process(); }
    window.addEventListener('pagehide', () => resetReader());
  }

  init();
  return {playerEl};
})();
