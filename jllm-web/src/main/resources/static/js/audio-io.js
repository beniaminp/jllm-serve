'use strict';
// Audio in the browser: decoding, microphone recording, playback and the calls to the server's audio API.
// Used by the Audio tab (audio.js, audio-editor.js) and the chat controls (audio-chat.js). Needs core.js and audio-edit.js.

const AudioState = (() => {
  const s = {models: [], available: false, listeners: [], loadedOnce: false};
  s.stt = () => s.models.filter(m => m.kind === 'speech-to-text');
  s.tts = () => s.models.filter(m => m.kind === 'text-to-speech');
  s.hasStt = () => s.stt().length > 0;
  s.hasTts = () => s.tts().length > 0;
  s.onChange = f => { s.listeners.push(f); };
  let busy = false;
  s.refresh = async () => {
    if (busy) return s.models;
    busy = true;
    try {
      const list = await api('/api/audio/models');
      const next = Array.isArray(list) ? list : [];
      s.available = true;
      const changed = JSON.stringify(next) !== JSON.stringify(s.models) || !s.loadedOnce;
      s.models = next;
      s.loadedOnce = true;
      if (changed) s.listeners.forEach(f => { try { f(s.models); } catch (e) { console.error(e); } });
    } catch { /* the server may not have the audio API; keep the last list */ } finally { busy = false; }
    return s.models;
  };
  return s;
})();

const AudioIO = (() => {
  const E = AudioEdit;
  const AC = window.AudioContext || window.webkitAudioContext;

  // ------------------------------------------------------------------ the shared playback context

  let shared = null;
  function ctx() {
    if (!AC) throw new Error('This browser has no Web Audio support.');
    if (!shared || shared.state === 'closed') shared = new AC();
    if (shared.state === 'suspended') shared.resume().catch(() => {});
    return shared;
  }

  // ------------------------------------------------------------------ decoding

  /** Decodes any audio file the browser can play (and WAV itself, at its own sample rate) into a clip. */
  async function decode(data) {
    const buf = data instanceof ArrayBuffer ? data : await data.arrayBuffer();
    if (E.isWav(buf)) {
      try { return E.decodeWav(buf); } catch { /* fall through to the browser's decoder */ }
    }
    const c = ctx();
    const ab = await new Promise((resolve, reject) => {
      const p = c.decodeAudioData(buf.slice(0), resolve, reject);   // callback form for older Safari
      if (p && p.catch) p.catch(reject);
    }).catch(() => { throw new Error('The browser cannot decode this file. Try a WAV, MP3, M4A, Ogg or FLAC file.'); });
    const chans = [];
    for (let k = 0; k < ab.numberOfChannels; k++) chans.push(ab.getChannelData(k).slice());
    return E.clip(ab.sampleRate, chans);
  }

  /** The WAV bytes the server wants: 16-bit mono at `rate` (default: the clip's rate). */
  function toWavBlob(clip, rate = 0, mono = true) {
    let c = mono ? E.toMono(clip) : clip;
    return new Blob([E.encodeWav(c, 16, rate || c.sampleRate)], {type: 'audio/wav'});
  }

  // ------------------------------------------------------------------ playback

  /** Makes an AudioBuffer for a clip (cached by clip object). */
  const bufCache = new WeakMap();
  function audioBuffer(clip) {
    let b = bufCache.get(clip);
    if (!b) {
      const n = Math.max(1, E.length(clip));
      b = ctx().createBuffer(clip.channels.length, n, Math.min(96000, Math.max(8000, clip.sampleRate)));
      clip.channels.forEach((ch, k) => b.copyToChannel(ch, k));
      bufCache.set(clip, b);
    }
    return b;
  }

  /**
   * A player for one clip at a time. play(clip, fromSec, toSec, loop) starts; position() is the playhead in seconds
   * (null when stopped); onEnd fires when playback finishes by itself (not on stop()).
   */
  function player() {
    let src = null, t0 = 0, from = 0, to = 0, loop = false;
    const p = {
      onEnd: null,
      playing: () => !!src,
      play(clip, fromSec = 0, toSec = null, lp = false) {
        p.stop();
        const dur = E.duration(clip);
        from = Math.min(Math.max(0, fromSec), dur);
        to = toSec == null ? dur : Math.min(Math.max(from, toSec), dur);
        if (to - from < 0.001) return false;
        loop = lp;
        const c = ctx();
        const s = c.createBufferSource();
        s.buffer = audioBuffer(clip);
        s.connect(c.destination);
        if (loop) { s.loop = true; s.loopStart = from; s.loopEnd = to; s.start(0, from); }
        else s.start(0, from, to - from);
        s.onended = () => { if (src === s) { src = null; p.onEnd?.(); } };
        src = s; t0 = c.currentTime;
        return true;
      },
      /** Stops; returns the playhead position (seconds) it had. */
      stop() {
        const pos = p.position();
        if (src) { const s = src; src = null; s.onended = null; try { s.stop(); } catch { /* already stopped */ } s.disconnect(); }
        return pos;
      },
      position() {
        if (!src) return null;
        const el = ctx().currentTime - t0;
        if (loop) return from + (el % Math.max(0.001, to - from));
        return Math.min(to, from + el);
      },
    };
    return p;
  }

  // ------------------------------------------------------------------ microphone

  const WORKLET = `class Rec extends AudioWorkletProcessor {
    process(inputs) { const ch = inputs[0] && inputs[0][0]; if (ch) this.port.postMessage(ch.slice(0)); return true; }
  }
  registerProcessor('jllm-rec', Rec);`;

  function micMessage(e) {
    if (!navigator.mediaDevices?.getUserMedia) return 'The microphone needs a secure page: open the console on https:// or http://localhost.';
    switch (e?.name) {
      case 'NotAllowedError': case 'SecurityError': return 'Microphone permission was denied. Allow it in the browser (address bar, site settings) and try again.';
      case 'NotFoundError': case 'OverconstrainedError': return 'No microphone was found.';
      case 'NotReadableError': case 'AbortError': return 'The microphone is in use by another application or cannot be opened.';
      default: return 'Cannot record: ' + (e?.message || e);
    }
  }

  /**
   * Records the microphone to a mono clip. start() asks for permission; stop() resolves to the clip (null when nothing
   * was captured). onLevel(peak 0..1) is called while recording, onTime(seconds) about every 100 ms.
   */
  class Recorder {
    constructor({onLevel, onTime, raw = false, maxSeconds = 1800} = {}) {
      Object.assign(this, {onLevel, onTime, raw, maxSeconds});
      this.chunks = []; this.frames = 0; this.active = false;
    }

    async start() {
      if (!navigator.mediaDevices?.getUserMedia || !AC) throw new Error(micMessage(null));
      try {
        this.stream = await navigator.mediaDevices.getUserMedia({audio: this.raw
          ? {echoCancellation: false, noiseSuppression: false, autoGainControl: false, channelCount: 1}
          : {channelCount: 1}});
      } catch (e) { throw new Error(micMessage(e)); }
      try {
        this.ac = new AC();
        if (this.ac.state === 'suspended') await this.ac.resume();
        this.rate = this.ac.sampleRate;
        this.src = this.ac.createMediaStreamSource(this.stream);
        this.sink = this.ac.createGain();
        this.sink.gain.value = 0;
        this.sink.connect(this.ac.destination);
        const take = ch => this._take(ch);
        let ok = false;
        if (this.ac.audioWorklet && window.AudioWorkletNode) {
          try {
            const url = URL.createObjectURL(new Blob([WORKLET], {type: 'application/javascript'}));
            await this.ac.audioWorklet.addModule(url);
            URL.revokeObjectURL(url);
            this.node = new AudioWorkletNode(this.ac, 'jllm-rec', {numberOfInputs: 1, numberOfOutputs: 1, channelCount: 1});
            this.node.port.onmessage = ev => take(ev.data);
            ok = true;
          } catch { this.node = null; }
        }
        if (!ok) {   // ScriptProcessor: deprecated but available everywhere
          this.node = this.ac.createScriptProcessor(4096, 1, 1);
          this.node.onaudioprocess = ev => take(ev.inputBuffer.getChannelData(0).slice(0));
        }
        this.src.connect(this.node);
        this.node.connect(this.sink);
        this.active = true;
        this.t0 = performance.now();
        this.timer = setInterval(() => this.onTime?.(this.frames / this.rate), 100);
      } catch (e) { this._release(); throw new Error('Cannot record: ' + (e?.message || e)); }
    }

    _take(ch) {
      if (!this.active) return;
      this.chunks.push(ch);
      this.frames += ch.length;
      if (this.onLevel) { let m = 0; for (let i = 0; i < ch.length; i++) { const v = Math.abs(ch[i]); if (v > m) m = v; } this.onLevel(m); }
      if (this.frames / this.rate >= this.maxSeconds && this.onLimit) this.onLimit();
    }

    _release() {
      this.active = false;
      clearInterval(this.timer);
      try { this.node?.disconnect(); this.src?.disconnect(); this.sink?.disconnect(); } catch { /* ignore */ }
      if (this.node?.port) this.node.port.onmessage = null;
      this.stream?.getTracks().forEach(t => t.stop());
      this.ac?.close().catch(() => {});
    }

    async stop() {
      if (!this.ac) return null;
      this._release();
      if (!this.frames) return null;
      const out = new Float32Array(this.frames);
      let at = 0;
      for (const ch of this.chunks) { out.set(ch, at); at += ch.length; }
      this.chunks = [];
      return E.clip(this.rate, [out]);
    }

    cancel() { this._release(); this.chunks = []; this.frames = 0; }
  }

  // ------------------------------------------------------------------ the server's audio API

  async function failure(res) {
    const text = await res.text();
    let msg = text;
    try { const j = JSON.parse(text); msg = j.error?.message || j.error || j.message || text; } catch { /* plain text */ }
    if (typeof msg !== 'string') msg = JSON.stringify(msg);
    const e = new Error(msg || res.statusText);
    e.status = res.status;
    try { e.code = JSON.parse(text).error?.code; } catch { /* none */ }
    return e;
  }

  /** Turns the server's verbose_json into {text, language, duration, segments:[{start,end,text}], words:[{word,start,end}], wordsApprox}. */
  function normalize(r) {
    const segments = (r.segments || []).map(s => ({start: s.start, end: s.end, text: s.text || '', words: s.words || null,
      avg_logprob: s.avg_logprob, no_speech_prob: s.no_speech_prob}));
    let words = Array.isArray(r.words) ? r.words.map(w => ({word: w.word, start: w.start, end: w.end})) : null;
    let approx = false;
    if (!words) {
      const fromSeg = segments.flatMap(s => s.words || []).map(w => ({word: w.word, start: w.start, end: w.end}));
      if (fromSeg.length) words = fromSeg;
    }
    if (!words || !words.length) { words = segments.flatMap(s => E.distributeWords(s)); approx = true; }
    return {text: (r.text || '').trim(), language: r.language || '', duration: r.duration, segments, words, wordsApprox: approx};
  }

  /**
   * POST /v1/audio/transcriptions (or translations). `wav` is a Blob. Resolves to the normalized verbose_json.
   * opts: {model, language, translate, prompt, words, temperature, signal}
   */
  async function transcribe(wav, opts = {}) {
    const f = new FormData();
    f.append('file', wav, 'audio.wav');
    if (opts.model) f.append('model', opts.model);
    if (opts.language && !opts.translate) f.append('language', opts.language);
    if (opts.prompt) f.append('prompt', opts.prompt);
    if (opts.temperature != null) f.append('temperature', String(opts.temperature));
    f.append('response_format', 'verbose_json');
    if (opts.words) f.append('timestamp_granularities[]', 'word');
    const res = await fetch(opts.translate ? '/v1/audio/translations' : '/v1/audio/transcriptions',
      {method: 'POST', headers: authHeaders(), body: f, signal: opts.signal});
    if (!res.ok) throw await failure(res);
    return normalize(await res.json());
  }

  /** POST /v1/audio/speech; resolves to {clip, wav (ArrayBuffer)}. */
  async function speak(opts) {
    const body = {input: opts.input, response_format: 'wav'};
    if (opts.model) body.model = opts.model;
    if (opts.voice) body.voice = opts.voice;
    if (opts.speed && opts.speed !== 1) body.speed = opts.speed;
    const res = await fetch('/v1/audio/speech', {method: 'POST', headers: {'Content-Type': 'application/json', ...authHeaders()},
      body: JSON.stringify(body), signal: opts.signal});
    if (!res.ok) {
      const e = await failure(res);
      if (e.status === 404 || e.code === 'model_not_found') e.message = 'No text-to-speech model is loaded. Load one (for example OuteAI/OuteTTS-0.3-500M) on the Models page.';
      throw e;
    }
    const wav = await res.arrayBuffer();
    return {wav, clip: await decode(wav)};
  }

  /** Stores a WAV blob in the server's clip store; resolves to {id, seconds, ...}. */
  async function uploadClip(blob, label = '') {
    const res = await fetch('/api/audio/files?label=' + encodeURIComponent(label), {method: 'POST',
      headers: {'Content-Type': 'audio/wav', ...authHeaders()}, body: blob});
    if (!res.ok) throw await failure(res);
    return res.json();
  }

  // blob URLs of stored clips, by id (an <audio src> cannot send the Authorization header)
  const clipUrls = new Map();
  function clipUrl(id) {
    let p = clipUrls.get(id);
    if (!p) {
      p = fetch('/api/audio/files/' + encodeURIComponent(id), {headers: authHeaders()})
        .then(async r => { if (!r.ok) throw await failure(r); return URL.createObjectURL(await r.blob()); });
      p.catch(() => clipUrls.delete(id));
      clipUrls.set(id, p);
    }
    return p;
  }

  async function voices(model) {
    return api('/api/audio/voices' + (model ? '?model=' + encodeURIComponent(model) : ''));
  }

  const LANG_NAMES = {en: 'English', zh: 'Chinese', de: 'German', es: 'Spanish', ru: 'Russian', ko: 'Korean', fr: 'French', ja: 'Japanese',
    pt: 'Portuguese', tr: 'Turkish', pl: 'Polish', ca: 'Catalan', nl: 'Dutch', ar: 'Arabic', sv: 'Swedish', it: 'Italian', id: 'Indonesian',
    hi: 'Hindi', fi: 'Finnish', vi: 'Vietnamese', he: 'Hebrew', uk: 'Ukrainian', el: 'Greek', ms: 'Malay', cs: 'Czech', ro: 'Romanian',
    da: 'Danish', hu: 'Hungarian', ta: 'Tamil', no: 'Norwegian', th: 'Thai', ur: 'Urdu', hr: 'Croatian', bg: 'Bulgarian', lt: 'Lithuanian',
    la: 'Latin', mi: 'Maori', ml: 'Malayalam', cy: 'Welsh', sk: 'Slovak', te: 'Telugu', fa: 'Persian', lv: 'Latvian', bn: 'Bengali',
    sr: 'Serbian', az: 'Azerbaijani', sl: 'Slovenian', kn: 'Kannada', et: 'Estonian', mk: 'Macedonian', br: 'Breton', eu: 'Basque',
    is: 'Icelandic', hy: 'Armenian', ne: 'Nepali', mn: 'Mongolian', bs: 'Bosnian', kk: 'Kazakh', sq: 'Albanian', sw: 'Swahili',
    gl: 'Galician', mr: 'Marathi', pa: 'Punjabi', si: 'Sinhala', km: 'Khmer', af: 'Afrikaans', be: 'Belarusian', ka: 'Georgian'};
  const langName = code => LANG_NAMES[code] ? `${LANG_NAMES[code]} (${code})` : code;

  function saveBlob(name, blob) {
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = name;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 4000);
  }

  /**
   * Wires a record / stop toggle button with a level meter (an element whose first child is the bar) and an elapsed time
   * element. onClip(clip) gets the finished recording. Returns {toggle, cancel, recording}.
   */
  function bindRecorder({button, meter, time, raw = false, onClip, onState, labelIdle = '● Record', labelOn = '■ Stop'}) {
    let rec = null, peak = 0, tick = 0;
    const setUi = on => {
      button.classList.toggle('recording', on);
      button.textContent = on ? labelOn : labelIdle;
      button.setAttribute('aria-pressed', String(on));
      meter?.classList.toggle('hidden', !on);
      time?.classList.toggle('hidden', !on);
      if (!on && meter?.firstElementChild) meter.firstElementChild.style.width = '0%';
      onState?.(on);
    };
    async function start() {
      if (rec) return;
      const r = new Recorder({raw, onLevel: l => { if (l > peak) peak = l; }, onTime: t => { if (time) time.textContent = E.clock(t); }});
      r.onLimit = () => { if (rec === r) stop(); };
      button.disabled = true;
      try { await r.start(); } catch (e) { toast(e.message, 'err', 9000); button.disabled = false; return; }
      button.disabled = false;
      rec = r;
      if (time) time.textContent = '0:00.0';
      tick = setInterval(() => { if (meter?.firstElementChild) meter.firstElementChild.style.width = Math.min(100, peak * 130) + '%'; peak *= 0.6; }, 50);
      setUi(true);
    }
    async function stop() {
      const r = rec;
      if (!r) return;
      rec = null;
      clearInterval(tick);
      setUi(false);
      const clip = await r.stop();
      if (clip) onClip(clip); else toast('Nothing was recorded.', 'err');
    }
    function cancel() { if (rec) { rec.cancel(); rec = null; clearInterval(tick); setUi(false); } }
    button.addEventListener('click', () => (rec ? stop() : start()));
    setUi(false);
    return {toggle: () => (rec ? stop() : start()), cancel, get recording() { return !!rec; }};
  }

  return {bindRecorder, ctx, decode, toWavBlob, audioBuffer, player, Recorder, micMessage, transcribe, speak, uploadClip, clipUrl, voices, normalize,
    langName, saveBlob, failure};
})();
