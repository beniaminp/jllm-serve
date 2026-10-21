'use strict';
// Pure audio editing functions for the console's Audio tab (no DOM, no Web Audio: runs in the browser and in node).
//
// A clip is {sampleRate, channels: [Float32Array, ...]} (all channels the same length). Every function returns a new clip
// (or new arrays) and never changes its arguments, so the editor can keep earlier clips as undo snapshots.
// Positions are frame (sample) indexes; ranges are half open [a, b).
(function (root) {
  const AudioEdit = {};

  // ------------------------------------------------------------------ basics

  const clamp = (x, lo, hi) => x < lo ? lo : x > hi ? hi : x;
  const dbToGain = db => Math.pow(10, db / 20);
  const gainToDb = g => g > 0 ? 20 * Math.log10(g) : -Infinity;

  const clip = (sampleRate, channels) => ({sampleRate, channels});
  const length = c => c.channels.length ? c.channels[0].length : 0;
  const duration = c => length(c) / c.sampleRate;
  const empty = (sampleRate, nch = 1) => clip(sampleRate, Array.from({length: nch}, () => new Float32Array(0)));
  const silent = (sampleRate, nch, n) => clip(sampleRate, Array.from({length: nch}, () => new Float32Array(n)));
  const range = (c, a, b) => {
    const n = length(c);
    a = clamp(Math.round(a ?? 0), 0, n);
    b = clamp(Math.round(b ?? n), 0, n);
    return a <= b ? [a, b] : [b, a];
  };
  Object.assign(AudioEdit, {clamp, dbToGain, gainToDb, clip, length, duration, empty, silent});

  AudioEdit.slice = (c, a, b) => {
    [a, b] = range(c, a, b);
    return clip(c.sampleRate, c.channels.map(ch => ch.slice(a, b)));
  };

  /** Appends clips of the same rate and channel count end to end. */
  AudioEdit.concat = parts => {
    const first = parts.find(p => p) || empty(44100);
    const nch = first.channels.length;
    const total = parts.reduce((s, p) => s + length(p), 0);
    const out = silent(first.sampleRate, nch, total);
    let at = 0;
    for (const p of parts) {
      for (let k = 0; k < nch; k++) out.channels[k].set(p.channels[k], at);
      at += length(p);
    }
    return out;
  };

  /** Mixes a clip down to one channel (average). */
  AudioEdit.toMono = c => {
    if (c.channels.length === 1) return c;
    const n = length(c), m = new Float32Array(n), k = c.channels.length;
    for (const ch of c.channels) for (let i = 0; i < n; i++) m[i] += ch[i];
    for (let i = 0; i < n; i++) m[i] /= k;
    return clip(c.sampleRate, [m]);
  };

  /** Gives a clip `nch` channels: mono is duplicated, more channels are averaged, extra channels are dropped. */
  AudioEdit.withChannels = (c, nch) => {
    if (c.channels.length === nch) return c;
    if (nch === 1) return AudioEdit.toMono(c);
    if (c.channels.length === 1) return clip(c.sampleRate, Array.from({length: nch}, () => c.channels[0]));
    const chans = [];
    for (let k = 0; k < nch; k++) chans.push(c.channels[Math.min(k, c.channels.length - 1)]);
    return clip(c.sampleRate, chans);
  };

  // ------------------------------------------------------------------ resampling

  /**
   * Windowed-sinc resampling of one channel (Lanczos kernel, a = 8 lobes); when going down the kernel is stretched so
   * that nothing above the new Nyquist frequency remains.
   */
  AudioEdit.resampleChannel = (x, from, to) => {
    if (from === to || x.length === 0) return x.slice();
    const ratio = to / from;
    const n = Math.max(1, Math.round(x.length * ratio));
    const out = new Float32Array(n);
    const lobes = 8;
    const scale = Math.min(1, ratio);          // < 1 when downsampling: widen the kernel
    const support = lobes / scale;
    for (let i = 0; i < n; i++) {
      const t = (i + 0.5) / ratio - 0.5;       // position in the input
      const lo = Math.max(0, Math.ceil(t - support)), hi = Math.min(x.length - 1, Math.floor(t + support));
      let acc = 0, wsum = 0;
      for (let j = lo; j <= hi; j++) {
        const d = (j - t) * scale;
        let w;
        if (d === 0) w = 1;
        else if (Math.abs(d) >= lobes) continue;
        else { const pd = Math.PI * d; w = lobes * Math.sin(pd) * Math.sin(pd / lobes) / (pd * pd); }
        acc += w * x[j];
        wsum += w;
      }
      out[i] = wsum !== 0 ? acc / wsum : 0;
    }
    return out;
  };

  AudioEdit.resample = (c, rate) => {
    rate = Math.round(rate);
    if (rate === c.sampleRate) return c;
    return clip(rate, c.channels.map(ch => AudioEdit.resampleChannel(ch, c.sampleRate, rate)));
  };

  // ------------------------------------------------------------------ range edits

  /** Removes [a, b). */
  AudioEdit.deleteRange = (c, a, b) => {
    [a, b] = range(c, a, b);
    return AudioEdit.concat([AudioEdit.slice(c, 0, a), AudioEdit.slice(c, b, length(c))]);
  };

  /** Brings `piece` to the rate and channel count of `c`. */
  AudioEdit.conform = (piece, c) => AudioEdit.withChannels(AudioEdit.resample(piece, c.sampleRate), c.channels.length);

  /** Inserts `piece` at frame `at` (the piece is converted to the clip's rate and channel count). */
  AudioEdit.insert = (c, at, piece) => {
    at = clamp(Math.round(at), 0, length(c));
    const p = AudioEdit.conform(piece, c);
    return AudioEdit.concat([AudioEdit.slice(c, 0, at), p, AudioEdit.slice(c, at, length(c))]);
  };

  /** Replaces [a, b) with `piece`. */
  AudioEdit.replace = (c, a, b, piece) => {
    [a, b] = range(c, a, b);
    const p = AudioEdit.conform(piece, c);
    return AudioEdit.concat([AudioEdit.slice(c, 0, a), p, AudioEdit.slice(c, b, length(c))]);
  };

  /** Inserts `frames` frames of silence at `at`. */
  AudioEdit.insertSilence = (c, at, frames) =>
    AudioEdit.insert(c, at, silent(c.sampleRate, c.channels.length, Math.max(0, Math.round(frames))));

  const mapRange = (c, a, b, fn) => {
    [a, b] = range(c, a, b);
    return clip(c.sampleRate, c.channels.map(ch => { const o = ch.slice(); fn(o, a, b); return o; }));
  };

  /** Fades [a, b) in (gain 0 to 1) or out (1 to 0); curve "linear" or "cosine" (smoother, equal-power-like). */
  AudioEdit.fade = (c, a, b, dir, curve = 'cosine') => mapRange(c, a, b, (o, a, b) => {
    const n = b - a;
    for (let i = 0; i < n; i++) {
      let g = n > 1 ? i / (n - 1) : 1;
      if (dir === 'out') g = 1 - g;
      if (curve === 'cosine') g = 0.5 - 0.5 * Math.cos(Math.PI * g);
      o[a + i] *= g;
    }
  });

  AudioEdit.gain = (c, a, b, db) => {
    const g = dbToGain(db);
    return mapRange(c, a, b, (o, a, b) => { for (let i = a; i < b; i++) o[i] *= g; });
  };

  AudioEdit.reverse = (c, a, b) => mapRange(c, a, b, (o, a, b) => {
    for (let i = a, j = b - 1; i < j; i++, j--) { const t = o[i]; o[i] = o[j]; o[j] = t; }
  });

  AudioEdit.peak = (c, a, b) => {
    [a, b] = range(c, a, b);
    let p = 0;
    for (const ch of c.channels) for (let i = a; i < b; i++) { const v = Math.abs(ch[i]); if (v > p) p = v; }
    return p;
  };

  AudioEdit.rms = (c, a, b) => {
    [a, b] = range(c, a, b);
    let s = 0;
    for (const ch of c.channels) for (let i = a; i < b; i++) s += ch[i] * ch[i];
    const n = (b - a) * c.channels.length;
    return n ? Math.sqrt(s / n) : 0;
  };

  /** Gain that brings the range to a peak or RMS target (dBFS); 1 when the range is silent. */
  AudioEdit.normalizeGain = (c, a, b, mode, targetDb) => {
    const level = mode === 'rms' ? AudioEdit.rms(c, a, b) : AudioEdit.peak(c, a, b);
    return level > 1e-9 ? dbToGain(targetDb) / level : 1;
  };

  /** Normalizes [a, b) to `targetDb` peak or RMS. With RMS the peaks may exceed 0 dBFS (the export clamps). */
  AudioEdit.normalize = (c, a, b, mode = 'peak', targetDb = -1) => {
    const g = AudioEdit.normalizeGain(c, a, b, mode, targetDb);
    return mapRange(c, a, b, (o, a, b) => { for (let i = a; i < b; i++) o[i] *= g; });
  };

  /** Adds a DC-free soft limiter to values beyond +-1 (tanh above the knee), used before export of mixes. */
  AudioEdit.limit = (c, knee = 0.9) => clip(c.sampleRate, c.channels.map(ch => {
    const o = ch.slice();
    const range1 = 1 - knee;
    for (let i = 0; i < o.length; i++) {
      const v = Math.abs(o[i]);
      if (v > knee) o[i] = Math.sign(o[i]) * (knee + range1 * Math.tanh((v - knee) / range1));
    }
    return o;
  }));

  /** Adds `other` (scaled by `db`) into `c` starting at frame `at`; the clip grows when the other one ends later. */
  AudioEdit.mix = (c, other, at, db = 0) => {
    at = clamp(Math.round(at), 0, length(c));
    const o = AudioEdit.conform(other, c);
    const g = dbToGain(db);
    const n = Math.max(length(c), at + length(o));
    const out = silent(c.sampleRate, c.channels.length, n);
    for (let k = 0; k < c.channels.length; k++) {
      out.channels[k].set(c.channels[k]);
      const dst = out.channels[k], src = o.channels[k];
      for (let i = 0; i < src.length; i++) dst[at + i] += src[i] * g;
    }
    return out;
  };

  // ------------------------------------------------------------------ silence detection

  /** Per-frame magnitude (max over channels) between a and b, smoothed by a short running maximum is not needed: callers window it. */
  const frameLevel = (c, i) => { let m = 0; for (const ch of c.channels) { const v = Math.abs(ch[i]); if (v > m) m = v; } return m; };

  /**
   * Finds the stretches of [a, b) whose level stays below `thresholdDb` for at least `minSec`. Level is the peak over 10 ms
   * windows so that a single quiet crossing inside speech does not count as silence. Returns [[start, end), ...].
   */
  AudioEdit.findSilences = (c, a, b, thresholdDb = -40, minSec = 0.5) => {
    [a, b] = range(c, a, b);
    const thr = dbToGain(thresholdDb);
    const win = Math.max(1, Math.round(c.sampleRate * 0.01));
    const minFrames = Math.max(1, Math.round(minSec * c.sampleRate));
    const out = [];
    let start = -1;
    for (let w = a; w < b; w += win) {
      const e = Math.min(b, w + win);
      let m = 0;
      for (let i = w; i < e; i++) { const v = frameLevel(c, i); if (v > m) m = v; }
      if (m < thr) { if (start < 0) start = w; }
      else if (start >= 0) { if (w - start >= minFrames) out.push([start, w]); start = -1; }
    }
    if (start >= 0 && b - start >= minFrames) out.push([start, b]);
    return out;
  };

  /**
   * Shortens every silence found by findSilences to `keepSec` (half at each edge, cross-faded). Returns
   * {clip, removedFrames, count}.
   */
  AudioEdit.removeSilence = (c, a, b, thresholdDb = -40, minSec = 0.5, keepSec = 0.1) => {
    [a, b] = range(c, a, b);
    const runs = AudioEdit.findSilences(c, a, b, thresholdDb, minSec);
    const keep = Math.round(keepSec * c.sampleRate);
    const cuts = [];
    for (const [s, e] of runs) {
      const half = Math.min(Math.floor(keep / 2), Math.floor((e - s) / 2));
      if (e - s - 2 * half > 0) cuts.push([s + half, e - half]);
    }
    if (!cuts.length) return {clip: c, removedFrames: 0, count: 0};
    const parts = [], joins = [];
    let at = 0;
    for (const [s, e] of cuts) { parts.push(AudioEdit.slice(c, at, s)); joins.push(Math.round(c.sampleRate * 0.005)); at = e; }
    parts.push(AudioEdit.slice(c, at, length(c)));
    const out = AudioEdit.concatCrossfade(parts, joins);
    return {clip: out, removedFrames: length(c) - length(out), count: cuts.length};
  };

  /** Frames of leading and trailing silence below `thresholdDb`: returns [first, last+1) of the non-silent part. */
  AudioEdit.activeRange = (c, thresholdDb = -45) => {
    const thr = dbToGain(thresholdDb), n = length(c);
    let a = 0, b = n;
    while (a < n && frameLevel(c, a) < thr) a++;
    while (b > a && frameLevel(c, b - 1) < thr) b--;
    return [a, b];
  };

  AudioEdit.trimSilence = (c, thresholdDb = -45, padSec = 0.01) => {
    let [a, b] = AudioEdit.activeRange(c, thresholdDb);
    if (b <= a) return c;
    const pad = Math.round(padSec * c.sampleRate);
    return AudioEdit.slice(c, a - pad, b + pad);
  };

  // ------------------------------------------------------------------ joins with cross-fades

  /**
   * Joins clips overlapping each join by joins[i] frames with an equal-power cross-fade (so the result is shorter by the
   * sum of the overlaps). A join of 0 is a plain concatenation. Overlaps are capped at half of the shorter neighbour.
   */
  AudioEdit.concatCrossfade = (parts, joins) => {
    parts = parts.filter(p => p);
    const first = parts[0];
    if (!first) return empty(44100);
    const nch = first.channels.length;
    let cur = first.channels.map(ch => Float32Array.from(ch));
    for (let p = 1; p < parts.length; p++) {
      const next = parts[p];
      let x = Math.round(joins[p - 1] || 0);
      x = Math.min(x, cur[0].length >> 1, next.channels[0].length >> 1);
      if (x < 0) x = 0;
      const outLen = cur[0].length + next.channels[0].length - x;
      const out = [];
      for (let k = 0; k < nch; k++) {
        const o = new Float32Array(outLen), a = cur[k], b = next.channels[k];
        o.set(a.subarray(0, a.length - x));
        for (let i = 0; i < x; i++) {
          const t = (i + 0.5) / x;
          o[a.length - x + i] = a[a.length - x + i] * Math.cos(t * Math.PI / 2) + b[i] * Math.sin(t * Math.PI / 2);
        }
        o.set(b.subarray(x), a.length);
        out.push(o);
      }
      cur = out;
    }
    return clip(first.sampleRate, cur);
  };

  // ------------------------------------------------------------------ speed and time stretching

  /** Plays [a, b) `factor` times faster by resampling (pitch changes with speed, like a tape). */
  AudioEdit.changeSpeed = (c, a, b, factor) => {
    [a, b] = range(c, a, b);
    factor = clamp(factor, 0.1, 10);
    const mid = AudioEdit.slice(c, a, b);
    // reading the same samples as if recorded at rate*factor, then resampling back to the clip's rate
    const fast = clip(Math.max(1, Math.round(c.sampleRate * factor)), mid.channels);
    const back = AudioEdit.resample(fast, c.sampleRate);
    return AudioEdit.concat([AudioEdit.slice(c, 0, a), clip(c.sampleRate, back.channels), AudioEdit.slice(c, b, length(c))]);
  };

  /**
   * WSOLA time stretch of one channel: the output is `factor` times faster (factor 2 = half as long) and keeps the pitch.
   * `frame` ~ 25 ms analysis windows, 50 % overlap, the best-matching source offset searched within +-`seek` samples.
   */
  AudioEdit.stretchChannel = (x, factor, sampleRate) => {
    if (Math.abs(factor - 1) < 1e-4 || x.length < 64) return x.slice();
    const frame = 2 * Math.max(16, Math.round(sampleRate * 0.0125));   // even
    const hop = frame >> 1;                                            // synthesis hop
    const seek = Math.max(8, Math.round(sampleRate * 0.006));
    const anaHop = hop * factor;                                       // nominal analysis hop
    const outLen = Math.max(1, Math.round(x.length / factor));
    const out = new Float32Array(outLen + frame), norm = new Float32Array(outLen + frame);
    const win = new Float32Array(frame);
    for (let i = 0; i < frame; i++) win[i] = 0.5 - 0.5 * Math.cos(2 * Math.PI * (i + 0.5) / frame);
    let prevPos = 0;                                                   // where the previous chosen frame started in x
    for (let o = 0, k = 0; o < outLen; o += hop, k++) {
      let pos = Math.round(k * anaHop);
      if (k > 0 && prevPos + 2 * hop <= x.length) {
        // the natural continuation of the previous frame is at prevPos + hop; look for the offset near pos whose start
        // best matches it (maximum cross-correlation of the first half of the frame)
        const target = prevPos + hop;
        let best = pos, bestScore = -Infinity;
        const lo = Math.max(0, pos - seek), hi = Math.min(x.length - frame, pos + seek);
        for (let cand = lo; cand <= hi; cand += 1) {
          let s = 0;
          for (let i = 0; i < hop; i += 2) s += x[cand + i] * x[target + i];
          if (s > bestScore) { bestScore = s; best = cand; }
        }
        if (hi >= lo) pos = best;
      }
      pos = clamp(pos, 0, Math.max(0, x.length - frame));
      for (let i = 0; i < frame; i++) {
        const v = pos + i < x.length ? x[pos + i] : 0;
        out[o + i] += v * win[i];
        norm[o + i] += win[i];
      }
      prevPos = pos;
    }
    const res = new Float32Array(outLen);
    for (let i = 0; i < outLen; i++) res[i] = norm[i] > 1e-3 ? out[i] / norm[i] : 0;
    return res;
  };

  /** Pitch-preserving speed change of [a, b) (WSOLA). */
  AudioEdit.timeStretch = (c, a, b, factor) => {
    [a, b] = range(c, a, b);
    factor = clamp(factor, 0.25, 4);
    const mid = c.channels.map(ch => AudioEdit.stretchChannel(ch.subarray(a, b), factor, c.sampleRate));
    return AudioEdit.concat([AudioEdit.slice(c, 0, a), clip(c.sampleRate, mid), AudioEdit.slice(c, b, length(c))]);
  };

  /** Stretches a whole clip to `frames` frames (pitch preserved when `keepPitch`, else by resampling). */
  AudioEdit.fitTo = (c, frames, keepPitch = true) => {
    const n = length(c);
    if (!n || !frames) return c;
    const factor = n / frames;
    const r = keepPitch ? AudioEdit.timeStretch(c, 0, n, factor) : AudioEdit.changeSpeed(c, 0, n, factor);
    return r;
  };

  // ------------------------------------------------------------------ peaks for drawing

  const BLOCK = 256;

  /** Min / max per block of 256 frames for every channel: a pyramid base so that long clips draw fast. */
  AudioEdit.buildPeaks = c => {
    const n = length(c), blocks = Math.ceil(n / BLOCK);
    const mins = [], maxs = [];
    for (const ch of c.channels) {
      const mn = new Float32Array(blocks), mx = new Float32Array(blocks);
      for (let b = 0; b < blocks; b++) {
        const s = b * BLOCK, e = Math.min(n, s + BLOCK);
        let lo = Infinity, hi = -Infinity;
        for (let i = s; i < e; i++) { const v = ch[i]; if (v < lo) lo = v; if (v > hi) hi = v; }
        mn[b] = lo; mx[b] = hi;
      }
      mins.push(mn); maxs.push(mx);
    }
    return {block: BLOCK, mins, maxs, frames: n};
  };

  /**
   * Min/max of channel `k` for `bins` equal slices of [start, end) (frames, may be fractional): returns {min, max}
   * Float32Arrays. Uses the raw samples when a bin covers fewer than 2 blocks, the block peaks otherwise.
   */
  AudioEdit.peaksFor = (c, pk, k, start, end, bins) => {
    const min = new Float32Array(bins), max = new Float32Array(bins);
    const ch = c.channels[k], n = ch.length, per = (end - start) / bins;
    for (let p = 0; p < bins; p++) {
      let s = Math.floor(start + p * per), e = Math.max(s + 1, Math.floor(start + (p + 1) * per));
      if (s >= n || e <= 0) { min[p] = 0; max[p] = 0; continue; }
      s = Math.max(0, s); e = Math.min(n, e);
      let lo = Infinity, hi = -Infinity;
      if (per < pk.block * 2) {
        for (let i = s; i < e; i++) { const v = ch[i]; if (v < lo) lo = v; if (v > hi) hi = v; }
      } else {
        const b0 = Math.ceil(s / pk.block), b1 = Math.floor(e / pk.block);
        for (let i = s; i < Math.min(e, b0 * pk.block); i++) { const v = ch[i]; if (v < lo) lo = v; if (v > hi) hi = v; }
        for (let b = b0; b < b1; b++) { if (pk.mins[k][b] < lo) lo = pk.mins[k][b]; if (pk.maxs[k][b] > hi) hi = pk.maxs[k][b]; }
        for (let i = Math.max(b1 * pk.block, b0 * pk.block); i < e; i++) { const v = ch[i]; if (v < lo) lo = v; if (v > hi) hi = v; }
      }
      min[p] = lo === Infinity ? 0 : lo; max[p] = hi === -Infinity ? 0 : hi;
    }
    return {min, max};
  };

  // ------------------------------------------------------------------ WAV

  const str = (dv, off, len) => { let s = ''; for (let i = 0; i < len; i++) s += String.fromCharCode(dv.getUint8(off + i)); return s; };

  /** True when the bytes start like a RIFF/WAVE file. */
  AudioEdit.isWav = buf => {
    if (buf.byteLength < 12) return false;
    const dv = new DataView(buf);
    return str(dv, 0, 4) === 'RIFF' && str(dv, 8, 4) === 'WAVE';
  };

  /** Decodes PCM 8/16/24/32-bit and 32/64-bit float WAV (also WAVE_FORMAT_EXTENSIBLE) into a clip at its own rate. */
  AudioEdit.decodeWav = buf => {
    const dv = new DataView(buf);
    if (!AudioEdit.isWav(buf)) throw new Error('not a WAV file');
    let fmt = null, dataOff = -1, dataLen = 0;
    for (let p = 12; p + 8 <= buf.byteLength;) {
      const id = str(dv, p, 4);
      let size = dv.getUint32(p + 4, true);
      if (id === 'fmt ') {
        fmt = {tag: dv.getUint16(p + 8, true), nch: dv.getUint16(p + 10, true), rate: dv.getUint32(p + 12, true),
          bits: dv.getUint16(p + 22, true)};
        if (fmt.tag === 0xFFFE && size >= 26) fmt.tag = dv.getUint16(p + 8 + 24, true);
      } else if (id === 'data') {
        dataOff = p + 8;
        if (size === 0xFFFFFFFF || dataOff + size > buf.byteLength) size = buf.byteLength - dataOff;   // streamed / truncated
        dataLen = size;
        break;
      }
      p += 8 + size + (size & 1);
    }
    if (!fmt || dataOff < 0) throw new Error('WAV without fmt or data chunk');
    const bytes = fmt.bits >> 3, frames = Math.floor(dataLen / (bytes * fmt.nch));
    const chans = Array.from({length: fmt.nch}, () => new Float32Array(frames));
    for (let i = 0; i < frames; i++) {
      for (let k = 0; k < fmt.nch; k++) {
        const o = dataOff + (i * fmt.nch + k) * bytes;
        let v;
        if (fmt.tag === 3) v = fmt.bits === 64 ? dv.getFloat64(o, true) : dv.getFloat32(o, true);
        else if (fmt.bits === 16) v = dv.getInt16(o, true) / 32768;
        else if (fmt.bits === 8) v = (dv.getUint8(o) - 128) / 128;
        else if (fmt.bits === 24) { let q = dv.getUint8(o) | (dv.getUint8(o + 1) << 8) | (dv.getUint8(o + 2) << 16); if (q & 0x800000) q |= ~0xFFFFFF; v = q / 8388608; }
        else if (fmt.bits === 32) v = dv.getInt32(o, true) / 2147483648;
        else throw new Error('unsupported WAV sample size ' + fmt.bits);
        chans[k][i] = v;
      }
    }
    return clip(fmt.rate, chans);
  };

  /** Encodes a clip as a WAV file: 16-bit PCM (values clamped, rounded) or 32-bit float. Returns an ArrayBuffer. */
  AudioEdit.encodeWav = (c, bits = 16, rate = null) => {
    if (rate && rate !== c.sampleRate) c = AudioEdit.resample(c, rate);
    const nch = c.channels.length, n = length(c), bytes = bits === 32 ? 4 : 2;
    const buf = new ArrayBuffer(44 + n * nch * bytes), dv = new DataView(buf);
    const w = (o, s) => { for (let i = 0; i < s.length; i++) dv.setUint8(o + i, s.charCodeAt(i)); };
    w(0, 'RIFF'); dv.setUint32(4, 36 + n * nch * bytes, true); w(8, 'WAVE'); w(12, 'fmt ');
    dv.setUint32(16, 16, true); dv.setUint16(20, bits === 32 ? 3 : 1, true); dv.setUint16(22, nch, true);
    dv.setUint32(24, c.sampleRate, true); dv.setUint32(28, c.sampleRate * nch * bytes, true);
    dv.setUint16(32, nch * bytes, true); dv.setUint16(34, bytes * 8, true); w(36, 'data'); dv.setUint32(40, n * nch * bytes, true);
    let o = 44;
    for (let i = 0; i < n; i++) {
      for (let k = 0; k < nch; k++) {
        const v = c.channels[k][i];
        if (bits === 32) dv.setFloat32(o, v, true);
        else { const s = v < -1 ? -1 : v > 1 ? 1 : v; dv.setInt16(o, Math.round(s < 0 ? s * 32768 : s * 32767), true); }
        o += bytes;
      }
    }
    return buf;
  };

  // ------------------------------------------------------------------ words, diff and subtitles

  /** Lower-cases a word and strips punctuation, so "Hello," equals "hello". */
  AudioEdit.normWord = w => w.toLowerCase().normalize('NFKC').replace(/[^\p{L}\p{N}']+/gu, '');

  /** Splits text into words (whitespace separated). */
  AudioEdit.tokenize = text => (text || '').split(/\s+/).filter(Boolean);

  /**
   * Word level diff by longest common subsequence (words compared without case and punctuation). Returns the changed
   * regions: [{type: 'delete'|'insert'|'replace', a0, a1, b0, b1}] with indexes into the old (a) and new (b) word
   * arrays; equal stretches are not listed.
   */
  AudioEdit.diffWords = (oldWords, newWords) => {
    const A = oldWords.map(AudioEdit.normWord), B = newWords.map(AudioEdit.normWord);
    const n = A.length, m = B.length;
    // trim the common prefix and suffix: speeds up the usual case of a few edits in a long text
    let pre = 0;
    while (pre < n && pre < m && A[pre] === B[pre]) pre++;
    let suf = 0;
    while (suf < n - pre && suf < m - pre && A[n - 1 - suf] === B[m - 1 - suf]) suf++;
    const an = n - pre - suf, bn = m - pre - suf;
    const L = Array.from({length: an + 1}, () => new Uint32Array(bn + 1));
    for (let i = an - 1; i >= 0; i--) {
      for (let j = bn - 1; j >= 0; j--) {
        L[i][j] = A[pre + i] === B[pre + j] ? L[i + 1][j + 1] + 1 : Math.max(L[i + 1][j], L[i][j + 1]);
      }
    }
    const ops = [];
    let i = 0, j = 0, cur = null;
    const flush = () => { if (cur) { cur.type = cur.a1 > cur.a0 && cur.b1 > cur.b0 ? 'replace' : cur.a1 > cur.a0 ? 'delete' : 'insert'; ops.push(cur); cur = null; } };
    while (i < an || j < bn) {
      if (i < an && j < bn && A[pre + i] === B[pre + j]) { flush(); i++; j++; continue; }
      if (!cur) cur = {a0: pre + i, a1: pre + i, b0: pre + j, b1: pre + j};
      if (j >= bn || (i < an && L[i + 1][j] >= L[i][j + 1])) { i++; cur.a1 = pre + i; }
      else { j++; cur.b1 = pre + j; }
    }
    flush();
    return ops;
  };

  /** Spreads a segment's time over its words by character count (used when the server sends no word times). */
  AudioEdit.distributeWords = seg => {
    const words = AudioEdit.tokenize(seg.text);
    const total = words.reduce((s, w) => s + w.length + 1, 0) || 1;
    let t = seg.start;
    return words.map(w => {
      const d = (seg.end - seg.start) * (w.length + 1) / total;
      const o = {word: w, start: t, end: t + d * (w.length / (w.length + 1))};
      t += d;
      return o;
    });
  };

  /**
   * Turns a word diff into regions of the audio to change. `words` are [{text, s, e}] with s/e in frames of the clip,
   * `ops` come from diffWords(old words, new words), `newWords` are the new tokens. Returns regions sorted by position:
   * {type, cutA, cutB, text, a0, a1}; text is the new words joined by spaces ('' for a deletion). A deletion also takes
   * the pause that follows the removed words (at most `maxPauseFrames` of it), so no double pause is left behind; an
   * insertion goes into the middle of the pause between its neighbours.
   */
  AudioEdit.planTextEdit = (words, ops, newWords, maxPauseFrames) => {
    const out = [];
    for (const op of ops) {
      const text = newWords.slice(op.b0, op.b1).join(' ');
      if (op.type === 'replace') {
        out.push({type: 'replace', cutA: words[op.a0].s, cutB: words[op.a1 - 1].e, text, a0: op.a0, a1: op.a1});
      } else if (op.type === 'delete') {
        const last = words[op.a1 - 1];
        let cutA = words[op.a0].s, cutB = last.e;
        if (op.a1 < words.length) cutB = Math.max(cutB, Math.min(words[op.a1].s, last.e + maxPauseFrames));
        else if (op.a0 > 0) cutA = Math.min(cutA, Math.max(words[op.a0 - 1].e, cutA - maxPauseFrames));
        out.push({type: 'delete', cutA, cutB, text: '', a0: op.a0, a1: op.a1});
      } else {
        let at;
        if (!words.length) at = 0;
        else if (op.a0 === 0) at = words[0].s;
        else if (op.a0 >= words.length) at = words[words.length - 1].e;
        else at = Math.round((words[op.a0 - 1].e + words[op.a0].s) / 2);
        out.push({type: 'insert', cutA: at, cutB: at, text, a0: op.a0, a1: op.a0});
      }
    }
    out.sort((x, y) => x.cutA - y.cutA || x.cutB - y.cutB);
    return out;
  };

  /**
   * Applies a plan: every region's [cutA, cutB) is removed and replaced by its piece (a clip, or null for a pure
   * deletion); the kept audio between regions is untouched. All joins are cross-faded by `xfadeFrames`. Returns
   * {clip, spans: [[start, end)] of every region in the new clip (a deletion has an empty span at its join)}.
   */
  AudioEdit.applyTextEdit = (c, regions, pieces, xfadeFrames) => {
    const parts = [], joins = [], spans = [];
    let pos = 0, L = 0;
    // mirrors concatCrossfade: appending a part overlaps the previous end by the (capped) cross-fade
    const push = part => {
      let x = 0;
      if (parts.length) { x = Math.min(xfadeFrames, L >> 1, length(part) >> 1); joins.push(xfadeFrames); }
      const at = L - x;
      parts.push(part);
      L = L + length(part) - x;
      return at;
    };
    for (let i = 0; i < regions.length; i++) {
      const r = regions[i];
      push(AudioEdit.slice(c, pos, Math.max(pos, r.cutA)));
      const piece = pieces[i] && length(pieces[i]) ? AudioEdit.conform(pieces[i], c) : null;
      if (piece) { const at = push(piece); spans.push([at, at + length(piece)]); }
      else spans.push([L, L]);
      pos = Math.max(pos, r.cutB);
    }
    push(AudioEdit.slice(c, pos, length(c)));
    const out = AudioEdit.concatCrossfade(parts, joins);
    return {clip: out, spans};
  };

  const pad = (n, w) => String(n).padStart(w, '0');
  const stamp = (sec, comma) => {
    const ms = Math.max(0, Math.round(sec * 1000));
    return `${pad(Math.floor(ms / 3600000), 2)}:${pad(Math.floor(ms / 60000) % 60, 2)}:${pad(Math.floor(ms / 1000) % 60, 2)}${comma ? ',' : '.'}${pad(ms % 1000, 3)}`;
  };
  AudioEdit.srtTime = s => stamp(s, true);
  AudioEdit.vttTime = s => stamp(s, false);
  AudioEdit.toSrt = segs => segs.map((s, i) => `${i + 1}\n${stamp(s.start, true)} --> ${stamp(s.end, true)}\n${s.text.trim()}\n`).join('\n');
  AudioEdit.toVtt = segs => 'WEBVTT\n\n' + segs.map(s => `${stamp(s.start, false)} --> ${stamp(s.end, false)}\n${s.text.trim()}\n`).join('\n');

  /** mm:ss.s for display. */
  AudioEdit.clock = (sec, decimals = 1) => {
    if (!isFinite(sec)) return '--:--';
    const neg = sec < 0 ? '-' : '';
    sec = Math.abs(sec);
    const m = Math.floor(sec / 60), s = sec - m * 60;
    return `${neg}${m}:${s.toFixed(decimals).padStart(decimals ? decimals + 3 : 2, '0')}`;
  };

  if (typeof module !== 'undefined' && module.exports) module.exports = AudioEdit;
  else root.AudioEdit = AudioEdit;
})(typeof globalThis !== 'undefined' ? globalThis : this);
