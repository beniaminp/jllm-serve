'use strict';
// Tests of the pure editing functions of the console's Audio tab. Run: node jllm-web/src/test/js/audio-edit-test.js
// (not part of the Maven build).
const assert = require('node:assert/strict');
const path = require('node:path');
const E = require(path.join(__dirname, '../../main/resources/static/js/audio-edit.js'));

let passed = 0;
const test = (name, fn) => { try { fn(); passed++; } catch (e) { console.error('FAIL ' + name + '\n', e); process.exitCode = 1; } };
const near = (a, b, eps = 1e-5, msg = '') => assert.ok(Math.abs(a - b) <= eps, `${msg} ${a} vs ${b}`);

const sine = (rate, freq, sec, amp = 0.5, nch = 1) => {
  const n = Math.round(rate * sec), ch = new Float32Array(n);
  for (let i = 0; i < n; i++) ch[i] = amp * Math.sin(2 * Math.PI * freq * i / rate);
  return E.clip(rate, Array.from({length: nch}, () => ch.slice()));
};
const ramp = n => E.clip(1000, [Float32Array.from({length: n}, (_, i) => i / n)]);

test('slice / concat / deleteRange / insert / replace', () => {
  const c = ramp(100);
  assert.equal(E.length(E.slice(c, 10, 30)), 20);
  assert.equal(E.length(E.deleteRange(c, 10, 30)), 80);
  assert.equal(E.deleteRange(c, 10, 30).channels[0][10], c.channels[0][30]);
  const ins = E.insert(c, 50, E.silent(1000, 1, 10));
  assert.equal(E.length(ins), 110);
  assert.equal(ins.channels[0][55], 0);
  assert.equal(ins.channels[0][60], c.channels[0][50]);
  const rep = E.replace(c, 10, 20, E.silent(1000, 1, 3));
  assert.equal(E.length(rep), 93);
  // inputs are never changed
  assert.equal(E.length(c), 100);
  near(c.channels[0][10], 0.1, 1e-7);
  // reversed / out of range bounds are normalized
  assert.equal(E.length(E.slice(c, 30, 10)), 20);
  assert.equal(E.length(E.slice(c, -5, 1000)), 100);
});

test('insert converts rate and channels', () => {
  const c = sine(8000, 440, 0.5, 0.5, 2);
  const p = sine(16000, 440, 0.25, 0.5, 1);
  const r = E.insert(c, 0, p);
  assert.equal(r.channels.length, 2);
  assert.ok(Math.abs(E.length(r) - (4000 + 2000)) <= 1);
});

test('fade in / out', () => {
  const c = E.clip(1000, [new Float32Array(100).fill(1)]);
  const f = E.fade(c, 0, 100, 'in', 'linear');
  near(f.channels[0][0], 0); near(f.channels[0][99], 1); near(f.channels[0][50], 50 / 99);
  const g = E.fade(c, 20, 80, 'out', 'cosine');
  near(g.channels[0][19], 1); near(g.channels[0][20], 1); near(g.channels[0][79], 0); near(g.channels[0][80], 1);
  assert.ok(g.channels[0][50] < 1 && g.channels[0][50] > 0);
});

test('gain, normalize, peak, rms', () => {
  const c = sine(8000, 100, 1, 0.25);
  near(E.peak(c), 0.25, 1e-3);
  const g = E.gain(c, 0, E.length(c), 6.0206);
  near(E.peak(g), 0.5, 1e-3);
  const n = E.normalize(c, 0, E.length(c), 'peak', -6.0206);
  near(E.peak(n), 0.5, 1e-3);
  const r = E.normalize(c, 0, E.length(c), 'rms', -20);
  near(E.gainToDb(E.rms(r)), -20, 0.01);
  // range-limited
  const h = E.gain(c, 0, 100, -120);
  assert.ok(E.peak(h, 0, 100) < 1e-5);
  near(h.channels[0][200], c.channels[0][200], 0);
  // silent range normalizes to nothing
  const z = E.normalize(E.silent(8000, 1, 10), 0, 10, 'peak', -1);
  assert.equal(E.peak(z), 0);
});

test('reverse', () => {
  const c = ramp(10);
  const r = E.reverse(c, 2, 8);
  assert.deepEqual([...r.channels[0].slice(0, 10)].map(x => Math.round(x * 10)), [0, 1, 7, 6, 5, 4, 3, 2, 8, 9]);
  assert.deepEqual([...E.reverse(r, 2, 8).channels[0]], [...c.channels[0]]);
});

test('insertSilence', () => {
  const c = ramp(10);
  const r = E.insertSilence(c, 5, 4);
  assert.equal(E.length(r), 14);
  assert.deepEqual([...r.channels[0].slice(5, 9)], [0, 0, 0, 0]);
});

test('mix extends and scales', () => {
  const a = E.clip(1000, [new Float32Array(10).fill(0.25)]);
  const b = E.clip(1000, [new Float32Array(10).fill(0.5)]);
  const m = E.mix(a, b, 5, -6.0206);
  assert.equal(E.length(m), 15);
  near(m.channels[0][4], 0.25); near(m.channels[0][5], 0.5, 1e-3); near(m.channels[0][14], 0.25, 1e-3);
});

test('findSilences / removeSilence', () => {
  const rate = 8000;
  const tone = sine(rate, 300, 0.5, 0.5);
  const gap = E.silent(rate, 1, rate);          // 1 s of silence
  const c = E.concat([tone, gap, tone, E.silent(rate, 1, 800), tone]);
  const runs = E.findSilences(c, 0, E.length(c), -40, 0.5);
  assert.equal(runs.length, 1);
  assert.ok(Math.abs(runs[0][0] - 4000) <= 80 && Math.abs(runs[0][1] - 12000) <= 80);
  assert.equal(E.findSilences(c, 0, E.length(c), -40, 0.05).length, 2);
  const r = E.removeSilence(c, 0, E.length(c), -40, 0.5, 0.1);
  assert.equal(r.count, 1);
  // 1 s silence shortened to 0.1 s: about 0.9 s (7200 frames) removed, minus the cross-fade overlap
  assert.ok(Math.abs(r.removedFrames - 7200) <= 400, 'removed ' + r.removedFrames);
  assert.equal(E.length(r.clip), E.length(c) - r.removedFrames);
  // a clip without a long silence is returned unchanged
  assert.equal(E.removeSilence(tone, 0, E.length(tone)).count, 0);
});

test('activeRange / trimSilence', () => {
  const rate = 8000, tone = sine(rate, 300, 0.25, 0.5);
  const c = E.concat([E.silent(rate, 1, 1000), tone, E.silent(rate, 1, 3000)]);
  const [a, b] = E.activeRange(c, -45);
  assert.ok(Math.abs(a - 1000) < 40 && Math.abs(b - 3000) < 40, `${a} ${b}`);
  assert.ok(E.length(E.trimSilence(c)) < 2200);
});

test('concatCrossfade is equal power and shortens by the overlap', () => {
  const a = E.clip(1000, [new Float32Array(100).fill(1)]);
  const b = E.clip(1000, [new Float32Array(100).fill(1)]);
  const r = E.concatCrossfade([a, b], [20]);
  assert.equal(E.length(r), 180);
  for (let i = 80; i < 100; i++) assert.ok(r.channels[0][i] > 0.99 && r.channels[0][i] < 1.42);
  near(r.channels[0][50], 1); near(r.channels[0][150], 1);
  assert.equal(E.length(E.concatCrossfade([a, b, a], [0, 0])), 300);
  // the overlap is capped at half of the shorter neighbour
  assert.equal(E.length(E.concatCrossfade([E.silent(1000, 1, 10), E.silent(1000, 1, 10)], [100])), 15);
});

test('resample keeps frequency, length and amplitude', () => {
  const c = sine(44100, 440, 1, 0.5);
  for (const to of [16000, 24000, 48000]) {
    const r = E.resample(c, to);
    assert.equal(r.sampleRate, to);
    assert.ok(Math.abs(E.length(r) - to) <= 1);
    const mid = E.slice(r, to * 0.1, to * 0.9);
    near(E.peak(mid), 0.5, 0.01, 'peak at ' + to);
    // zero crossings: 440 Hz over 0.8 s ~ 704 crossings
    let z = 0;
    const ch = mid.channels[0];
    for (let i = 1; i < ch.length; i++) if ((ch[i - 1] < 0) !== (ch[i] < 0)) z++;
    assert.ok(Math.abs(z - 704) <= 3, `crossings ${z} at ${to}`);
  }
});

test('resample removes content above the new Nyquist frequency', () => {
  const hi = sine(48000, 15000, 0.5, 0.5);      // above the Nyquist frequency of 16 kHz / 2 = 8 kHz
  const r = E.resample(hi, 16000);
  assert.ok(E.peak(E.slice(r, 200, E.length(r) - 200)) < 0.05, 'aliasing ' + E.peak(r));
});

test('changeSpeed by resampling', () => {
  const c = sine(8000, 200, 1, 0.5);
  const r = E.changeSpeed(c, 0, E.length(c), 2);
  assert.ok(Math.abs(E.length(r) - 4000) <= 1);
  const half = E.changeSpeed(c, 0, E.length(c), 0.5);
  assert.ok(Math.abs(E.length(half) - 16000) <= 1);
  // only a range: the rest is untouched
  const part = E.changeSpeed(c, 2000, 4000, 2);
  assert.ok(Math.abs(E.length(part) - 7000) <= 1);
  near(part.channels[0][100], c.channels[0][100], 0);
});

test('WSOLA stretch changes length and keeps the pitch', () => {
  const rate = 16000, freq = 220;
  const c = sine(rate, freq, 2, 0.5);
  for (const f of [0.6, 1.5, 2]) {
    const r = E.timeStretch(c, 0, E.length(c), f);
    assert.ok(Math.abs(E.length(r) - E.length(c) / f) <= 2, 'length for ' + f);
    const mid = E.slice(r, E.length(r) * 0.2, E.length(r) * 0.8);
    let z = 0;
    const ch = mid.channels[0];
    for (let i = 1; i < ch.length; i++) if ((ch[i - 1] < 0) !== (ch[i] < 0)) z++;
    const hz = z / 2 / (ch.length / rate);
    assert.ok(Math.abs(hz - freq) < freq * 0.03, `pitch ${hz} for factor ${f}`);
    assert.ok(E.peak(mid) > 0.4 && E.peak(mid) < 0.6, 'level ' + E.peak(mid));
  }
  assert.ok(Math.abs(E.length(E.fitTo(c, 20000)) - 20000) <= 2);
});

test('peaks pyramid agrees with the raw samples', () => {
  const c = sine(8000, 50, 3, 0.7, 2);
  const pk = E.buildPeaks(c);
  const raw = E.peaksFor(c, {block: 1 << 30, mins: [], maxs: []}, 0, 0, E.length(c), 100);
  const fast = E.peaksFor(c, pk, 0, 0, E.length(c), 100);
  for (let i = 0; i < 100; i++) { near(fast.min[i], raw.min[i], 1e-6); near(fast.max[i], raw.max[i], 1e-6); }
  // zoomed in on a few samples, fractional range, out of range
  const z = E.peaksFor(c, pk, 1, 10.5, 40.5, 30);
  assert.equal(z.min.length, 30);
  const out = E.peaksFor(c, pk, 0, E.length(c) + 10, E.length(c) + 100, 5);
  assert.deepEqual([...out.max], [0, 0, 0, 0, 0]);
});

test('WAV round trip 16 bit, 32 bit float, stereo', () => {
  const c = sine(22050, 440, 0.1, 0.6, 2);
  c.channels[1] = c.channels[1].map(v => -v);
  const w16 = E.decodeWav(E.encodeWav(c, 16));
  assert.equal(w16.sampleRate, 22050); assert.equal(w16.channels.length, 2); assert.equal(E.length(w16), E.length(c));
  for (let i = 0; i < E.length(c); i += 37) { near(w16.channels[0][i], c.channels[0][i], 1 / 32000); near(w16.channels[1][i], c.channels[1][i], 1 / 32000); }
  const w32 = E.decodeWav(E.encodeWav(c, 32));
  for (let i = 0; i < E.length(c); i += 37) near(w32.channels[0][i], c.channels[0][i], 1e-7);
  // clamping
  const loud = E.clip(8000, [Float32Array.from([2, -2, 0.5])]);
  const d = E.decodeWav(E.encodeWav(loud, 16));
  near(d.channels[0][0], 32767 / 32768, 1e-6); near(d.channels[0][1], -1, 1e-6);
  // encode with another rate
  assert.equal(E.decodeWav(E.encodeWav(c, 16, 11025)).sampleRate, 11025);
  assert.ok(E.isWav(E.encodeWav(c)));
  assert.ok(!E.isWav(new ArrayBuffer(100)));
  assert.throws(() => E.decodeWav(new ArrayBuffer(100)));
});

test('decode WAV with extra chunks, 24 bit and a truncated data size', () => {
  const buf = new ArrayBuffer(12 + 8 + 16 + 8 + 4 + 8 + 6), dv = new DataView(buf);
  const w = (o, s) => { for (let i = 0; i < s.length; i++) dv.setUint8(o + i, s.charCodeAt(i)); };
  w(0, 'RIFF'); dv.setUint32(4, buf.byteLength - 8, true); w(8, 'WAVE');
  w(12, 'fmt '); dv.setUint32(16, 16, true); dv.setUint16(20, 1, true); dv.setUint16(22, 1, true); dv.setUint32(24, 8000, true);
  dv.setUint32(28, 24000, true); dv.setUint16(32, 3, true); dv.setUint16(34, 24, true);
  w(36, 'LIST'); dv.setUint32(40, 4, true); w(44, 'abcd');
  w(48, 'data'); dv.setUint32(52, 0xFFFFFFFF, true);
  // two 24-bit samples: +0.5 and -0.5
  dv.setUint8(56, 0x00); dv.setUint8(57, 0x00); dv.setUint8(58, 0x40);
  dv.setUint8(59, 0x00); dv.setUint8(60, 0x00); dv.setUint8(61, 0xC0);
  const c = E.decodeWav(buf);
  assert.equal(E.length(c), 2); near(c.channels[0][0], 0.5); near(c.channels[0][1], -0.5);
});

test('diffWords: replace, delete, insert, mixed, punctuation-insensitive', () => {
  const old = 'the quick brown fox jumps over the lazy dog'.split(' ');
  let ops = E.diffWords(old, 'the quick brown fox jumps over the lazy dog'.split(' '));
  assert.equal(ops.length, 0);
  ops = E.diffWords(old, 'the quick red fox jumps over the lazy dog'.split(' '));
  assert.deepEqual(ops, [{type: 'replace', a0: 2, a1: 3, b0: 2, b1: 3}]);
  ops = E.diffWords(old, 'the quick fox jumps over the lazy dog'.split(' '));
  assert.deepEqual(ops, [{type: 'delete', a0: 2, a1: 3, b0: 2, b1: 2}]);
  ops = E.diffWords(old, 'the very quick brown fox jumps over the lazy dog'.split(' '));
  assert.deepEqual(ops, [{type: 'insert', a0: 1, a1: 1, b0: 1, b1: 2}]);
  ops = E.diffWords(old, 'a quick brown fox leaps over the lazy cat today'.split(' '));
  assert.deepEqual(ops.map(o => [o.type, o.a0, o.a1, o.b0, o.b1]),
    [['replace', 0, 1, 0, 1], ['replace', 4, 5, 4, 5], ['replace', 8, 9, 8, 10]]);
  assert.equal(E.diffWords(['Hello,', 'World.'], ['hello', 'world']).length, 0);
  assert.deepEqual(E.diffWords([], ['a', 'b']).map(o => o.type), ['insert']);
  assert.deepEqual(E.diffWords(['a', 'b'], []).map(o => o.type), ['delete']);
  // everything replaced
  assert.deepEqual(E.diffWords(['a', 'b'], ['c', 'd', 'e']), [{type: 'replace', a0: 0, a1: 2, b0: 0, b1: 3}]);
});

test('distributeWords covers the segment in order', () => {
  const w = E.distributeWords({start: 10, end: 14, text: ' Hello there, general Kenobi'});
  assert.equal(w.length, 4);
  assert.ok(w[0].start >= 10 && w[3].end <= 14.0001);
  for (let i = 1; i < w.length; i++) assert.ok(w[i].start >= w[i - 1].end);
});

test('subtitles and clock', () => {
  const segs = [{start: 0, end: 2.5, text: ' Hello. '}, {start: 3661.007, end: 3662, text: 'Bye'}];
  assert.equal(E.toSrt(segs), '1\n00:00:00,000 --> 00:00:02,500\nHello.\n\n2\n01:01:01,007 --> 01:01:02,000\nBye\n');
  assert.equal(E.toVtt(segs), 'WEBVTT\n\n00:00:00.000 --> 00:00:02.500\nHello.\n\n01:01:01.007 --> 01:01:02.000\nBye\n');
  assert.equal(E.clock(75.34), '1:15.3');
  assert.equal(E.clock(5, 0), '0:05');
});

test('planTextEdit + applyTextEdit: delete, replace, insert', () => {
  const rate = 1000;
  // five words of 200 frames each with 100 frame pauses: "one two three four five"
  const texts = ['one', 'two', 'three', 'four', 'five'];
  const words = texts.map((t, i) => ({text: t, s: i * 300, e: i * 300 + 200}));
  const total = 5 * 300;
  const c = E.clip(rate, [Float32Array.from({length: total}, (_, i) => (i % 300 < 200 ? 0.5 : 0))]);
  const edit = (newText, pieces) => {
    const nw = E.tokenize(newText);
    const ops = E.diffWords(texts, nw);
    const plan = E.planTextEdit(words, ops, nw, 250);
    const ps = plan.map(r => (r.text ? pieces(r) : null));
    return {plan, res: E.applyTextEdit(c, plan, ps, 0)};
  };
  const tone = n => E.clip(rate, [new Float32Array(n).fill(0.9)]);
  // delete "three": cuts the word and the pause after it
  let r = edit('one two four five', () => null);
  assert.deepEqual(r.plan.map(x => [x.type, x.cutA, x.cutB]), [['delete', 600, 900]]);
  assert.equal(E.length(r.res.clip), total - 300);
  // replace "two" by a 150 frame piece
  r = edit('one deux three four five', () => tone(150));
  assert.deepEqual(r.plan.map(x => [x.type, x.cutA, x.cutB, x.text]), [['replace', 300, 500, 'deux']]);
  assert.equal(E.length(r.res.clip), total - 200 + 150);
  assert.deepEqual(r.res.spans, [[300, 450]]);
  near(r.res.clip.channels[0][320], 0.9, 1e-6); near(r.res.clip.channels[0][460], 0, 1e-6);
  // insert "very" before "three": into the middle of the pause (frame 550)
  r = edit('one two very three four five', () => tone(100));
  assert.deepEqual(r.plan.map(x => [x.type, x.cutA, x.cutB]), [['insert', 550, 550]]);
  assert.equal(E.length(r.res.clip), total + 100);
  // two regions, with cross-fades of 10 frames
  const nw = E.tokenize('uno two three quatro five'), ops = E.diffWords(texts, nw);
  const plan = E.planTextEdit(words, ops, nw, 250);
  assert.equal(plan.length, 2);
  const res = E.applyTextEdit(c, plan, [tone(100), tone(100)], 10);
  // kept 700 + 400 and two pieces of 100 = 1300, minus three cross-fades of 10 (the first join follows an empty part)
  assert.equal(E.length(res.clip), 1270);
  assert.equal(res.spans.length, 2);
  // delete at the end takes the pause before the removed words, not beyond the clip
  r = edit('one two three four', () => null);
  assert.deepEqual(r.plan.map(x => [x.type, x.cutA, x.cutB]), [['delete', 1100, 1400]]);
});

test('limit keeps values within +-1', () => {
  const c = E.clip(8000, [Float32Array.from([0.5, 1.5, -3, 0.95])]);
  const l = E.limit(c);
  for (const v of l.channels[0]) assert.ok(Math.abs(v) < 1.0001);
  near(l.channels[0][0], 0.5, 0);
});

console.log(passed + ' passed' + (process.exitCode ? ', some FAILED' : ''));
