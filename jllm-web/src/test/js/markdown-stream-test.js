'use strict';
// Tests of the console's Markdown renderer while an answer streams: Markdown.stream must show exactly what
// Markdown.render shows for the same text, however the text is cut into pieces, and must render a finished
// block only once. Run: node jllm-web/src/test/js/markdown-stream-test.js (not part of the Maven build).
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

// markdown.js is a browser script: give it the two globals it uses and a document that takes its click listener
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[c]));
const src = fs.readFileSync(path.join(__dirname, '../../main/resources/static/js/markdown.js'), 'utf8');
const Markdown = vm.runInNewContext(src + '\n;Markdown', {esc, copyText() {}, document: {addEventListener() {}}});

// the little of an element that stream() uses: innerHTML, firstChild and insertAdjacentHTML('beforebegin') on the tail
function fakeEl() {
  const el = {settled: [], tail: null, settledWrites: 0, tailWrites: 0};
  Object.defineProperty(el, 'innerHTML', {
    set(html) {
      assert.equal(html, '<div class="md-tail"></div>');
      el.settled = [];
      el.tail = {html: '', insertAdjacentHTML(where, h) { assert.equal(where, 'beforebegin'); el.settled.push(h); el.settledWrites++; }};
      Object.defineProperty(el.tail, 'innerHTML', {set(h) { this.html = h; el.tailWrites++; }, get() { return this.html; }});
    },
  });
  Object.defineProperty(el, 'firstChild', {get() { return el.tail; }});
  el.shown = () => el.settled.join('') + el.tail.html;
  return el;
}

let passed = 0;
const test = (name, fn) => { try { fn(); passed++; } catch (e) { console.error('FAIL ' + name + '\n', e); process.exitCode = 1; } };

const DOC = [
  '# Title', '', 'A paragraph with **bold**, `code`, _em_ and a [link](https://example.com/a).', 'Its second line.', '',
  '1. one', '2. two', '   continued', '  - nested', '3. three', '', '4. four after a blank line', '',
  '- a', '- b', '', '```java', 'record V(double x) {', '', '    // a blank line above, inside the fence', '}', '```', '',
  '| a | b |', '|---|--:|', '| 1 | 2 |', '', '> quoted **text**', '> second line', '', '---', '',
  '  ```', 'indented fence', '', '```not a closing line', '  ```  ', 'after the fence', '',
  'text <b>escaped</b> & "quoted"', '', '```', 'an unclosed fence', '', 'that runs to the end',
].join('\n');

test('settledEnd: blank lines outside fences only', () => {
  assert.equal(Markdown.settledEnd('abc', 0), 0);
  assert.equal(Markdown.settledEnd('abc\n', 0), 0);
  assert.equal(Markdown.settledEnd('abc\n\n', 0), 5);
  assert.equal(Markdown.settledEnd('abc\n\ndef', 0), 5);
  assert.equal(Markdown.settledEnd('abc\n  \t\ndef\n', 0), 8);
  assert.equal(Markdown.settledEnd('```\na\n\nb', 0), 0, 'a blank line inside an open fence');
  assert.equal(Markdown.settledEnd('```\na\n\nb\n```\n\nc', 0), 14);
  assert.equal(Markdown.settledEnd('a\n\nb\n\nc', 3), 6, 'from an offset');
  assert.equal(Markdown.settledEnd('a\n\n```\n\n', 3), 3, 'nothing settles after the offset');
});

test('a stream fed in pieces shows what render shows', () => {
  for (const step of [1, 2, 3, 5, 7, 13, 64, DOC.length]) {
    const el = fakeEl();
    const s = Markdown.stream(el);
    for (let n = step; ; n += step) {
      const text = DOC.slice(0, Math.min(n, DOC.length));
      s.update(text);
      assert.equal(el.shown(), Markdown.render(text), `step ${step}, ${text.length} characters`);
      if (n >= DOC.length) break;
    }
  }
});

test('random documents, random cuts', () => {
  let seed = 12345;
  const rnd = n => { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return seed % n; };
  const LINES = ['', '', '  ', 'plain text', '# heading', '- item', '  - sub', '1. first', '7) seventh', '   continuation', '```', '```js', '  ```  ',
    '| a | b |', '|---|---|', '| 1 | 2 |', '> quote', '---', '**bold** `code`', '![x](/v1/images/files/0123456789abcdef.png)', 'a\r'];
  for (let k = 0; k < 400; k++) {
    const text = Array.from({length: 1 + rnd(40)}, () => LINES[rnd(LINES.length)]).join('\n');
    const el = fakeEl();
    const s = Markdown.stream(el);
    for (let n = 0; n < text.length;) {
      n = Math.min(text.length, n + 1 + rnd(9));
      s.update(text.slice(0, n));
      assert.equal(el.shown(), Markdown.render(text.slice(0, n)), JSON.stringify(text.slice(0, n)));
    }
  }
});

test('a finished block is rendered once', () => {
  const el = fakeEl();
  const s = Markdown.stream(el);
  const text = Array.from({length: 50}, (_, i) => `paragraph ${i}\n\n`).join('');
  for (let n = 1; n <= text.length; n++) s.update(text.slice(0, n));
  assert.equal(el.settledWrites, 50, 'one append per paragraph');
  assert.equal(el.shown(), Markdown.render(text));
  const writes = el.tailWrites;
  s.update(text);   // the same text again: nothing to do
  assert.equal(el.tailWrites, writes);
  assert.equal(el.settledWrites, 50);
});

test('a text that does not extend the previous one starts over', () => {
  const el = fakeEl();
  const s = Markdown.stream(el);
  s.update('first answer\n\nmore');
  s.update('');
  assert.equal(el.shown(), '');
  s.update('second\n\nanswer');
  assert.equal(el.shown(), Markdown.render('second\n\nanswer'));
  s.update('second\n\nrewritten');   // the settled part is a prefix: only the tail changes
  assert.equal(el.shown(), Markdown.render('second\n\nrewritten'));
  s.update('other\n\ntext');
  assert.equal(el.shown(), Markdown.render('other\n\ntext'));
});

test('an ordered list keeps its first number', () => {
  assert.equal(Markdown.render('1. a\n2. b'), '<ol><li>a</li><li>b</li></ol>');
  assert.equal(Markdown.render('3. c\n4. d'), '<ol start="3"><li>c</li><li>d</li></ol>');
  assert.equal(Markdown.render('1. a\n\n2. b'), '<ol><li>a</li></ol><ol start="2"><li>b</li></ol>');
  assert.equal(Markdown.render('- a\n- b'), '<ul><li>a</li><li>b</li></ul>');
});

test('a code fence under a list item is a code block, not item text', () => {
  const html = Markdown.render('1. run\n   ```sh\n   ls\n   ```\n2. done');
  assert.equal(html, '<ol><li>run</li></ol><div class="codeblock"><div class="cb-head"><span>sh</span><button data-copy>copy</button></div>'
    + '<pre><code>   ls</code></pre></div><ol start="2"><li>done</li></ol>');
});

test('a table whose separator has no leading pipe renders (it used to throw)', () => {
  assert.equal(Markdown.render('| a |\n---'), '<table><thead><tr><th>a</th></tr></thead><tbody></tbody></table>');
  assert.equal(Markdown.render('| a | b |\n--- | ---\n| 1 | 2 |'),
    '<table><thead><tr><th>a</th><th>b</th></tr></thead><tbody><tr><td>1</td><td>2</td></tr></tbody></table>');
});

console.log(`${passed} tests passed`);
