'use strict';
// Measures what the chat page costs the browser while an answer streams (docs/29 §29.5). Not a test and not part
// of the Maven build: it needs Playwright with a Chromium (npm i playwright, or a machine that has one).
//
//   node jllm-web/src/test/js/chat-stream-bench.js [tokens=4000] [gapMs=2]
//
// It serves the console's static files itself, with a scripted backend in place of the engine: /api/chat streams
// a reasoning block and then a long Markdown answer (headings, paragraphs, lists, code, tables, quotes), 4
// characters per event, one event every gapMs. Chromium's own counters then say how long the page's main thread
// was busy during the stream (TaskDuration; ScriptDuration, LayoutDuration and RecalcStyleDuration are parts of
// it), and the page counts its animation frames. To compare two versions of the page, run it on each checkout.
const http = require('node:http');
const fs = require('node:fs');
const path = require('node:path');

const ROOT = path.join(__dirname, '../../main/resources/static');
const TOKENS = +process.argv[2] || 4000;
const GAP_MS = +process.argv[3] || 2;
const TYPES = {'.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css'};

function answer() {
  let text = '';
  for (let k = 0; text.length < TOKENS * 4; k++) {
    text += `## Section ${k}\n\nThis is **paragraph** ${k} with \`inline code\`, a [link](https://example.com/${k}) and _emphasis_. `
      + 'Lorem ipsum dolor sit amet, consectetur adipiscing elit. '.repeat(4) + '\n\n'
      + `1. first item ${k}\n2. second item with **bold**\n3. third item\n\n`
      + '```java\n' + `record Vec${k}(double x, double y) {\n\n    Vec${k} add(Vec${k} o) { return new Vec${k}(x + o.x, y + o.y); }\n}\n` + '```\n\n'
      + `| a | b | c |\n|---|:-:|--:|\n| ${k} | x | y |\n| 1 | 2 | 3 |\n\n> a quote for section ${k}\n\n`;
  }
  return '<think>\n' + 'thinking about it. '.repeat(100) + '\n</think>\n\n' + text.slice(0, TOKENS * 4);
}

const server = http.createServer((req, res) => {
  const url = req.url.split('?')[0];
  if (url === '/api/chat') {
    const text = answer();
    const event = o => res.write('data: ' + JSON.stringify(o) + '\n\n');
    res.writeHead(200, {'Content-Type': 'text/event-stream'});
    event({type: 'start', model: 'm'});
    let i = 0;
    const timer = setInterval(() => {
      if (i < text.length) { event({type: 'delta', content: text.slice(i, i + 4)}); i += 4; return; }
      clearInterval(timer);
      event({type: 'usage', usage: {prompt_tokens: 10, cached_tokens: 0, completion_tokens: TOKENS, tokens_per_second: 1}});
      event({type: 'message', message: {role: 'assistant', content: text}});
      event({type: 'done', finish_reason: 'stop'});
      res.end();
    }, GAP_MS);
    return;
  }
  if (url.startsWith('/api/') || url.startsWith('/v1/')) {   // just enough for the page to start with one model loaded
    const body = url === '/api/models/loaded' ? [{id: 'm', kind: 'text', contextLength: 32768}]
      : url === '/api/status' ? {loadedModels: 1, weightBytes: 1, heapUsed: 1, heapMax: 2, processors: 4, uptimeMs: 1, thermalState: -1,
        runningJobs: 0, settingsFile: 'x', apiKeyRequired: false}
      : url === '/api/tools' ? {tools: []} : [];
    res.writeHead(200, {'Content-Type': 'application/json'});
    res.end(JSON.stringify(body));
    return;
  }
  const file = path.join(ROOT, path.normalize(url === '/' ? 'index.html' : url));
  fs.readFile(file, (err, bytes) => {
    if (err || !file.startsWith(ROOT)) { res.writeHead(404); res.end(); return; }
    res.writeHead(200, {'Content-Type': TYPES[path.extname(file)] || 'application/octet-stream'});
    res.end(bytes);
  });
});

server.listen(0, '127.0.0.1', async () => {
  const {chromium} = require('playwright');
  const browser = await chromium.launch();
  try {
    const page = await browser.newPage({viewport: {width: 1280, height: 900}});
    const errors = [];
    page.on('pageerror', e => errors.push(e.message));
    await page.goto(`http://127.0.0.1:${server.address().port}/`);
    await page.waitForFunction(() => document.querySelector('#chat-model')?.value === 'm');
    const cdp = await page.context().newCDPSession(page);
    await cdp.send('Performance.enable');
    const metrics = async () => Object.fromEntries((await cdp.send('Performance.getMetrics')).metrics.map(m => [m.name, m.value]));
    await page.evaluate(() => {
      window.benchFrames = 0; window.benchWorstGap = 0;
      let last = performance.now();
      const tick = t => { window.benchFrames++; window.benchWorstGap = Math.max(window.benchWorstGap, t - last); last = t; requestAnimationFrame(tick); };
      requestAnimationFrame(tick);
    });
    const before = await metrics();
    const t0 = Date.now();
    await page.fill('#input', 'go');
    await page.keyboard.press('Enter');
    await page.waitForSelector('#stop:not(.hidden)');
    await page.waitForSelector('#send:not(.hidden)', {timeout: 30 * 60 * 1000});   // the answer is complete
    const after = await metrics();
    const seconds = (Date.now() - t0) / 1000;
    const shown = await page.evaluate(() => ({frames: window.benchFrames, worstGap: Math.round(window.benchWorstGap),
      blocks: document.querySelectorAll('#messages .turn.assistant .md > *').length}));
    const d = k => +(after[k] - before[k]).toFixed(2);
    console.log(JSON.stringify({tokens: TOKENS, streamSeconds: +seconds.toFixed(1), mainThreadBusySeconds: d('TaskDuration'),
      scriptSeconds: d('ScriptDuration'), layoutSeconds: d('LayoutDuration'), styleSeconds: d('RecalcStyleDuration'),
      framesPerSecond: +(shown.frames / seconds).toFixed(1), worstFrameGapMs: shown.worstGap, blocksShown: shown.blocks, pageErrors: errors}, null, 1));
  } finally {
    await browser.close();
    server.close();
  }
});
