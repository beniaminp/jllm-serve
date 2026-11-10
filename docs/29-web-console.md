# 29. The web console's page: what the browser does, and what it costs

The other chapters are about the engine. This one is about the page a person looks at: the console that
`jllm-web` serves at `http://127.0.0.1:8090/`. What its tabs can do is listed in the
[README](../README.md#web-console-jllm-web); the server side of a chat request (slots, the scheduler, priorities) is
[chapter 10](10-engine-and-server.md) and [chapter 25](25-scheduler.md). Here: how the page is put together, the
event stream between the server and the chat, how an answer is drawn while it is generated, what the page asks the
server when nobody types, and how its files reach the browser.

| | | where |
|---|---|---|
| [29.1](#291-in-plain-words) | the idea, told as a story | |
| [29.2](#292-how-the-page-is-put-together) | the files, the tabs, what is kept in the browser, the event stream of `/api/chat` | `static/index.html`, `js/core.js`, `js/chat.js`, `ChatController` |
| [29.3](#293-drawing-an-answer-while-it-is-generated) | one paint per frame, the two parts of the running turn, the streaming Markdown renderer and why it shows exactly what the full renderer shows (a worked example) | `js/chat.js` `paintLive`, `js/markdown.js` `stream` |
| [29.4](#294-what-the-page-asks-when-nobody-types) | the timers, the hidden page, the log view, the files of the page (cache and compression) | `js/core.js` `startApp`, `js/admin.js`, `application.properties` |
| [29.5](#295-measured) | the main thread of the browser during a 4 000 and a 12 000 token answer, before and after | `chat-stream-bench.js` |
| [29.6](#296-why-this-design) | the alternatives that were rejected | |
| [29.7](#297-settings) | the application properties that matter here | |
| [29.8](#298-checks-and-limits) | the tests, the corrections that came with this work, and what is not done | `markdown-stream-test.js` |

---

## 29.1 In plain words

🧸 A friend dictates a long letter to you, one word at a time, and you write it on a whiteboard for everyone to
read. The slow way: every time a new word comes, you wipe the **whole** board and write the letter again from the
first line, now one word longer. At the beginning that is quick. By the tenth paragraph you spend all your time
wiping and rewriting, the people reading lose their place every time the board goes blank, and you fall behind
your friend.

The better way: a paragraph that is **finished** stays on the board and is never touched again. Only the paragraph
your friend is still dictating is rewritten, and it is short. How do you know a paragraph is finished? Your friend
leaves an empty line. (One exception: inside a boxed piece of computer code an empty line means nothing, the box is
finished only when it is closed.)

The console's chat used the slow way: for every screen refresh it built the whole answer again, and with it
everything else the turn already showed. It now uses the better way. The answer looks the same; the browser does a
quarter of the work on a long answer and stays smooth, and on a computer whose processor is also running the model,
the work the browser no longer does is left for the model.

The same idea, twice more: the **log** view adds the new lines under the old ones instead of writing 2 500 lines
again, and a console that is **not looked at** (a tab in the background) stops asking the server how it is doing.

## 29.2 How the page is put together

📐 The console is one HTML file, five style sheets and nineteen scripts, served as they are from
`jllm-web/src/main/resources/static/`: no framework, no build step, no dependency. Every script is an ordinary
`<script src>` at the end of `index.html`; each one is an immediately invoked function that returns the few names
the others use (`Chat`, `Markdown`, `Vision`, `Models`, …). The order of the tags is the dependency order.

| file | lines | what it is |
|---|---:|---|
| `js/core.js` | 134 | helpers (`$`, `esc`, `store`, `api`, `toast`), the tabs, the theme, the status box, the timers |
| `js/markdown.js` | 179 | the Markdown renderer: `render(text)` and `stream(element)` ([29.3](#293-drawing-an-answer-while-it-is-generated)) |
| `js/chat.js` | 563 | conversations, the request to `/api/chat`, drawing the turns |
| `js/vision.js` | 265 | pictures in the chat ([docs/20](20-vision.md)) |
| `js/models.js`, `js/tools.js`, `js/admin.js` | 318, 193, 221 | the Models and Tools tabs; Settings, Logs and Connect, then `startApp()` |
| `js/images*.js` (5 files) | 2 121 | the Images tab, its gallery and editor ([docs/22](22-image-generation.md)) |
| `js/video*.js` (2 files) | 756 | the Video tab ([docs/23](23-video-generation.md)) |
| `js/audio*.js` (5 files) | 2 854 | the Audio tab, its editor and the audio buttons of the chat ([docs/21](21-audio.md)) |

**Tabs.** `showTab(name)` in `core.js` shows one `<section class="tab">` and writes the name into the address
(`#models`), so a reload returns to the same tab. A tab registers two callbacks: `onShow[name]`, called when it
becomes visible, and `onTick[name]`, called every 1.5 s while it is the visible tab.

**What is kept in the browser.** The server keeps no conversation. Everything a person sees again after a reload
is in the browser's `localStorage` (`store.get` / `store.set`, JSON values):

| key | value |
|---|---|
| `jllm.convs`, `jllm.current` | the conversations (title, messages in the OpenAI format plus `usage`, `finish`, `model`) and the id of the open one |
| `jllm.params`, `jllm.system`, `jllm.priority`, `jllm.think`, `jllm.model` | the sampling fields, the system prompt, interactive or background, the thinking switch, the chosen model |
| `jllm.tools` | the tools a new chat starts with |
| `jllm.theme`, `jllm.convsCollapsed`, `jllm.apiKey` | the theme, the folded sidebar, the API key when the server asks for one |

The pictures of a chat, the generated images and the clips are too large for that: they live in IndexedDB
(`vision.js`, `images-store.js`, `video-store.js`).

**The event stream.** The chat sends one `POST /api/chat` per turn with the whole conversation and reads the
response as it arrives (`fetch`, `res.body.getReader()`). The response is `text/event-stream`: one `data: {json}`
line per event, an empty line between events, written and flushed by `ChatSupport.Sse` as soon as the engine has a
piece of text.

```
{"type":"start","model":"Qwen/Qwen3-0.6B","tools":[]}
{"type":"delta","content":"Hello"}                      text as it is generated
{"type":"tool_call","id":…,"name":…,"arguments":{…}}    the server is about to run a tool
{"type":"tool_result","id":…,"content":…,"error":false,"ms":123}
{"type":"usage","usage":{…}}                            after every generation pass
{"type":"message","message":{…}}                        a message to keep in the conversation
{"type":"error","message":"…"}
{"type":"done","finish_reason":"stop|length|cancelled|tool_limit|error"}
```

`run()` in `chat.js` keeps the state of the turn being generated in one object:

```js
running = {abort, conv, text: '', calls: {}, usage: null, error: null, toolRunning: false}
```

A `delta` appends to `running.text`. A `message` moves a finished message into the conversation (and empties
`running.text` when it is the assistant's). `tool_call` and `tool_result` record the state of a tool in
`running.calls`. After every event the page is asked to paint (`renderLive()`), which leads to the next section.

## 29.3 Drawing an answer while it is generated

**One paint per frame.** Tokens arrive faster than a screen refreshes: 30 tokens per second from a small model on a
CPU, several hundred from a GPU. `renderLive()` therefore does not paint; it asks for one animation frame
(`requestAnimationFrame`) unless one is already asked for. However many events arrive in 16.7 ms, the page paints
once.

**What a paint used to cost.** The turn being generated was one string of HTML, built from scratch for every paint
and assigned to `innerHTML`: every finished message of the turn (an agent turn has several: answer, tool call,
result, answer, …), every tool card (its arguments parsed and formatted again), and the Markdown of the whole text
so far. For an answer of `n` characters painted `f` times that is `n · f / 2` characters through the renderer: the
work grows with the **square** of the length. Replacing the HTML also throws away what the browser had built: the
text selection, the layout of every block, the pictures and clips of a tool result (loaded again 60 times per
second), a reasoning block the reader had just folded.

**What a paint costs now.** The bubble of the running turn has two parts (`paintLive` in `chat.js`), both
`display: contents`, so they add no box and the turn looks as before:

```
<div class="bubble">
  <div class="settled">   the turn's finished messages and tool cards
  <div class="live">      the text being generated
    <details class="think">    reasoning: one text node that grows
    <div class="md">           the answer: Markdown.stream
    <span class="cursor">
    <div class="error">
```

* `.settled` is rendered again only when its **key** changes: the number of messages of the turn and the state of
  its tools (`msgs.length + '|' + JSON.stringify(running.calls)`). While tokens arrive the key does not change, so
  this part is not touched. The cards a reader has opened stay open across a change.
* The **reasoning** (`<think>…`) is plain text. The new characters are appended to the text node that is already
  there (`appendData`); nothing is escaped or laid out again. The block is opened when thinking starts and folded
  when the answer starts; in between it obeys the reader.
* The **answer** goes through `Markdown.stream`, below.
* The cursor and the error line are shown and hidden, not created.

When the stream ends, `run()` stores the conversation and `renderMessages()` draws the finished turn once with the
ordinary renderer, as it does for every turn of a conversation that is opened.

### The streaming Markdown renderer

`Markdown.render(text)` escapes the text, splits it into lines and walks them once. A paragraph, a list, a table
and a quote each end at a blank line; a fenced code block (` ``` `) ends at its closing fence and may contain blank
lines. Nothing else carries state from one line to the next. So for a blank line that is **outside** a fence:

```
render(a + blankLine + b)  ===  render(a) + render(b)
```

That is the whole idea. `Markdown.stream(el)` returns an object with one method, `update(text)`:

1. `settledEnd(text, from)` scans the lines after the part that is already settled, keeps track of being inside a
   fence, and returns the offset just after the last blank line outside a fence.
2. The text between the old and the new offset is rendered **once** and inserted before the tail
   (`insertAdjacentHTML('beforebegin', …)`). It is never rendered again and its nodes are never replaced.
3. The rest, the *tail*, is the block still being written. It is rendered into `<div class="md-tail">` when it
   changed.

A text that does not start with the settled part (a new answer after a tool call, an edited message) starts over.

📐 *Worked example.* The model has written this so far (`␤` is a line break):

~~~
  The command is:␤
  ␤
  ```sh␤
  javac Main.java␤
  ␤
  java Ma
~~~

| update | text ends with | settled offset | rendered in this update |
|---|---|---:|---|
| 1 | `The command` | 0 | tail: `<p>The command</p>` |
| 2 | `…is:␤␤` | 17 | settled: `<p>The command is:</p>` (appended, final); tail: empty |
| 3 | `` ```sh␤javac Main.java␤␤java Ma `` | 17 | tail: the code block with `javac Main.java`, an empty line and `java Ma`. The blank line is inside the fence: nothing settles |
| 4 | `` …java Main␤```␤␤Done `` | after the fence's blank line | settled: the finished code block; tail: `<p>Done</p>` |

The scan in step 1 starts at the settled offset, not at 0, so an update costs in proportion to the tail: the block
being written, typically a paragraph or one code block. The cost of a whole answer is proportional to its length,
not to its square.

**Why the two renderers cannot disagree.** The claim `render(a + blank + b) === render(a) + render(b)` holds only
if `settledEnd` and `render` agree on what a fence is. They did not at first: `render` took an indented ` ``` `
line after a list item as more text of that item, while the scan saw a fence opening. The renderer was changed
(a fence ends the list and becomes a code block, which is also what a model means when it writes a command under a
numbered step), and `markdown-stream-test.js` checks the equality on a document with every construct and on 400
random documents cut at random places, after every single update.

## 29.4 What the page asks when nobody types

**The timers.** `startApp()` starts three, and the Audio tab one:

| every | request | for |
|---|---|---|
| 5 s | `GET /api/status` | the box at the bottom left: models, weights, heap, uptime, running jobs |
| 1.5 s | the visible tab's `onTick` | Models: loaded models, jobs, memory. Logs: `GET /api/logs?after=<last>`. Tools: only while an MCP server connects. Images, Video: only while no model is loaded |
| 6 s | `GET /api/models/loaded` | the model list of the chat (not on the Models tab, whose tick loads it) |
| 6 s | the audio models | the audio buttons of the chat (not on the Audio tab) |

All four now check `document.hidden` first. A page in a background tab or a minimized window asks nothing; when it
is shown again (`visibilitychange`) it refreshes the status, the models and the visible tab at once. A console that
is left open all day beside a coding agent sent about 2 000 requests an hour from the Chat tab (several times that
from the Models tab) to a server that is busy generating; hidden, it now sends none. A generation that is running is not affected: its stream is one open response, not a timer.

**The Models tab.** Its tick asks for the loaded models, the jobs and the memory report every 1.5 s, and most
answers repeat the last one. Each list is drawn again only when its data changed (`changed()` in `models.js`
compares the JSON with the one it was last drawn from), so a text selection survives, a button does not vanish
under a click and the tab is not laid out 40 times a minute. The Audio rows of that tab already worked this way.

**The log view.** `GET /api/logs?after=n` returns only the lines after sequence number `n`. The page used to draw
all of them again whenever one arrived: up to 2 500 `<span>` elements every 1.5 s during a download or a
conversion, which are exactly the moments the log is watched. `appendLog` now adds the new lines at the end and
removes the oldest at the top, with the filter applied to the new lines. The full `renderLog` remains for the
first time, a changed filter, *Clear view* and a restarted server (its sequence numbers start again).

**The files of the page.** 25 files, 465 kB. Two settings in `application.properties`:

* `spring.web.resources.cache.cachecontrol.no-cache=true`: the browser keeps the files but asks whether one changed
  before using it; the answer for an unchanged file is `304` without a body. Without the header a browser chooses
  by itself how long a file stays fresh (usually a tenth of its age), so after an upgrade it could run a new
  `chat.js` against an old `markdown.js`. The scripts depend on each other, so they must come from one version.
* `server.compression.*`: the HTML, CSS and JavaScript are compressed (465 kB → 129 kB with gzip). On the same
  machine this saves nothing worth measuring; it is for a console opened from another machine. The list of content
  types is short on purpose: `text/event-stream` is not in it, so a token never waits in a compression buffer, and
  neither is the JSON of the API nor a generated picture, clip or sound.

The same idea in the engine's own page (`./jllm serve`, `WebUi.java`, one small HTML string): a piece of the answer
is appended to a text node instead of setting the text of the whole answer again, and the page scrolls once per
network read instead of once per token.

## 29.5 Measured

`node jllm-web/src/test/js/chat-stream-bench.js <tokens> 2` serves the console's files with a scripted backend that
streams a reasoning block and then Markdown (headings, paragraphs, numbered lists, code blocks, tables, quotes), 4
characters every 2 ms, and reads Chromium's own counters for the page's main thread. Headless Chromium on a 4-core
x86 Linux machine, nothing else running; "before" is the same script on the previous commit.

| answer | | main thread busy | of which script | layout | style | frames per second | longest gap between frames |
|---|---|---:|---:|---:|---:|---:|---:|
| 4 000 tokens (16 kB, 161 blocks), 10 s | before | 5.03 s | 0.92 s | 1.46 s | 0.70 s | 59.8 | 33 ms |
| | after | **1.77 s** | 0.17 s | 0.18 s | 0.15 s | 60.0 | 17 ms |
| 12 000 tokens (48 kB, 482 blocks), 27 s | before | 22.07 s | 4.68 s | 7.92 s | 3.33 s | 47.5 | 67 ms |
| | after | **6.43 s** | 0.63 s | 0.70 s | 0.88 s | 60.0 | 50 ms |

Reading the table:

* Three times the length cost 4.4 times the work before (5.03 → 22.07 s; the quadratic term on top of a part that
  grows with the length) and 3.6 times after (1.77 → 6.43 s): what remains is close to proportional. It is the cost
  of receiving 12 000 events, painting 1 600 frames and scrolling, which no renderer avoids.
* At 12 000 tokens the old page was busy 82 % of the time and dropped a fifth of its frames; the new one is busy
  24 % of the time and drops none.
* The HTML of the finished turn is byte for byte the same in both versions (compared for both lengths).
* The stream took the same time in both: the page was never the bottleneck of the *stream*, it was a second
  consumer of the processor. On a machine where the browser and a CPU backend share the cores, the seconds in the
  first column are seconds one core does not give to the model. That effect was not measured here.

With a real model (Qwen3-0.6B, Q4_0, CPU backend, 4 threads, the console and the browser on the same machine) a
454-token answer with a reasoning block, a list, a code block and a table streamed at 29.7 tokens per second and
finished without a page error; the response headers are the ones of [29.4](#294-what-the-page-asks-when-nobody-types)
(`Cache-Control: no-cache` and `Content-Encoding: gzip` on `js/chat.js`, a `304` for a conditional request, no
`Content-Encoding` on `/api/chat` and `/api/status`).

## 29.6 Why this design

* **A Markdown library with an incremental mode, or a virtual DOM.** Both solve the problem and both end the
  console's "no build step, no dependency" rule (`index.html` loads files, nothing is bundled). The renderer here
  is 180 lines, its incremental part 50 of them with the comments; the property that makes it incremental was already true of it.
* **Rendering less often** (every 100 ms instead of every frame). It divides the constant and keeps the square: a
  long answer is still rendered whole, six times less often, and the text visibly arrives in steps.
* **Rendering only new characters** (appending text, formatting at the end). It is the cheapest, and the reader
  watches raw `**asterisks**` and table pipes until the answer is complete.
* **Settling at every line break instead of every blank line.** A paragraph's second line, a list's next item and
  a table's next row change the HTML of the lines before them (the same `<p>`, the same `<ul>`, the same
  `<table>`); only a blank line outside a fence is a boundary the renderer cannot cross.
* **Keeping the reasoning as HTML.** It is shown as plain text (`white-space: pre-wrap`), so one growing text node
  is all it needs.
* **`Cache-Control: max-age` with versioned file names** would save the 25 conditional requests of a page load.
  It needs a build step that renames files and rewrites `index.html`; a `304` on the loopback interface costs
  well under a millisecond.

What the design costs: the running turn and a finished turn are drawn by two code paths (`paintLive` and
`assistantTurnHtml`). They share the renderer and the tool cards, the benchmark compares their final HTML, and the
finished turn is always drawn by the ordinary path, so a difference could only be visible while a turn streams.

## 29.7 Settings

These are application properties of the console (`jllm-web/src/main/resources/application.properties`, or
`--name=value` after `./jllm-web/run.sh`), not engine settings.

| property | default | meaning |
|---|---|---|
| `server.compression.enabled` | `true` | compress the console's own files |
| `server.compression.mime-types` | `text/html,text/css,text/javascript,application/javascript` | which responses; keep `text/event-stream` out of it |
| `server.compression.min-response-size` | `2048` | smaller files are sent as they are |
| `spring.web.resources.cache.cachecontrol.no-cache` | `true` | the browser revalidates the console's files before using them |
| `spring.servlet.multipart.max-file-size`, `…max-request-size` | `4GB` | the container's ceiling for uploads; the real limits are the engine settings `audio.max-upload-mb` and `vision.max-image-bytes` |
| `jllm.web.log-lines` | `2000` | lines the server keeps for the Logs tab (the page shows up to 2 500 of what it has received) |

## 29.8 Checks and limits

**Tests.**

* `node jllm-web/src/test/js/markdown-stream-test.js` (8 tests, no browser): `settledEnd` on blank lines inside
  and outside fences; `stream` against `render` after every update, for a document with every construct fed 1, 2,
  3, 5, 7, 13 and 64 characters at a time and for 400 random documents; one append per finished block; starting
  over; the three corrections below.
* `node jllm-web/src/test/js/chat-stream-bench.js` ([29.5](#295-measured)) reports page errors, so it is also a
  smoke test of the chat page in a real browser.
* `cd jllm-web && mvn test`: `ImagesApiTest.edits` covers the upload limit below. The console's other tests do not
  touch the page.

**Corrections that came with this work.**

* *A table could stop the chat from drawing.* A line with pipes followed by a separator without a leading pipe
  (`| a |` then `---`, or `--- | ---`) made `render` read a row that was not there and throw. During a stream the
  turn stopped updating; afterwards the conversation could not be drawn at all, because it is drawn again from the
  stored messages. The header and the separator are now taken as they are.
* *Numbered lists restarted at 1.* Models often leave a blank line between numbered items; each item is then its
  own list, and every one was numbered 1. A list now starts at its first number (`<ol start="2">`).
* *A code fence under a list item* was shown as text of the item, backticks included. It now ends the list and
  becomes a code block (see [29.3](#293-drawing-an-answer-while-it-is-generated)).
* *The upload limit of `/v1/images/edits` was 4 GB, not 64 MB.* `application.properties` set
  `spring.servlet.multipart.max-file-size` twice (64 MB for pictures, 4 GB for audio); in a properties file the
  last one wins, and the endpoint read every part into memory without a check of its own. It now refuses a file
  part over `vision.max-image-bytes` (32 MB by default, the limit of a chat's pictures) and a text field over
  1 MB with a 413, before reading it. The audio endpoints already checked `audio.max-upload-mb`.

**Not done.**

* The conversation is still drawn whole when it is opened or a turn ends (once, not per frame). A conversation
  with hundreds of long turns takes a noticeable moment to open; drawing only the turns in view would fix that.
* Scrolling to the end reads the height of the message list once per frame, which makes the browser lay out what
  changed. It is what remains of "layout" in the table.
* `localStorage` holds every conversation in one value (`jllm.convs`), written when a turn ends. Browsers allow
  about 5 MB per origin. A write that does not fit used to be ignored, and a reload then showed the conversations
  as they were at the last write that fit; the chat now says that the storage is full (once, until a write works
  again). Moving the conversations to IndexedDB, as the pictures and clips are, would lift the limit.
* The effect on the tokens per second of a CPU backend that shares its cores with the browser was not measured.
