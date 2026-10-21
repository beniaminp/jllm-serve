'use strict';
// Images: one generation at a time. Drives a streamed request and reports into the slots / progress widgets of
// the view that started it (Create or Editor).

const ImgRun = (() => {
  let current = null;
  const busyListeners = [];
  const notify = () => busyListeners.forEach(f => f(!!current));

  // path: /v1/images/generations | /v1/images/edits. body: the JSON request (stream is forced on).
  // ui: {slots: [ImgSlot] (one per image), progress(text, frac), image(i, blob, ev), done(ev), end(error|null, doneEvent)}
  async function start(path, body, ui) {
    if (current) return;
    const ctrl = new AbortController();
    current = {ctrl};
    notify();
    const n = body.n || 1;
    let finished = 0;
    ui.progress?.('starting…', 0);
    try {
      const done = await ImgApi.stream(path, {...body, stream: true}, {
        progress: ev => {
          const frac = (ev.image + (ev.steps ? ev.step / ev.steps : 0)) / n;
          ui.progress?.(`image ${ev.image + 1}/${n} · step ${ev.step}/${ev.steps}`, Math.min(1, frac));
          ui.slots?.[ev.image]?.progress(ev.steps ? ev.step / ev.steps : 0);
          if (ev.step === 0) ui.slots?.[ev.image]?.text('step 0…');
        },
        partial: (ev, blob) => ui.slots?.[ev.image]?.preview(blob),
        image: (ev, blob) => { finished++; ui.image?.(ev.image, blob, ev); ui.progress?.(finished < n ? `image ${finished + 1}/${n}` : 'finishing…', finished / n); },
        done: ev => ui.done?.(ev),
      }, ctrl.signal);
      ui.end?.(null, done);
    } catch (e) {
      const aborted = e.name === 'AbortError';
      (ui.slots || []).forEach(s => { if (!s.blob) s.fail(aborted ? 'stopped' : 'failed'); });
      ui.end?.(aborted ? 'stopped' : e, null);
    } finally {
      current = null;
      notify();
    }
  }

  return {
    start,
    stop: () => current?.ctrl.abort(),
    busy: () => !!current,
    onBusy: f => busyListeners.push(f),
  };
})();
