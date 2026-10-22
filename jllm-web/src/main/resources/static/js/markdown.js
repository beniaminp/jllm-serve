'use strict';
// A small, safe Markdown renderer for model output: everything is escaped first, then a line-based pass builds
// headings, fenced code (with a copy button), tables, lists, quotes and rules; inline code, bold, italics,
// strikethrough and http(s) links. Incomplete constructs (while streaming) degrade to text.

const Markdown = (() => {
  function inline(s) {
    const codes = [];
    s = s.replace(/`([^`\n]+)`/g, (_, c) => { codes.push(c); return `\u0000${codes.length - 1}\u0000`; });
    // images: begin
    const imgs = [];   // pictures kept by the server (/v1/images/files/<hex>.png) and nothing else
    s = s.replace(/!\[([^\]\n]*)\]\((\/v1\/images\/files\/[0-9a-f]{8,64}\.png)\)/g, (_, alt, u) => {
      imgs.push(`<a href="${u}" target="_blank" rel="noopener"><img class="md-img" src="${u}" alt="${alt.replace(/[*_`~]/g, '')}" loading="lazy"></a>`);
      return `\u0001${imgs.length - 1}\u0001`;
    });
    // images: end
    s = s
      .replace(/\*\*([^*\n]+?)\*\*/g, '<strong>$1</strong>')
      .replace(/__([^_\n]+?)__/g, '<strong>$1</strong>')
      .replace(/(^|[^*\w])\*([^*\n]+?)\*(?!\w)/g, '$1<em>$2</em>')
      .replace(/(^|[^_\w])_([^_\n]+?)_(?!\w)/g, '$1<em>$2</em>')
      .replace(/~~([^~\n]+)~~/g, '<del>$1</del>')
      .replace(/\[([^\]\n]+)\]\((https?:\/\/[^)\s]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>')
      .replace(/(^|[\s(])(https?:\/\/[^\s<)]+[^\s<).,;:!?'"])/g, '$1<a href="$2" target="_blank" rel="noopener noreferrer">$2</a>');
    return s.replace(/\u0000(\d+)\u0000/g, (_, i) => `<code>${codes[+i]}</code>`).replace(/\u0001(\d+)\u0001/g, (_, i) => imgs[+i]) /* images */;
  }

  function table(rows) {
    const cells = r => r.replace(/^\s*\|/, '').replace(/\|\s*$/, '').split('|').map(c => c.trim());
    const head = cells(rows[0]);
    const align = cells(rows[1]).map(c => c.startsWith(':') && c.endsWith(':') ? 'center' : c.endsWith(':') ? 'right' : '');
    const td = (tag, c, i) => `<${tag}${align[i] ? ` style="text-align:${align[i]}"` : ''}>${inline(c)}</${tag}>`;
    return `<table><thead><tr>${head.map((c, i) => td('th', c, i)).join('')}</tr></thead><tbody>`
      + rows.slice(2).map(r => `<tr>${cells(r).map((c, i) => td('td', c, i)).join('')}</tr>`).join('') + '</tbody></table>';
  }

  function render(src) {
    const lines = esc(src).split('\n');
    const out = [];
    let para = [];
    const flush = () => { if (para.length) { out.push(`<p>${inline(para.join('\n')).replace(/\n/g, '<br>')}</p>`); para = []; } };
    for (let i = 0; i < lines.length; i++) {
      const line = lines[i];
      const fence = line.match(/^\s*```\s*([\w+#.-]*)/);
      if (fence) {
        flush();
        const body = [];
        i++;
        while (i < lines.length && !/^\s*```\s*$/.test(lines[i])) body.push(lines[i++]);
        out.push(`<div class="codeblock"><div class="cb-head"><span>${fence[1] || 'code'}</span><button data-copy>copy</button></div>`
          + `<pre><code>${body.join('\n')}</code></pre></div>`);
        continue;
      }
      if (/^\s*$/.test(line)) { flush(); continue; }
      const h = line.match(/^(#{1,6})\s+(.*)$/);
      if (h) { flush(); const n = Math.min(4, h[1].length); out.push(`<h${n}>${inline(h[2])}</h${n}>`); continue; }
      if (/^\s*([-*_])(\s*\1){2,}\s*$/.test(line)) { flush(); out.push('<hr>'); continue; }
      if (/^\s*\|.*\|\s*$/.test(line) && i + 1 < lines.length && /^\s*\|?\s*:?-{2,}/.test(lines[i + 1])) {
        flush();
        const rows = [];
        while (i < lines.length && /^\s*\|.*\|?\s*$/.test(lines[i]) && lines[i].includes('|')) rows.push(lines[i++]);
        i--;
        out.push(table(rows));
        continue;
      }
      if (/^\s*&gt;\s?/.test(line)) {
        flush();
        const q = [];
        while (i < lines.length && /^\s*&gt;\s?/.test(lines[i])) q.push(lines[i++].replace(/^\s*&gt;\s?/, ''));
        i--;
        out.push(`<blockquote>${render(q.join('\n').replace(/&amp;/g, '&').replace(/&lt;/g, '<').replace(/&gt;/g, '>').replace(/&quot;/g, '"').replace(/&#39;/g, "'"))}</blockquote>`);
        continue;
      }
      const li = line.match(/^(\s*)([-*+]|\d+[.)])\s+(.*)$/);
      if (li) {
        flush();
        const ordered = /\d/.test(li[2]);
        const items = [];
        while (i < lines.length) {
          const m = lines[i].match(/^(\s*)([-*+]|\d+[.)])\s+(.*)$/);
          if (m) { items.push({depth: m[1].length >= 2 ? 1 : 0, text: m[3]}); i++; continue; }
          if (/^\s{2,}\S/.test(lines[i]) && items.length) { items[items.length - 1].text += '<br>' + lines[i].trim(); i++; continue; }
          break;
        }
        i--;
        let html = ordered ? '<ol>' : '<ul>';
        let nested = false;
        for (const it of items) {
          if (it.depth && !nested) { html += '<ul>'; nested = true; }
          if (!it.depth && nested) { html += '</ul>'; nested = false; }
          html += `<li>${inline(it.text)}</li>`;
        }
        if (nested) html += '</ul>';
        out.push(html + (ordered ? '</ol>' : '</ul>'));
        continue;
      }
      para.push(line);
    }
    flush();
    return out.join('');
  }

  // copy buttons work through delegation, so re-rendered blocks need no wiring
  document.addEventListener('click', e => {
    const b = e.target.closest('[data-copy]');
    if (b) copyText(b.closest('.codeblock').querySelector('code').textContent, b);
  });

  return {render};
})();
