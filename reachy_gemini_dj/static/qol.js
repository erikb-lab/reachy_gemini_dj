/* Quality-of-life UI: "Brain:" status chip, quiet-mode bar with a Wake button, and the Conversation card.
   Self-contained; reads window.reachyApi state (conversation, brain.chip, quiet). Vanilla JS, no dependencies. */
(function () {
  'use strict';
  var api = window.reachyApi;
  var feed = document.getElementById('feed');
  var bar = document.getElementById('statusbar');
  if (!api || !feed || !bar) return;

  var css = document.createElement('link');
  css.rel = 'stylesheet'; css.href = '/static/qol.css';
  document.head.appendChild(css);

  function h(tag, cls, text) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    return e;
  }

  /* brain chip, next to the backend chip in the header */
  var chip = h('span', 'stat'), dot = h('span', 'dot warn'), chipText = h('span', '', 'Brain: …');
  chip.id = 'st-brain'; chip.title = 'Local voice brain (on the laptop)';
  chip.append(dot, chipText);
  var anchorChip = document.getElementById('st-backend');
  if (anchorChip && anchorChip.parentNode) anchorChip.parentNode.insertBefore(chip, anchorChip.nextSibling);

  /* quiet bar, under the header */
  var qbar = h('div', 'hide'), qtext = h('span'), qwake = h('button', 'btn', 'Wake');
  qbar.id = 'quiet-bar'; qbar.setAttribute('role', 'status'); qwake.type = 'button';
  qbar.append(qtext, qwake);
  bar.appendChild(qbar);
  qwake.addEventListener('click', function () {
    qwake.disabled = true;
    api.post('/api/quiet/wake', {}).then(function () { api.toast('Awake'); })
      .catch(function () { api.toast('Could not wake Reachy'); })
      .then(function () { qwake.disabled = false; });
  });

  /* conversation card, above Recent events */
  var anchor = feed.closest('section') || feed.parentNode;
  var card = h('section', 'card span2');
  card.setAttribute('aria-labelledby', 'h-cv');
  var title = h('h2', '', 'Conversation '), badge = h('span', 'badge', '0');
  title.id = 'h-cv'; title.appendChild(badge);
  var list = h('ul', 'cv-list'); list.setAttribute('aria-live', 'polite');
  card.append(title, h('p', 'sub', 'What Reachy heard and answered, newest first. Kept in memory only, never saved.'), list);
  anchor.parentNode.insertBefore(card, anchor);

  var lastSig = '';
  function renderConversation(items) {
    var sig = JSON.stringify(items);
    if (sig === lastSig) return;
    lastSig = sig;
    badge.textContent = items.length;
    list.replaceChildren();
    items.forEach(function (t) {
      var li = h('li'), meta = h('div', 'cv-meta');
      meta.append(h('span', '', t.t || ''), h('span', '', t.backend === 'gemini' ? 'Gemini' : (t.backend === 'local' ? 'local brain' : '')));
      if (t.latency_s != null) meta.append(h('span', 'cv-lat', t.latency_s.toFixed(1) + ' s'));
      li.appendChild(meta);
      if (t.heard) li.appendChild(h('div', 'cv-h', '🗣 ' + t.heard));
      if (t.said) li.appendChild(h('div', 'cv-s', '🤖 ' + t.said));
      list.appendChild(li);
    });
    if (!items.length) list.appendChild(h('li', 'empty', 'Nothing yet.'));
  }

  api.subscribe(function (s) {
    var c = (s.brain && s.brain.chip) || null;
    if (c) {
      chipText.textContent = c.text;
      dot.className = 'dot ' + (c.level || 'warn');
    }
    var q = s.quiet || {};
    qbar.classList.toggle('hide', !q.active);
    if (q.active) qtext.textContent = '😴 Quiet until ' + q.until_text;
    renderConversation(Array.isArray(s.conversation) ? s.conversation : []);
  });
})();
