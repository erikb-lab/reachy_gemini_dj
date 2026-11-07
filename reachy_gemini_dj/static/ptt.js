/* Push-to-talk: a big hold-to-talk button (mouse + touch) and Space-bar talk. Self-contained; shown only when the
   robot is in listen_mode "push_to_talk". Talks to POST /api/ptt {"down": bool}. Vanilla JS, no dependencies. */
(function () {
  'use strict';
  var card, btn, label, hint, held = false, serverActive = false, mode = 'always', chain = Promise.resolve();

  function send(down) {
    // serialised so a release can never overtake its press; the server also auto-releases after 30 s
    chain = chain.then(function () {
      return fetch('/api/ptt', { method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ down: down }), keepalive: true })
        .then(function (r) { if (!r.ok) throw new Error('HTTP ' + r.status); })
        .catch(function () {
          if (!down) return fetch('/api/ptt', { method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ down: false }), keepalive: true }).catch(function () {});
        });
    });
  }
  function paint() {
    if (!card) return;
    var live = held || serverActive;
    card.classList.toggle('hide', mode !== 'push_to_talk');
    btn.classList.toggle('live', live);
    btn.setAttribute('aria-pressed', live ? 'true' : 'false');
    label.textContent = live ? 'Listening…' : 'Hold to talk';
    hint.textContent = live ? 'Release when you are done.' : 'Press and hold (or hold the Space bar), speak, then let go.';
  }
  function down() { if (held) return; held = true; send(true); paint(); }
  function up() { if (!held) return; held = false; send(false); paint(); }

  function typing(t) {
    if (!t || !t.tagName) return false;
    var n = t.tagName;
    return n === 'INPUT' || n === 'TEXTAREA' || n === 'SELECT' || t.isContentEditable;
  }

  function build() {
    var grid = document.querySelector('#p-dj .grid');
    if (!grid) return false;
    card = document.createElement('section');
    card.className = 'card span2 ptt-card hide';
    card.setAttribute('aria-labelledby', 'h-ptt');
    card.innerHTML = '<h2 id="h-ptt">Push to talk</h2>' +
      '<button type="button" class="ptt-btn" id="b-ptt" aria-pressed="false"><span class="ptt-ic" aria-hidden="true">&#127908;</span>' +
      '<span class="ptt-l">Hold to talk</span></button><p class="ptt-hint sub"></p>';
    grid.insertBefore(card, grid.firstChild);
    btn = card.querySelector('#b-ptt'); label = card.querySelector('.ptt-l'); hint = card.querySelector('.ptt-hint');

    btn.addEventListener('pointerdown', function (e) { if (e.button > 0) return; e.preventDefault(); down(); });
    ['pointerup', 'pointercancel', 'pointerleave'].forEach(function (t) { btn.addEventListener(t, up); });
    btn.addEventListener('contextmenu', function (e) { e.preventDefault(); });
    btn.addEventListener('keydown', function (e) { if (e.key === 'Enter') { e.preventDefault(); if (!e.repeat) down(); } });
    btn.addEventListener('keyup', function (e) { if (e.key === 'Enter') up(); });
    return true;
  }

  // Space bar holds-to-talk on desktop (not while typing in a field)
  document.addEventListener('keydown', function (e) {
    if (mode !== 'push_to_talk' || (e.key !== ' ' && e.code !== 'Space') || typing(e.target) || e.ctrlKey || e.metaKey || e.altKey) return;
    e.preventDefault();
    if (!e.repeat) down();
  });
  document.addEventListener('keyup', function (e) {
    if ((e.key !== ' ' && e.code !== 'Space') || typing(e.target)) return;
    if (held) { e.preventDefault(); up(); }
  });
  // never leave the mic open when the page goes away
  document.addEventListener('visibilitychange', function () { if (document.hidden) up(); });
  window.addEventListener('blur', up);
  window.addEventListener('pagehide', up);

  function render(s) {
    if (!s) return;
    mode = s.listen_mode || 'always';
    serverActive = !!s.ptt_active;
    if (mode !== 'push_to_talk' && held) up();
    paint();
  }
  function init() {
    if (!build()) return;
    var tries = 0;
    (function attach() {
      if (window.reachyApi && typeof window.reachyApi.subscribe === 'function') window.reachyApi.subscribe(render);
      else if (tries++ < 100) setTimeout(attach, 100);
    })();
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init); else init();
})();
