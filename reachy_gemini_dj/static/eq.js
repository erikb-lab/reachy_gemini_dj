/* Music EQ card (Settings tab): three bands + presets. Reads state.config.music_eq_* via reachyApi.subscribe and
   saves with POST /api/config on slider release / preset click. Self-contained, vanilla JS. */
(function () {
  'use strict';
  var BANDS = [['bass', 'Bass'], ['mid', 'Mid'], ['treble', 'Treble']];
  var PRESETS = [
    ['Flat', [0, 0, 0]],
    ['Bass boost', [6, 0, 0]],
    ['Vocal', [-2, 4, 2]],
    ['Treble boost', [0, 0, 6]],
    ['Small speaker', [5, 1, 3]]
  ];
  var sliders = {}, outs = {}, dragging = null;

  function fmt(v) { v = Math.round(v); return (v > 0 ? '+' : '') + v + ' dB'; }
  function el(tag, cls, text) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text !== undefined) e.textContent = text;
    return e;
  }
  function values() { return BANDS.map(function (b) { return +sliders[b[0]].value; }); }

  function save(vals) {
    var body = { music_eq_bass: vals[0], music_eq_mid: vals[1], music_eq_treble: vals[2] };
    var p = window.reachyApi && window.reachyApi.post ? window.reachyApi.post('/api/config', body)
      : fetch('/api/config', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
    if (p && p.catch) p.catch(function (e) {
      if (window.reachyApi && window.reachyApi.toast) window.reachyApi.toast('EQ save failed: ' + e.message);
    });
  }
  function setAll(vals) {
    BANDS.forEach(function (b, i) { sliders[b[0]].value = vals[i]; outs[b[0]].textContent = fmt(vals[i]); });
  }

  function build() {
    var panel = document.getElementById('p-settings');
    if (!panel || document.getElementById('eq-card')) return !!panel;
    var card = el('section', 'card');
    card.id = 'eq-card';
    card.setAttribute('aria-labelledby', 'h-eq');
    var h = el('h2', null, 'Music EQ'); h.id = 'h-eq';
    card.appendChild(h);
    card.appendChild(el('p', 'sub', 'Applies to the current song after a short restart and to every song after.'));

    BANDS.forEach(function (b) {
      var field = el('div', 'sp-field');
      var lab = el('label', null, b[1]);
      var inp = document.createElement('input');
      inp.type = 'range'; inp.min = -12; inp.max = 12; inp.step = 1; inp.value = 0;
      inp.id = 'eq-' + b[0];
      lab.htmlFor = inp.id;
      var out = el('output', null, fmt(0));
      inp.addEventListener('input', function () { dragging = b[0]; out.textContent = fmt(+inp.value); });
      inp.addEventListener('change', function () { save(values()); dragging = null; });
      sliders[b[0]] = inp; outs[b[0]] = out;
      var rng = el('div', 'sp-range');
      rng.appendChild(inp); rng.appendChild(out);
      field.appendChild(lab); field.appendChild(rng);
      card.appendChild(field);
    });

    card.appendChild(el('h3', null, 'Presets'));
    var pr = el('div', 'eq-presets');
    pr.setAttribute('role', 'group');
    pr.setAttribute('aria-label', 'EQ presets');
    PRESETS.forEach(function (p) {
      var btn = el('button', 'btn ghost eq-pill', p[0]);
      btn.type = 'button';
      btn.addEventListener('click', function () { setAll(p[1]); save(p[1]); });
      pr.appendChild(btn);
    });
    var reset = el('button', 'btn ghost eq-pill eq-reset', 'Reset');
    reset.type = 'button';
    reset.addEventListener('click', function () { setAll([0, 0, 0]); save([0, 0, 0]); });
    pr.appendChild(reset);
    card.appendChild(pr);

    var first = panel.querySelector('details.acc');
    if (first) panel.insertBefore(card, first); else panel.appendChild(card);
    return true;
  }

  function render(s) {
    var c = s && s.config;
    if (!c || !sliders.bass) return;
    BANDS.forEach(function (b) {
      var v = c['music_eq_' + b[0]];
      if (typeof v !== 'number' || dragging === b[0]) return;   // never fight the user's drag
      sliders[b[0]].value = Math.round(v);
      outs[b[0]].textContent = fmt(v);
    });
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
