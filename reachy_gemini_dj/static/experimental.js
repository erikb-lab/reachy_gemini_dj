/* Settings > Experimental: opt-in features that are still rough. Currently the gamepad: the on/off switch, the speed
   multiplier, and the controller mapping table (built by gamepad.js through window.reachyGamepad.mountMapping).
   Vanilla JS, no dependencies. */
(function () {
  'use strict';
  var els = {}, editing = false;

  function h(tag, attrs, kids) {
    var e = document.createElement(tag);
    Object.keys(attrs || {}).forEach(function (k) {
      if (k === 'class') e.className = attrs[k];
      else if (k === 'text') e.textContent = attrs[k];
      else e.setAttribute(k, attrs[k]);
    });
    (kids || []).forEach(function (c) { if (c) e.appendChild(typeof c === 'string' ? document.createTextNode(c) : c); });
    return e;
  }
  function post(path, body) {
    if (window.reachyApi && window.reachyApi.post) return window.reachyApi.post(path, body);
    return fetch(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })
      .then(function (r) { if (!r.ok) throw new Error('HTTP ' + r.status); return r.json(); });
  }
  function toast(msg) { if (window.reachyApi && window.reachyApi.toast) window.reachyApi.toast(msg); }
  function fmtX(v) { return (+v).toFixed(2) + 'x'; }
  function save(patch, okMsg) {
    return post('/api/config', patch).then(function () { if (okMsg) toast(okMsg); })
      .catch(function (e) { toast('Save failed: ' + e.message); });
  }

  function build() {
    var panel = document.getElementById('p-settings');
    if (!panel) return false;
    if (document.getElementById('acc-experimental')) return true;
    var acc = h('details', { class: 'acc', id: 'acc-experimental' });
    acc.appendChild(h('summary', { text: 'Experimental' }));
    var body = h('div');
    body.appendChild(h('p', { class: 'xp-warn', role: 'note', text: 'Experimental features move the motors directly. Keep hands clear of ' +
      'the head and antennas. If a motor overloads, use Motor health on the Robot tab to clear it.' }));

    els.on = h('input', { type: 'checkbox', id: 'x_gamepad_enabled' });
    els.on.addEventListener('change', function () {
      editing = true;
      save({ gamepad_enabled: els.on.checked }, els.on.checked ? 'Gamepad mode enabled' : 'Gamepad mode disabled')
        .then(function () { editing = false; });
    });
    body.appendChild(h('div', { class: 'sw first' }, [
      h('label', { for: 'x_gamepad_enabled' }, [h('b', { text: 'Gamepad mode' }), h('br'),
        h('span', { class: 'sub', text: 'Drive the head, body and antennas with a game controller. Adds a Gamepad card to the Robot tab.' })]),
      els.on]));

    els.speed = h('input', { type: 'range', id: 'x_gamepad_speed', min: '0.25', max: '2', step: '0.05', value: '1' });
    els.out = h('output', { for: 'x_gamepad_speed', text: fmtX(1) });
    els.speed.addEventListener('input', function () { editing = true; els.out.textContent = fmtX(els.speed.value); });
    els.speed.addEventListener('change', function () {
      save({ gamepad_speed: +els.speed.value }).then(function () { editing = false; });
    });
    body.appendChild(h('div', { class: 'sp-field' }, [
      h('label', { for: 'x_gamepad_speed', text: 'Gamepad speed' }),
      h('div', { class: 'sp-range' }, [els.speed, els.out])]));

    body.appendChild(h('h3', { text: 'Controls and button mapping' }));
    els.map = h('div', { id: 'x-gamepad-map' });
    body.appendChild(els.map);

    acc.appendChild(body);
    var accs = panel.querySelectorAll('details.acc');
    var diag = accs.length ? accs[accs.length - 1] : null;
    if (diag && diag.parentNode) diag.parentNode.insertBefore(acc, diag); else panel.appendChild(acc);

    var tries = 0;
    (function mount() {
      if (window.reachyGamepad && window.reachyGamepad.mountMapping) window.reachyGamepad.mountMapping(els.map);
      else if (tries++ < 100) setTimeout(mount, 100);
    })();
    return true;
  }

  function render(s) {
    var c = s && s.config;
    if (!c || !els.on || editing) return;     // don't fight the user mid-click or mid-drag
    if (typeof c.gamepad_enabled === 'boolean') els.on.checked = c.gamepad_enabled;
    if (typeof c.gamepad_speed === 'number' && isFinite(c.gamepad_speed)) {
      els.speed.value = String(c.gamepad_speed); els.out.textContent = fmtX(c.gamepad_speed);
    }
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
