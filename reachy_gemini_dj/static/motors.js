/* Motor health card (Robot tab): shows motors reporting a hardware error (Overload, Overheating, ...) and clears
   them with a motor reboot through GET /api/motors/health and POST /api/motors/clear. Checks every 15 s while the
   Robot tab is open. Vanilla JS, no dependencies. */
(function () {
  'use strict';
  var els = {}, busy = false, timer = 0;

  function h(tag, attrs, kids) {
    var e = document.createElement(tag);
    Object.keys(attrs || {}).forEach(function (k) {
      if (k === 'class') e.className = attrs[k];
      else if (k === 'text') e.textContent = attrs[k];
      else if (k.slice(0, 2) === 'on') e.addEventListener(k.slice(2), attrs[k]);
      else e.setAttribute(k, attrs[k]);
    });
    (kids || []).forEach(function (c) { if (c) e.appendChild(typeof c === 'string' ? document.createTextNode(c) : c); });
    return e;
  }
  function toast(msg, isErr) {
    var box = document.getElementById('pn-toasts');
    if (!box) { box = h('div', { id: 'pn-toasts', class: 'pn-toasts', role: 'status', 'aria-live': 'polite' }); document.body.appendChild(box); }
    var t = h('div', { class: 'pn-toast' + (isErr ? ' err' : ''), text: msg });
    box.appendChild(t);
    setTimeout(function () { t.remove(); }, isErr ? 6000 : 2500);
  }

  function build() {
    var root = document.getElementById('robot-panel');
    if (!root) return false;
    var card = h('div', { class: 'pn-card', id: 'rp-motors' });
    card.appendChild(h('h2', { text: 'Motor health' }));
    els.status = h('div', { class: 'mh-status', role: 'status', 'aria-live': 'polite', text: 'Not checked yet.' });
    els.list = h('ul', { class: 'mh-list' });
    els.check = h('button', { type: 'button', class: 'pn-btn', text: 'Check now', onclick: check });
    els.clear = h('button', { type: 'button', class: 'pn-btn secondary', text: 'Clear motor errors', onclick: clear });
    card.appendChild(els.status);
    card.appendChild(els.list);
    card.appendChild(h('div', { class: 'gp-row mh-row' }, [els.check, els.clear]));
    card.appendChild(h('p', { class: 'pn-muted', text: 'A motor that is pushed too hard (for example an antenna held against ' +
      'its stop) shuts itself off with an "Overload" error. Clear restarts just that motor, the same as a power cycle ' +
      'would, then holds it where it is. If the error comes straight back, something is blocking it: check it by hand.' }));
    root.appendChild(card);
    var css = document.createElement('style');
    css.textContent = '.mh-status{display:flex;align-items:center;gap:8px;font-size:14px;font-weight:600;margin-bottom:12px}' +
      '.mh-list{list-style:none;margin:0 0 12px;padding:0;font-size:13.5px}.mh-list li{padding:8px 0;border-top:1px solid var(--p-border)}' +
      '.mh-list li:first-child{border-top:0}.mh-list b{display:inline-block;min-width:120px}' +
      '.mh-row{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:12px}';
    document.head.appendChild(css);
    paint(null);
    return true;
  }

  function paint(r) {
    els.list.textContent = '';
    els.status.textContent = '';
    if (!r) { els.status.appendChild(h('span', { class: 'pn-dot' })); els.status.appendChild(document.createTextNode('Not checked yet.')); }
    else if (!r.ok) {
      els.status.appendChild(h('span', { class: 'pn-dot bad' }));
      els.status.appendChild(document.createTextNode(r.error || 'Could not read the motors.'));
    } else {
      var faults = r.faults || [], motors = r.motors || {};
      var unreadable = Object.keys(motors).filter(function (n) { return motors[n].unreadable; });
      els.status.appendChild(h('span', { class: 'pn-dot ' + (faults.length ? 'bad' : unreadable.length ? '' : 'ok') }));
      els.status.appendChild(document.createTextNode(faults.length ? faults.length + ' motor' + (faults.length > 1 ? 's' : '') +
        ' reporting an error' : 'All ' + Object.keys(motors).length + ' motors OK'));
      faults.forEach(function (n) {
        els.list.appendChild(h('li', {}, [h('b', { text: motors[n].label }), (motors[n].errors || []).join(', ')]));
      });
      unreadable.forEach(function (n) {
        els.list.appendChild(h('li', { class: 'pn-muted' }, [h('b', { text: motors[n].label }), 'no reply']));
      });
    }
    var hasFault = !!(r && r.ok && (r.faults || []).length);
    els.clear.disabled = busy || !hasFault;
    els.check.disabled = busy;
  }

  function check() {
    if (busy) return Promise.resolve();
    busy = true; paint(els.last || null);
    return fetch('/api/motors/health', { cache: 'no-store' }).then(function (r) { return r.json(); })
      .then(function (r) { els.last = r; })
      .catch(function (e) { els.last = { ok: false, error: 'Could not reach the app (' + e.message + ')' }; })
      .then(function () { busy = false; paint(els.last); });
  }
  function clear() {
    if (busy) return;
    busy = true; paint(els.last || null);
    fetch('/api/motors/clear', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}' })
      .then(function (r) { return r.json(); })
      .then(function (r) {
        var done = Object.keys(r.cleared || {}).map(function (n) { return r.cleared[n].label; });
        if (r.ok) toast(done.length ? 'Cleared: ' + done.join(', ') : 'Nothing to clear');
        else toast(r.error || ('Still faulty: ' + (r.faults || []).map(function (n) { return r.cleared[n].label; }).join(', ') +
          '. Check it is not blocked, then try again or power-cycle.'), true);
      })
      .catch(function (e) { toast('Could not reach the app (' + e.message + ')', true); })
      .then(function () { busy = false; return check(); });
  }

  function visible() {
    var p = document.getElementById('p-robot');
    return !document.hidden && p && !p.classList.contains('hide');
  }
  function poll() {
    clearTimeout(timer);
    if (visible()) check();
    timer = setTimeout(poll, 15000);
  }
  function init() {
    if (!build()) return;
    var tab = document.getElementById('tab-robot');
    if (tab) tab.addEventListener('click', function () { setTimeout(poll, 50); });
    poll();
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init); else init();
})();
