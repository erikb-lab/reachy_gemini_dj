/* Gamepad mode (Robot tab): drive Reachy's motors directly with a game controller.
   The controller pairs over Bluetooth (or USB) with the phone or computer showing this page; the browser's Gamepad
   API reads it, and this script streams a target pose to POST /api/drive/target. The robot walks the motors towards
   the target at a capped speed (motion.py), so a lost packet or a "centre" press never makes it jump.
   Vanilla JS, no dependencies. */
(function () {
  'use strict';
  var DEAD = 0.15;                       // stick dead zone
  var RATE = { yaw: 90, pitch: 70, roll: 60, z: 30, x: 30, body: 90, ant: 180 };   // per second at full deflection
  var FINE = 0.3;                        // fine mode speed factor
  var SEND_MS = 60;
  var TICK_MS = 25;                     // controller poll interval (a timer: animation frames can stall in embedded views)
  var LIMITS = { x: [-15, 15], y: [-15, 15], z: [-15, 20], roll: [-25, 25], pitch: [-30, 30], yaw: [-60, 60],
                 body: [-150, 150], ant_l: [-120, 120], ant_r: [-120, 120] };
  var KEYS = ['x', 'y', 'z', 'roll', 'pitch', 'yaw', 'body', 'ant_l', 'ant_r'];

  var on = false, busy = false, fine = false, target = null, lastSent = '', lastSendT = 0, lastFrame = 0;
  var prevButtons = [], padIndex = null, raf = 0;
  var els = {};

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
    setTimeout(function () { t.remove(); }, isErr ? 5000 : 2500);
  }
  function post(path, body) {
    return fetch(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })
      .then(function (r) { if (!r.ok) throw new Error('HTTP ' + r.status); return r.json(); });
  }
  function clamp(k, v) { var l = LIMITS[k]; return Math.min(l[1], Math.max(l[0], v)); }
  function dz(v) { if (Math.abs(v) < DEAD) return 0; return (v - Math.sign(v) * DEAD) / (1 - DEAD); }

  function pad() {
    var list = navigator.getGamepads ? navigator.getGamepads() : [];
    if (padIndex != null && list[padIndex]) return list[padIndex];
    for (var i = 0; i < list.length; i++) if (list[i] && list[i].connected) { padIndex = i; return list[i]; }
    return null;
  }

  /* ---------------- UI ---------------- */
  function build() {
    var root = document.getElementById('robot-panel');
    if (!root) return false;
    var card = h('div', { class: 'pn-card', id: 'rp-gamepad' });
    card.appendChild(h('h2', { text: 'Gamepad' }));
    els.pad = h('div', { class: 'gp-pad', role: 'status' });
    card.appendChild(els.pad);
    els.btn = h('button', { type: 'button', class: 'pn-btn primary', 'aria-pressed': 'false', onclick: function () { setMode(!on); } });
    els.fine = h('button', { type: 'button', class: 'pn-btn', 'aria-pressed': 'false', onclick: function () { setFine(!fine); } });
    els.center = h('button', { type: 'button', class: 'pn-btn', text: 'Centre', onclick: centre });
    card.appendChild(h('div', { class: 'gp-row' }, [els.btn, els.fine, els.center]));
    els.read = h('dl', { class: 'gp-read', 'aria-live': 'off' });
    card.appendChild(els.read);
    card.appendChild(h('details', { class: 'gp-help' }, [
      h('summary', { text: 'Controls' }),
      h('ul', {}, [
        h('li', { text: 'Left stick: look left/right and up/down' }),
        h('li', { text: 'Right stick: turn the body, raise/lower the head' }),
        h('li', { text: 'LB / RB: tilt the head' }),
        h('li', { text: 'LT / RT: lean the head back / forward' }),
        h('li', { text: 'D-pad up/down: both antennas; left/right: antennas opposite ways' }),
        h('li', { text: 'A: save pose · B: centre everything · X: head straight · Y: fine / fast' }),
        h('li', { text: 'Start: gamepad mode on / off' })
      ]),
      h('p', { class: 'pn-muted', text: 'Pair the controller (Xbox, PlayStation, Switch Pro, 8BitDo…) with this phone or computer in its ' +
        'own Bluetooth settings, then press any button on it so the browser can see it. Voice moves, dances and emotions ' +
        'pause while gamepad mode is on; it turns itself off after 3 minutes without input and Reachy keeps the pose.' })
    ]));
    var poses = document.getElementById('rp-poses');
    if (poses && poses.parentNode === root) root.insertBefore(card, poses); else root.appendChild(card);

    var css = document.createElement('style');
    css.textContent =
      '.gp-pad{display:flex;align-items:center;gap:8px;margin:0 0 12px;color:var(--p-muted)}' +
      '.gp-pad.ok{color:var(--p-fg);font-weight:600}' +
      '.gp-row{display:flex;flex-wrap:wrap;gap:8px}.gp-row .pn-btn{text-transform:none}' +
      '#rp-gamepad .pn-btn[aria-pressed=true]{background:var(--p-accent);border-color:var(--p-accent);color:#fff}' +
      '.gp-read{display:grid;grid-template-columns:repeat(auto-fill,minmax(92px,1fr));gap:8px;margin:14px 0 4px}' +
      '.gp-read div{background:var(--p-bg);border:1px solid var(--p-border);border-radius:10px;padding:6px 10px}' +
      '.gp-read dt{color:var(--p-muted);font-size:.78rem}.gp-read dd{margin:0;font-weight:700;font-variant-numeric:tabular-nums}' +
      '.gp-help{margin-top:12px}.gp-help summary{cursor:pointer;font-weight:600;min-height:32px}' +
      '.gp-help ul{margin:6px 0 8px;padding-left:20px}.gp-help li{margin:3px 0}';
    document.head.appendChild(css);
    paint();
    return true;
  }

  function paint() {
    if (!els.btn) return;
    var p = pad();
    els.pad.textContent = '';
    els.pad.className = 'gp-pad' + (p ? ' ok' : '');
    els.pad.appendChild(h('span', { class: 'pn-dot ' + (p ? 'ok' : '') }));
    els.pad.appendChild(document.createTextNode(p ? 'Controller: ' + p.id.replace(/\s*\(.*\)\s*$/, '').slice(0, 60)
      : 'No controller yet. Pair one with this device, then press any button on it.'));
    els.btn.textContent = on ? 'Stop gamepad mode' : 'Start gamepad mode';
    els.btn.setAttribute('aria-pressed', on ? 'true' : 'false');
    els.btn.disabled = busy;
    els.fine.textContent = fine ? 'Fine: on' : 'Fine: off';
    els.fine.setAttribute('aria-pressed', fine ? 'true' : 'false');
    els.center.disabled = !on;
    paintRead();
  }
  function paintRead() {
    if (!els.read) return;
    var t = target || {};
    var rows = [['Look', fmt(t.yaw, '°')], ['Up/down', fmt(t.pitch, '°')], ['Tilt', fmt(t.roll, '°')],
                ['Height', fmt(t.z, ' mm')], ['Lean', fmt(t.x, ' mm')], ['Body', fmt(t.body, '°')],
                ['Antenna L', fmt(t.ant_l, '°')], ['Antenna R', fmt(t.ant_r, '°')]];
    var sig = JSON.stringify(rows);
    if (els.read.dataset.sig === sig) return;
    els.read.dataset.sig = sig;
    els.read.textContent = '';
    rows.forEach(function (r) { els.read.appendChild(h('div', {}, [h('dt', { text: r[0] }), h('dd', { text: r[1] })])); });
  }
  function fmt(v, unit) { return typeof v === 'number' ? Math.round(v) + unit : '–'; }

  /* ---------------- mode ---------------- */
  function setMode(want) {
    if (busy) return;
    busy = true; paint();
    post('/api/drive', { on: want }).then(function (r) {
      if (r && r.ok) {
        on = !!r.manual;
        if (on) { target = r.target || target; lastSent = JSON.stringify(target); toast('Gamepad mode on'); loop(); }
        else toast('Gamepad mode off: Reachy holds the pose');
      } else toast((r && r.error) || 'Could not change gamepad mode', true);
    }).catch(function (e) { toast('Could not reach the app (' + e.message + ')', true); })
      .then(function () { busy = false; paint(); });
  }
  function setFine(v) { fine = v; paint(); }
  function centre() {
    if (!on || !target) return;
    KEYS.forEach(function (k) { target[k] = 0; });
    paintRead();
  }
  function headStraight() {
    if (!on || !target) return;
    ['x', 'y', 'z', 'roll', 'pitch', 'yaw'].forEach(function (k) { target[k] = 0; });
    paintRead();
  }
  function savePose() {
    var name = window.prompt('Name for this pose?');
    if (!name || !name.trim()) return;
    post('/api/poses/save', { name: name.trim() }).then(function (r) {
      if (r && r.ok) toast('Saved pose "' + r.name + '"'); else toast((r && r.error) || 'Save failed', true);
    }).catch(function (e) { toast('Save failed: ' + e.message, true); });
  }

  /* ---------------- input loop ---------------- */
  function pressed(b, i) { return !!(b[i] && (b[i].pressed || b[i].value > 0.5)); }
  function edge(b, i) { return pressed(b, i) && !prevButtons[i]; }

  function frame(now) {
    raf = 0;
    var dt = lastFrame ? Math.min(0.1, (now - lastFrame) / 1000) : 0;
    lastFrame = now;
    var p = pad();
    if (p) {
      var b = p.buttons || [], a = p.axes || [];
      if (edge(b, 9)) setMode(!on);
      if (on && target) {
        if (edge(b, 0)) savePose();
        if (edge(b, 1)) centre();
        if (edge(b, 2)) headStraight();
        if (edge(b, 3)) setFine(!fine);
        var f = fine ? FINE : 1, d = dt * f;
        var lx = dz(a[0] || 0), ly = dz(a[1] || 0), rx = dz(a[2] || 0), ry = dz(a[3] || 0);
        var lt = b[6] ? b[6].value || 0 : 0, rt = b[7] ? b[7].value || 0 : 0;
        target.yaw = clamp('yaw', target.yaw - lx * RATE.yaw * d);
        target.pitch = clamp('pitch', target.pitch + ly * RATE.pitch * d);
        target.body = clamp('body', target.body - rx * RATE.body * d);
        target.z = clamp('z', target.z - ry * RATE.z * d);
        target.x = clamp('x', target.x + (rt - lt) * RATE.x * d);
        if (pressed(b, 4)) target.roll = clamp('roll', target.roll - RATE.roll * d);
        if (pressed(b, 5)) target.roll = clamp('roll', target.roll + RATE.roll * d);
        var both = (pressed(b, 12) ? 1 : 0) - (pressed(b, 13) ? 1 : 0);
        var opp = (pressed(b, 15) ? 1 : 0) - (pressed(b, 14) ? 1 : 0);
        if (both || opp) {
          target.ant_l = clamp('ant_l', target.ant_l + (both + opp) * RATE.ant * d);
          target.ant_r = clamp('ant_r', target.ant_r + (both - opp) * RATE.ant * d);
        }
      }
      prevButtons = b.map(function (x) { return !!(x && (x.pressed || x.value > 0.5)); });
    }
    if (on && target) {
      send(now);
      paintRead();
    }
    if (on || p) raf = setTimeout(tick, TICK_MS);
    else lastFrame = 0;
  }
  function tick() { frame(performance.now()); }
  function loop() { if (!raf) { lastFrame = 0; raf = setTimeout(tick, 0); } }

  function send(now) {
    if (now - lastSendT < SEND_MS) return;
    var body = {};
    KEYS.forEach(function (k) { body[k] = Math.round(target[k] * 10) / 10; });
    var sig = JSON.stringify(body);
    if (sig === lastSent) return;
    lastSent = sig; lastSendT = now;
    post('/api/drive/target', body).then(function (r) {
      if (r && r.ok === false && on) { on = false; toast(r.error || 'Gamepad mode stopped', true); paint(); }
    }).catch(function () { /* next change retries */ });
  }

  /* ---------------- wiring ---------------- */
  window.addEventListener('gamepadconnected', function (e) {
    padIndex = e.gamepad.index; prevButtons = (e.gamepad.buttons || []).map(function () { return true; });
    toast('Controller connected'); paint(); loop();
  });
  window.addEventListener('gamepaddisconnected', function (e) {
    if (padIndex === e.gamepad.index) padIndex = null;
    toast('Controller disconnected', true); paint();
  });

  function render(s) {
    var m = (s && s.motion) || {};
    if (typeof m.manual === 'boolean' && m.manual !== on && !busy) {
      on = m.manual;
      if (on && !target) {           // turned on elsewhere: fetch where the robot is
        post('/api/drive', { on: true }).then(function (r) { if (r && r.ok) { target = r.target; loop(); paint(); } }).catch(function () {});
      }
      paint();
    }
  }
  function init() {
    if (!build()) return;
    var tries = 0;
    (function attach() {
      if (window.reachyApi && typeof window.reachyApi.subscribe === 'function') window.reachyApi.subscribe(render);
      else if (tries++ < 100) setTimeout(attach, 100);
    })();
    if (pad()) loop();
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init); else init();
})();
