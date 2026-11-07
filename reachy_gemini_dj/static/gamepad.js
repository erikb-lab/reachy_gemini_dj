/* Gamepad mode (Robot tab): drive Reachy's motors directly with a game controller.
   The controller pairs over Bluetooth (or USB) with the phone or computer showing this page; the browser's Gamepad
   API reads it, and this script streams a target pose to POST /api/drive/target. The robot walks the motors towards
   the target at a capped speed (motion.py), so a lost packet or a "centre" press never makes it jump.

   Every control can be remapped (Settings > Experimental): tap a control in the table, then press a button or move a stick or trigger.
   Analog inputs stay analog (sticks and L2/R2 triggers give proportional speed). The mapping is remembered per
   controller in this browser. Vanilla JS, no dependencies. */
(function () {
  'use strict';
  var DEAD = 0.15;                       // stick dead zone
  var TRIG_DEAD = 0.05;                  // trigger / button dead zone
  var FINE = 0.3;                        // fine mode speed factor
  var SEND_MS = 60;
  var TICK_MS = 25;                      // controller poll interval (a timer: animation frames can stall in embedded views)
  // Official limits (sent by the robot when gamepad mode starts; these are only the fallback): head pitch/roll
  // +-40, head yaw +-180, body +-160, head yaw within 65 of the body; antennas = Pollen's recorded move range.
  var LIMITS = { x: [-15, 15], y: [-15, 15], z: [-15, 20], roll: [-40, 40], pitch: [-40, 40], yaw: [-180, 180],
                 body: [-160, 160], ant_l: [-60, 194], ant_r: [-182, 60] };
  var MAX_REL_YAW = 65;
  var KEYS = ['x', 'y', 'z', 'roll', 'pitch', 'yaw', 'body', 'ant_l', 'ant_r'];
  var STORE = 'reachy-gamepad-map:';

  /* Analog actions. plus/minus are the two directions (from Reachy's point of view). keys: which pose values move,
     with a sign per key (yaw and body are + to the left, pitch is + down, so "right" and "up" are negative). */
  var ANALOG = [
    { id: 'look_h', name: 'Look', plus: 'right', minus: 'left', rate: 90, keys: { yaw: -1 } },
    { id: 'look_v', name: 'Look', plus: 'up', minus: 'down', rate: 70, keys: { pitch: -1 } },
    { id: 'body', name: 'Turn body', plus: 'right', minus: 'left', rate: 90, keys: { body: -1 } },
    { id: 'height', name: 'Head height', plus: 'up', minus: 'down', rate: 30, keys: { z: 1 } },
    { id: 'lean', name: 'Lean head', plus: 'forward', minus: 'back', rate: 30, keys: { x: 1 } },
    { id: 'tilt', name: 'Tilt head', plus: 'right', minus: 'left', rate: 60, keys: { roll: 1 } },
    { id: 'slide', name: 'Slide head', plus: 'left', minus: 'right', rate: 30, keys: { y: 1 } },
    { id: 'ant', name: 'Both antennas', plus: 'one way', minus: 'other way', rate: 90, keys: { ant_l: 1, ant_r: 1 } },
    { id: 'ant_opp', name: 'Antennas opposite', plus: 'one way', minus: 'other way', rate: 90, keys: { ant_l: 1, ant_r: -1 } },
    { id: 'ant_l', name: 'Left antenna', plus: 'one way', minus: 'other way', rate: 90, keys: { ant_l: 1 } },
    { id: 'ant_r', name: 'Right antenna', plus: 'one way', minus: 'other way', rate: 90, keys: { ant_r: 1 } }
  ];
  var BUTTONS = [
    { id: 'toggle', name: 'Gamepad mode on / off' },
    { id: 'save', name: 'Save pose' },
    { id: 'centre', name: 'Centre everything' },
    { id: 'straight', name: 'Head straight' },
    { id: 'fine', name: 'Fine control on / off' }
  ];
  /* Inputs: {a: axis, d: +1/-1} centred stick axis; {t: axis} trigger reported as an axis resting at -1;
     {b: button} button (its value is analog for L2/R2 in the standard layout). Defaults = standard layout
     (DualShock 4 / DualSense / Xbox in Chrome, Edge, Safari): sticks look and turn, L2/R2 lean, L1/R1 tilt,
     D-pad antennas, Cross save, Circle centre, Square straight, Triangle fine, Options on/off. */
  var DEFAULTS = {
    look_h: { plus: { a: 0, d: 1 } }, look_v: { plus: { a: 1, d: -1 } },
    body: { plus: { a: 2, d: 1 } }, height: { plus: { a: 3, d: -1 } },
    lean: { plus: { b: 7 }, minus: { b: 6 } }, tilt: { plus: { b: 5 }, minus: { b: 4 } },
    ant: { plus: { b: 12 }, minus: { b: 13 } }, ant_opp: { plus: { b: 15 }, minus: { b: 14 } },
    toggle: { b: 9 }, save: { b: 0 }, centre: { b: 1 }, straight: { b: 2 }, fine: { b: 3 }
  };
  var NAMES = {
    ps: ['✕ Cross', '○ Circle', '□ Square', '△ Triangle', 'L1', 'R1', 'L2', 'R2', 'Share', 'Options', 'L3', 'R3',
         'D-pad ↑', 'D-pad ↓', 'D-pad ←', 'D-pad →', 'PS', 'Touchpad'],
    xbox: ['A', 'B', 'X', 'Y', 'LB', 'RB', 'LT', 'RT', 'View', 'Menu', 'LS', 'RS',
           'D-pad ↑', 'D-pad ↓', 'D-pad ←', 'D-pad →', 'Xbox'],
    nin: ['B', 'A', 'Y', 'X', 'L', 'R', 'ZL', 'ZR', '−', '+', 'LS', 'RS', 'D-pad ↑', 'D-pad ↓', 'D-pad ←', 'D-pad →', 'Home', 'Capture']
  };
  var STICKS = [['Left stick', '→', '←'], ['Left stick', '↓', '↑'], ['Right stick', '→', '←'], ['Right stick', '↓', '↑']];

  var on = false, busy = false, fine = false, target = null, lastSent = '', lastSendT = 0, lastFrame = 0;
  var prevButtons = [], padIndex = null, timer = 0, map = null, mapFor = '';
  var enabled = false, speed = 1;       // from state.config (Settings > Experimental)
  var capture = null;                    // {kind:'analog'|'button', id, slot, base:{axes, buttons}, until}
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
  function dz(v, d) { if (Math.abs(v) < d) return 0; return (v - Math.sign(v) * d) / (1 - d); }
  function copy(o) { return JSON.parse(JSON.stringify(o)); }

  function pad() {
    var list = navigator.getGamepads ? navigator.getGamepads() : [];
    if (padIndex != null && list[padIndex]) return list[padIndex];
    for (var i = 0; i < list.length; i++) if (list[i] && list[i].connected) { padIndex = i; return list[i]; }
    return null;
  }
  function family(p) {
    if (!p || p.mapping !== 'standard') return '';
    var id = p.id || '';
    if (/054c|dualshock|dualsense|playstation|wireless controller/i.test(id)) return 'ps';
    if (/057e|nintendo|pro controller|joy-con/i.test(id)) return 'nin';
    return 'xbox';
  }
  function shortId(p) { return p ? p.id.replace(/\s*\(.*\)\s*$/, '').slice(0, 60) : ''; }

  /* ---------------- mapping storage ---------------- */
  function loadMap(p) {
    var key = p ? shortId(p) : '';
    if (map && mapFor === key) return map;
    mapFor = key; map = copy(DEFAULTS);
    try {
      var raw = localStorage.getItem(STORE + key);
      if (raw) { var saved = JSON.parse(raw); if (saved && typeof saved === 'object') map = Object.assign(copy(DEFAULTS), saved); }
    } catch (e) { /* storage blocked: defaults */ }
    return map;
  }
  function saveMap() {
    try { localStorage.setItem(STORE + mapFor, JSON.stringify(map)); } catch (e) { /* not remembered */ }
  }

  /* ---------------- reading inputs ---------------- */
  function inputValue(inp, p) {
    if (!inp || !p) return 0;
    var a = p.axes || [], b = p.buttons || [];
    if (inp.a != null) return dz((a[inp.a] || 0) * (inp.d || 1), DEAD);                 // -1..1
    if (inp.t != null) return Math.max(0, dz(((a[inp.t] == null ? -1 : a[inp.t]) + 1) / 2, TRIG_DEAD));  // 0..1
    if (inp.b != null) { var x = b[inp.b]; return x ? Math.max(0, dz(x.value || (x.pressed ? 1 : 0), TRIG_DEAD)) : 0; }
    return 0;
  }
  function analog(id, p) {
    var m = map[id] || {};
    return Math.max(-1, Math.min(1, inputValue(m.plus, p) - inputValue(m.minus, p)));
  }
  function buttonDown(id, p) {
    return inputValue(map[id], p) > 0.5;
  }
  function label(inp, p) {
    if (!inp) return '—';
    var fam = family(p);
    if (inp.a != null) {
      var s = fam && STICKS[inp.a];
      return s ? s[0] + ' ' + (inp.d > 0 ? s[1] : s[2]) : 'Axis ' + inp.a + (inp.d > 0 ? ' +' : ' −');
    }
    if (inp.t != null) return 'Axis ' + inp.t + ' (trigger)';
    if (inp.b != null) return (fam && NAMES[fam][inp.b]) || 'Button ' + inp.b;
    return '—';
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

    card.appendChild(h('p', { class: 'pn-muted gp-hint' }, ['Mapping and speed: ',
      h('button', { type: 'button', class: 'gp-link', text: 'Settings > Experimental', onclick: openExperimental })]));
    els.card = card;
    card.classList.toggle('hide', !enabled);

    var poses = document.getElementById('rp-poses');
    if (poses && poses.parentNode === root) root.insertBefore(card, poses); else root.appendChild(card);

    var css = document.createElement('style');
    css.textContent =
      '.gp-pad{display:flex;align-items:center;gap:8px;margin:0 0 12px;font-size:13.5px;color:var(--p-muted)}' +
      '.gp-pad.ok{color:var(--p-fg);font-weight:600}' +
      '.gp-row{display:flex;flex-wrap:wrap;gap:8px}' +
      '#rp-gamepad .pn-btn[aria-pressed=true]{background:var(--p-accent);border-color:var(--p-accent);color:var(--accent-ink,#fff)}' +
      '.gp-read{display:grid;grid-template-columns:repeat(auto-fill,minmax(108px,1fr));gap:0 16px;margin:16px 0 0;font-size:13.5px}' +
      '.gp-read div{display:flex;justify-content:space-between;align-items:baseline;gap:8px;padding:6px 0;border-top:1px solid var(--p-border)}' +
      '.gp-read dt{color:var(--p-muted)}.gp-read dd{margin:0;font-weight:600;color:var(--p-fg);font-variant-numeric:tabular-nums}' +
      '.gp-hint{margin:14px 0 0}.gp-link{background:none;border:0;padding:0;font:inherit;color:var(--p-accent);text-decoration:underline;cursor:pointer}' +
      '.gp-map{display:grid;gap:6px;margin:8px 0}' +
      '.gp-mrow{display:grid;grid-template-columns:minmax(110px,1fr) minmax(0,1.4fr);gap:6px 10px;align-items:center;' +
        'padding:6px 0;border-top:1px solid var(--p-border)}' +
      '.gp-mrow:first-child{border-top:0}.gp-mname{font-weight:600}' +
      '.gp-slots{display:flex;flex-wrap:wrap;gap:6px}' +
      '.gp-chip{min-height:40px;padding:4px 10px;border-radius:10px;border:1px solid var(--p-border);background:var(--p-bg);' +
        'color:var(--p-fg);font:inherit;font-size:.88rem;cursor:pointer;text-align:left}' +
      '.gp-chip small{display:block;color:var(--p-muted);font-size:.72rem}' +
      '.gp-chip.wait{border-color:var(--p-accent);box-shadow:0 0 0 2px var(--p-accent)}' +
      '.gp-cap{display:flex;flex-wrap:wrap;gap:8px;align-items:center;padding:10px;border-radius:10px;' +
        'background:var(--p-bg);border:1px solid var(--p-accent);margin:8px 0;font-weight:600}' +
      '.gp-cap .pn-btn{min-height:36px}' +
      '@media (max-width:480px){.gp-mrow{grid-template-columns:1fr}}';
    document.head.appendChild(css);
    paint();
    renderMap();
    return true;
  }

  /* Builds the mapping UI (capture bar, remap table, reset, notes) into el. Called by experimental.js. */
  function mountMapping(el) {
    var box = h('div', { class: 'gp-mapbox' });
    els.capBar = h('div', { class: 'gp-cap hide', role: 'status', 'aria-live': 'polite' });
    box.appendChild(els.capBar);
    els.table = h('div', { class: 'gp-map' });
    box.appendChild(els.table);
    box.appendChild(h('div', { class: 'gp-row', style: 'margin-top:10px' }, [
      h('button', { type: 'button', class: 'pn-btn', text: 'Reset to defaults', onclick: function () {
        if (!window.confirm('Reset all controller mappings to the defaults?')) return;
        map = copy(DEFAULTS); saveMap(); renderMap(); toast('Mapping reset');
      } })
    ]));
    els.mapNote = h('p', { class: 'pn-muted' });
    box.appendChild(els.mapNote);
    box.appendChild(h('p', { class: 'pn-muted', text: 'Tap a control, then press a button or move a stick or trigger ' +
      'in the direction you want. Sticks and L2/R2 are analog: push further to move faster. Directions are from ' +
      'Reachy\'s point of view. The mapping is remembered in this browser for this controller.' }));
    box.appendChild(h('p', { class: 'pn-muted', text: 'Pair the controller with this phone or computer in its ' +
      'own Bluetooth settings (DualShock 4: hold Share + PS until the light flashes), then press any button so the ' +
      'browser sees it. Voice moves, dances and emotions pause while gamepad mode is on; it turns itself off after ' +
      '3 minutes without input and Reachy keeps the pose.' }));
    el.textContent = '';
    el.appendChild(box);
    renderMap();
    if (pad()) loop();
  }
  function openExperimental() {
    var tab = document.getElementById('tab-settings');
    if (tab) tab.click();
    var acc = document.getElementById('acc-experimental');
    if (acc) { acc.open = true; if (acc.scrollIntoView) acc.scrollIntoView({ behavior: 'smooth', block: 'start' }); }
  }
  window.reachyGamepad = { mountMapping: mountMapping };

  function paint() {
    if (!els.btn) return;
    var p = pad();
    els.pad.textContent = '';
    els.pad.className = 'gp-pad' + (p ? ' ok' : '');
    els.pad.appendChild(h('span', { class: 'pn-dot ' + (p ? 'ok' : '') }));
    var fam = family(p);
    els.pad.appendChild(document.createTextNode(p ? 'Controller: ' + shortId(p) +
      (fam === 'ps' ? ' (PlayStation buttons)' : fam === 'nin' ? ' (Nintendo buttons)' : '')
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

  function chip(kind, id, slot, top, inp, p) {
    var waiting = capture && capture.id === id && capture.slot === slot;
    return h('button', { type: 'button', class: 'gp-chip' + (waiting ? ' wait' : ''),
      'aria-label': 'Change: ' + top + ', now ' + label(inp, p),
      onclick: function () { startCapture(kind, id, slot, top); } },
      [h('small', { text: top }), waiting ? 'Press or move…' : label(inp, p)]);
  }
  function renderMap() {
    if (!els.table) return;
    var p = pad();
    loadMap(p);
    els.table.textContent = '';
    ANALOG.forEach(function (a) {
      var m = map[a.id] || {};
      var slots = [chip('analog', a.id, 'plus', a.plus, m.plus, p)];
      var plusIsStick = m.plus && m.plus.a != null;
      if (!plusIsStick || m.minus) slots.push(chip('analog', a.id, 'minus', a.minus, m.minus, p));
      els.table.appendChild(h('div', { class: 'gp-mrow' }, [h('span', { class: 'gp-mname', text: a.name }),
        h('div', { class: 'gp-slots' }, slots)]));
    });
    BUTTONS.forEach(function (b) {
      els.table.appendChild(h('div', { class: 'gp-mrow' }, [h('span', { class: 'gp-mname', text: b.name }),
        h('div', { class: 'gp-slots' }, [chip('button', b.id, '', 'button', map[b.id], p)])]));
    });
    els.mapNote.textContent = p && p.mapping !== 'standard'
      ? 'This browser reports a non-standard layout for this controller, so the defaults may be wrong: tap each control and set it.'
      : '';
    els.mapNote.className = 'pn-muted' + (els.mapNote.textContent ? ' pn-err' : '');
  }

  /* ---------------- remapping ---------------- */
  function startCapture(kind, id, slot, top) {
    var p = pad();
    if (!p) { toast('Connect a controller first: press any button on it', true); return; }
    if (capture && capture.id === id && capture.slot === slot) { endCapture(); return; }
    capture = { kind: kind, id: id, slot: slot, until: Date.now() + 10000,
      base: { axes: (p.axes || []).slice(), buttons: (p.buttons || []).map(function (b) { return b ? b.value || (b.pressed ? 1 : 0) : 0; }) } };
    var act = (kind === 'analog' ? ANALOG : BUTTONS).filter(function (x) { return x.id === id; })[0];
    els.capBar.textContent = '';
    els.capBar.appendChild(document.createTextNode('Set ' + act.name + (kind === 'analog' ? ' (' + top + ')' : '') +
      ': press a button or move a stick or trigger…'));
    els.capBar.appendChild(h('button', { type: 'button', class: 'pn-btn', text: 'Clear', onclick: function () {
      assign(null); } }));
    els.capBar.appendChild(h('button', { type: 'button', class: 'pn-btn', text: 'Cancel', onclick: endCapture }));
    els.capBar.classList.remove('hide');
    renderMap();
    loop();
  }
  function endCapture() {
    capture = null;
    if (els.capBar) els.capBar.classList.add('hide');
    renderMap();
  }
  function assign(inp) {
    if (!capture) return;
    var old = map[capture.id];
    if (capture.kind === 'analog') {
      var m = map[capture.id] = Object.assign({}, map[capture.id] || {});
      if (inp) m[capture.slot] = inp; else delete m[capture.slot];
      if (inp && inp.a != null && capture.slot === 'plus' && m.minus && m.minus.a === inp.a) delete m.minus;
    } else if (inp && inp.a != null) {
      map[capture.id] = inp;               // a stick direction used as a button
    } else {
      if (inp) map[capture.id] = inp; else delete map[capture.id];
    }
    // the same physical button must not also fire another button action: that action gets this one's old button
    if (inp && capture.kind === 'button') BUTTONS.forEach(function (b) {
      var o = map[b.id];
      if (b.id !== capture.id && o && JSON.stringify(o) === JSON.stringify(inp)) {
        if (old) map[b.id] = old; else delete map[b.id];
      }
    });
    saveMap();
    toast(inp ? 'Mapped to ' + label(inp, pad()) : 'Cleared');
    endCapture();
  }
  function detect(p) {
    var a = p.axes || [], b = p.buttons || [], base = capture.base;
    for (var i = 0; i < b.length; i++) {
      var v = b[i] ? b[i].value || (b[i].pressed ? 1 : 0) : 0;
      if (v > 0.6 && (base.buttons[i] || 0) < 0.3) return { b: i };
    }
    for (var j = 0; j < a.length; j++) {
      var v0 = base.axes[j] == null ? 0 : base.axes[j], v1 = a[j] || 0;
      if (Math.abs(v1 - v0) < 0.6) continue;
      if (v0 < -0.8) return { t: j };     // an axis resting at -1 is a trigger
      return { a: j, d: v1 > 0 ? 1 : -1 };
    }
    return null;
  }

  /* ---------------- mode ---------------- */
  function setMode(want) {
    if (busy) return;
    busy = true; paint();
    post('/api/drive', { on: want }).then(function (r) {
      if (r && r.ok) {
        on = !!r.manual;
        if (r.limits) LIMITS = r.limits;
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
  function frame(now) {
    timer = 0;
    var dt = lastFrame ? Math.min(0.1, (now - lastFrame) / 1000) : 0;
    lastFrame = now;
    var p = pad();
    if (p) {
      loadMap(p);
      if (capture) {
        if (Date.now() > capture.until) { toast('Nothing pressed; mapping unchanged', true); endCapture(); }
        else { var got = detect(p); if (got) assign(got); }
        prevButtons = BUTTONS.map(function (b) { return true; });   // swallow the press that was just mapped
      } else {
        var down = {};
        BUTTONS.forEach(function (b, i) {
          down[b.id] = buttonDown(b.id, p);
          if (down[b.id] && !prevButtons[i]) fire(b.id);
        });
        prevButtons = BUTTONS.map(function (b) { return down[b.id]; });
        if (on && target) {
          var f = (fine ? FINE : 1) * speed;
          ANALOG.forEach(function (a) {
            var v = analog(a.id, p);
            if (!v) return;
            Object.keys(a.keys).forEach(function (k) { target[k] = clamp(k, target[k] + v * a.keys[k] * a.rate * f * dt); });
          });
          target.yaw = Math.min(target.body + MAX_REL_YAW, Math.max(target.body - MAX_REL_YAW, target.yaw));
        }
      }
    }
    if (on && target) { send(now); paintRead(); }
    if (on || p || capture) timer = setTimeout(tick, TICK_MS);
    else lastFrame = 0;
  }
  function fire(id) {
    if (id === 'toggle') { if (enabled || on) setMode(!on); return; }
    if (!on || !target) return;
    if (id === 'save') savePose();
    else if (id === 'centre') centre();
    else if (id === 'straight') headStraight();
    else if (id === 'fine') setFine(!fine);
  }
  function tick() { frame(performance.now()); }
  function loop() { if (!timer) { lastFrame = 0; timer = setTimeout(tick, 0); } }

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
    padIndex = e.gamepad.index;
    prevButtons = BUTTONS.map(function () { return true; });   // the press that woke it up does nothing
    toast('Controller connected'); map = null; paint(); renderMap(); loop();
  });
  window.addEventListener('gamepaddisconnected', function (e) {
    if (padIndex === e.gamepad.index) padIndex = null;
    if (capture) endCapture();
    toast('Controller disconnected', true); paint(); renderMap();
  });

  function render(s) {
    var cfg = s && s.config;
    if (cfg && typeof cfg.gamepad_enabled === 'boolean') {
      enabled = cfg.gamepad_enabled;
      if (els.card) els.card.classList.toggle('hide', !enabled);
      if (!enabled && on) { on = false; paint(); }
    }
    if (cfg && typeof cfg.gamepad_speed === 'number' && isFinite(cfg.gamepad_speed)) {
      speed = Math.min(2, Math.max(0.25, cfg.gamepad_speed));
    }
    var m = (s && s.motion) || {};
    if (typeof m.manual === 'boolean' && m.manual !== on && !busy && (enabled || !m.manual)) {
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
