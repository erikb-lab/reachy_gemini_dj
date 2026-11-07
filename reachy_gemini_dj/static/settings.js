/* Settings panel: mounts into #settings-panel. Vanilla JS, no deps. */
(function () {
  'use strict';
  var root, fields = {}, pending = {}, timers = {}, roBox = {}, keyState, voiceSig = '';
  var DEBOUNCE = 600;

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

  // Save one key (debounced). Marks field pending so state polls don't overwrite typing.
  function save(key, value, delay, label) {
    pending[key] = true;
    clearTimeout(timers[key]);
    timers[key] = setTimeout(function () {
      var b = {}; b[key] = value();
      post('/api/config', b).then(function (r) {
        if (r && r.ok === false) throw new Error(r.error || 'rejected');
        toast((label || 'Setting') + ' saved');
      }).catch(function (e) { toast((label || 'Setting') + ' not saved: ' + e.message, true); })
        .then(function () { delete pending[key]; });
    }, delay);
  }

  var FMT = {
    pct: function (v) { return Math.round(v * 100) + '%'; },
    x: function (v) { return (+v).toFixed(2) + '×'; },
    n1: function (v) { return (+v).toFixed(1); }
  };

  function field(id, label, hint, control) {
    var wrap = h('div', { class: 'sp-field' }, [
      h('label', { for: id }, [label, hint ? h('span', { class: 'hint', text: hint }) : null]), control]);
    return wrap;
  }
  function textField(key, label, hint, opts) {
    opts = opts || {};
    var inp = h('input', { class: 'sp-input', id: 'sp-' + key, type: opts.type || 'text', autocomplete: 'off',
      spellcheck: 'false', placeholder: opts.placeholder || '' });
    inp.addEventListener('input', function () { save(key, function () { return inp.value.trim(); }, DEBOUNCE, label); });
    fields[key] = { el: inp, set: function (v) { inp.value = v == null ? '' : v; } };
    return field('sp-' + key, label, hint, inp);
  }
  function rangeField(key, label, hint, min, max, step, fmt) {
    var inp = h('input', { type: 'range', id: 'sp-' + key, min: min, max: max, step: step });
    var out = h('output', { for: 'sp-' + key });
    inp.addEventListener('input', function () {
      out.textContent = fmt(+inp.value);
      save(key, function () { return +inp.value; }, DEBOUNCE, label);
    });
    fields[key] = { el: inp, set: function (v) { inp.value = v; out.textContent = fmt(+v); } };
    return field('sp-' + key, label, hint, h('div', { class: 'sp-range' }, [inp, out]));
  }
  function switchRow(key, label, hint, first) {
    var inp = h('input', { type: 'checkbox', id: 'sp-' + key });
    inp.addEventListener('change', function () { save(key, function () { return inp.checked; }, 150, label); });
    fields[key] = { el: inp, set: function (v) { inp.checked = !!v; } };
    return h('label', { class: 'pn-sw' + (first ? ' first' : ''), for: 'sp-' + key }, [
      h('span', { class: 'txt' }, [h('b', { text: label }), h('span', { text: hint })]), inp]);
  }

  function build() {
    root.textContent = '';

    // --- Gemini ---
    var c1 = h('div', { class: 'pn-card' }, [h('h2', { text: 'Gemini' })]);
    var keyIn = h('input', { class: 'sp-input', id: 'sp-api_key', type: 'password', autocomplete: 'new-password',
      placeholder: 'Paste a new key to replace the stored one', spellcheck: 'false' });
    var keyBtn = h('button', { type: 'button', class: 'pn-btn primary', text: 'Save key' });
    keyState = h('span', { class: 'hint', text: 'checking…' });
    function saveKey() {
      var v = keyIn.value.trim();
      if (!v) { toast('Paste a key first', true); return; }
      keyBtn.disabled = true;
      post('/api/config', { api_key: v }).then(function (r) {
        if (r && r.ok === false) throw new Error(r.error || 'rejected');
        keyIn.value = ''; toast('API key saved');
      }).catch(function (e) { toast('Key not saved: ' + e.message, true); })
        .then(function () { keyBtn.disabled = false; });
    }
    keyBtn.addEventListener('click', saveKey);
    keyIn.addEventListener('keydown', function (e) { if (e.key === 'Enter') saveKey(); });
    c1.appendChild(h('div', { class: 'sp-field' }, [
      h('label', { for: 'sp-api_key' }, ['Gemini API key ', keyState]),
      h('div', { class: 'sp-row' }, [keyIn, keyBtn]),
      h('span', { class: 'hint', text: 'Stored locally on the robot and never shown again.' })]));
    c1.appendChild(textField('model', 'Model', 'Gemini live model name', { placeholder: 'gemini-2.5-flash-native-audio-latest' }));
    var voice = h('select', { id: 'sp-voice' });
    voice.addEventListener('change', function () { save('voice', function () { return voice.value; }, 150, 'Voice'); });
    fields.voice = { el: voice, set: function (v) { voice.value = v; } };
    c1.appendChild(field('sp-voice', 'Voice', 'How Reachy sounds', voice));
    var ta = h('textarea', { id: 'sp-extra_instructions', placeholder: 'e.g. Call me Sam. I like lo-fi when I work.' });
    ta.addEventListener('input', function () { save('extra_instructions', function () { return ta.value; }, DEBOUNCE + 400, 'Instructions'); });
    fields.extra_instructions = { el: ta, set: function (v) { ta.value = v || ''; } };
    c1.appendChild(field('sp-extra_instructions', 'Extra instructions', 'Anything Reachy should know about you (optional)', ta));
    root.appendChild(c1);

    // --- Sound ---
    var c2 = h('div', { class: 'pn-card' }, [h('h2', { text: 'Sound & interaction' })]);
    c2.appendChild(rangeField('music_volume', 'Music volume', 'Loudness of played music', 0, 2.5, 0.05, FMT.pct));
    c2.appendChild(rangeField('duck_level', 'Duck level', 'Music volume while someone is talking', 0, 1, 0.01, FMT.pct));
    c2.appendChild(rangeField('voice_gain', 'Voice loudness', 'Gain applied to Reachy’s voice', 0.2, 2.5, 0.05, FMT.x));
    c2.appendChild(rangeField('echo_guard', 'Echo guard', 'Raise if music interrupts Reachy; lower if it ignores you', 0.5, 5, 0.1, FMT.n1));
    c2.appendChild(switchRow('web_search', 'Web search in Gemini mode', 'Lets Gemini look things up. The local brain always has free web search and weather.', true));
    var lm = h('select', { id: 'sp-listen_mode' });
    [['always', 'Always listening'], ['push_to_talk', 'Push to talk (loud venues)']].forEach(function (o) {
      lm.appendChild(h('option', { value: o[0], text: o[1] }));
    });
    lm.addEventListener('change', function () { save('listen_mode', function () { return lm.value; }, 150, 'Listening mode'); });
    fields.listen_mode = { el: lm, set: function (v) { lm.value = v || 'always'; } };
    c2.appendChild(field('sp-listen_mode', 'Listening', 'Push to talk: Reachy only hears you while you hold the talk button (or Space) on the DJ tab', lm));
    c2.appendChild(switchRow('greet_on_start', 'Say “I’m ready” at start', 'Reachy says it once when it can listen after the app starts (the first warm-up can take a couple of minutes).', false));
    root.appendChild(c2);

    // --- Backend ---
    var c3 = h('div', { class: 'pn-card' }, [h('h2', { text: 'Voice backend' })]);
    var be = h('select', { id: 'sp-backend' });
    [['auto', 'Auto (local when available, else Gemini)'], ['local', 'Local only'], ['gemini', 'Gemini only']].forEach(function (o) {
      be.appendChild(h('option', { value: o[0], text: o[1] }));
    });
    be.addEventListener('change', function () { save('backend', function () { return be.value; }, 150, 'Backend'); });
    fields.backend = { el: be, set: function (v) { be.value = v || 'auto'; } };
    c3.appendChild(field('sp-backend', 'Backend', 'Which engine handles conversation', be));
    c3.appendChild(textField('brain_url', 'Brain URL', 'Address of the brain computer running reachy-brain', { type: 'url', placeholder: 'ws://192.168.1.50:8765' }));
    c3.appendChild(switchRow('allow_gemini_assist', 'Allow Gemini assist', 'Let the local backend ask Gemini for help with hard questions.', true));
    roBox.dl = h('dl', { class: 'sp-ro', 'aria-live': 'polite' });
    roBox.active = h('dd'); roBox.brain = h('dd');
    roBox.dl.append(h('dt', { text: 'Active backend' }), roBox.active, h('dt', { text: 'Brain status' }), roBox.brain);
    c3.appendChild(roBox.dl);
    root.appendChild(c3);

    // --- FRC (FIRST Robotics): Nexus QR scan, queuing alerts, Blue Alliance ---
    var c4 = h('div', { class: 'pn-card' }, [h('h2', { text: 'FRC robotics' })]);
    var scanBtn = h('button', { type: 'button', class: 'pn-btn primary', text: 'Scan QR code' });
    var scanMsg = h('span', { class: 'hint', 'aria-live': 'polite', text: 'Hold your team\u2019s Nexus code up to Reachy\u2019s camera.' });
    scanBtn.addEventListener('click', function () {
      scanBtn.disabled = true; scanMsg.textContent = 'Looking for a QR code\u2026';
      post('/api/frc/scan', {}).then(function (r) {
        if (r.ok) { scanMsg.textContent = r.result || ('Read: ' + (r.qr_text || '')); toast(r.result || 'QR code read'); }
        else { scanMsg.textContent = r.error || 'No QR code found'; toast(r.error || 'No QR code found', true); }
      }).catch(function (e) { scanMsg.textContent = 'Scan failed: ' + e.message; })
        .then(function () { scanBtn.disabled = false; });
    });
    c4.appendChild(h('div', { class: 'sp-field' }, [h('div', { class: 'sp-row' }, [scanBtn]), scanMsg]));
    c4.appendChild(textField('frc_team', 'Team number', 'Filled in by the Nexus QR code', { placeholder: 'e.g. 1234' }));
    c4.appendChild(textField('frc_event', 'Event code', 'Blue Alliance / Nexus event key', { placeholder: 'e.g. 2026mokc' }));
    c4.appendChild(secretField('nexus_api_key', 'nexus_key_set', 'Nexus API key', 'Live queuing. Get one at frc.nexus/api'));
    c4.appendChild(secretField('tba_api_key', 'tba_key_set', 'The Blue Alliance key', 'Rankings and results. Get one at thebluealliance.com/account'));
    c4.appendChild(switchRow('frc_alerts', 'Queuing alerts', 'Reachy says out loud when our match is queuing soon, now queuing or on deck.', true));
    frcBox.dl = h('dl', { class: 'sp-ro', 'aria-live': 'polite' });
    frcBox.q = h('dd'); frcBox.next = h('dd');
    frcBox.dl.append(h('dt', { text: 'Now queuing' }), frcBox.q, h('dt', { text: 'Our next match' }), frcBox.next);
    c4.appendChild(frcBox.dl);
    var credit = h('p', { class: 'hint' }, ['Match data from ',
      h('a', { href: 'https://frc.nexus', target: '_blank', rel: 'noopener', text: 'FRC Nexus' }), ' and ',
      h('a', { href: 'https://www.thebluealliance.com', target: '_blank', rel: 'noopener', text: 'The Blue Alliance' }), '.']);
    c4.appendChild(credit);
    root.appendChild(c4);
  }

  var frcBox = {}, secretStates = {};
  function secretField(key, setKey, label, hint) {
    var inp = h('input', { class: 'sp-input', id: 'sp-' + key, type: 'password', autocomplete: 'new-password',
      spellcheck: 'false', placeholder: 'Paste a key to replace the stored one' });
    var btn = h('button', { type: 'button', class: 'pn-btn primary', text: 'Save' });
    var st = h('span', { class: 'hint', text: '' });
    secretStates[setKey] = st;
    function go() {
      var v = inp.value.trim();
      if (!v) { toast('Paste a key first', true); return; }
      var b = {}; b[key] = v; btn.disabled = true;
      post('/api/config', b).then(function () { inp.value = ''; toast(label + ' saved'); })
        .catch(function (e) { toast(label + ' not saved: ' + e.message, true); })
        .then(function () { btn.disabled = false; });
    }
    btn.addEventListener('click', go);
    inp.addEventListener('keydown', function (e) { if (e.key === 'Enter') go(); });
    return h('div', { class: 'sp-field' }, [h('label', { for: 'sp-' + key }, [label + ' ', st]),
      h('div', { class: 'sp-row' }, [inp, btn]), h('span', { class: 'hint', text: hint })]);
  }

  function brainText(b) {
    if (b == null || b === '') return 'not available yet';
    if (typeof b === 'string') return b;
    var parts = [];
    var st = b.status || b.state || (b.ok === true ? 'ok' : (b.ok === false ? 'unreachable' : ''));
    if (st) parts.push(String(st));
    if (b.model) parts.push(String(b.model));
    if (b.connected !== undefined) parts.push(b.connected ? 'connected' : 'not connected');
    if (b.reason) parts.push(String(b.reason));
    if (b.error) parts.push('error: ' + b.error);
    if (b.latency_ms != null) parts.push(Math.round(b.latency_ms) + ' ms');
    return parts.join(' · ') || JSON.stringify(b);
  }

  function render(s) {
    if (!s || !root || !fields.model) return;
    var c = s.config || {};
    var voices = Array.isArray(s.voices) ? s.voices : [];
    var sig = voices.join('|');
    if (sig !== voiceSig) {
      voiceSig = sig;
      var v = fields.voice.el;
      v.textContent = '';
      voices.forEach(function (n) { v.appendChild(h('option', { value: n, text: n })); });
    }
    if (c.voice && voices.indexOf(c.voice) < 0 && !pending.voice) {
      fields.voice.el.appendChild(h('option', { value: c.voice, text: c.voice }));
    }
    Object.keys(fields).forEach(function (k) {
      if (pending[k] || document.activeElement === fields[k].el && fields[k].el.tagName !== 'SELECT') return;
      if (k in c) fields[k].set(c[k]);
    });
    keyState.textContent = c.api_key_set ? '✓ set' : '✗ not set';
    keyState.style.color = c.api_key_set ? 'var(--p-ok)' : 'var(--p-bad)';
    roBox.active.textContent = s.active_backend ? String(s.active_backend) : 'not available yet';
    roBox.brain.textContent = brainText(s.brain);
    Object.keys(secretStates).forEach(function (k) {
      secretStates[k].textContent = c[k] ? '\u2713 set' : '\u2717 not set';
      secretStates[k].style.color = c[k] ? 'var(--p-ok)' : 'var(--p-muted)';
    });
    var f = s.frc || {};
    if (frcBox.q) {
      var need = !c.frc_event ? 'scan your Nexus QR or set the event' : (!c.nexus_key_set ? 'add a Nexus API key for live queuing' : '');
      frcBox.q.textContent = need || (f.error ? 'error: ' + f.error : (f.now_queuing || (f.updated ? 'nothing yet' : 'checking\u2026')));
      frcBox.next.textContent = need ? '\u2014' : (f.next_match || (c.frc_team ? 'none scheduled' : 'set a team number'));
    }
  }

  function init() {
    root = document.getElementById('settings-panel');
    if (!root) return;
    build();
    var tries = 0;
    (function attach() {
      if (window.reachyApi && typeof window.reachyApi.subscribe === 'function') {
        window.reachyApi.subscribe(render);
        try { render(window.reachyApi.state()); } catch (e) { /* no state yet */ }
      } else if (tries++ < 100) setTimeout(attach, 100);
    })();
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init); else init();
})();
