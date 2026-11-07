/* Robot panel: mounts into #robot-panel. Vanilla JS, no deps. */
(function () {
  'use strict';
  var root, movesTries = 0, moves = { emotions: [], dances: [] };
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
  function move(kind, value) {
    return post('/api/robot/move', { kind: kind, value: value }).then(function (r) {
      if (r && r.ok) toast('Reachy: ' + (r.result || 'ok')); else toast((r && r.error) || 'Move failed', true);
    }).catch(function (e) { toast('Could not reach the app (' + e.message + ')', true); });
  }
  function saveToggle(key, input) {
    var b = {}; b[key] = input.checked;
    post('/api/config', b).then(function () { toast('Saved'); })
      .catch(function (e) { input.checked = !input.checked; toast('Save failed: ' + e.message, true); });
  }

  function build() {
    root.textContent = '';
    var card = h('div', { class: 'pn-card' });
    card.appendChild(h('h2', { text: 'Robot' }));

    els.status = h('div', { class: 'rp-status', 'aria-live': 'polite' });
    card.appendChild(els.status);

    card.appendChild(h('h3', { text: 'Head direction' }));
    var pad = h('div', { class: 'rp-pad', role: 'group', 'aria-label': 'Head direction' });
    [['up', '↑', 'Look up'], ['left', '←', 'Look left'], ['front', 'front', 'Look front'],
     ['right', '→', 'Look right'], ['down', '↓', 'Look down']].forEach(function (d) {
      pad.appendChild(h('button', { type: 'button', class: 'pn-btn ' + d[0], 'aria-label': d[2], title: d[2],
        onclick: function () { move('head', d[0]); } }, [d[1]]));
    });
    card.appendChild(pad);

    card.appendChild(h('h3', { text: 'Emotions' }));
    els.emos = h('div', { class: 'rp-grid' });
    card.appendChild(els.emos);

    card.appendChild(h('h3', { text: 'Dance' }));
    els.danceSel = h('select', { id: 'rp-dance-sel', 'aria-label': 'Dance' });
    els.danceBtn = h('button', { type: 'button', class: 'pn-btn primary', text: 'Dance',
      onclick: function () { move('dance', els.danceSel.value); } });
    els.danceHint = h('div', { class: 'pn-muted' });
    card.appendChild(h('div', { class: 'rp-dance' }, [els.danceSel, els.danceBtn]));
    card.appendChild(els.danceHint);

    card.appendChild(h('h3', { text: 'Motion settings' }));
    els.sw = {};
    [['motion_enabled', 'Motion enabled', 'Master switch for all robot movement.'],
     ['expressive', 'Expressive', 'Reachy reacts with emotions while talking.'],
     ['dance_enabled', 'Dancing allowed', 'Lets Reachy dance on its own to music.']].forEach(function (s, i) {
      var inp = h('input', { type: 'checkbox', id: 'rp-' + s[0], onchange: function () { saveToggle(s[0], inp); } });
      els.sw[s[0]] = inp;
      card.appendChild(h('label', { class: 'pn-sw' + (i ? '' : ' first'), for: 'rp-' + s[0] }, [
        h('span', { class: 'txt' }, [h('b', { text: s[1] }), h('span', { text: s[2] })]), inp]));
    });
    root.appendChild(card);
    buildPoses();
    renderMoves();
  }

  /* ---- Learn poses: relax motors, pose Reachy by hand, save (written to the robot's disk at once) ---- */
  var poseNames = null, learning = false;
  function buildPoses() {
    var card = h('div', { class: 'pn-card', id: 'rp-poses' });
    card.appendChild(h('h2', { text: 'Learn poses' }));
    card.appendChild(h('div', { class: 'pn-muted', text: 'Learn mode turns the motors off so you can pose Reachy by hand. Lock turns them back on and holds the pose; Save locks and stores it on the robot right away. ' +
      'Play saved poses here or say "do your <name> pose".' }));
    els.learnBtn = h('button', { type: 'button', class: 'pn-btn primary rp-learn', 'aria-pressed': 'false',
      onclick: function () { setLearn(!learning); } });
    els.learnHint = h('div', { class: 'rp-learn-hint', 'aria-live': 'polite' });
    card.appendChild(h('div', { class: 'rp-learn-row' }, [els.learnBtn]));
    card.appendChild(els.learnHint);
    els.poseName = h('input', { type: 'text', id: 'rp-pose-name', maxlength: '40', placeholder: 'Pose name, e.g. wave',
      'aria-label': 'Pose name', autocomplete: 'off',
      onkeydown: function (e) { if (e.key === 'Enter') savePose(); } });
    els.poseSave = h('button', { type: 'button', class: 'pn-btn primary', text: 'Save pose', onclick: savePose });
    card.appendChild(h('h3', { text: 'Save current pose' }));
    card.appendChild(h('div', { class: 'rp-dance' }, [els.poseName, els.poseSave]));
    card.appendChild(h('h3', { text: 'Saved poses' }));
    els.poseList = h('div', { class: 'rp-pose-list' });
    card.appendChild(els.poseList);
    root.appendChild(card);
    renderLearn();
    fetch('/api/poses').then(function (r) { return r.json(); }).then(function (r) {
      learning = !!r.learning; renderLearn(); renderPoses((r.poses || []).map(function (p) { return p.name; }));
    }).catch(function () { renderPoses([]); });
  }
  function setLearn(on) {
    els.learnBtn.disabled = true;
    post('/api/poses/learn', { on: on }).then(function (r) {
      if (r && r.ok) { learning = !!r.learning; toast(on ? 'Motors off: pose Reachy by hand' : 'Motors on: pose locked'); }
      else toast((r && r.error) || 'Could not change learn mode', true);
    }).catch(function (e) { toast('Could not reach the app (' + e.message + ')', true); })
      .then(function () { els.learnBtn.disabled = false; renderLearn(); });
  }
  function renderLearn() {
    els.learnBtn.textContent = learning ? 'Lock pose (motors on)' : 'Learn: motors off';
    els.learnBtn.setAttribute('aria-pressed', learning ? 'true' : 'false');
    els.learnBtn.classList.toggle('on', learning);
    els.learnHint.textContent = learning ? 'Motors are OFF. Support the head while you pose it, then press Lock pose or Save pose.' : '';
    els.learnHint.className = 'rp-learn-hint' + (learning ? ' on' : '');
  }
  function savePose() {
    var name = els.poseName.value.trim();
    if (!name) { toast('Give the pose a name first', true); els.poseName.focus(); return; }
    els.poseSave.disabled = true;
    post('/api/poses/save', { name: name }).then(function (r) {
      if (r && r.ok) { if (learning) { learning = false; renderLearn(); } toast('Locked and saved "' + r.name + '"'); els.poseName.value = '';
        renderPoses((r.poses || []).map(function (p) { return p.name; })); }
      else toast((r && r.error) || 'Save failed', true);
    }).catch(function (e) { toast('Save failed: ' + e.message, true); })
      .then(function () { els.poseSave.disabled = false; });
  }
  function renderPoses(names) {
    var key = JSON.stringify(names || []);
    if (key === poseNames) return;           // only rebuild when the list changed (keeps focus)
    poseNames = key;
    els.poseList.textContent = '';
    if (!names.length) { els.poseList.appendChild(h('div', { class: 'pn-muted', text: 'No poses saved yet.' })); return; }
    names.forEach(function (n) {
      els.poseList.appendChild(h('div', { class: 'rp-pose' }, [
        h('span', { class: 'rp-pose-name', text: n }),
        h('button', { type: 'button', class: 'pn-btn primary', text: 'Play', 'aria-label': 'Play pose ' + n,
          onclick: function () {
            post('/api/poses/play', { name: n }).then(function (r) {
              if (r && r.ok) { if (learning) { learning = false; renderLearn(); } toast('Reachy: pose ' + r.name); }
              else toast((r && r.error) || 'Could not play pose', true);
            }).catch(function (e) { toast('Could not reach the app (' + e.message + ')', true); });
          } }),
        h('button', { type: 'button', class: 'pn-btn', text: 'Delete', 'aria-label': 'Delete pose ' + n,
          onclick: function () {
            if (!window.confirm('Delete the pose "' + n + '"?')) return;
            post('/api/poses/delete', { name: n }).then(function (r) {
              if (r && r.ok) { toast('Deleted "' + n + '"'); renderPoses((r.poses || []).map(function (p) { return p.name; })); }
              else toast((r && r.error) || 'Delete failed', true);
            }).catch(function (e) { toast('Delete failed: ' + e.message, true); });
          } })
      ]));
    });
  }

  function renderMoves() {
    els.emos.textContent = '';
    moves.emotions.forEach(function (e) {
      els.emos.appendChild(h('button', { type: 'button', class: 'pn-btn', text: e === 'random' ? 'surprise me' : e.replace(/_/g, ' '),
        onclick: function () { move('emotion', e); } }));
    });
    var cur = els.danceSel.value;
    els.danceSel.textContent = '';
    els.danceSel.appendChild(h('option', { value: '', text: 'Surprise me' }));
    moves.dances.forEach(function (d) { els.danceSel.appendChild(h('option', { value: d, text: d.replace(/_/g, ' ') })); });
    els.danceSel.value = cur;
  }

  function retry() { if (movesTries++ < 20) setTimeout(loadMoves, 5000); }
  function loadMoves() {
    fetch('/api/robot/moves').then(function (r) { return r.json(); }).then(function (m) {
      moves = { emotions: m.emotions || [], dances: m.dances || [] };
      renderMoves();
      var ready = moves.dances.length > 0 || (m.status && m.status.dances_ready);
      els.danceHint.textContent = ready ? '' : 'Dance library still loading…';
      if (!ready) retry();
    }).catch(retry);
  }

  function render(s) {
    if (!s || !root || !els.status) return;
    var m = s.motion || {}, c = s.config || {};
    els.status.textContent = '';
    var dotCls = m.error ? 'bad' : (m.busy ? 'busy' : (m.enabled === false ? '' : 'ok'));
    els.status.appendChild(h('span', {}, [h('span', { class: 'pn-dot ' + dotCls }),
      m.enabled === false ? 'Motion off' : (m.busy ? 'Moving…' : 'Idle')]));
    if (m.last) els.status.appendChild(h('span', { class: 'pn-muted', text: 'Last: ' + m.last }));
    if (m.error) els.status.appendChild(h('span', { class: 'pn-err', role: 'alert', text: 'Error: ' + m.error }));
    els.status.appendChild(h('span', { class: 'pn-muted', text: 'Emotions ' + (m.emotions_ready ? 'ready' : 'loading') +
      ' · dances ' + (m.dances_ready ? 'ready' : 'loading') }));
    Object.keys(els.sw).forEach(function (k) {
      if (k in c && document.activeElement !== els.sw[k]) els.sw[k].checked = !!c[k];
    });
    els.danceBtn.disabled = c.motion_enabled === false;
    if (els.learnBtn && typeof m.learning === 'boolean' && !els.learnBtn.disabled && m.learning !== learning) {
      learning = m.learning; renderLearn();
    }
    if (els.poseList && Array.isArray(m.poses)) renderPoses(m.poses);
  }

  function init() {
    root = document.getElementById('robot-panel');
    if (!root) return;
    build();
    loadMoves();
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
