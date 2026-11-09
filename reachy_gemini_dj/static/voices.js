/* Voices tab: add a voice from a recording of someone who agreed, keep a list, switch between voices.
   GET /api/voices lists them, POST /api/voices makes one (the recording is read in the browser and sent as base64, the
   brain computer makes the voice), POST /api/voices/<id>/use switches, DELETE /api/voices/<id> removes. The cloning
   happens on the brain computer; this page only relays. Vanilla JS, no dependencies. */
(function () {
  'use strict';
  var API = '/api/voices';
  var MAX_BYTES = 5 * 1024 * 1024;
  var els = {}, st = { voices: [], active: 'default', canClone: false, connected: false, ok: false, error: '' };
  var busy = false, loaded = false;

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
  function api(method, path, body) {
    var opts = { method: method, cache: 'no-store' };
    if (body !== undefined) { opts.headers = { 'Content-Type': 'application/json' }; opts.body = JSON.stringify(body); }
    return fetch(API + path, opts).then(function (r) { return r.json(); });
  }
  function fmtDate(iso) {
    if (!iso) return '';
    var d = new Date(iso);
    if (isNaN(d.getTime())) return '';
    try { return d.toLocaleDateString(undefined, { dateStyle: 'medium' }); } catch (e) { return d.toLocaleDateString(); }
  }
  function apply(res) {
    st.ok = !!res.ok; st.error = res.ok ? '' : (res.error || 'Something went wrong.');
    st.connected = res.connected !== undefined ? !!res.connected : st.connected;
    if (res.ok) {
      st.voices = res.voices || st.voices;
      st.active = res.active || st.active;
      if (res.can_clone !== undefined) st.canClone = !!res.can_clone;
    }
  }

  /* ---------------------------------------------------------------- list */
  function row(v) {
    var isDefault = v.id === 'default', active = st.active === v.id;
    var meta = [];
    if (isDefault) meta.push("Reachy's built-in voice");
    else {
      if (v.seconds) meta.push(Math.round(v.seconds) + ' s of speech');
      if (fmtDate(v.created)) meta.push('added ' + fmtDate(v.created));
      meta.push(v.consent && v.consent.confirmed ? 'speaker agreed (confirmed when added)' : 'no consent record');
    }
    var use = h('button', { type: 'button', class: 'pn-btn sm ' + (active ? 'quiet' : 'primary'), text: active ? 'In use' : 'Use this voice' });
    if (active) use.disabled = true;
    use.addEventListener('click', function () { act(use, 'POST', '/' + encodeURIComponent(v.id) + '/use', 'Switched. Reachy says hello in the new voice.'); });
    var kids = [use];
    if (!isDefault) {
      var del = h('button', { type: 'button', class: 'pn-btn sm danger', text: 'Delete' }), timer = 0;
      del.addEventListener('click', function () {
        if (!del.dataset.sure) {
          del.dataset.sure = '1'; del.textContent = 'Really delete?';
          timer = setTimeout(function () { delete del.dataset.sure; del.textContent = 'Delete'; }, 4000);
          return;
        }
        clearTimeout(timer);
        act(del, 'DELETE', '/' + encodeURIComponent(v.id), 'Voice deleted');
      });
      kids.push(del);
    }
    return h('li', { class: 'vc-item' + (active ? ' on' : '') }, [
      h('div', { class: 'vc-info' }, [
        h('div', { class: 'vc-name' }, [v.name, active ? h('span', { class: 'vc-badge', text: 'active' }) : null]),
        h('div', { class: 'vc-meta pn-muted', text: meta.join(' · ') })]),
      h('div', { class: 'vc-actions' }, kids)]);
  }
  function render() {
    els.count.textContent = st.voices.length ? String(st.voices.length) : '';
    var msg = '';
    if (!loaded) msg = '';
    else if (!st.connected) msg = 'The voice brain is not connected, so voices cannot be listed or changed. Start the brain on your computer and wait for the robot to connect.';
    else if (!st.ok) msg = st.error;
    els.status.hidden = !msg; els.status.textContent = msg;
    var noClone = loaded && st.connected && st.ok && !st.canClone;
    els.noclone.hidden = !noClone;
    els.list.textContent = '';
    if (loaded && st.connected && st.ok) {
      els.list.appendChild(row({ id: 'default', name: 'Default voice' }));
      st.voices.forEach(function (v) { els.list.appendChild(row(v)); });
    }
    updateForm();
  }
  function refresh() {
    return api('GET', '').then(function (r) { loaded = true; apply(r); render(); })
      .catch(function (e) { loaded = true; st.ok = false; st.connected = false; st.error = 'Could not reach the app (' + e.message + ')'; render(); });
  }
  function act(btn, method, path, okMsg) {
    if (busy) return;
    busy = true; btn.disabled = true;
    api(method, path).then(function (r) {
      if (!r.ok) { toast(r.error || 'That did not work.', true); return; }
      apply(r); render(); toast(okMsg);
    }).catch(function (e) { toast('Request failed: ' + e.message, true); })
      .then(function () { busy = false; render(); });
  }

  /* ---------------------------------------------------------------- form */
  function updateForm() {
    var ready = st.connected && st.ok && st.canClone && els.file.files.length && els.name.value.trim() && els.consent.checked && !busy;
    els.make.disabled = !ready;
  }
  function readB64(file) {
    return new Promise(function (resolve, reject) {
      var r = new FileReader();
      r.onload = function () { var s = String(r.result || ''); resolve(s.slice(s.indexOf(',') + 1)); };
      r.onerror = function () { reject(new Error('Could not read that file.')); };
      r.readAsDataURL(file);
    });
  }
  function make() {
    var f = els.file.files[0];
    els.err.hidden = true;
    if (!f) return;
    if (f.size > MAX_BYTES) { fail('That file is too big. A 30 second clip is plenty (5 MB at most).'); return; }
    busy = true; els.make.disabled = true; els.prog.hidden = false;
    readB64(f).then(function (b64) {
      return api('POST', '', { name: els.name.value.trim(), consent: els.consent.checked, filename: f.name, audio_b64: b64 });
    }).then(function (r) {
      if (!r.ok) { fail(r.error || 'The voice could not be made.'); return; }
      apply(r); els.file.value = ''; els.name.value = ''; els.consent.checked = false;
      toast('Voice added: ' + (r.voice && r.voice.name || ''));
    }).catch(function (e) { fail(e.message); })
      .then(function () { busy = false; els.prog.hidden = true; render(); });
  }
  function fail(msg) { els.err.hidden = false; els.err.textContent = msg; busy = false; els.prog.hidden = true; updateForm(); }

  /* ---------------------------------------------------------------- build */
  function build() {
    var root = document.getElementById('voices-panel');
    if (!root) return false;
    var card = h('div', { class: 'pn-card', id: 'rp-voices' });
    els.count = h('span', { class: 'vc-count' });
    card.appendChild(h('h2', {}, ['Voices ', els.count]));
    card.appendChild(h('p', { class: 'pn-muted', text: "Pick the voice Reachy speaks with, or add a new one from a short recording of someone talking. " +
      'The voice is made on the brain computer, so it works only while the brain is connected.' }));
    els.status = h('div', { class: 'vc-note', role: 'status', hidden: '' });
    card.appendChild(els.status);
    els.noclone = h('div', { class: 'vc-note', hidden: '', text: 'Making new voices is not set up on the brain yet: the Kyutai voice model needs its terms accepted on ' +
      'Hugging Face and a read login on the brain computer (see the README). You can still switch between the voices below.' });
    card.appendChild(els.noclone);
    els.list = h('ul', { class: 'vc-list', 'aria-live': 'polite' });
    card.appendChild(els.list);

    card.appendChild(h('h3', { text: 'Add a voice' }));
    els.file = h('input', { type: 'file', id: 'vc-file', accept: 'audio/*,.mp3,.wav,.m4a,.ogg,.flac,.aac', onchange: updateForm });
    els.name = h('input', { type: 'text', id: 'vc-name', maxlength: '40', placeholder: 'Name, for example "Dan"', autocomplete: 'off', oninput: updateForm });
    els.consent = h('input', { type: 'checkbox', id: 'vc-consent', onchange: updateForm });
    els.make = h('button', { type: 'button', class: 'pn-btn primary', text: 'Make this voice', onclick: make });
    els.make.disabled = true;
    card.appendChild(h('div', { class: 'vc-form' }, [
      h('label', { for: 'vc-file', text: 'Recording (mp3, wav or m4a, 10 to 30 seconds of one person speaking clearly)' }), els.file,
      h('label', { for: 'vc-name', text: 'Name' }), els.name,
      h('label', { class: 'vc-consent', for: 'vc-consent' }, [els.consent,
        h('span', {}, [h('b', { text: 'The speaker agreed to their voice being used here.' }),
          h('span', { class: 'pn-muted', text: ' Only add the voice of someone who said yes. The recording is not kept: only the small voice file stays on the brain computer, and you can delete it here at any time.' })])]),
      h('div', { class: 'pano-row' }, [els.make])]));
    els.prog = h('div', { class: 'pano-prog', hidden: '' }, [h('div', { class: 'pano-bar indet' }, [h('i')]),
      h('div', { class: 'pano-prog-text pn-muted', text: 'Making the voice. This can take up to a minute.' })]);
    card.appendChild(els.prog);
    els.err = h('div', { class: 'pano-err', role: 'alert', hidden: '' });
    card.appendChild(els.err);
    root.appendChild(card);
    return true;
  }
  function visible() {
    var p = document.getElementById('p-voices');
    return !document.hidden && p && !p.classList.contains('hide');
  }
  function init() {
    if (!build()) return;
    var tab = document.getElementById('tab-voices');
    if (tab) tab.addEventListener('click', function () { setTimeout(function () { if (visible() && !busy) refresh(); }, 50); });
    document.addEventListener('visibilitychange', function () { if (visible() && !busy) refresh(); });
    if (visible()) refresh();
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init); else init();
})();
