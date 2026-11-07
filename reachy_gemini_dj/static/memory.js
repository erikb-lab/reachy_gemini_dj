/* Memory card (Settings tab): list, add and delete remembered facts. Vanilla JS, no deps. */
(function () {
  'use strict';
  var list = document.getElementById('mem-list'), badge = document.getElementById('mem-n'),
      form = document.getElementById('f-mem'), input = document.getElementById('mem_q');
  if (!list || !form) return;

  function toast(msg) { if (window.reachyApi) window.reachyApi.toast(msg); }
  function call(path, body) {
    return fetch(path, body === undefined ? { cache: 'no-store' } :
      { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })
      .then(function (r) { if (!r.ok) throw new Error('HTTP ' + r.status); return r.json(); });
  }
  function render(facts) {
    list.textContent = '';
    badge.textContent = facts.length;
    if (!facts.length) {
      var e = document.createElement('li'); e.className = 'sub'; e.textContent = 'Nothing remembered yet.'; list.appendChild(e);
      return;
    }
    facts.forEach(function (f) {
      var li = document.createElement('li'), t = document.createElement('span'), b = document.createElement('button');
      t.className = 'grow'; t.textContent = f.text;
      b.className = 'btn ghost'; b.type = 'button'; b.textContent = 'Delete';
      b.setAttribute('aria-label', 'Forget: ' + f.text);
      b.addEventListener('click', function () {
        b.disabled = true;
        call('/api/memory/delete', { id: f.id }).then(function (r) {
          if (r.facts) render(r.facts);
          toast(r.ok ? 'Forgotten' : (r.error || 'Could not delete'));
        }).catch(function (err) { b.disabled = false; toast('Request failed: ' + err.message); });
      });
      li.appendChild(t); li.appendChild(b); list.appendChild(li);
    });
  }
  function load() {
    call('/api/memory').then(function (r) { if (r.facts) render(r.facts); }).catch(function () {});
  }
  form.addEventListener('submit', function (e) {
    e.preventDefault();
    var text = input.value.trim(); if (!text) return;
    call('/api/memory', { text: text }).then(function (r) {
      if (r.facts) render(r.facts);
      if (r.ok) { input.value = ''; toast(r.status === 'duplicate' ? 'Already remembered' : r.status === 'updated' ? 'Updated' : 'Remembered'); }
      else toast(r.error || 'Could not save');
    }).catch(function (err) { toast('Request failed: ' + err.message); });
  });
  load();
  setInterval(function () { if (!document.hidden && document.activeElement !== input) load(); }, 15000);   // voice edits show up
})();
