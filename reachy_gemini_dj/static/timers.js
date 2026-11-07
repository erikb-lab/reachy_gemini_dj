/* Timers card: self-contained. Mounts itself into the DJ tab, renders state.timers, posts cancel/stop. */
(function () {
  'use strict';
  var api = window.reachyApi;
  var feed = document.getElementById('feed');
  if (!api || !feed) return;
  var anchor = feed.closest('section') || feed.parentNode;

  var css = document.createElement('style');
  css.textContent =
    '.tm-list{list-style:none;margin:0;padding:0}' +
    '.tm-list li{display:flex;gap:12px;align-items:center;justify-content:space-between;padding:10px 0;border-top:1px solid var(--line)}' +
    '.tm-list li:first-child{border-top:0;padding-top:0}.tm-list li:last-child{padding-bottom:0}.tm-list .tm-t{min-width:0;overflow-wrap:anywhere;font-size:14.5px}' +
    '.tm-list small{display:block;color:var(--mut);font-size:13px}.tm-list .tm-left{font-weight:700;font-size:15px;font-variant-numeric:tabular-nums}' +
    '.tm-list .row{flex-wrap:nowrap;gap:12px}.tm-list .btn{min-height:36px;padding:0 12px}' +
    '.tm-list li.ring .tm-left{color:var(--bad)}.tm-list .empty{color:var(--mut);border:0}' +
    '#tm-stop{width:100%;margin-bottom:12px}';
  document.head.appendChild(css);

  var card = document.createElement('section');
  card.className = 'card';
  card.setAttribute('aria-labelledby', 'h-tm');
  card.innerHTML = '<h2 id="h-tm">Timers <span class="badge" id="tm-n">0</span></h2>' +
    '<button class="btn danger hide" id="tm-stop" type="button">Stop alarm</button>' +
    '<ul class="tm-list" id="tm-list" aria-live="polite"></ul>';
  anchor.parentNode.insertBefore(card, anchor);

  var list = card.querySelector('#tm-list'), stop = card.querySelector('#tm-stop'), count = card.querySelector('#tm-n');

  function left(s) {
    s = Math.max(0, Math.ceil(s));
    var h = Math.floor(s / 3600), m = Math.floor(s % 3600 / 60), x = s % 60, p2 = function (n) { return String(n).padStart(2, '0'); };
    return h ? h + ':' + p2(m) + ':' + p2(x) : m + ':' + p2(x);
  }
  function title(t) {
    if (t.reminder) return 'Reminder: ' + t.label;
    return (t.kind === 'alarm' ? 'Alarm' : 'Timer') + (t.label ? ': ' + t.label : '');
  }
  function post(path, body) {
    api.post(path, body).then(function (r) { if (r && r.error) api.toast(r.error); }).catch(function (e) { api.toast('Request failed: ' + e.message); });
  }

  api.subscribe(function (s) {
    var d = s.timers || { timers: [], ringing: false }, rows = d.timers || [];
    count.textContent = rows.filter(function (t) { return !t.ringing; }).length;
    stop.classList.toggle('hide', !d.ringing);
    list.replaceChildren();
    rows.forEach(function (t) {
      var li = document.createElement('li'), box = document.createElement('div'), name = document.createElement('div'), sub = document.createElement('small');
      li.className = t.ringing ? 'ring' : '';
      box.className = 'tm-t';
      name.textContent = title(t);
      sub.textContent = t.ringing ? 'Ringing' : (t.kind === 'alarm' || t.reminder ? t.at : '');
      box.appendChild(name);
      if (sub.textContent) box.appendChild(sub);
      var right = document.createElement('div');
      right.className = 'row';
      var lt = document.createElement('span');
      lt.className = 'tm-left';
      lt.textContent = t.ringing ? 'Now' : left(t.left);
      right.appendChild(lt);
      if (!t.ringing) {
        var b = document.createElement('button');
        b.className = 'btn ghost'; b.type = 'button'; b.textContent = 'Cancel';
        b.setAttribute('aria-label', 'Cancel ' + title(t));
        b.onclick = function () { post('/api/timers/cancel', { id: t.id }); };
        right.appendChild(b);
      }
      li.appendChild(box); li.appendChild(right);
      list.appendChild(li);
    });
    if (!rows.length) { var e = document.createElement('li'); e.className = 'empty'; e.textContent = 'No timers or alarms.'; list.appendChild(e); }
  });
  stop.onclick = function () { post('/api/timers/stop', {}); };
})();
