/* Music queue card: self-contained. Replaces the plain queue list with per-item controls, shuffle/clear,
   autoplay toggle, save-song heart and a favorites list; makes the progress bar click-to-seek. */
(function () {
  'use strict';
  var api = window.reachyApi;
  var oldQ = document.getElementById('queue');
  if (!api || !oldQ) return;
  var oldCard = oldQ.closest('section') || oldQ.parentNode;
  oldCard.classList.add('hide');

  var css = document.createElement('style');
  css.textContent =
    '.mq-bar{display:flex;flex-wrap:wrap;gap:8px;margin:0 0 8px}.mq-bar .btn{min-height:36px;padding:0 12px}' +
    '.mq-list{list-style:none;margin:0;padding:0}.mq-list li{display:flex;gap:8px;align-items:center;padding:10px 0;border-top:1px solid var(--line)}' +
    '.mq-list li:first-child{border-top:0}.mq-t{flex:1;min-width:0;overflow-wrap:anywhere;font-size:14.5px}.mq-t small{display:block;color:var(--mut);font-size:13px}' +
    '.mq-list .btn{min-width:36px;min-height:36px;padding:0 10px}.mq-list .empty{color:var(--mut);border:0}' +
    '.mq-h{margin:18px 0 8px;font-size:12px;font-weight:700;text-transform:uppercase;letter-spacing:.06em;color:var(--mut)}#progbar{cursor:pointer}';
  document.head.appendChild(css);

  var card = document.createElement('section');
  card.className = 'card';
  card.setAttribute('aria-labelledby', 'h-mq');
  card.innerHTML = '<h2 id="h-mq">Queue <span class="badge" id="mq-n">0</span></h2>' +
    '<div class="mq-bar">' +
    '<button class="btn ghost" id="mq-fav" type="button" aria-label="Save current song to favorites">&#9829; Save song</button>' +
    '<button class="btn ghost" id="mq-shuf" type="button">Shuffle</button>' +
    '<button class="btn ghost" id="mq-clear" type="button">Clear</button>' +
    '<button class="btn ghost" id="mq-auto" type="button" aria-pressed="false">Keep going: on</button></div>' +
    '<ol class="mq-list" id="mq-list" aria-live="polite"></ol>' +
    '<h3 class="mq-h">Favorites <span class="badge" id="mq-fn">0</span></h3>' +
    '<div class="mq-bar"><button class="btn ghost" id="mq-favplay" type="button">Play all (shuffled)</button></div>' +
    '<ul class="mq-list" id="mq-favs"></ul>';
  oldCard.parentNode.insertBefore(card, oldCard);

  function q(id) { return card.querySelector('#' + id); }
  function el(tag, cls, text) { var e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; }
  function btn(label, aria, fn) { var b = el('button', 'btn ghost', label); b.type = 'button'; b.setAttribute('aria-label', aria); b.onclick = fn; return b; }

  function call(path, body, method) {
    var p = method === 'GET' ? fetch(path, { cache: 'no-store' }).then(function (r) { return r.json(); }) : api.post(path, body);
    return p.then(function (r) {
      if (r && r.error) api.toast(r.error); else if (r && r.say && method !== 'GET') api.toast(r.say);
      return r;
    }).catch(function (e) { api.toast('Request failed: ' + e.message); });
  }
  function queueAct(body) { call('/api/music/queue', body).then(function () { refresh(); }); }
  function favAct(body) { call('/api/music/favorite', body).then(loadFavs); }

  q('mq-shuf').onclick = function () { queueAct({ action: 'shuffle' }); };
  q('mq-clear').onclick = function () { queueAct({ action: 'clear' }); };
  q('mq-fav').onclick = function () { favAct({ action: 'save' }); };
  q('mq-favplay').onclick = function () { favAct({ action: 'play' }); };
  q('mq-auto').onclick = function () {
    var on = q('mq-auto').getAttribute('aria-pressed') === 'true';
    queueAct({ action: on ? 'autoplay_off' : 'autoplay_on' });
  };

  function refresh() { if (api.state()) render(api.state()); }

  function render(st) {
    var m = st && st.music; if (!m) return;
    var items = m.queue_items || [];
    q('mq-n').textContent = items.length;
    var auto = !!m.autoplay;
    var ab = q('mq-auto'); ab.setAttribute('aria-pressed', auto); ab.textContent = 'Keep going: ' + (auto ? 'on' : 'off');
    q('mq-fav').disabled = m.state === 'idle';
    var key = JSON.stringify(items);
    var list = q('mq-list');
    if (list.dataset.key === key) return;
    list.dataset.key = key;
    list.replaceChildren();
    items.forEach(function (it) {
      var li = el('li'), t = el('div', 'mq-t', it.title || '?');
      if (it.artist) t.appendChild(el('small', '', it.artist));
      li.appendChild(t);
      li.appendChild(btn('↑', 'Play next', function () { queueAct({ action: 'play_next', id: it.id }); }));
      li.appendChild(btn('✕', 'Remove from queue', function () { queueAct({ action: 'remove', id: it.id }); }));
      list.appendChild(li);
    });
    if (!items.length) list.appendChild(el('li', 'empty', 'Nothing queued.'));
  }
  api.subscribe(render);

  function loadFavs() {
    call('/api/music/favorites', undefined, 'GET').then(function (r) {
      var favs = (r && r.favorites) || [], ul = q('mq-favs');
      q('mq-fn').textContent = favs.length;
      q('mq-favplay').disabled = !favs.length;
      ul.replaceChildren();
      favs.slice().reverse().forEach(function (f) {
        var id = f.id || f.title, li = el('li'), t = el('div', 'mq-t', f.title || '?');
        if (f.artist) t.appendChild(el('small', '', f.artist));
        li.appendChild(t);
        li.appendChild(btn('▶', 'Queue this favorite', function () { favAct({ action: 'play', id: id }); }));
        li.appendChild(btn('✕', 'Remove favorite', function () { favAct({ action: 'remove', id: id }); }));
        ul.appendChild(li);
      });
      if (!favs.length) ul.appendChild(el('li', 'empty', 'No favorites yet. Tap ♥ while a song plays.'));
    });
  }
  loadFavs();

  /* click the progress bar to jump to that point */
  var bar = document.getElementById('progbar');
  if (bar) {
    bar.title = 'Click to jump';
    bar.addEventListener('click', function (ev) {
      var st = api.state(), m = st && st.music;
      if (!m || m.state === 'idle' || !m.duration || m.is_live) return;
      var r = bar.getBoundingClientRect();
      var secs = Math.round(Math.max(0, Math.min(1, (ev.clientX - r.left) / r.width)) * m.duration);
      call('/api/music/seek', { to: Math.floor(secs / 60) + ':' + String(secs % 60).padStart(2, '0') });
    });
  }
})();
