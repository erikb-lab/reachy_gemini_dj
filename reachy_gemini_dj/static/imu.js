/* Sensors card: tilt, posture, moving/still and internal temperature, from state.imu (the watcher's latest values).
   Self-contained; temperature turns amber at 70 C or above. Vanilla JS, no dependencies. */
(function () {
  'use strict';
  var api = window.reachyApi;
  var feed = document.getElementById('feed');
  if (!api || !feed) return;
  var anchor = feed.closest('section') || feed.parentNode;

  var css = document.createElement('style');
  css.textContent =
    '.im-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(110px,1fr));gap:12px;margin:0}' +
    '.im-grid div{display:grid;gap:2px}.im-grid dt{color:var(--mut);font-size:13px}' +
    '.im-grid dd{margin:0;font-weight:700;font-size:15px;font-variant-numeric:tabular-nums}' +
    '.im-grid dd.hot{color:var(--warn)}.im-none{color:var(--mut);margin:0}';
  document.head.appendChild(css);

  var card = document.createElement('section');
  card.className = 'card';
  card.setAttribute('aria-labelledby', 'h-im');
  card.innerHTML = '<h2 id="h-im">Sensors</h2>' +
    '<p class="im-none" id="im-none">Motion sensor not available.</p>' +
    '<dl class="im-grid hide" id="im-grid" aria-live="off">' +
    '<div><dt>Tilt</dt><dd id="im-tilt">-</dd></div>' +
    '<div><dt>Posture</dt><dd id="im-pose">-</dd></div>' +
    '<div><dt>Motion</dt><dd id="im-move">-</dd></div>' +
    '<div><dt>Temperature</dt><dd id="im-temp">-</dd></div></dl>';
  anchor.parentNode.insertBefore(card, anchor);

  var none = card.querySelector('#im-none'), grid = card.querySelector('#im-grid');
  var tilt = card.querySelector('#im-tilt'), pose = card.querySelector('#im-pose');
  var move = card.querySelector('#im-move'), temp = card.querySelector('#im-temp');

  api.subscribe(function (s) {
    var d = s.imu || {};
    var ok = !!d.available;
    none.classList.toggle('hide', ok);
    grid.classList.toggle('hide', !ok);
    if (!ok) return;
    tilt.textContent = Math.round(d.tilt_deg) + '°';
    pose.textContent = d.posture;
    move.textContent = d.held ? 'Being held' : (d.moving ? 'Moving' : 'Still');
    if (d.temperature_c == null) {
      temp.textContent = '-';
      temp.className = '';
    } else {
      temp.textContent = Math.round(d.temperature_c) + ' °C' + (d.hot ? ' (hot)' : '');
      temp.className = d.hot ? 'hot' : '';
    }
  });
})();
