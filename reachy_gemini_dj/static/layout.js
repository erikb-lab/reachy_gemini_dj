/* Responsive layout for the DJ tab. Loaded last. Moves the DJ cards (some are added by other scripts) into three
   fixed columns; layout.css decides how those columns show on a phone (one list), laptop (two) or desktop (three).
   Cards are matched by their heading id, so other scripts need no changes. Vanilla JS, no dependencies. */
(function () {
  'use strict';
  var grid = document.querySelector('#p-dj .grid');
  if (!grid) return;
  var COLS = [
    ['a', ['h-ptt', 'h-np', 'h-search', 'h-mq', 'h-q']],   // talk + music
    ['b', ['h-mic', 'h-cv']],                              // what Reachy heard and said
    ['c', ['h-tm', 'h-im', 'h-ev']]                         // timers, sensors, events
  ];
  var cols = {};
  COLS.forEach(function (c) {
    var d = document.createElement('div');
    d.className = 'col col-' + c[0];
    cols[c[0]] = d;
    grid.appendChild(d);
  });

  function card(id) { return grid.querySelector('section[aria-labelledby="' + id + '"]'); }

  function place() {
    COLS.forEach(function (c) {
      c[1].forEach(function (id) {
        var s = card(id);
        if (s) cols[c[0]].appendChild(s);   // appending in list order keeps the order fixed
      });
    });
    // anything new and unknown lands at the end of the last column
    Array.prototype.slice.call(grid.children).forEach(function (el) {
      if (!el.classList.contains('col')) cols.c.appendChild(el);
    });
  }

  place();
  // a card added later straight into the grid gets placed too
  new MutationObserver(function () {
    var stray = Array.prototype.some.call(grid.children, function (el) { return !el.classList.contains('col'); });
    if (stray) place();
  }).observe(grid, { childList: true });
})();
