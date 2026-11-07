/* Settings sub-tabs: shows one group of Settings sections at a time instead of one long page.
   Desktop: a sticky list of sections on the left. Phone/laptop: a row of chips that scrolls sideways.
   Sections are matched by id (some are built later by other scripts, so it re-applies when they appear).
   Opening an accordion from elsewhere (e.g. the Gamepad card's "Settings > Experimental" link sets
   #acc-experimental.open) switches to its group. Remembers the last group per browser. Vanilla JS. */
(function () {
  'use strict';
  var GROUPS = [
    { id: 'sound', label: 'Sound', ids: ['sc-sound', 'eq-card'] },
    { id: 'voice', label: 'Voice & AI', ids: ['sc-backend', 'sc-gemini', 'acc-quick'] },
    { id: 'memory', label: 'Memory', ids: ['memory-card'] },
    { id: 'frc', label: 'FRC', ids: ['sc-frc'] },
    { id: 'google', label: 'Google', ids: ['acc-google'], open: true },
    { id: 'experimental', label: 'Experimental', ids: ['acc-experimental'], open: true },
    { id: 'diag', label: 'Diagnostics', ids: ['acc-diag'], open: true }
  ];
  var KEY = 'reachy-settings-group';
  var panel, nav, buttons = {}, active = 'sound', applying = false;

  function groupOf(id) {
    for (var i = 0; i < GROUPS.length; i++) if (GROUPS[i].ids.indexOf(id) >= 0) return GROUPS[i];
    return null;
  }
  function sections() {
    // the Diagnostics accordion has no id in the page markup: name it so it can be grouped
    var accs = panel.querySelectorAll(':scope > details.acc');
    var last = accs.length ? accs[accs.length - 1] : null;
    if (last && !last.id) last.id = 'acc-diag';
    var out = [];
    GROUPS.forEach(function (g) { g.ids.forEach(function (id) { var el = document.getElementById(id); if (el) out.push(el); }); });
    return out;
  }

  function apply() {
    if (!panel) return;
    applying = true;
    GROUPS.forEach(function (grp) {
      var on = grp.id === active;
      grp.ids.forEach(function (id) {
        var el = document.getElementById(id);
        if (!el) return;
        el.classList.toggle('sn-off', !on);
        if (on && grp.open && el.tagName === 'DETAILS') el.open = true;
      });
      var b = buttons[grp.id];
      if (b) { b.setAttribute('aria-selected', on ? 'true' : 'false'); b.tabIndex = on ? 0 : -1; }
    });
    applying = false;
  }

  function select(id, focus) {
    if (!GROUPS.some(function (g) { return g.id === id; })) return;
    active = id;
    try { localStorage.setItem(KEY, id); } catch (e) { /* not remembered */ }
    apply();
    if (focus && buttons[id]) buttons[id].focus();
  }

  function build() {
    panel = document.getElementById('p-settings');
    if (!panel) return false;
    nav = document.createElement('nav');
    nav.className = 'sn-nav';
    nav.setAttribute('aria-label', 'Settings sections');
    var list = document.createElement('div');
    list.className = 'sn-list';
    list.setAttribute('role', 'tablist');
    list.setAttribute('aria-orientation', 'vertical');
    GROUPS.forEach(function (g) {
      var b = document.createElement('button');
      b.type = 'button';
      b.className = 'sn-tab';
      b.id = 'sn-' + g.id;
      b.setAttribute('role', 'tab');
      b.textContent = g.label;
      b.addEventListener('click', function () { select(g.id); });
      buttons[g.id] = b;
      list.appendChild(b);
    });
    list.addEventListener('keydown', function (e) {
      var i = GROUPS.findIndex(function (g) { return g.id === active; });
      if (e.key === 'ArrowDown' || e.key === 'ArrowRight') i = (i + 1) % GROUPS.length;
      else if (e.key === 'ArrowUp' || e.key === 'ArrowLeft') i = (i - 1 + GROUPS.length) % GROUPS.length;
      else if (e.key === 'Home') i = 0;
      else if (e.key === 'End') i = GROUPS.length - 1;
      else return;
      e.preventDefault();
      select(GROUPS[i].id, true);
    });
    nav.appendChild(list);
    panel.insertBefore(nav, panel.firstChild);

    try { var saved = localStorage.getItem(KEY); if (saved) active = saved; } catch (e) { /* default group */ }
    if (!GROUPS.some(function (g) { return g.id === active; })) active = 'sound';

    // an accordion opened from elsewhere (deep link) brings its group to the front
    panel.addEventListener('toggle', function (e) {
      var el = e.target;
      if (applying || !el.open || !el.id) return;
      var g = groupOf(el.id);
      if (g && g.id !== active) select(g.id);
    }, true);
    // sections built later by other scripts (Music EQ, Experimental, the #settings-panel cards)
    new MutationObserver(function () { sections(); apply(); }).observe(panel, { childList: true, subtree: true });
    sections();
    apply();
    return true;
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', build); else build();
})();
