/* Reachy Gemini DJ controller UI. Vanilla JS, no dependencies. */
(function () {
'use strict';
const $ = id => document.getElementById(id);
const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; };

let S = null, lastState = null, fails = 0, lastOk = 0, picking = null, gSel = { calendar: true, tasks: true, gmail: false };
const subs = new Set();
const localQueue = [];            // queries sent from this page, shown until music state shows them consumed
let lastQueued = 0;

/* ---------- public API for robot.js / settings.js ---------- */
window.reachyApi = {
  state: () => lastState,
  subscribe(fn) { subs.add(fn); if (lastState) { try { fn(lastState); } catch (e) { console.error(e); } } return () => subs.delete(fn); },
  post: (path, body) => api(path, body === undefined ? {} : body),
  toast,
};

async function api(path, body) {
  const r = await fetch(path, body === undefined ? { cache: 'no-store' } :
    { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  if (!r.ok) throw new Error('HTTP ' + r.status);
  return r.json();
}
function toast(t) { const e = $('toast'); e.textContent = t; e.classList.add('on'); clearTimeout(toast.t); toast.t = setTimeout(() => e.classList.remove('on'), 2600); }
const post = (path, body, ok) => api(path, body).then(r => { if (ok) toast(ok); poll(); return r; }).catch(e => { toast('Request failed: ' + e.message); });
const fmt = s => { s = Math.max(0, Math.floor(s || 0)); return Math.floor(s / 60) + ':' + String(s % 60).padStart(2, '0'); };

/* ---------- tabs ---------- */
const tabs = ['dj', 'robot', 'settings'];
function showTab(name, focus) {
  tabs.forEach(t => {
    const b = $('tab-' + t), on = t === name;
    b.setAttribute('aria-selected', on); b.tabIndex = on ? 0 : -1; $('p-' + t).classList.toggle('hide', !on);
    if (on && focus) b.focus();
  });
  try { localStorage.setItem('reachy_tab', name); } catch (e) {}
}
tabs.forEach(t => $('tab-' + t).addEventListener('click', () => showTab(t)));
$('tabs').addEventListener('keydown', e => {
  const i = tabs.findIndex(t => $('tab-' + t).getAttribute('aria-selected') === 'true');
  if (e.key === 'ArrowRight') showTab(tabs[(i + 1) % 3], true);
  else if (e.key === 'ArrowLeft') showTab(tabs[(i + 2) % 3], true);
});
try { const t = localStorage.getItem('reachy_tab'); if (tabs.includes(t)) showTab(t); } catch (e) {}

/* ---------- placeholders for the other panels ---------- */
['robot-panel', 'settings-panel'].forEach(id => {
  const m = $(id);
  new MutationObserver(() => {
    const real = [...m.children].some(c => !c.hasAttribute('data-placeholder'));
    m.querySelectorAll('[data-placeholder]').forEach(p => { if (real) p.remove(); });
  }).observe(m, { childList: true });
});
function scriptMissing(src, mountId, what) {
  const s = document.querySelector('script[src="' + src + '"]');
  if (!s) return;
  s.addEventListener('error', () => { const p = $(mountId).querySelector('[data-placeholder]'); if (p) p.textContent = what + ' panel is not installed yet.'; });
}
scriptMissing('/static/robot.js', 'robot-panel', 'Robot');
scriptMissing('/static/settings.js', 'settings-panel', 'Settings');

/* ---------- transport ---------- */
$('b-play').onclick = () => {
  const st = S && S.music.state;
  post('/api/music/' + (st === 'playing' ? 'pause' : 'resume'), {});
};
$('b-skip').onclick = () => post('/api/music/skip', {});
$('b-stop').onclick = () => post('/api/music/stop', {});
$('f-play').addEventListener('submit', e => {
  e.preventDefault();
  const i = $('track_q'), q = i.value.trim(); if (!q) return;
  api('/api/music/play', { query: q }).then(r => {
    i.value = '';
    if (r.error) toast(r.error); else { localQueue.push(q); toast(r.result || 'Queued'); }
    poll();
  }).catch(err => toast('Request failed: ' + err.message));
});
$('b-mute').onclick = () => post('/api/mute', { muted: !(S && S.muted) });
$('b-newconv').onclick = () => post('/api/new_conversation', {}, 'Fresh conversation started');

/* ---------- settings (built-in quick settings) ---------- */
const SL = { music_volume: v => Math.round(v * 100) + '%', duck_level: v => Math.round(v * 100) + '%', voice_gain: v => v.toFixed(2) + 'x', echo_guard: v => v.toFixed(1) };
Object.keys(SL).forEach(id => {
  const inp = $(id), out = id === 'music_volume' ? $('o-vol') : inp.parentNode.querySelector('output');
  inp.addEventListener('input', () => { picking = id; out.textContent = SL[id](+inp.value); });
  inp.addEventListener('change', () => api('/api/config', { [id]: +inp.value }).catch(e => toast('Save failed: ' + e.message)).finally(() => { picking = null; }));
});
document.querySelectorAll('input[data-cfg]').forEach(c => c.addEventListener('change', () =>
  post('/api/config', { [c.dataset.cfg]: c.checked }, 'Saved')));
$('q_voice').addEventListener('change', e => post('/api/config', { voice: e.target.value }, 'Voice changed'));
$('b-key').onclick = () => { const v = $('q_api_key').value.trim(); if (!v) return toast('Paste a key first'); api('/api/config', { api_key: v }).then(() => { $('q_api_key').value = ''; toast('Key saved'); poll(); }).catch(e => toast('Save failed: ' + e.message)); };
$('b-text').onclick = () => post('/api/config', { extra_instructions: $('q_extra_instructions').value, model: $('q_model').value }, 'Saved');

/* ---------- google ---------- */
function gErr(t) { const e = $('g-err'); e.textContent = t || ''; e.classList.toggle('hide', !t); }
function renderGoogle(g) {
  $('g-on').classList.toggle('hide', !g.connected);
  $('g-setup').classList.toggle('hide', g.connected && !g.needs_reconnect);
  $('g-reconnect').classList.toggle('hide', !g.needs_reconnect);
  $('g-badge').textContent = g.connected ? (g.needs_reconnect ? 'reconnect' : 'connected') : 'not connected';
  $('g-who').textContent = g.email ? 'Signed in as ' + g.email : 'Signed in';
  $('g-svcs').replaceChildren(...(g.services || []).map(s => el('span', 'chip', '✓ ' + (g.available[s] || s))));
  const sv = $('g-saved'); sv.classList.toggle('hide', !g.client_set); sv.textContent = g.client_set ? 'Saved client ' + g.client_id_hint + '… (paste again to replace)' : '';
  const pick = $('g-pick'), n = String(Object.keys(g.available || {}).length);
  if (!pick.children.length || pick.dataset.n !== n) {
    pick.dataset.n = n;
    pick.replaceChildren(...Object.entries(g.available || {}).map(([k, label]) => {
      const b = el('button', 'chip'); b.type = 'button';
      const paint = () => { b.textContent = (gSel[k] ? '✓ ' : '') + label; b.setAttribute('aria-pressed', !!gSel[k]); };
      b.onclick = () => { gSel[k] = !gSel[k]; paint(); }; paint(); return b;
    }));
  }
}
$('b-gstart').onclick = async () => {
  gErr('');
  try {
    const id = $('gid').value.trim(), sec = $('gsec').value.trim();
    if (id || sec) { const r = await api('/api/google/client', { client_id: id, client_secret: sec }); if (!r.ok) return gErr(r.error); }
    const svc = Object.keys(gSel).filter(k => gSel[k]); if (!svc.length) return gErr('Pick at least one thing to connect.');
    const r = await api('/api/google/start', { services: svc }); if (!r.ok) return gErr(r.error);
    window.open(r.url, '_blank', 'noopener'); toast('Approve in the Google tab, then come back'); $('gurl').focus();
  } catch (e) { gErr('Request failed: ' + e.message); }
};
$('b-gfinish').onclick = async () => {
  gErr(''); const u = $('gurl').value.trim(); if (!u) return gErr('Paste the address first.');
  try {
    const r = await api('/api/google/finish', { redirect_url: u }); if (!r.ok) return gErr(r.error);
    $('gurl').value = ''; $('gid').value = ''; $('gsec').value = ''; toast('Google connected'); poll();
  } catch (e) { gErr('Request failed: ' + e.message); }
};
$('b-gdis').onclick = async () => { if (!confirm('Disconnect Google from Reachy?')) return; await post('/api/google/disconnect', {}); };

/* ---------- rendering ---------- */
const BACKEND = { listening: ['ok', 'Listening'], connecting: ['warn', 'Connecting…'], error: ['bad', 'Problem'], needs_key: ['warn', 'Needs API key'], starting: ['warn', 'Starting…'] };
function setStat(id, cls, text) { const e = $(id); e.firstElementChild.className = 'dot ' + cls; e.lastElementChild.textContent = text; }

function renderConn() {
  const stale = fails >= 3;
  if (fails === 0) setStat('st-conn', 'ok', 'Online');
  else setStat('st-conn', stale ? 'bad' : 'warn', stale ? 'Offline' : 'Reconnecting…');
  const b = $('banner');
  b.classList.toggle('hide', !stale);
  b.className = stale ? 'bad' : 'hide';
  if (stale) b.textContent = 'Cannot reach the robot app' + (lastOk ? ' (last update ' + Math.round((Date.now() - lastOk) / 1000) + 's ago)' : '') + '. Retrying…';
  ['b-play', 'b-skip', 'b-stop'].forEach(i => { if (stale) $(i).disabled = true; });
}

function render() {
  const c = S.config, m = S.music;
  const [cls, txt] = BACKEND[S.gemini] || ['warn', S.gemini];
  setStat('st-backend', cls, txt);
  $('st-model').firstElementChild.textContent = 'model: ' + (S.model || '?');
  const err = S.error || (m && m.error) || '';
  $('st-err').classList.toggle('hide', !err); $('st-err').lastElementChild.textContent = err;

  /* now playing */
  const idle = m.state === 'idle';
  $('trk').textContent = idle ? 'Nothing playing' : (m.track || 'Loading…');
  $('art').textContent = idle ? 'Ask Reachy for some music' : ((m.artist || '') + (m.state === 'paused' ? ' · paused' : '') + (m.state === 'loading' ? ' · loading' : ''));
  const pct = m.duration ? Math.min(100, 100 * m.elapsed / m.duration) : 0;
  $('prog').style.width = pct + '%'; $('progbar').setAttribute('aria-valuenow', Math.round(pct));
  $('t-el').textContent = fmt(m.elapsed); $('t-du').textContent = m.duration ? fmt(m.duration) : '--:--';
  const playing = m.state === 'playing';
  $('disc').classList.toggle('spin', playing);
  const bp = $('b-play'); bp.innerHTML = playing ? '&#10074;&#10074;' : '&#9654;';
  bp.setAttribute('aria-label', playing ? 'Pause' : 'Play'); bp.disabled = idle || m.state === 'loading';
  $('b-skip').disabled = idle; $('b-stop').disabled = idle;
  const me = $('merr'); me.textContent = m.error || ''; me.classList.toggle('hide', !m.error);

  /* queue: backend exposes a count (and a list if present); otherwise show what this page queued */
  const q = m.queue_items || m.queue_list || (Array.isArray(m.queue) ? m.queue : null);
  const n = q ? q.length : (m.queued || 0);
  while (!q && localQueue.length > n) localQueue.shift();
  $('qn').textContent = n;
  const ql = $('queue'); ql.replaceChildren();
  if (q) q.forEach(i => ql.append(el('li', '', typeof i === 'string' ? i : (i.title || i.query || JSON.stringify(i)))));
  else {
    for (let i = 0; i < n; i++) ql.append(el('li', '', localQueue[localQueue.length - n + i] || 'Queued request ' + (i + 1)));
  }
  if (!n) ql.append(el('li', 'empty', 'Nothing queued.'));

  /* mic + transcript */
  const mb = $('b-mute'); mb.setAttribute('aria-pressed', !!S.muted); mb.textContent = S.muted ? 'Unmute mic' : 'Mute mic';
  const lvl = Math.min(100, (S.mic_level || 0) * 900);
  $('lvl').style.width = (S.muted ? 0 : lvl) + '%'; $('meter').classList.toggle('muted', !!S.muted);
  $('meter').setAttribute('aria-valuenow', Math.round(S.muted ? 0 : lvl));
  $('heard').textContent = S.heard || ''; $('said').textContent = S.said || '';

  /* events */
  const f = $('feed'); f.replaceChildren();
  (S.events || []).slice(0, 40).forEach(e => { const li = el('li', e.kind === 'error' ? 'error' : ''); li.append(el('time', '', e.t), el('span', '', e.text)); f.append(li); });
  if (!(S.events || []).length) f.append(el('li', 'empty', 'Nothing yet.'));

  /* settings */
  if (picking !== 'music_volume') { $('music_volume').value = c.music_volume; $('o-vol').textContent = SL.music_volume(+c.music_volume); }
  ['duck_level', 'voice_gain', 'echo_guard'].forEach(id => { if (picking === id) return; const i = $(id); i.value = c[id]; i.parentNode.querySelector('output').textContent = SL[id](+c[id]); });
  document.querySelectorAll('input[data-cfg]').forEach(x => { if (document.activeElement !== x) x.checked = !!c[x.dataset.cfg]; });
  const vs = $('q_voice'); if (!vs.options.length) (S.voices || []).forEach(v => vs.add(new Option(v, v)));
  if (document.activeElement !== vs) vs.value = c.voice;
  [['q_extra_instructions', c.extra_instructions], ['q_model', c.model]].forEach(([id, v]) => { if (document.activeElement !== $(id)) $(id).value = v || ''; });
  const ks = $('keystate'); ks.textContent = c.api_key_set ? '✓ saved' : '⚠ not set'; ks.style.color = c.api_key_set ? 'var(--ok)' : 'var(--warn)';
  $('curmodel').textContent = '(using ' + S.model + ')';
  renderGoogle(S.google);
  const mo = S.motion || {};
  $('diag').textContent = 'Gemini: ' + S.gemini + (S.error ? ' (' + S.error + ')' : '') + ' | Audio decoder: ' + (S.ffmpeg ? 'ready' : 'missing') +
    ' | Emotions: ' + (mo.emotions_ready ? 'ready' : 'loading') + ' | Dances: ' + (mo.dances_ready ? 'ready' : 'loading') + ' | v' + (S.version || '');
}

/* ---------- polling ---------- */
let inflight = false;
async function poll() {
  if (inflight) return; inflight = true;
  try {
    const s = await api('/api/state');
    S = lastState = s; fails = 0; lastOk = Date.now();
    render(); renderConn();
    subs.forEach(fn => { try { fn(s); } catch (e) { console.error(e); } });
  } catch (e) {
    fails++; renderConn();
  } finally { inflight = false; }
}
renderConn();
poll(); setInterval(poll, 1000);
document.addEventListener('visibilitychange', () => { if (!document.hidden) poll(); });
})();
