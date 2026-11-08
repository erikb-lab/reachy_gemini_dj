/* 360° photo card (Robot tab) and fullscreen 360° viewer. The card starts a capture with POST /api/panorama/capture,
   follows GET /api/panorama/status (about every 700 ms while capturing or stitching, otherwise only when the Robot
   tab opens), and lists finished photos from GET /api/panorama/list. The viewer is a small WebGL2 renderer: one
   full-screen triangle, the view ray per pixel is computed in the fragment shader and the equirectangular image is
   sampled by longitude and latitude (exact at the poles, no seam at the wrap). Without WebGL2 it shows the flat image
   in a horizontally scrolling box. Vanilla JS, no dependencies, no network use beyond the app's own API. */
(function () {
  'use strict';
  var API = '/api/panorama/';
  var els = {}, items = [], timer = 0, wasActive = false, cancelling = false;

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
  function api(path, opts) {
    opts = opts || {};
    if (opts.method === 'POST') { opts.headers = { 'Content-Type': 'application/json' }; opts.body = opts.body || '{}'; }
    return fetch(API + path, Object.assign({ cache: 'no-store' }, opts)).then(function (r) { return r.json(); });
  }
  function fmtDate(iso, id) {
    var d = iso ? new Date(iso) : null;
    if (!d || isNaN(d.getTime())) return id || '';
    try { return d.toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' }); } catch (e) { return d.toLocaleString(); }
  }
  function fmtSize(b) {
    if (!b && b !== 0) return '';
    return b >= 1048576 ? (b / 1048576).toFixed(1) + ' MB' : Math.max(1, Math.round(b / 1024)) + ' kB';
  }
  function fmtEta(s) {
    if (s == null || !isFinite(s)) return '';
    s = Math.max(0, Math.round(s));
    return s >= 60 ? 'about ' + Math.floor(s / 60) + ' min ' + (s % 60) + ' s left' : 'about ' + s + ' s left';
  }

  /* ---------------------------------------------------------------- card */
  function build() {
    var root = document.getElementById('robot-panel');
    if (!root) return false;
    var card = h('div', { class: 'pn-card', id: 'rp-panorama' });
    card.appendChild(h('h2', { text: '360° photos' }));
    els.take = h('button', { type: 'button', class: 'pn-btn primary', text: 'Take 360° photo', onclick: start });
    els.cancel = h('button', { type: 'button', class: 'pn-btn quiet', text: 'Cancel', onclick: cancel, hidden: '' });
    card.appendChild(h('div', { class: 'pano-row' }, [els.take, els.cancel]));
    els.bar = h('div', { class: 'pano-bar', role: 'progressbar', 'aria-label': 'Progress', 'aria-valuemin': '0', 'aria-valuemax': '100', 'aria-valuenow': '0' }, [h('i')]);
    els.step = h('span', { class: 'pano-step' });
    els.eta = h('span', { class: 'pano-eta pn-muted' });
    els.prog = h('div', { class: 'pano-prog', hidden: '' }, [els.bar, h('div', { class: 'pano-prog-text', role: 'status', 'aria-live': 'polite' }, [els.step, els.eta])]);
    card.appendChild(els.prog);
    els.err = h('div', { class: 'pano-err', role: 'alert', hidden: '' });
    card.appendChild(els.err);
    card.appendChild(h('p', { class: 'pn-muted', text: 'The robot turns all the way around, taking about 34 sharp full-resolution photos (live video pauses for about 1 minute), then stitches them on the robot. Stand clear of it ' +
      'and step out of the shot, or you will show up in the photo.' }));
    els.count = h('span', { class: 'pano-count' });
    card.appendChild(h('h3', {}, ['Gallery ', els.count]));
    els.empty = h('p', { class: 'pn-muted pano-empty', text: 'No 360° photos yet. Take one and it will show up here.' });
    els.grid = h('div', { class: 'pano-grid' });
    card.appendChild(els.empty);
    card.appendChild(els.grid);
    root.appendChild(card);
    return true;
  }

  function paintGallery() {
    els.grid.textContent = '';
    els.count.textContent = items.length ? String(items.length) : '';
    els.empty.hidden = items.length > 0;
    items.forEach(function (it) {
      var when = fmtDate(it.created, it.id);
      var meta = [it.width && it.height ? it.width + '×' + it.height : '', it.frames ? it.frames + ' photos' : '', fmtSize(it.size_bytes)]
        .filter(Boolean).join(' · ');
      var thumb = h('button', { type: 'button', class: 'pano-thumb', 'aria-label': 'View 360° photo from ' + when, onclick: function (ev) { openViewer(it, ev.currentTarget); } },
        [h('img', { src: it.thumb_url, alt: '', loading: 'lazy', draggable: 'false' }), h('span', { class: 'pano-badge', text: '360°' })]);
      var del = h('button', { type: 'button', class: 'pn-btn sm danger', text: 'Delete' });
      var delTimer = 0;
      del.addEventListener('click', function () {
        if (!del.dataset.sure) {
          del.dataset.sure = '1'; del.textContent = 'Really delete?';
          delTimer = setTimeout(function () { delete del.dataset.sure; del.textContent = 'Delete'; }, 4000);
          return;
        }
        clearTimeout(delTimer); del.disabled = true;
        api(it.id, { method: 'DELETE' }).then(function (r) {
          if (!r.ok) throw new Error(r.error || 'not deleted');
          toast('Photo deleted'); return refreshList();
        }).catch(function (e) { del.disabled = false; toast('Could not delete: ' + e.message, true); });
      });
      var dl = h('a', { class: 'pn-btn sm quiet', href: it.url, download: 'panorama-' + it.id + '.jpg', text: 'Download' });
      var view = h('button', { type: 'button', class: 'pn-btn sm secondary', text: 'View', onclick: function (ev) { openViewer(it, ev.currentTarget); } });
      els.grid.appendChild(h('div', { class: 'pano-item' }, [thumb,
        h('div', { class: 'pano-info' }, [h('div', { class: 'pano-when', text: when }), h('div', { class: 'pn-muted pano-meta', text: meta })]),
        h('div', { class: 'pano-actions' }, [view, dl, del])]));
    });
  }

  function refreshList() {
    return api('list').then(function (r) { items = r.items || []; paintGallery(); })
      .catch(function () { /* keep the old list */ });
  }

  function paintStatus(st) {
    var active = st.state === 'capturing' || st.state === 'stitching';
    els.take.disabled = active;
    els.take.textContent = active ? (st.state === 'stitching' ? 'Stitching…' : 'Capturing…') : 'Take 360° photo';
    els.cancel.hidden = st.state !== 'capturing';
    els.cancel.disabled = cancelling;
    els.prog.hidden = !active;
    if (active) {
      var pct = Math.round(Math.max(0, Math.min(1, st.progress || 0)) * 100);
      els.bar.firstChild.style.width = pct + '%';
      els.bar.setAttribute('aria-valuenow', String(pct));
      els.bar.classList.toggle('indet', st.state === 'stitching' && !(st.progress > 0));
      els.step.textContent = (st.step || (st.state === 'stitching' ? 'Stitching the photos' : 'Capturing')) + (pct ? ' · ' + pct + '%' : '');
      els.eta.textContent = fmtEta(st.eta_s);
    }
    if (st.state === 'error' && st.error) { els.err.hidden = false; els.err.textContent = st.error; }
    else if (st.state !== 'error') els.err.hidden = true;
  }

  function check() {
    return api('status').then(function (st) {
      var active = st.state === 'capturing' || st.state === 'stitching';
      paintStatus(st);
      if (wasActive && !active) {
        cancelling = false;
        if (st.state === 'done') {
          toast('360° photo ready');
          refreshList().then(function () {
            var it = items.filter(function (x) { return x.id === st.id; })[0];
            if (it) openViewer(it, els.take);
          });
        } else if (st.state === 'error') { toast(st.error || 'The 360° photo failed', true); refreshList(); }
        else { toast('Capture stopped'); refreshList(); }
      }
      wasActive = active;
      clearTimeout(timer);
      if (active) timer = setTimeout(check, document.hidden ? 2000 : 700);
    }).catch(function () {
      clearTimeout(timer);
      if (wasActive) timer = setTimeout(check, 2000);
    });
  }

  function start() {
    els.take.disabled = true; els.err.hidden = true;
    api('capture', { method: 'POST' }).then(function (r) {
      if (!r.ok) { els.take.disabled = false; els.err.hidden = false; els.err.textContent = r.error || 'Could not start the capture.'; return; }
      wasActive = true; paintStatus({ state: 'capturing', progress: 0, step: 'Starting…' });
      check();
    }).catch(function (e) { els.take.disabled = false; els.err.hidden = false; els.err.textContent = 'Could not reach the app (' + e.message + ')'; });
  }
  function cancel() {
    cancelling = true; els.cancel.disabled = true;
    api('cancel', { method: 'POST' }).then(function () { check(); })
      .catch(function (e) { cancelling = false; els.cancel.disabled = false; toast('Could not cancel (' + e.message + ')', true); });
  }

  /* -------------------------------------------------------------- viewer */
  var V = null; // state of the open viewer, null when closed

  var VS = '#version 300 es\nin vec2 p;out vec2 v;void main(){v=p;gl_Position=vec4(p,0.,1.);}';
  var FS = '#version 300 es\nprecision highp float;in vec2 v;out vec4 o;uniform sampler2D tex;uniform vec2 ang;uniform vec2 sc;\n' +
    'const float PI=3.14159265358979;\n' +
    'void main(){float cy=cos(ang.x),sy=sin(ang.x),cp=cos(ang.y),sp=sin(ang.y);\n' +
    ' vec3 f=vec3(sy*cp,sp,cy*cp),r=vec3(cy,0.,-sy),u=vec3(-sy*sp,cp,-cy*sp);\n' +
    ' vec3 d=normalize(f+v.x*sc.x*r+v.y*sc.y*u);\n' +
    ' vec2 uv=vec2(atan(d.x,d.z)/(2.*PI)+.5,.5-asin(clamp(d.y,-1.,1.))/PI);\n' +
    ' vec2 gx=dFdx(uv),gy=dFdy(uv);gx.x-=floor(gx.x+.5);gy.x-=floor(gy.x+.5);\n' + // undo the jump of atan at the wrap
    ' o=vec4(textureGrad(tex,uv,gx,gy).rgb,1.);}';

  function compile(gl, type, src) {
    var s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
    return s;
  }
  function initGL(canvas) {
    var gl = canvas.getContext('webgl2', { antialias: false, alpha: false });
    if (!gl) return null;
    var pr = gl.createProgram();
    gl.attachShader(pr, compile(gl, gl.VERTEX_SHADER, VS));
    gl.attachShader(pr, compile(gl, gl.FRAGMENT_SHADER, FS));
    gl.linkProgram(pr);
    if (!gl.getProgramParameter(pr, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(pr));
    gl.useProgram(pr);
    var buf = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
    var loc = gl.getAttribLocation(pr, 'p'); gl.enableVertexAttribArray(loc); gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
    return { gl: gl, ang: gl.getUniformLocation(pr, 'ang'), sc: gl.getUniformLocation(pr, 'sc') };
  }
  function uploadTexture(g, img) {
    var gl = g.gl, w = img.naturalWidth, hh = img.naturalHeight;
    var max = Math.min(gl.getParameter(gl.MAX_TEXTURE_SIZE), 8192), src = img;
    if (w > max || hh > max) { // too big for this GPU: shrink through a 2D canvas
      var k = max / Math.max(w, hh), c = document.createElement('canvas');
      c.width = Math.max(1, Math.floor(w * k)); c.height = Math.max(1, Math.floor(hh * k));
      var cx = c.getContext('2d'); cx.imageSmoothingQuality = 'high'; cx.drawImage(img, 0, 0, c.width, c.height);
      src = c;
    }
    var t = gl.createTexture(); gl.bindTexture(gl.TEXTURE_2D, t);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, src);
    gl.generateMipmap(gl.TEXTURE_2D);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    var an = gl.getExtension('EXT_texture_filter_anisotropic');
    if (an) gl.texParameterf(gl.TEXTURE_2D, an.TEXTURE_MAX_ANISOTROPY_EXT, Math.min(8, gl.getParameter(an.MAX_TEXTURE_MAX_ANISOTROPY_EXT)));
  }

  var DEF = { yaw: 0, pitch: 0, fov: 80 }, FOV_MIN = 30, FOV_MAX = 100;
  function clamp(x, a, b) { return Math.max(a, Math.min(b, x)); }

  function openViewer(it, opener) {
    if (V) return;
    var title = h('div', { class: 'pv-title', text: fmtDate(it.created, it.id) });
    var btnReset = h('button', { type: 'button', class: 'pv-btn', text: 'Reset view', onclick: function () { V.reset(); } });
    var btnDl = h('a', { class: 'pv-btn', href: it.url, download: 'panorama-' + it.id + '.jpg', text: 'Download' });
    var btnClose = h('button', { type: 'button', class: 'pv-btn pv-close', 'aria-label': 'Close viewer', text: 'Close', onclick: function () { closeViewer(true); } });
    var bar = h('div', { class: 'pv-top' }, [title, btnReset, btnDl, btnClose]);
    var canvas = h('canvas', { class: 'pv-canvas', 'aria-label': '360° photo. Drag or use the arrow keys to look around, scroll or plus and minus to zoom.', tabindex: '0' });
    var status = h('div', { class: 'pv-status', role: 'status', 'aria-live': 'polite' }, [h('span', { class: 'pv-spin' }), h('span', { text: 'Loading photo…' })]);
    var hint = h('div', { class: 'pv-hint', text: 'Drag to look around · scroll or pinch to zoom' });
    var root = h('div', { class: 'pv', role: 'dialog', 'aria-modal': 'true', 'aria-label': '360° photo viewer' }, [canvas, bar, status, hint]);
    document.body.appendChild(root);
    document.body.classList.add('pv-open');
    V = { it: it, root: root, canvas: canvas, opener: opener || null, st: { yaw: DEF.yaw, pitch: DEF.pitch, fov: DEF.fov }, vel: { x: 0, y: 0 },
      auto: true, dirty: true, raf: 0, last: 0, ptrs: {}, pinch: 0, g: null, ready: false, closing: false, status: status, hint: hint, dragging: false, keys: null };
    V.reset = function () { V.st.yaw = DEF.yaw; V.st.pitch = DEF.pitch; V.st.fov = DEF.fov; V.vel.x = V.vel.y = 0; V.auto = true; V.dirty = true; };
    try { V.g = initGL(canvas); } catch (e) { V.g = null; }
    history.pushState({ pano: it.id }, '');
    root.addEventListener('keydown', onKey);
    window.addEventListener('popstate', onPop);
    btnClose.focus();

    var img = new Image();
    img.onload = function () {
      if (!V || V.it !== it) return;
      if (V.g) {
        try { uploadTexture(V.g, img); V.ready = true; V.status.hidden = true; bindInput(); resize(); if (document.activeElement === btnClose) canvas.focus({ preventScroll: true }); V.raf = requestAnimationFrame(frame); return; }
        catch (e) { V.g = null; }
      }
      flatFallback(img);
    };
    img.onerror = function () {
      if (!V) return;
      V.status.classList.add('err'); V.status.lastChild.textContent = 'Could not load the photo.';
    };
    img.src = it.url;
  }

  function flatFallback(img) {
    V.canvas.hidden = true; V.status.hidden = true;
    V.hint.textContent = 'No WebGL2 in this browser: showing the flat photo, scroll sideways.';
    var box = h('div', { class: 'pv-flat', tabindex: '0' }, [img]);
    V.root.insertBefore(box, V.canvas);
    box.scrollLeft = (box.scrollWidth - box.clientWidth) / 2;
  }

  function resize() {
    if (!V || !V.g || !V.ready) return;
    var c = V.canvas, r = c.getBoundingClientRect(), dpr = Math.min(window.devicePixelRatio || 1, 2);
    var w = Math.max(1, Math.round(r.width * dpr)), hh = Math.max(1, Math.round(r.height * dpr));
    if (w * hh > 6e6) { var k = Math.sqrt(6e6 / (w * hh)); w = Math.round(w * k); hh = Math.round(hh * k); }
    if (c.width !== w || c.height !== hh) { c.width = w; c.height = hh; V.g.gl.viewport(0, 0, w, hh); }
    V.dirty = true;
  }
  function draw() {
    var g = V.g, gl = g.gl, asp = V.canvas.width / V.canvas.height, tf = Math.tan(V.st.fov * Math.PI / 360);
    gl.uniform2f(g.ang, V.st.yaw * Math.PI / 180, V.st.pitch * Math.PI / 180);
    gl.uniform2f(g.sc, asp >= 1 ? tf * asp : tf, asp >= 1 ? tf : tf / asp); // fov applies to the shorter screen side
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }
  function degPerPx() { var r = V.canvas.getBoundingClientRect(); return V.st.fov / Math.max(1, Math.min(r.width, r.height)); }
  function frame(t) {
    if (!V) return;
    var dt = Math.min(0.05, (t - (V.last || t)) / 1000), moving = false; V.last = t;
    if (V.auto && !V.dragging) { V.st.yaw += 5 * dt; moving = true; }
    if (!V.dragging && (Math.abs(V.vel.x) > 0.01 || Math.abs(V.vel.y) > 0.01)) {
      V.st.yaw += V.vel.x * dt; V.st.pitch += V.vel.y * dt;
      var decay = Math.pow(0.04, dt); V.vel.x *= decay; V.vel.y *= decay; moving = true;
    }
    var k = V.keys;
    if (k && (k.left || k.right || k.up || k.down)) { // held arrow keys
      var sp = V.st.fov * 0.9 * dt;
      V.st.yaw += (k.right - k.left) * sp; V.st.pitch += (k.up - k.down) * sp; moving = true;
    }
    V.st.pitch = clamp(V.st.pitch, -90, 90);
    V.st.yaw = ((V.st.yaw + 180) % 360 + 360) % 360 - 180;
    if (moving || V.dirty) { V.dirty = false; draw(); }
    V.raf = requestAnimationFrame(frame);
  }
  function interact() { V.auto = false; V.hint.classList.add('gone'); }

  function bindInput() {
    var c = V.canvas, lastT = 0;
    function dist() { var k = Object.keys(V.ptrs), a = V.ptrs[k[0]], b = V.ptrs[k[1]]; return Math.hypot(a.x - b.x, a.y - b.y); }
    c.addEventListener('pointerdown', function (e) {
      c.setPointerCapture(e.pointerId); V.ptrs[e.pointerId] = { x: e.clientX, y: e.clientY };
      interact(); V.vel.x = V.vel.y = 0; lastT = e.timeStamp;
      if (Object.keys(V.ptrs).length === 2) V.pinch = dist();
      V.dragging = true; c.classList.add('grab');
    });
    c.addEventListener('pointermove', function (e) {
      var p = V.ptrs[e.pointerId]; if (!p) return;
      if (Object.keys(V.ptrs).length >= 2) {
        p.x = e.clientX; p.y = e.clientY;
        var d = dist(); if (V.pinch > 0 && d > 0) { V.st.fov = clamp(V.st.fov * V.pinch / d, FOV_MIN, FOV_MAX); V.dirty = true; }
        V.pinch = d; return;
      }
      var dx = e.clientX - p.x, dy = e.clientY - p.y, k = degPerPx(), dt = Math.max(1, e.timeStamp - lastT) / 1000;
      p.x = e.clientX; p.y = e.clientY; lastT = e.timeStamp;
      V.st.yaw -= dx * k; V.st.pitch = clamp(V.st.pitch + dy * k, -90, 90);
      V.vel.x = V.vel.x * 0.5 + (-dx * k / dt) * 0.5; V.vel.y = V.vel.y * 0.5 + (dy * k / dt) * 0.5;
      V.dirty = true;
    });
    function up(e) {
      if (!V.ptrs[e.pointerId]) return;
      delete V.ptrs[e.pointerId];
      var left = Object.keys(V.ptrs).length;
      if (left < 2) V.pinch = 0;
      if (!left) { V.dragging = false; c.classList.remove('grab'); if (e.timeStamp - lastT > 80) V.vel.x = V.vel.y = 0; }
      else V.vel.x = V.vel.y = 0;
    }
    c.addEventListener('pointerup', up); c.addEventListener('pointercancel', up);
    c.addEventListener('wheel', function (e) {
      e.preventDefault(); interact();
      V.st.fov = clamp(V.st.fov * Math.exp(e.deltaY * (e.deltaMode === 1 ? 0.03 : 0.0012)), FOV_MIN, FOV_MAX); V.dirty = true;
    }, { passive: false });
    c.addEventListener('dblclick', function () { V.reset(); V.auto = false; });
    window.addEventListener('resize', resize);
    if (window.ResizeObserver) { V.ro = new ResizeObserver(resize); V.ro.observe(c); }
    V.keys = { left: 0, right: 0, up: 0, down: 0 };
    window.addEventListener('keyup', onKeyUp);
  }

  var ARROWS = { ArrowLeft: 'left', ArrowRight: 'right', ArrowUp: 'up', ArrowDown: 'down' };
  function onKey(e) {
    if (!V) return;
    if (e.key === 'Escape') { e.preventDefault(); e.stopPropagation(); closeViewer(true); return; }
    if (e.key === 'Tab') { // keep focus inside the dialog
      var f = Array.prototype.slice.call(V.root.querySelectorAll('button, a[href], [tabindex="0"]')).filter(function (x) { return !x.hidden && x.offsetParent !== null; });
      if (!f.length) return;
      var i = f.indexOf(document.activeElement);
      if (e.shiftKey && i <= 0) { e.preventDefault(); f[f.length - 1].focus(); }
      else if (!e.shiftKey && i === f.length - 1) { e.preventDefault(); f[0].focus(); }
      return;
    }
    if (!V.ready) return;
    var tag = document.activeElement && document.activeElement.tagName;
    if (ARROWS[e.key] && tag !== 'BUTTON' && tag !== 'A') { V.keys[ARROWS[e.key]] = 1; interact(); e.preventDefault(); }
    else if (e.key === '+' || e.key === '=') { V.st.fov = clamp(V.st.fov - 6, FOV_MIN, FOV_MAX); V.dirty = true; interact(); }
    else if (e.key === '-' || e.key === '_') { V.st.fov = clamp(V.st.fov + 6, FOV_MIN, FOV_MAX); V.dirty = true; interact(); }
    else if (e.key === '0' || e.key === 'r') { V.reset(); V.auto = false; }
  }
  function onKeyUp(e) { if (V && V.keys && ARROWS[e.key]) V.keys[ARROWS[e.key]] = 0; }
  function onPop() { if (V && !V.closing) closeViewer(false); }

  function closeViewer(popHistory) {
    if (!V) return;
    var v = V; V = null; v.closing = true;
    cancelAnimationFrame(v.raf);
    window.removeEventListener('popstate', onPop); window.removeEventListener('keyup', onKeyUp); window.removeEventListener('resize', resize);
    if (v.ro) v.ro.disconnect();
    if (v.g) { var ext = v.g.gl.getExtension('WEBGL_lose_context'); if (ext) ext.loseContext(); }
    v.root.remove();
    document.body.classList.remove('pv-open');
    if (popHistory && history.state && history.state.pano) history.back();
    if (v.opener && document.contains(v.opener)) v.opener.focus();
  }

  /* ---------------------------------------------------------------- init */
  function visible() {
    var p = document.getElementById('p-robot');
    return !document.hidden && p && !p.classList.contains('hide');
  }
  function init() {
    if (!build()) return;
    var tab = document.getElementById('tab-robot');
    if (tab) tab.addEventListener('click', function () { setTimeout(function () { if (visible()) { check(); refreshList(); } }, 50); });
    document.addEventListener('visibilitychange', function () { if (visible() && !wasActive) check(); });
    refreshList(); check();
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init); else init();
})();
