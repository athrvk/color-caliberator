// ── Patch tiles ──
const LEVELS = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1];
const tileRow = document.getElementById('tile-row');
LEVELS.forEach((lv, i) => {
  const d = document.createElement('div');
  d.className = 'pt';
  const v = Math.round(lv * 255);
  d.dataset.fill = `rgb(${v},${v},${v})`;
  d.dataset.i = i;
  tileRow.appendChild(d);
});
let activeTile = -1;
const allTiles = () => document.querySelectorAll('.pt');

function highlightTile(idx) {
  allTiles().forEach((t, i) => {
    t.classList.remove('current');
    if (i === idx) t.classList.add('current');
  });
  activeTile = idx;
}

function captureTile(idx) {
  const tiles = allTiles();
  if (idx < 0 || idx >= tiles.length) return;
  const t = tiles[idx];
  t.classList.remove('current');
  t.classList.add('captured');
  t.style.background = t.dataset.fill;
  if (LEVELS[idx] > 0.5) {
    t.style.boxShadow = 'inset 0 1px 0 rgba(255,255,255,0.15)';
  }
}

function resetTiles() {
  allTiles().forEach(t => {
    t.className = 'pt';
    t.style.background = '';
    t.style.boxShadow = '';
  });
  activeTile = -1;
}

// ── SSNR pips ──
const pips = document.querySelectorAll('.pip');
function setPips(state) {
  pips.forEach((p, i) => {
    p.className = 'pip';
    if (state === 'waiting') setTimeout(() => p.classList.add('amber'), i * 70);
    else if (state === 'stable') p.classList.add('teal');
  });
}

// ── ΔE counter animation ──
function countUp(target, ms = 1300) {
  const el = document.getElementById('de-number');
  const t0 = performance.now();
  const step = ts => {
    const p = Math.min((ts - t0) / ms, 1);
    const ease = 1 - Math.pow(1 - p, 3);
    el.textContent = (target * ease).toFixed(2);
    if (p < 1) requestAnimationFrame(step);
    else el.textContent = target.toFixed(2);
  };
  requestAnimationFrame(step);
}

// ── Screen management ──
const S = {
  error:     document.getElementById('s-error'),
  qr:        document.getElementById('s-qr'),
  setup:     document.getElementById('s-setup'),
  measuring: document.getElementById('s-measuring'),
  applying:  document.getElementById('s-applying'),
  result:    document.getElementById('s-result'),
};

function show(name) {
  document.body.style.background = '';
  Object.values(S).forEach(s => s.classList.remove('active'));
  if (name) S[name].classList.add('active');
}

// ── Connection dot ──
const cdot   = document.getElementById('conn-dot');
const clabel = document.getElementById('conn-label');
function setConn(cls, label) {
  cdot.className = 'conn-dot ' + cls;
  clabel.textContent = label;
}

// ── Install toggle ──
document.getElementById('install-toggle').onclick = function () {
  this.classList.toggle('open');
  const box = document.getElementById('install-box');
  box.style.display = box.style.display === 'block' ? 'none' : 'block';
};

// ── Emergency reset (error screen + result-failure banner) ──
function requestDisplayReset(btn) {
  btn.disabled = true;
  const originalText = btn.textContent;
  btn.textContent = 'Resetting…';
  const restore = (msg) => { btn.textContent = msg; setTimeout(() => { btn.disabled = false; btn.textContent = originalText; }, 2000); };
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: 'reset_display' }));
    btn.dataset.restoreText = originalText;
  } else {
    // WS dead — fall back to HTTPS endpoint (works from any PC on LAN).
    fetch('/reset_display', { method: 'POST' })
      .then(r => restore(r.ok ? 'Reset OK' : 'Reset failed'))
      .catch(() => restore('Reset failed'));
  }
}

document.getElementById('reset-display-btn').onclick = function () { requestDisplayReset(this); };
document.getElementById('fail-reset-btn').onclick = function () { requestDisplayReset(this); };

// ── Result screen ──
function showResult(msg) {
  document.body.style.background = '';
  const gradeEl    = document.getElementById('de-grade');
  const noteEl     = document.getElementById('de-note');
  const dlBtn      = document.getElementById('download-btn');
  const failBanner = document.getElementById('fail-banner');

  if (msg.mode === 'color') {
    document.getElementById('de-number').textContent = '—';
    gradeEl.textContent = 'Color Profile'; gradeEl.className = 'de-grade grade-a';
    noteEl.textContent = 'Matrix-shaper profile with white point and primary chromaticities.';
    failBanner.hidden = true;
  } else {
    const de = msg.delta_e ?? 0;
    countUp(de);
    if (de < 2) {
      gradeEl.textContent = 'Excellent'; gradeEl.className = 'de-grade grade-a';
      noteEl.textContent = 'Outstanding accuracy. Your display is well characterised.';
      failBanner.hidden = true;
    } else if (de < 4) {
      gradeEl.textContent = 'Good'; gradeEl.className = 'de-grade grade-b';
      noteEl.textContent = 'Good result. Most colour errors will be invisible in normal use.';
      failBanner.hidden = true;
    } else if (de < 10) {
      gradeEl.textContent = 'Fair'; gradeEl.className = 'de-grade grade-c';
      noteEl.textContent = 'Visible improvement but some inaccuracy remains. Retry in a darker room.';
      failBanner.hidden = true;
    } else {
      gradeEl.textContent = 'Failed'; gradeEl.className = 'de-grade grade-c';
      noteEl.textContent = '';
      failBanner.hidden = false;
      dlBtn.disabled = true;
    }
  }

  if (msg.before_b64) document.getElementById('before-img').src = `data:image/png;base64,${msg.before_b64}`;
  if (msg.after_b64)  document.getElementById('after-img').src  = `data:image/png;base64,${msg.after_b64}`;

  const bytes = Uint8Array.from(atob(msg.icc_b64), c => c.charCodeAt(0));
  const url   = URL.createObjectURL(new Blob([bytes], { type: 'application/vnd.iccprofile' }));
  dlBtn.onclick = () => {
    Object.assign(document.createElement('a'), { href: url, download: 'color-calibrator.icc' }).click();
  };

  // Whole-desktop live preview toggle (TruHu-style). Server holds the final
  // LUTs in memory and flips the VideoLUT on click.
  const liveBox = document.getElementById('live-preview');
  if (msg.live_preview && liveBox) {
    const previewCorrected = document.getElementById('preview-corrected-btn');
    const failed = msg.mode !== 'color' && (msg.delta_e ?? 0) >= 10;
    liveBox.hidden = false;
    if (failed) {
      previewCorrected.disabled = true;
      previewCorrected.title = 'Calibration failed — preview disabled to avoid wrecking your display.';
    }
    previewCorrected.onclick = () => {
      ws.send(JSON.stringify({ type: 'preview_corrected' }));
    };
    document.getElementById('preview-original-btn').onclick = () => {
      ws.send(JSON.stringify({ type: 'preview_original' }));
    };
  }

  show('result');
}

// ── WebSocket ──
const wsScheme = location.protocol === 'https:' ? 'wss:' : 'ws:';
const ws = new WebSocket(`${wsScheme}//${location.host}/ws/pc`);
ws.onopen  = () => setConn('pulsing', 'connected');
ws.onclose = () => setConn('', 'disconnected');
ws.onerror = (e) => { console.warn('PC WS error', e); setConn('', 'connection error'); };

ws.onmessage = ({ data }) => {
  let msg;
  try { msg = JSON.parse(data); }
  catch (e) { console.warn('non-JSON WS frame ignored', e); return; }

  if (msg.type === 'error') {
    document.getElementById('error-text').textContent = msg.message;
    const eyebrow = document.getElementById('error-eyebrow');
    const hints = {
      backend_unavailable: { eyebrow: 'Setup Required',       hint: 'error-hint-backend' },
      mobile_disconnected: { eyebrow: 'Mobile Disconnected',  hint: 'error-hint-mobile'  },
      calibration_failed:  { eyebrow: 'Calibration Failed',   hint: 'error-hint-runtime' },
      runtime:             { eyebrow: 'Error',                hint: 'error-hint-runtime' },
    };
    const cfg = hints[msg.kind] || hints.runtime;
    eyebrow.textContent = cfg.eyebrow;
    ['error-hint-backend', 'error-hint-mobile', 'error-hint-runtime'].forEach(id => {
      document.getElementById(id).hidden = (id !== cfg.hint);
    });
    show('error'); return;
  }

  if (msg.type === 'qr_code') {
    document.getElementById('qr-img').src = `data:image/png;base64,${msg.png_b64}`;
    document.getElementById('mobile-url').textContent = msg.url;
    show('qr'); return;
  }

  if (msg.type === 'mobile_connected') {
    setConn('live', 'mobile connected');
    show('setup');
    document.getElementById('begin-btn').onclick = async () => {
      // Request fullscreen on this click (user-gesture context). Browser
      // chrome — tabs, address bar, taskbar — leaks light into the camera
      // and contaminates patch measurements. If the browser refuses, fall
      // back to a banner asking the user to press F11 / ⌃⌘F manually.
      try {
        await document.documentElement.requestFullscreen({ navigationUI: 'hide' });
      } catch (e) {
        const hint = document.getElementById('fs-hint');
        if (hint) hint.style.display = 'block';
      }
      const mode = document.querySelector('input[name="mode"]:checked').value;
      ws.send(JSON.stringify({ type: 'start_calibration', mode }));
      const btn = document.getElementById('begin-btn');
      btn.disabled = true; btn.textContent = 'Starting…';
    };
    return;
  }

  if (msg.type === 'show_patch') {
    const [r, g, b] = msg.rgb;
    document.body.style.background = `rgb(${r},${g},${b})`;
    Object.values(S).forEach(s => s.classList.remove('active'));
    return;
  }

  if (msg.type === 'capturing') {
    if (msg.round > 0) document.getElementById('round-tag').textContent = `Round ${msg.round} of 3`;
    if (msg.n) highlightTile(msg.n - 1);
    setPips('waiting');
    document.getElementById('ssnr-text').textContent = 'Waiting for stable frame';
    return;
  }

  if (msg.type === 'patch_done') {
    document.body.style.background = '';
    document.getElementById('patch-num').textContent = msg.n;
    document.getElementById('patch-total').textContent = msg.total;
    if (msg.round > 0) document.getElementById('round-tag').textContent = `Round ${msg.round} of 3`;
    captureTile(msg.n - 1);
    setPips('stable');
    document.getElementById('ssnr-text').textContent = 'Captured ✓';
    show('measuring'); return;
  }

  if (msg.type === 'holdout_started') {
    document.getElementById('round-tag').textContent = 'Verifying calibration';
    document.getElementById('patch-total').textContent = msg.total;
    document.getElementById('patch-num').textContent = '0';
    resetTiles();
    show('measuring'); return;
  }

  if (msg.type === 'round_done') {
    document.getElementById('apply-round').textContent = msg.round;
    resetTiles();
    show('applying'); return;
  }

  if (msg.type === 'result') {
    showResult(msg); return;
  }

  if (msg.type === 'display_reset') {
    const btn = document.getElementById('reset-display-btn');
    btn.textContent = msg.ok ? 'Reset OK' : 'Reset failed';
    setTimeout(() => { btn.disabled = false; btn.textContent = 'Reset Display Gamma'; }, 2000);
    return;
  }

  if (msg.type === 'mobile_disconnected_soft') {
    // Calibration is paused waiting for mobile to come back. Don't navigate.
    setConn('pulsing', 'mobile reconnecting…');
    return;
  }

  if (msg.type === 'mobile_reconnected') {
    setConn('live', 'mobile connected');
    return;
  }
};
