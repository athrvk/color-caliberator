const video        = document.getElementById('viewfinder');
const canvas       = document.getElementById('canvas');
const ctx          = canvas.getContext('2d');
const statusText   = document.getElementById('status-text');
const connDot      = document.getElementById('conn-dot');
const reticle      = document.getElementById('reticle');
const ssnrWrap     = document.getElementById('ssnr-wrap');
const ssnrFill     = document.getElementById('ssnr-fill');
const patchDots    = document.getElementById('patch-dots');
const ssnrPips     = document.getElementById('ssnr-pips');
const startCamBtn  = document.getElementById('start-cam-btn');
const readyBtn     = document.getElementById('ready-btn');
const uploadRawBtn = document.getElementById('upload-raw-btn');
const uploadBar    = document.getElementById('upload-bar');
const uploadProg   = document.getElementById('upload-progress');
const rawInput     = document.getElementById('raw-input');
const flash        = document.getElementById('flash');
const splash       = document.getElementById('splash');
const doneScreen   = document.getElementById('done-screen');
const pips         = ssnrPips.querySelectorAll('.pip');

let ws;
let capturing = false;
let captureInterval = null;
let totalPatches = 0;
let donePatches  = 0;
let videoTrack  = null;
let lockSupported = false;  // true if camera exposes manual WB/exposure/focus
let isIOS = /iPad|iPhone|iPod/.test(navigator.userAgent) || /CriOS/.test(navigator.userAgent);

// ── SSNR arc (r=80, circumference ≈ 502) ──
const CIRC = 2 * Math.PI * 80;

function setSsnrArc(fraction, stable) {
  const offset = CIRC * (1 - Math.min(1, Math.max(0, fraction)));
  ssnrFill.style.strokeDashoffset = offset;
  ssnrFill.style.stroke = stable ? 'var(--accent)' : 'var(--amber)';
}

function setPips(fraction, stable) {
  const filled = Math.round(fraction * pips.length);
  pips.forEach((p, i) => {
    p.className = 'pip';
    if (i < filled) p.classList.add(stable ? 'stable' : 'waiting');
  });
}

function setStatus(txt) {
  statusText.textContent = txt;
}

// ── Camera ──
async function startCamera() {
  if (!navigator.mediaDevices?.getUserMedia) {
    setStatus('Camera API unavailable. Use HTTPS + Chrome.');
    return;
  }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'environment', width: { ideal: 1920 }, height: { ideal: 1080 } }
    });
    video.srcObject = stream;
    await video.play();
    canvas.width  = video.videoWidth  || 1280;
    canvas.height = video.videoHeight || 720;
    videoTrack = stream.getVideoTracks()[0];

    // Detect manual lock capability. Chrome Android exposes whiteBalanceMode,
    // exposureMode, focusMode in track capabilities. iOS Chrome (WebKit
    // engine) does NOT — Apple gates this off; users must fall back to
    // native-camera tap-and-hold AE/AF lock workflow.
    const caps = (typeof videoTrack.getCapabilities === 'function')
      ? videoTrack.getCapabilities()
      : {};
    lockSupported =
      Array.isArray(caps.whiteBalanceMode) && caps.whiteBalanceMode.includes('manual') &&
      Array.isArray(caps.exposureMode)     && caps.exposureMode.includes('manual') &&
      Array.isArray(caps.focusMode)        && caps.focusMode.includes('manual');
    console.log('camera capabilities:', caps, 'lockSupported:', lockSupported);

    video.classList.add('visible');
    splash.classList.add('gone');
    setStatus('Camera live. Connecting to PC…');
    connectWs();
  } catch (e) {
    setStatus('Camera denied: ' + (e.message || e.name));
  }
}

async function lockCamera() {
  /* Pin WB/exposure/focus at their current values. Chrome Android only. */
  if (!videoTrack || !lockSupported) return false;
  try {
    await videoTrack.applyConstraints({
      advanced: [{
        whiteBalanceMode: 'manual',
        exposureMode:     'manual',
        focusMode:        'manual',
      }],
    });
    return true;
  } catch (e) {
    console.warn('applyConstraints lock failed', e);
    return false;
  }
}

startCamBtn.onclick = startCamera;

// ── WebSocket ──
function connectWs() {
  const scheme = location.protocol === 'https:' ? 'wss:' : 'ws:';
  ws = new WebSocket(`${scheme}//${location.host}/ws/mobile`);
  ws.onopen    = () => { connDot.classList.add('live'); setStatus('Connected — waiting for PC…'); };
  ws.onmessage = (evt) => {
    let msg;
    try { msg = JSON.parse(evt.data); }
    catch (e) { console.warn('non-JSON WS frame ignored', e); return; }
    try { handleMsg(msg); }
    catch (e) { console.error('handleMsg crashed', e); }
  };
  ws.onclose   = () => { connDot.classList.remove('live'); stopCapturing(); setStatus('Disconnected. Reload to reconnect.'); };
  ws.onerror   = (e) => { console.warn('mobile WS error', e); };
}

document.addEventListener('visibilitychange', () => {
  if (document.visibilityState === 'visible' &&
      ws && (ws.readyState === WebSocket.CLOSED || ws.readyState === WebSocket.CLOSING)) {
    setStatus('Reconnecting…');
    connectWs();
  }
});

// ── Message handler ──
function handleMsg(msg) {
  if (msg.type === 'show_white_for_wb') {
    reticle.classList.add('visible');
    ssnrWrap.classList.add('visible');
    if (lockSupported) {
      setStatus('Aim at the white screen. Tap Ready to auto-lock WB + exposure + focus.');
    } else if (isIOS) {
      setStatus('iOS Chrome can\'t auto-lock the camera. Open the native Camera app, tap-and-hold the white area until AE/AF LOCK appears, return here, then tap Ready.');
    } else {
      setStatus('Manual lock unavailable on this device. Hold the phone perfectly still on the white area and tap Ready.');
    }
    readyBtn.hidden = false;
    readyBtn.onclick = async () => {
      readyBtn.disabled = true;
      const locked = await lockCamera();
      ws.send(JSON.stringify({ type: 'ready' }));
      readyBtn.hidden = true;
      readyBtn.disabled = false;
      setStatus(locked
        ? 'Camera locked. Waiting for first patch…'
        : 'Ready sent. Keep the phone steady — auto-lock unavailable on this device.');
    };
    return;
  }

  if (msg.type === 'request_raw') {
    reticle.classList.remove('visible');
    ssnrWrap.classList.remove('visible');
    setStatus(`Shoot a RAW (DNG) of the ${msg.label} patch.\niOS: Save to Files → upload from Files.\nAndroid: pick from camera roll.`);
    uploadRawBtn.textContent = `// Upload ${msg.label} RAW`;
    uploadRawBtn.hidden = false;
    uploadRawBtn.onclick = () => rawInput.click();
    rawInput.onchange = (e) => {
      const file = e.target.files[0];
      if (file) uploadFile(file, msg.seq);
    };
    return;
  }

  if (msg.type === 'capture') {
    totalPatches = msg.total;
    donePatches  = msg.n - 1;
    buildPatchDots(totalPatches, donePatches);
    patchDots.hidden = false;
    ssnrPips.hidden  = false;
    reticle.classList.add('visible');
    ssnrWrap.classList.add('visible');
    setSsnrArc(0, false);
    setPips(0, false);
    setStatus(`Patch ${msg.n} / ${msg.total} — hold steady`);
    capturing = true;
    startCapturing();
    return;
  }

  if (msg.type === 'ssnr') {
    const frac   = Math.min(1, (msg.db - 5) / 20);
    const stable = msg.db >= 20;
    setSsnrArc(frac, stable);
    setPips(frac, stable);
    return;
  }

  if (msg.type === 'stop_capture') { stopCapturing(); return; }

  if (msg.type === 'patch_done') {
    stopCapturing();
    donePatches = msg.n;
    buildPatchDots(totalPatches, donePatches);
    setSsnrArc(1, true);
    setPips(1, true);
    setStatus(`Patch ${msg.n} / ${msg.total} captured`);
    triggerFlash();
    return;
  }

  if (msg.type === 'round_done') {
    patchDots.hidden = true;
    ssnrPips.hidden  = true;
    reticle.classList.remove('visible');
    ssnrWrap.classList.remove('visible');
    setStatus('Round complete — applying correction…');
    return;
  }

  if (msg.type === 'all_done') {
    // Release the manual locks so the user's camera goes back to auto when
    // they leave the page. Best-effort; OK if it fails.
    if (videoTrack && lockSupported) {
      videoTrack.applyConstraints({
        advanced: [{ whiteBalanceMode: 'continuous', exposureMode: 'continuous', focusMode: 'continuous' }],
      }).catch(() => {});
    }
    doneScreen.classList.add('show');
  }
}

// ── Patch dots ──
function buildPatchDots(total, done) {
  patchDots.innerHTML = '';
  for (let i = 0; i < total; i++) {
    const d = document.createElement('span');
    d.className = 'pdot' + (i < done ? ' done' : i === done ? ' active' : '');
    patchDots.appendChild(d);
  }
}

// ── Capture loop ──
function startCapturing() {
  if (captureInterval) return;
  captureInterval = setInterval(sendFrame, 200);
}

function stopCapturing() {
  capturing = false;
  if (captureInterval) { clearInterval(captureInterval); captureInterval = null; }
}

function sendFrame() {
  if (!capturing || !ws || ws.readyState !== WebSocket.OPEN) return;
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  const dataUrl = canvas.toDataURL('image/jpeg', 0.85);
  ws.send(JSON.stringify({ type: 'frame', data: dataUrl.split(',')[1] }));
}

// ── Flash ──
function triggerFlash() {
  flash.style.opacity = '0.45';
  setTimeout(() => { flash.style.opacity = '0'; }, 140);
}

// ── RAW upload ──
function uploadFile(file, seq) {
  const form = new FormData();
  form.append('file', file);
  uploadProg.style.display = 'block';
  uploadBar.style.width = '0%';
  setStatus(`Uploading ${file.name}…`);

  const xhr = new XMLHttpRequest();
  xhr.open('POST', `/upload/raw/${seq}`);
  xhr.upload.onprogress = (ev) => {
    if (!ev.lengthComputable) return;
    const pct = Math.round((ev.loaded / ev.total) * 100);
    uploadBar.style.width = pct + '%';
    setStatus(`Uploading ${file.name}… ${pct}%`);
  };
  xhr.onload = () => {
    uploadProg.style.display = 'none';
    if (xhr.status >= 200 && xhr.status < 300) {
      setStatus('Uploaded. Waiting for next instruction…');
      uploadRawBtn.hidden = true;
      rawInput.value = '';
    } else {
      setStatus('Upload failed: ' + xhr.statusText);
    }
  };
  xhr.onerror = () => { uploadProg.style.display = 'none'; setStatus('Upload failed: network error'); };
  xhr.send(form);
}
