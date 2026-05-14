// Sessions list + detail viewer. Phase 1: save / list / view. No rebuild yet.

const listEl   = document.getElementById('list');
const detailEl = document.getElementById('detail');
let selectedId = null;

function fmtBytes(n) {
  if (!Number.isFinite(n)) return '—';
  const u = ['B', 'KB', 'MB', 'GB'];
  let i = 0;
  while (n >= 1024 && i < u.length - 1) { n /= 1024; i++; }
  return `${n.toFixed(n < 10 ? 1 : 0)} ${u[i]}`;
}

function fmtDate(iso) {
  try {
    const d = new Date(iso);
    return d.toLocaleString();
  } catch { return iso; }
}

async function loadList() {
  try {
    const r = await fetch('/api/sessions');
    const data = await r.json();
    renderList(data.sessions || []);
  } catch (e) {
    listEl.innerHTML = `<div class="list-empty">Failed to load sessions: ${e}</div>`;
  }
}

function renderList(sessions) {
  if (!sessions.length) {
    listEl.innerHTML = '<div class="list-empty">No saved sessions yet. Run a calibration on the home page to record one.</div>';
    return;
  }
  listEl.innerHTML = '';
  for (const s of sessions) {
    const btn = document.createElement('button');
    btn.className = 'item' + (s.id === selectedId ? ' selected' : '');
    const badgeClass = s.mode === 'color' ? 'color' : 'gamma';
    const dePart = (s.mode !== 'color' && typeof s.delta_e === 'number')
      ? `ΔE ${s.delta_e.toFixed(2)}`
      : (s.mode === 'color' ? 'matrix-shaper' : '—');
    btn.innerHTML = `
      <div class="item-id">${s.id}</div>
      <div class="item-row">
        <span class="badge ${badgeClass}">${s.mode || 'gamma'}</span>
        <span style="font-family:var(--mono); font-size:0.78rem; color:var(--text);">${dePart}</span>
      </div>
      <div class="item-meta">${fmtDate(s.started_at)} · ${fmtBytes(s.size_bytes)}</div>
    `;
    btn.onclick = () => selectSession(s.id);
    listEl.appendChild(btn);
  }
}

async function selectSession(id) {
  selectedId = id;
  document.querySelectorAll('.item').forEach(el => el.classList.remove('selected'));
  const target = Array.from(document.querySelectorAll('.item')).find(el => el.querySelector('.item-id')?.textContent === id);
  if (target) target.classList.add('selected');

  detailEl.innerHTML = '<div class="detail-empty">Loading…</div>';
  try {
    const r = await fetch(`/api/sessions/${encodeURIComponent(id)}`);
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const data = await r.json();
    renderDetail(id, data);
  } catch (e) {
    detailEl.innerHTML = `<div class="detail-empty">Failed to load: ${e}</div>`;
  }
}

function renderDetail(id, data) {
  const m = data.meta || {};
  const s = data.summary || {};
  const cd = s.color_data || {};
  const isColor = m.mode === 'color';
  const okBadge = m.ok ? '✓ Completed' : '✗ Incomplete';
  const deLine = isColor
    ? 'Matrix-shaper profile'
    : (typeof s.delta_e === 'number' ? `Final ΔE: ${s.delta_e.toFixed(2)}` : 'ΔE not recorded');

  let gammaLine = '';
  if (isColor && cd.gamma_r != null) {
    gammaLine = `<div class="k">Display γ (R / G / B)</div>
                 <div class="v">${cd.gamma_r.toFixed(2)} / ${cd.gamma_g.toFixed(2)} / ${cd.gamma_b.toFixed(2)}</div>`;
  }

  detailEl.innerHTML = `
    <h2>Session ${id}</h2>
    <div class="meta-grid">
      <div class="k">Mode</div><div class="v">${m.mode || '—'}</div>
      <div class="k">Started</div><div class="v">${fmtDate(m.started_at)}</div>
      <div class="k">Finished</div><div class="v">${fmtDate(m.finished_at)}</div>
      <div class="k">Status</div><div class="v">${okBadge}</div>
      <div class="k">Result</div><div class="v">${deLine}</div>
      ${gammaLine}
    </div>
    <div class="actions">
      <button class="btn" id="dl-btn">Download .icc</button>
      <button class="btn ghost" id="rebuild-btn" title="Re-run the fit on saved measurements with the current code.">Rebuild</button>
      <button class="btn ghost" id="open-folder-btn" title="Print the on-disk path to the console.">Show path</button>
      <button class="btn danger" id="del-btn">Delete</button>
    </div>
    <h2>Before / After ${isColor ? '(simulated via software CMM)' : ''}</h2>
    <div class="comparison">
      <div class="cmp-half">
        ${data.before_b64 ? `<img src="data:image/png;base64,${data.before_b64}">` : ''}
        <div class="cmp-label">Before</div>
      </div>
      <div class="cmp-half">
        ${data.after_b64 ? `<img src="data:image/png;base64,${data.after_b64}">` : ''}
        <div class="cmp-label">After</div>
      </div>
    </div>
    <footer>ID ${m.id || id}</footer>
  `;

  document.getElementById('dl-btn').onclick = () => {
    const bytes = Uint8Array.from(atob(data.icc_b64 || ''), c => c.charCodeAt(0));
    const url = URL.createObjectURL(new Blob([bytes], { type: 'application/vnd.iccprofile' }));
    Object.assign(document.createElement('a'), { href: url, download: `chroma-${id}.icc` }).click();
  };
  document.getElementById('rebuild-btn').onclick = async () => {
    const btn = document.getElementById('rebuild-btn');
    const oldText = btn.textContent;
    btn.disabled = true; btn.textContent = 'Rebuilding…';
    try {
      const r = await fetch(`/api/sessions/${encodeURIComponent(id)}/rebuild`, { method: 'POST' });
      if (!r.ok) {
        const err = await r.json().catch(() => ({ detail: `HTTP ${r.status}` }));
        alert(`Rebuild failed: ${err.detail || r.status}`);
        return;
      }
      const fresh = await r.json();
      // Mark the visible result as rebuilt — keep the saved record untouched.
      renderDetail(id, fresh);
      // Banner so user knows what they're looking at.
      const banner = document.createElement('div');
      banner.style.cssText = 'font-family:var(--mono); font-size:0.72rem; color:var(--accent); background:var(--accent-glow, rgba(45,226,196,0.12)); padding:0.5rem 0.85rem; border-radius:6px; margin-bottom:1rem;';
      banner.textContent = '↻ Showing rebuilt result from current code. On-disk session unchanged.';
      detailEl.insertBefore(banner, detailEl.firstChild.nextSibling);
    } catch (e) {
      alert(`Rebuild failed: ${e}`);
    } finally {
      const after = document.getElementById('rebuild-btn');
      if (after) { after.disabled = false; after.textContent = oldText; }
    }
  };
  document.getElementById('open-folder-btn').onclick = () => {
    console.log(`Session on disk: .sessions/${id}/`);
    alert(`Saved at:\n.sessions/${id}/`);
  };
  document.getElementById('del-btn').onclick = async () => {
    if (!confirm(`Delete session ${id}? This removes all saved frames and the ICC.`)) return;
    const r = await fetch(`/api/sessions/${encodeURIComponent(id)}`, { method: 'DELETE' });
    if (r.ok) {
      selectedId = null;
      detailEl.innerHTML = '<div class="detail-empty">Select a session to view the result.</div>';
      loadList();
    } else {
      alert(`Delete failed: HTTP ${r.status}`);
    }
  };
}

loadList();
