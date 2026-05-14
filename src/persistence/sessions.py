"""Session persistence.

Saves every artifact a calibration produces — raw camera frames, averaged
frames, per-patch measurements, color-mode DNG anchors, final ICC, and
CMM-rendered before/after preview — under `.sessions/<id>/`. A session is
considered "complete" only after `meta.json` is written; partial sessions
are filtered out of listings so an interrupted run never appears.

This phase intentionally only implements **save + list + view**. A rebuild
pipeline (re-running fit_* over stored cooked measurements) is deferred.

Layout:
    .sessions/<id>/
      meta.json                    (written LAST — atomic completion flag)
      white_ref/
        raw/frame_NNNN.jpg
        averaged.jpg
      patches/
        round_<r>/
          patch_<idx>_level_<lvl>/
            raw/frame_NNNN.jpg
            averaged.jpg
            measured.json
        holdout/
          patch_<idx>_level_<lvl>/...
      anchors/                     (color mode only)
        seq_<n>_<label>.dng
      result/
        profile.icc
        before.png
        after.png
        summary.json
"""

from __future__ import annotations

import io
import json
import secrets
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image


def _project_root() -> Path:
    # src/persistence/sessions.py → repo root is parents[2].
    return Path(__file__).resolve().parents[2]


def sessions_dir() -> Path:
    """Resolve the on-disk sessions directory, creating it lazily."""
    d = _project_root() / ".sessions"
    d.mkdir(parents=True, exist_ok=True)
    return d


def new_session_id() -> str:
    """Sortable + unique id: `YYYYMMDD-HHMMSS-<6hex>`."""
    now = datetime.now(tz=timezone.utc)
    return f"{now.strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(3)}"


# ─────────────────────────────────────────────────────────────────────────────
# Recorder
# ─────────────────────────────────────────────────────────────────────────────


class SessionRecorder:
    """Writes session artifacts incrementally.

    iterate.py treats a `None` recorder as a no-op, so tests and headless
    runs pay zero cost. All I/O is best-effort: a failed write logs and
    continues — calibration must not break because the disk is full.
    """

    def __init__(self, root: Path, session_id: str, mode: str):
        self.root = root
        self.id = session_id
        self.mode = mode
        self.started_at = datetime.now(tz=timezone.utc).isoformat()
        self._white_raw_count = 0
        self._patch_raw_counts: dict[str, int] = {}

    # ── construction / lifecycle ──

    @classmethod
    def create(cls, mode: str, base_dir: Path | None = None) -> "SessionRecorder":
        base = (base_dir or sessions_dir())
        sid = new_session_id()
        root = base / sid
        root.mkdir(parents=True, exist_ok=False)
        (root / "white_ref" / "raw").mkdir(parents=True, exist_ok=True)
        (root / "patches").mkdir(parents=True, exist_ok=True)
        if mode == "color":
            (root / "anchors").mkdir(parents=True, exist_ok=True)
        (root / "result").mkdir(parents=True, exist_ok=True)
        return cls(root=root, session_id=sid, mode=mode)

    # ── frame recording ──

    def record_white_raw_frame(self, jpeg_bytes: bytes) -> None:
        try:
            self._white_raw_count += 1
            path = self.root / "white_ref" / "raw" / f"frame_{self._white_raw_count:04d}.jpg"
            path.write_bytes(jpeg_bytes)
        except Exception:
            pass

    def record_white_averaged(self, frame_uint8: np.ndarray) -> None:
        _save_array_jpeg(frame_uint8, self.root / "white_ref" / "averaged.jpg")

    def _patch_dir(self, phase: str, round_num: int | None, index: int, level: float) -> Path:
        if phase == "holdout":
            sub = self.root / "patches" / "holdout"
        elif phase.startswith("color_"):
            # Color channels iterate 0..N per channel — keep them in
            # per-channel subdirs so patch_index can't collide across R/G/B.
            ch = phase.split("_", 1)[1]
            sub = self.root / "patches" / "color" / ch
        else:
            sub = self.root / "patches" / f"round_{round_num}"
        d = sub / f"patch_{index:02d}_level_{level:.2f}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "raw").mkdir(parents=True, exist_ok=True)
        return d

    def record_patch_raw_frame(
        self,
        phase: str,
        round_num: int | None,
        index: int,
        level: float,
        jpeg_bytes: bytes,
    ) -> None:
        try:
            d = self._patch_dir(phase, round_num, index, level)
            key = str(d)
            self._patch_raw_counts[key] = self._patch_raw_counts.get(key, 0) + 1
            n = self._patch_raw_counts[key]
            (d / "raw" / f"frame_{n:04d}.jpg").write_bytes(jpeg_bytes)
        except Exception:
            pass

    def record_patch_averaged(
        self,
        phase: str,
        round_num: int | None,
        index: int,
        level: float,
        frame_uint8: np.ndarray | None,
        measured: dict[str, Any],
    ) -> None:
        try:
            d = self._patch_dir(phase, round_num, index, level)
            if frame_uint8 is not None:
                _save_array_jpeg(frame_uint8, d / "averaged.jpg")
            (d / "measured.json").write_text(json.dumps(measured, default=_json_default, indent=2))
        except Exception:
            pass

    def record_anchor(self, seq: int, label: str, dng_bytes: bytes) -> None:
        try:
            (self.root / "anchors" / f"seq_{seq}_{label}.dng").write_bytes(dng_bytes)
        except Exception:
            pass

    # ── final result ──

    def record_result(
        self,
        icc_bytes: bytes,
        before_png_bytes: bytes,
        after_png_bytes: bytes,
        summary: dict[str, Any],
    ) -> None:
        try:
            (self.root / "result" / "profile.icc").write_bytes(icc_bytes)
            (self.root / "result" / "before.png").write_bytes(before_png_bytes)
            (self.root / "result" / "after.png").write_bytes(after_png_bytes)
            (self.root / "result" / "summary.json").write_text(
                json.dumps(summary, default=_json_default, indent=2)
            )
        except Exception:
            pass

    def finalize(self, ok: bool, summary: dict[str, Any]) -> None:
        """Write meta.json LAST. This file's presence marks completion;
        listings filter sessions without it, so a crashed run never shows.
        """
        meta = {
            "id": self.id,
            "mode": self.mode,
            "started_at": self.started_at,
            "finished_at": datetime.now(tz=timezone.utc).isoformat(),
            "ok": ok,
            **summary,
        }
        try:
            (self.root / "meta.json").write_text(
                json.dumps(meta, default=_json_default, indent=2)
            )
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Listing / loading
# ─────────────────────────────────────────────────────────────────────────────


def list_sessions(base_dir: Path | None = None) -> list[dict[str, Any]]:
    """Return completed sessions, newest first. Partial (no meta.json) skipped."""
    base = (base_dir or sessions_dir())
    if not base.exists():
        return []
    entries: list[dict[str, Any]] = []
    for d in base.iterdir():
        if not d.is_dir():
            continue
        meta = d / "meta.json"
        if not meta.exists():
            continue
        try:
            data = json.loads(meta.read_text())
        except Exception:
            continue
        size = _dir_size_bytes(d)
        entries.append({**data, "size_bytes": size, "path": str(d)})
    entries.sort(key=lambda e: e.get("started_at", ""), reverse=True)
    return entries


def load_session(session_id: str, base_dir: Path | None = None) -> dict[str, Any] | None:
    base = (base_dir or sessions_dir())
    d = base / session_id
    if not (d / "meta.json").exists():
        return None
    try:
        meta = json.loads((d / "meta.json").read_text())
    except Exception:
        return None
    summary_path = d / "result" / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    icc_path = d / "result" / "profile.icc"
    before_path = d / "result" / "before.png"
    after_path = d / "result" / "after.png"
    return {
        "meta": meta,
        "summary": summary,
        "icc_bytes": icc_path.read_bytes() if icc_path.exists() else b"",
        "before_png": before_path.read_bytes() if before_path.exists() else b"",
        "after_png": after_path.read_bytes() if after_path.exists() else b"",
    }


def delete_session(session_id: str, base_dir: Path | None = None) -> bool:
    base = (base_dir or sessions_dir())
    d = base / session_id
    if not d.exists() or not d.is_dir():
        return False
    # Defence-in-depth: refuse to delete anything outside `.sessions/`.
    if base.resolve() not in d.resolve().parents:
        return False
    shutil.rmtree(d)
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _save_array_jpeg(arr: np.ndarray, path: Path) -> None:
    Image.fromarray(arr).save(path, format="JPEG", quality=92)


def _json_default(o: Any) -> Any:
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


def _dir_size_bytes(d: Path) -> int:
    total = 0
    for p in d.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except OSError:
                pass
    return total
