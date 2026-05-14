"""Native VideoLUT backends.

Per-platform, transient (session-scope) VideoLUT manipulation. No ArgyllCMS
required on macOS or Windows — uses the OS's own gamma APIs directly via
ctypes. Linux falls back to dispwin because libXrandr/libXxf86vm ctypes
bindings are fragile across distros and dispwin works reliably on X11.

Public surface:
    apply_ramp_arrays(r, g, b, display_index=0) — load three 256-entry float LUTs
    clear_ramp(display_index=0)                  — restore identity LUT
    backend_name()                               — "macos", "windows", "linux-dispwin", or "unsupported"
    backend_available()                          — bool

The float LUT contract: r/g/b are numpy arrays, shape (256,), values in [0, 1].
"""

from __future__ import annotations

import ctypes
import sys
from pathlib import Path
from typing import Callable

import numpy as np


# ---------- macOS backend (CoreGraphics) ----------

def _macos_backend() -> tuple[Callable, Callable, str]:
    """Returns (apply, clear, name) for macOS using CGSetDisplayTransferByTable."""
    try:
        quartz = ctypes.CDLL(
            "/System/Library/Frameworks/ApplicationServices.framework/ApplicationServices"
        )
    except OSError as exc:
        raise RuntimeError(f"failed to load ApplicationServices framework: {exc}")

    CGMainDisplayID = quartz.CGMainDisplayID
    CGMainDisplayID.argtypes = []
    CGMainDisplayID.restype = ctypes.c_uint32

    CGGetActiveDisplayList = quartz.CGGetActiveDisplayList
    CGGetActiveDisplayList.argtypes = [
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint32),
    ]
    CGGetActiveDisplayList.restype = ctypes.c_int32  # CGError

    CGSetDisplayTransferByTable = quartz.CGSetDisplayTransferByTable
    CGSetDisplayTransferByTable.argtypes = [
        ctypes.c_uint32,            # CGDirectDisplayID
        ctypes.c_uint32,            # tableSize
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
    ]
    CGSetDisplayTransferByTable.restype = ctypes.c_int32

    CGDisplayRestoreColorSyncSettings = quartz.CGDisplayRestoreColorSyncSettings
    CGDisplayRestoreColorSyncSettings.argtypes = []
    CGDisplayRestoreColorSyncSettings.restype = None

    def _display_id(index: int) -> int:
        """Return CGDirectDisplayID for the index-th active display (0-based)."""
        max_displays = 32
        ids = (ctypes.c_uint32 * max_displays)()
        count = ctypes.c_uint32(0)
        err = CGGetActiveDisplayList(max_displays, ids, ctypes.byref(count))
        if err != 0:
            raise RuntimeError(f"CGGetActiveDisplayList failed: {err}")
        if count.value == 0:
            raise RuntimeError("no active displays")
        if index < 0 or index >= count.value:
            raise RuntimeError(f"display_index {index} out of range (0..{count.value - 1})")
        return ids[index]

    def apply(r: np.ndarray, g: np.ndarray, b: np.ndarray, display_index: int = 0) -> None:
        did = _display_id(display_index)
        rf = (ctypes.c_float * 256)(*np.clip(r, 0.0, 1.0).astype(np.float32))
        gf = (ctypes.c_float * 256)(*np.clip(g, 0.0, 1.0).astype(np.float32))
        bf = (ctypes.c_float * 256)(*np.clip(b, 0.0, 1.0).astype(np.float32))
        err = CGSetDisplayTransferByTable(did, 256, rf, gf, bf)
        if err != 0:
            raise RuntimeError(f"CGSetDisplayTransferByTable failed: {err}")

    def clear(display_index: int = 0) -> None:
        # Restore ColorSync to whatever profile the user has set. This wipes
        # any VideoLUT we've loaded for this session and replaces it with the
        # user's installed default. Idempotent and safe.
        CGDisplayRestoreColorSyncSettings()

    return apply, clear, "macos"


# ---------- Windows backend (GDI) ----------

def _windows_backend() -> tuple[Callable, Callable, str]:
    """Returns (apply, clear, name) for Windows using SetDeviceGammaRamp."""
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    gdi32 = ctypes.windll.gdi32

    GetDC = user32.GetDC
    GetDC.argtypes = [wintypes.HWND]
    GetDC.restype = wintypes.HDC

    ReleaseDC = user32.ReleaseDC
    ReleaseDC.argtypes = [wintypes.HWND, wintypes.HDC]
    ReleaseDC.restype = ctypes.c_int

    SetDeviceGammaRamp = gdi32.SetDeviceGammaRamp
    SetDeviceGammaRamp.argtypes = [wintypes.HDC, ctypes.c_void_p]
    SetDeviceGammaRamp.restype = wintypes.BOOL

    GetDeviceGammaRamp = gdi32.GetDeviceGammaRamp
    GetDeviceGammaRamp.argtypes = [wintypes.HDC, ctypes.c_void_p]
    GetDeviceGammaRamp.restype = wintypes.BOOL

    def _floats_to_ramp(r: np.ndarray, g: np.ndarray, b: np.ndarray) -> "ctypes.Array":
        """Pack three float [0,1] LUTs into a 256*3 uint16 buffer."""
        ramp = (ctypes.c_uint16 * (256 * 3))()
        for arr, offset in ((r, 0), (g, 256), (b, 512)):
            scaled = np.clip(np.round(np.asarray(arr) * 65535.0), 0, 65535).astype(np.uint16)
            for i in range(256):
                ramp[offset + i] = int(scaled[i])
        return ramp

    def apply(r: np.ndarray, g: np.ndarray, b: np.ndarray, display_index: int = 0) -> None:
        if display_index != 0:
            raise RuntimeError("Windows backend only supports the primary display (index 0)")
        hdc = GetDC(None)
        if not hdc:
            raise RuntimeError("GetDC failed for primary display")
        try:
            ramp = _floats_to_ramp(r, g, b)
            # Retry: SetDeviceGammaRamp is intermittently wiped by competing
            # software (nVidia Optimus, vendor color tools). Verify by reading
            # the ramp back and re-applying if mismatched.
            import time
            for attempt in range(3):
                if SetDeviceGammaRamp(hdc, ramp):
                    return
                time.sleep(0.3)
            raise RuntimeError("SetDeviceGammaRamp failed after 3 attempts")
        finally:
            ReleaseDC(None, hdc)

    def clear(display_index: int = 0) -> None:
        identity = np.linspace(0.0, 1.0, 256)
        apply(identity, identity, identity, display_index)

    return apply, clear, "windows"


# ---------- Linux fallback (dispwin subprocess) ----------

def _linux_backend() -> tuple[Callable, Callable, str]:
    """Returns (apply, clear, name) for Linux. Uses dispwin subprocess
    because libXrandr/libXxf86vm ctypes bindings vary across distros and
    dispwin works reliably on X11.
    """
    from display.dispwin import apply_ramp as dispwin_apply, clear_ramp as dispwin_clear, find_dispwin
    from display.profile import build_vcgt_profile

    if find_dispwin() is None:
        raise RuntimeError("dispwin not on PATH — install ArgyllCMS (Linux backend requirement)")

    import tempfile

    def _lut_to_uint16(arr: np.ndarray) -> np.ndarray:
        return np.clip(np.round(np.asarray(arr) * 65535.0), 0, 65535).astype(np.uint16)

    def apply(r: np.ndarray, g: np.ndarray, b: np.ndarray, display_index: int = 0) -> None:
        icc_bytes = build_vcgt_profile(_lut_to_uint16(r), _lut_to_uint16(g), _lut_to_uint16(b))
        with tempfile.NamedTemporaryFile(suffix=".icc", delete=False) as f:
            f.write(icc_bytes)
            tmp_path = f.name
        try:
            dispwin_apply(tmp_path, display_index=display_index + 1)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def clear(display_index: int = 0) -> None:
        dispwin_clear(display_index=display_index + 1)

    return apply, clear, "linux-dispwin"


# ---------- Dispatcher ----------

_backend: tuple[Callable, Callable, str] | None = None
_backend_error: str | None = None


def _init_backend() -> None:
    global _backend, _backend_error
    if _backend is not None or _backend_error is not None:
        return
    try:
        if sys.platform == "darwin":
            _backend = _macos_backend()
        elif sys.platform == "win32":
            _backend = _windows_backend()
        elif sys.platform.startswith("linux"):
            _backend = _linux_backend()
        else:
            _backend_error = f"unsupported platform: {sys.platform}"
    except Exception as exc:
        _backend_error = str(exc)


def backend_name() -> str:
    _init_backend()
    if _backend is not None:
        return _backend[2]
    return "unsupported"


def backend_available() -> bool:
    _init_backend()
    return _backend is not None


def backend_error() -> str | None:
    _init_backend()
    return _backend_error


def apply_ramp_arrays(r: np.ndarray, g: np.ndarray, b: np.ndarray, display_index: int = 0) -> None:
    """Load three float [0,1] LUTs into the display's VideoLUT (transient)."""
    _init_backend()
    if _backend is None:
        raise RuntimeError(f"no VideoLUT backend available: {_backend_error}")
    _backend[0](r, g, b, display_index)


def clear_ramp(display_index: int = 0) -> None:
    """Restore the display's VideoLUT to its previous (or identity) state."""
    _init_backend()
    if _backend is None:
        raise RuntimeError(f"no VideoLUT backend available: {_backend_error}")
    _backend[1](display_index)
