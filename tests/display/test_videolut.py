"""Dispatcher + sanity tests for native VideoLUT backends.

We don't actually touch hardware here — that's done by smoke runs on real
machines. These tests just verify the dispatcher selects a backend per
platform and that ctypes function pointers wire up without import errors.
"""

import sys

import numpy as np
import pytest

from display import videolut


def test_backend_name_matches_platform():
    name = videolut.backend_name()
    if sys.platform == "darwin":
        assert name == "macos"
    elif sys.platform == "win32":
        assert name == "windows"
    elif sys.platform.startswith("linux"):
        assert name in ("linux-dispwin", "unsupported")
    else:
        assert name == "unsupported"


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS-only backend")
def test_macos_backend_loads_quartz():
    # Module-level CDLL bind succeeded if backend_available() is True; we don't
    # actually invoke the LUT setter here (would alter the user's display).
    assert videolut.backend_available()


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-only backend")
def test_windows_backend_loads_gdi():
    assert videolut.backend_available()


def test_apply_ramp_arrays_rejects_when_no_backend(monkeypatch):
    # Force the dispatcher into "no backend" state and verify the public
    # surface raises with the recorded error.
    monkeypatch.setattr(videolut, "_backend", None)
    monkeypatch.setattr(videolut, "_backend_error", "synthetic test")
    with pytest.raises(RuntimeError, match="synthetic test"):
        ident = np.linspace(0, 1, 256)
        videolut.apply_ramp_arrays(ident, ident, ident)


def test_clear_ramp_rejects_when_no_backend(monkeypatch):
    monkeypatch.setattr(videolut, "_backend", None)
    monkeypatch.setattr(videolut, "_backend_error", "synthetic test")
    with pytest.raises(RuntimeError, match="synthetic test"):
        videolut.clear_ramp()
