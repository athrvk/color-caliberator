from unittest.mock import MagicMock, patch

from display.dispwin import apply_ramp, clear_ramp, find_dispwin


def _mock_run_ok():
    """Return a mock subprocess.CompletedProcess with returncode=0."""
    m = MagicMock()
    m.returncode = 0
    m.stderr = ""
    m.stdout = ""
    return m


def test_find_dispwin_returns_none_when_absent():
    with patch("shutil.which", return_value=None):
        assert find_dispwin() is None


def test_find_dispwin_returns_path_when_present():
    with patch("shutil.which", return_value="/usr/bin/dispwin"):
        assert find_dispwin() == "/usr/bin/dispwin"


def test_apply_ramp_calls_dispwin_with_correct_args():
    with patch("subprocess.run", return_value=_mock_run_ok()) as mock_run:
        apply_ramp("/tmp/cal.icc", display_index=1)
        args = mock_run.call_args[0][0]
        assert args == ["dispwin", "-d1", "/tmp/cal.icc"]


def test_apply_ramp_uses_display_index():
    with patch("subprocess.run", return_value=_mock_run_ok()) as mock_run:
        apply_ramp("/tmp/cal.icc", display_index=2)
        args = mock_run.call_args[0][0]
        assert "-d2" in args


def test_clear_ramp_calls_dispwin_reset():
    with patch("subprocess.run", return_value=_mock_run_ok()) as mock_run:
        clear_ramp(display_index=1)
        args = mock_run.call_args[0][0]
        assert args == ["dispwin", "-d1", "-c"]


def test_apply_ramp_raises_on_nonzero_exit():
    m = MagicMock()
    m.returncode = 1
    m.stderr = "Failed to set VideoLUTs"
    m.stdout = ""
    with patch("subprocess.run", return_value=m):
        try:
            apply_ramp("/tmp/cal.icc")
            assert False, "Should have raised"
        except RuntimeError as e:
            assert "Failed to set VideoLUTs" in str(e)
