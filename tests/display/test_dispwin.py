from unittest.mock import patch

from display.dispwin import apply_ramp, clear_ramp, find_dispwin


def test_find_dispwin_returns_none_when_absent():
    with patch("shutil.which", return_value=None):
        assert find_dispwin() is None


def test_find_dispwin_returns_path_when_present():
    with patch("shutil.which", return_value="/usr/bin/dispwin"):
        assert find_dispwin() == "/usr/bin/dispwin"


def test_apply_ramp_calls_dispwin_with_correct_args():
    with patch("subprocess.run") as mock_run:
        apply_ramp("/tmp/cal.icc", display_index=1)
        mock_run.assert_called_once_with(
            ["dispwin", "-d1", "-I", "/tmp/cal.icc"], check=True
        )


def test_apply_ramp_uses_display_index():
    with patch("subprocess.run") as mock_run:
        apply_ramp("/tmp/cal.icc", display_index=2)
        args = mock_run.call_args[0][0]
        assert "-d2" in args


def test_clear_ramp_calls_dispwin_reset():
    with patch("subprocess.run") as mock_run:
        clear_ramp(display_index=1)
        mock_run.assert_called_once_with(
            ["dispwin", "-d1", "-c"], check=True
        )
