import shutil
import subprocess
import time


def find_dispwin() -> str | None:
    """Return path to dispwin binary, or None if not on PATH."""
    return shutil.which("dispwin")


def _run(args: list[str], retries: int = 3, delay: float = 0.5) -> None:
    """Run dispwin, retrying on failure (Windows SetDeviceGammaRamp is occasionally flaky)."""
    last_err = ""
    for attempt in range(retries):
        result = subprocess.run(args, capture_output=True, text=True)
        if result.returncode == 0:
            return
        last_err = result.stderr.strip() or result.stdout.strip() or "(no output)"
        if attempt < retries - 1:
            time.sleep(delay)
    raise RuntimeError(f"dispwin failed after {retries} attempts: {last_err}\nCommand: {args}")


def apply_ramp(profile_path: str, display_index: int = 1) -> None:
    """Load an ICC profile's VCGT tag as the display's VideoLUT."""
    _run(["dispwin", f"-d{display_index}", profile_path])


def clear_ramp(display_index: int = 1) -> None:
    """Reset the display VideoLUT to linear (identity)."""
    _run(["dispwin", f"-d{display_index}", "-c"])
