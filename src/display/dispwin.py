import shutil
import subprocess


def find_dispwin() -> str | None:
    """Return path to dispwin binary, or None if not on PATH."""
    return shutil.which("dispwin")


def apply_ramp(profile_path: str, display_index: int = 1) -> None:
    """Load an ICC profile's VCGT tag as the display's VideoLUT."""
    subprocess.run(["dispwin", f"-d{display_index}", "-I", profile_path], check=True)


def clear_ramp(display_index: int = 1) -> None:
    """Reset the display VideoLUT to linear (identity)."""
    subprocess.run(["dispwin", f"-d{display_index}", "-c"], check=True)
