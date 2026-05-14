import shutil
import subprocess


def find_dispwin() -> str | None:
    """Return path to dispwin binary, or None if not on PATH."""
    return shutil.which("dispwin")


def apply_ramp(profile_path: str, display_index: int = 1) -> None:
    """Load an ICC profile's VCGT tag as the display's VideoLUT."""
    # Pass the .icc as a positional arg to load into VideoLUT without installing system-wide.
    # -I installs into the OS profile store (requires elevated rights); positional arg does not.
    subprocess.run(["dispwin", f"-d{display_index}", profile_path], check=True)


def clear_ramp(display_index: int = 1) -> None:
    """Reset the display VideoLUT to linear (identity)."""
    subprocess.run(["dispwin", f"-d{display_index}", "-c"], check=True)
