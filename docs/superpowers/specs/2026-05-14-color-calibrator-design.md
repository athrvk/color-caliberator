# Color Calibrator — Design Spec
**Date:** 2026-05-14

## Overview

A free, open-source display **gamma and gray-balance corrector** that uses a mobile phone camera as the measurement device. A local Python web server serves two browser pages: a PC page that displays color patches and a mobile page that captures them via camera. The server uses an **iterative feedback loop** — measure, apply gamma correction, measure again — to converge on an accurate correction without needing the camera to be absolutely calibrated. The final correction is exported as an ICC v2 profile (with VCGT tag) and applied to the display.

What this tool corrects: per-channel gamma curves and gray balance (tone response). What it does not correct: white point, color gamut shape, or hue errors — those require a hardware colorimeter.

Target accuracy: ~2–5 ΔE on gray patches. Suitable for prosumer use — significantly better than uncalibrated displays, not a replacement for hardware colorimeters (<1 ΔE).

---

## Why Iterative Feedback

The fundamental problem with a one-shot approach: the phone camera's own color response is unknown. Fitting a camera RGB→XYZ matrix against pre-defined target XYZ values produces an ideal sRGB profile regardless of what the actual display does.

The iterative approach sidesteps this entirely. The camera only needs to be a **consistent relative sensor** — it compares the same display against itself across rounds. Each round measures error and applies a corrective gamma ramp. The loop converges on the display's actual tone response without ever needing the camera to be absolutely calibrated.

---

## Architecture

Single FastAPI server launched via `uv run python main.py`. Detects local network IP at startup and serves both pages. PC and mobile communicate with the server via WebSocket. The server orchestrates measurement rounds and applies gamma ramps to the display via `dispwin` (ArgyllCMS) subprocess calls.

**External dependency:** ArgyllCMS must be installed on the PC. Server checks for `dispwin` at startup and shows a clear install link if missing.

No build step. No frontend framework. Plain HTML/JS/CSS.

---

## Project Structure

```
color-calibrator/
├── main.py                        # Entry point: starts uvicorn, checks dispwin, prints QR
├── pyproject.toml                 # uv project config
└── src/
    ├── web/
    │   ├── server.py              # FastAPI routes, WebSocket session manager
    │   └── static/
    │       ├── pc.html            # PC calibration wizard page
    │       ├── mobile.html        # Mobile camera capture page
    │       ├── test_chart.png     # Reference image for before/after comparison
    │       └── style.css
    ├── calibration/
    │   ├── patches.py             # 11 gray patch definitions + 3 holdout patches
    │   ├── capture.py             # Frame stability detection via SSNR (numpy)
    │   ├── iterate.py             # Iterative feedback loop: measure → compute ramp → apply
    │   └── ramp.py                # Gamma ramp computation (float LUTs, curve fitting)
    ├── platform/
    │   ├── dispwin.py             # Cross-platform dispwin subprocess wrapper
    │   └── profile.py             # ICC v2 profile + VCGT tag generation (python-lcms2)
    └── util/
        └── qr.py                  # QR code generation for mobile URL
```

---

## Libraries

| Purpose | Library |
|---|---|
| Web server + WebSockets | `fastapi`, `uvicorn` |
| Gamma ramp math, SSNR, curve fitting | `numpy`, `scipy` |
| ICC v2 profile + VCGT tag generation | `python-lcms2` |
| Image I/O, before/after rendering | `Pillow` |
| QR code generation | `qrcode[pil]` |
| Display gamma ramp application | `dispwin` (ArgyllCMS, subprocess) |

`colour-science` is not needed — the iterative approach avoids absolute camera characterization math.

---

## Patch Set (14 total)

**11 gray patches** (calibration targets):
- Input levels: 0%, 10%, 20%, 30%, 40%, 50%, 60%, 70%, 80%, 90%, 100%
- These drive the per-channel gamma ramp correction via curve fitting

**3 holdout gray patches** (post-calibration verification only):
- Input levels: 25%, 50%, 75% (not used in ramp fitting)
- Measured after all rounds complete; ΔE reported to user

Color patches are excluded: they have no role in the iterative gray-balance loop, and reducing patch count shortens capture time.

Each patch is defined by a **display RGB value** and a **target luminance ratio** — the relative brightness a correctly gamma-2.2-corrected display should produce (e.g., 50% gray input → ~18% of peak luminance). Ratios are camera-independent.

---

## Calibration Workflow

### Startup Check
- Server checks for `dispwin` binary on PATH
- If missing: PC page shows install instructions for ArgyllCMS (Windows/Mac/Linux) and blocks start

### Session Setup
1. Server starts, detects local IP, generates QR code for `/mobile`
2. PC page loads, shows QR code + "Scan with your phone"
3. Mobile connects via WebSocket → PC advances to setup screen
4. Setup instructions:
   - Darken the room
   - Position phone 30 cm from screen center, perpendicular
   - **White balance lock:** PC shows full-screen white patch; user locks exposure + WB on phone, taps Ready
5. Server captures 5 stable frames of white patch → records camera's white RGB as session reference (used to normalize all subsequent measurements relative to white)

### Iterative Calibration Loop (up to 3 rounds)

Each round:

**Measurement pass (×11 patches):**
1. Server resets display to current gamma ramp (identity on round 1, corrected on subsequent rounds) via `dispwin -c` then `dispwin -I current_ramp.icc`
2. For each patch:
   - Server sends `show_patch` to PC → full-screen gray patch rendered
   - Server sends `capture` to mobile → mobile streams JPEG frames
   - Server applies SSNR check (≥ 20 dB over 5 consecutive frames, 10s timeout)
   - Server averages stable frames, crops center 25%, records measured RGB
   - Normalizes measured RGB against session white reference to cancel camera WB drift

**Ramp computation (`ramp.py`):**
- For each gray patch, fit the display's effective gamma `γ_d` against the model `measured_luminance ≈ input ^ γ_d` using `scipy.curve_fit`
- Build a 256-entry per-channel float LUT: `LUT(x) = x ^ (γ_target / γ_d)` where `γ_target = 2.2` — this is the pre-warp that makes `display(LUT(x))` track the target gamma
- Compose with the current LUT (if round > 1) via `np.interp`

**Apply ramp (`platform/dispwin.py` + `platform/profile.py`):**
- Convert composed float LUT to uint16 (multiply by 65535, clip, cast) for VCGT emission only
- Embed the 256-entry LUT as a VCGT tag in a temporary ICC profile
- Call `dispwin -I temp_ramp.icc` to apply to display immediately
- Display now shows corrected output for the next measurement round

**Convergence check:**
- Compute mean ΔE between measured and target luminance ratios for all gray patches
- If mean ΔE < 1.0 or round == 3: stop iterating
- Otherwise: run another round

### Post-Calibration Verification
- Server measures 3 holdout patches (not used in ramp fitting)
- Computes ΔE between measured and target luminance ratios
- Reports to user: "Average ΔE: 2.1 (lower is better; hardware colorimeters achieve <1)"
- Note: ΔE here measures gray tone response accuracy, not perceptual color accuracy vs. an absolute reference

### Profile Export
- Final float LUT converted to uint16 and embedded as VCGT tag in ICC v2 profile via `python-lcms2`
- Profile also includes: white point (D65), sRGB primaries as baseline matrix, `desc` and `cprt` tags
- Server sends profile as base64 to PC page for download

---

## PC Page — UI Flow

| Step | Screen |
|---|---|
| 1 | ArgyllCMS check — if missing, show install link; if present, show QR code |
| 2 | QR code + "Scan with your phone" |
| 3 | Setup: room darkening + phone position + WB lock step (white patch, user locks, taps Ready) |
| 4 | Round indicator (Round 1 of up to 3) + patch progress (N of 11) + SSNR quality indicator |
| 5 | Between rounds: "Applying correction…" feedback while dispwin runs |
| 6 | Before/after comparison: `test_chart.png` rendered side-by-side via Pillow — uncorrected left, ramp-corrected right. ΔE verification shown below. |
| 7 | Download `.icc` button + platform-specific install + activation instructions |

---

## Mobile Page — UI Flow

| Step | Screen |
|---|---|
| 1 | Camera viewfinder fullscreen |
| 2 | "Point at the white screen, lock exposure and white balance, then tap Ready" |
| 3 | During each round: patch progress (N of 11) + SSNR quality bar (green = stable, yellow = waiting) |
| 4 | Green flash on each successful patch capture |
| 5 | "Round complete — applying correction, please wait" between rounds |
| 6 | "All done — check your PC" screen |

---

## Data Flow Diagram

```
PC browser              Server                    Mobile browser
─────────               ──────                    ──────────────
Load /             →    Check dispwin
                   ←    QR code (local IP)
                                          ←       Scan QR, load /mobile
                   ←    WS: mobile_connected
Show white patch
User locks WB
                   →    WS: ready
                        Capture white ref (5 frames, SSNR check)
                        Record white_rgb

[Round 1..3]
                   ←    WS: show_patch(rgb)        WS: capture →
                                                   Stream JPEG frames
                        SSNR check
                        Average + normalize vs white_rgb
                   ←    WS: patch_done(n/11)   ←   WS: patch_done(n/11)
[11 patches done]
                        ramp.py: fit correction curve
                        platform/: apply ramp to display
                   ←    WS: round_done(delta_e, round_n)
[if not converged: next round]

                        Measure 3 holdout patches → ΔE
                        platform/profile.py: build ICC v2 + VCGT
                   ←    WS: result(icc_b64, delta_e)
Show before/after + ΔE
Download .icc
```

---

## Gamma Ramp Computation (`calibration/ramp.py`)

LUTs stay as float arrays (values in [0.0, 1.0]) throughout the pipeline. Conversion to uint16 happens only at VCGT export.

```python
import numpy as np
from scipy.optimize import curve_fit

# input_levels: [0.0, 0.1, 0.2, ..., 1.0] — display RGB input (skip 0.0 for curve_fit)
# measured_luma: camera-measured luminance for each patch (normalized vs white)
# Model: measured = input^γ_d. Solve for γ_d, then LUT(x) = x^(2.2/γ_d).

def fit_correction(input_levels, measured_luma, target_gamma=2.2):
    # Exclude black (input=0) — camera black level is unreliable
    mask = input_levels > 0
    (gamma_d,), _ = curve_fit(
        lambda x, g: x**g, input_levels[mask], measured_luma[mask],
        p0=[2.2], bounds=(0.5, 5.0),
    )
    exponent = target_gamma / gamma_d
    lut = np.clip(np.linspace(0, 1, 256) ** exponent, 0, 1)
    return lut  # float [0, 1]

def compose_luts(prev_lut: np.ndarray, new_lut: np.ndarray) -> np.ndarray:
    """Apply new_lut on top of prev_lut. Both are float [0, 1], length 256."""
    x = np.linspace(0, 1, 256)
    return np.interp(new_lut, x, prev_lut)

def lut_to_vcgt(lut: np.ndarray) -> np.ndarray:
    """Convert float [0, 1] LUT to uint16 for ICC VCGT tag."""
    return np.clip(lut * 65535, 0, 65535).astype(np.uint16)
```

---

## dispwin Wrapper (`platform/dispwin.py`)

```python
import subprocess, shutil

def find_dispwin() -> str | None:
    return shutil.which('dispwin')

def apply_ramp(profile_path: str, display_index: int = 1):
    subprocess.run(['dispwin', f'-d{display_index}', '-I', profile_path], check=True)

def clear_ramp(display_index: int = 1):
    subprocess.run(['dispwin', f'-d{display_index}', '-c'], check=True)
```

---

## SSNR Formula

```python
# For 5 consecutive frames:
luminance = [0.299*R + 0.587*G + 0.114*B for each frame center crop]
ssnr_db = 20 * np.log10(np.mean(luminance) / np.std(luminance))
# Accept if ssnr_db >= 20
```

---

## Error Handling

| Scenario | Behavior |
|---|---|
| `dispwin` not found at startup | Block start; show ArgyllCMS install link for each platform |
| `dispwin` subprocess fails mid-session | Surface error; offer to retry or skip ramp application |
| Mobile disconnects mid-capture | Reset session; PC shows QR to reconnect |
| SSNR < 20 dB after 10s | Skip patch, flag it; if > 3 skipped warn user |
| White reference frame too dark | Block start; prompt user to check room and phone position |
| ΔE doesn't converge after 3 rounds | Inform user; offer to download best-effort profile anyway |

---

## ICC Profile Output

- Format: ICC v2 (`.icc`)
- VCGT tag: 256-entry per-channel gamma ramp (the correction)
- Matrix: sRGB primaries (baseline; VCGT handles the actual correction)
- White point: D65
- Install + activation instructions per platform:
  - **Windows:** double-click → Install Profile → open `colorcpl` (Win+R) → Devices tab → select display → tick "Use my settings" → set as default
  - **Mac:** double-click → ColorSync Utility installs and activates automatically
  - **Linux (X11):** `xcalib profile.icc` or copy to `~/.local/share/icc/` and activate via GNOME/KDE display settings
  - **Linux (Wayland):** copy to `~/.local/share/icc/`, activate via `colormgr`

---

## Known Limitations (surfaced in UI)

- Corrects gamma and gray balance only — does not correct white point, gamut, or hue errors
- Accuracy ~2–5 ΔE on gray tone response; hardware colorimeters achieve <1 ΔE
- Requires ArgyllCMS (`dispwin`) installed on the calibrating PC
- Room must be darkened during calibration
- Camera RAW not supported — browsers do not expose RAW capture
- Black level (near-zero) measurement unreliable on phone cameras; excluded from curve fitting
- VCGT gamma ramp is a 1D correction per channel — cannot correct hue errors or gamut shape
