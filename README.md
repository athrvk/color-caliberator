# color-calibrator

**Your phone is now a colorimeter.** Almost.

A free, open-source display calibrator that uses your phone camera to fix the
gamma and gray balance of your monitor. No $300 hardware puck. No subscription.
Just a phone, a PC, and a dark room.

---

## What it does (the short version)

1. Your phone watches your PC screen.
2. Your PC shows a bunch of gray patches.
3. The server measures how wrong each one looks, generates a correction curve,
   and pushes it to your display via ArgyllCMS.
4. Repeat until the screen stops lying to you.
5. Out pops an `.icc` profile you can install with one double-click.

You get **~2–5 ΔE accuracy on grays**. A real hardware colorimeter hits <1 ΔE.
For most prosumers staring at miscalibrated panels, this is a huge upgrade
over "trust the factory."

## What it does NOT do

This is a **1D per-channel tone correction** — gamma and gray balance only. It
cannot fix:

- White point shifts
- Color gamut shape
- Hue errors

Those require a spectrophotometer or proper colorimeter. If you're grading
Hollywood films, buy an i1 Display Pro. If you're a developer/photographer who
just wants your monitor to stop looking yellow at 50% gray, you're in the
right place.

---

## Two modes

**Gamma mode (default).** Automated, ~2 min. Fixes per-channel gamma and gray
balance from 11 gray patches. Output: VCGT-only ICC. ΔE ~3-5 on grays.

**Color mode (opt-in).** Adds 4 manual RAW (DNG) captures plus 33 patches.
~5 minutes total. Fixes gray balance AND extends to white-point + primary
chromaticities. Output: matrix-shaper ICC with TRC + VCGT. ΔE ~3 across the
full gamut. Requires a phone that can capture DNG (iPhone Pro with ProRAW
enabled, or Android via Open Camera / Halide / native pro modes).

**Color mode prerequisites:**
- Phone must capture DNG (RAW).
- Phone JPEG output must be **sRGB**, not Display P3. iOS: Settings → Camera → Formats → Most Compatible. Otherwise the JPEG-stream patches are decoded with the wrong reverse curve and color accuracy drops to ~6 ΔE.
- **iOS only:** the iOS Photos app silently transcodes DNG → JPEG when you pick a file in the browser. After shooting RAW, open Photos, share the photo, **Save to Files**, then upload from Files (not Photos). On Android: most file pickers preserve DNG; use whichever app saves the original.

Choose mode on the PC setup screen.

---

## Quick start (monkey mode)

```bash
# 0. Install uv (if you don't have it)
curl -LsSf https://astral.sh/uv/install.sh | sh   # macOS / Linux
# Windows: https://docs.astral.sh/uv/getting-started/installation/

# 1. (Linux only) install ArgyllCMS — provides the `dispwin` binary we use
#    for VideoLUT manipulation on X11. macOS and Windows use native OS APIs
#    (CoreGraphics / GDI) directly — no extra install needed.
sudo apt install argyll         # Linux (Debian/Ubuntu)
# Other distros: install the `argyll` or `argyllcms` package.

# 2. Clone, install deps, and run
git clone https://github.com/athrvk/color-caliberator.git
cd color-caliberator
uv sync
uv run python main.py
```

The server prints two URLs. Open the **first** on your PC, scan the QR with
your phone, follow the on-screen wizard. The whole process takes ~2 minutes.

Both browsers will yell about a self-signed certificate. **Click through.** The
cert is generated locally for your LAN IP and only exists so mobile Safari
will hand over the camera (`getUserMedia` requires HTTPS).

---

## How it actually works (dev mode)

The fundamental trick: **the camera does not need to be calibrated.**

A naive approach would point the phone at the screen, fit a camera RGB → XYZ
matrix against known target XYZs, and call it a day. This produces a generic
sRGB profile regardless of what the actual display does — the camera's own
gamma cancels everything out.

We sidestep that with an **iterative feedback loop**:

```
white reference  ──┐
                   ▼
   ┌──► show 11 gray patches
   │    measure relative luminance vs white
   │    fit  measured = level ^ γ_d
   │    build  LUT(x) = x ^ (γ_target / γ_d)
   │    compose with previous LUT
   │    push to display via dispwin
   │       │
   └───────┘  up to 3 rounds, or until ΔE < 1.0
              │
              ▼
   3 holdout patches → final ΔE report
              │
              ▼
   ICC v2 profile with VCGT tag → download
```

The camera only needs to be **consistent across rounds**. It's comparing the
display against itself, not against an absolute reference. Each round halves
the residual error.

### Color mode (Option B hybrid)

When color mode is selected, the loop adds an **anchor phase** before measurement:

```
manual capture: white + R + G + B DNGs  ──┐
                                          ▼
   read DNG ForwardMatrix2 / AsShotNeutral
   build camera-RGB → XYZ_D50 transform
                                          │
                                          ▼
   automated patch stream (33 single-channel patches)
   reverse sRGB; normalize vs white; project to primary XYZ
   fit per-channel TRC in XYZ space
                                          │
                                          ▼
   matrix-shaper ICC v2 (rXYZ/gXYZ/bXYZ + rTRC/gTRC/bTRC + VCGT)
```

The DNG tags give us factory-calibrated camera spectral data — same trick
Apple TV Color Balance uses internally with iPhone sensors.

### Stack

| Layer | Tech |
|---|---|
| Server | FastAPI + uvicorn, async, single-session |
| Transport | WebSockets (one reader task per socket, queue dispatch) |
| Math | numpy + scipy (`curve_fit` power-law) |
| Display I/O | macOS `CGSetDisplayTransferByTable` / Windows `SetDeviceGammaRamp` (ctypes) / Linux `dispwin` |
| ICC output | Raw bytes via `struct`, includes Apple VCGT tag |
| TLS | Self-signed cert via `cryptography`, auto-regen on LAN-IP change |
| Mobile | Plain HTML/JS, `getUserMedia`, JPEG frames at 5 fps over WS |
| QR | `qrcode[pil]` |
| Tests | pytest + pytest-asyncio (54 tests) |

### Architecture in 30 seconds

```
main.py                       # backend probe, TLS cert, uvicorn
src/calibration/
  patches.py                  # 11 gray + 3 holdout patch definitions
  capture.py                  # SSNR stability, BT.601 luma, JPEG decode
  ramp.py                     # Gamma fit, LUT compose, VCGT conversion
  iterate.py                  # Async calibration loop (the brain)
src/display/
  videolut.py                 # Native VideoLUT backends (macOS / Windows / Linux dispatcher)
  dispwin.py                  # ArgyllCMS subprocess wrapper (Linux backend)
  profile.py                  # ICC v2 byte-level builder + VCGT tag
src/util/
  qr.py                       # QR codes for the mobile URL
  tls.py                      # Self-signed cert covering LAN IP
src/web/
  server.py                   # FastAPI app + WebSocket session
  static/pc.html              # PC wizard
  static/mobile.html          # Mobile camera page
tests/                        # 54 tests, all green
```

### Key design notes for contributors

- **Single-reader invariant on each WebSocket.** Concurrent `receive_text`
  calls raise `RuntimeError` in Starlette. One reader task per socket pushes
  parsed JSON into an `asyncio.Queue`; everything else consumes from there.
- **`asyncio.to_thread` around VideoLUT calls** — macOS/Windows ctypes calls
  and the Linux dispwin subprocess can stall briefly. Don't block the event
  loop while frames are still streaming in.
- **Stale frames are drained** between patches. Mobile streams at 5 fps; a few
  frames are always in flight when `stop_capture` lands. Skip them.
- **Black level is dropped** from the curve fit. Phone cameras are too noisy
  near zero for the fit to be meaningful.
- **Best LUTs are checkpointed by ΔE** across rounds. If round 3 diverges
  (rare, but possible with a wobbly camera), the final profile uses the best
  intermediate.

---

## Requirements

- Python 3.11+ and [`uv`](https://docs.astral.sh/uv/)
- macOS or Windows: nothing extra. Linux/X11: ArgyllCMS (`dispwin` on PATH). Linux/Wayland: not supported — log into an X11 session.
- A phone with a modern browser (Chrome / Safari iOS 14.5+)
- PC and phone on the same LAN
- A room you can darken

## Testing

```bash
uv run pytest -v
```

## Known caveats

- 1D correction only — gamma and gray balance, nothing else.
- `dispwin -d1` is hard-coded for the primary display. Multi-monitor users
  need to edit `src/display/dispwin.py`.
- Browsers do not expose RAW camera data, so we work in 8-bit JPEG. Good
  enough.
- iOS Safari needs the **Start Camera** tap because `getUserMedia` requires a
  user gesture in secure-but-self-signed contexts.

## License

Open source. Use it. Hack it. Send pull requests. Don't blame me if your
monitor still looks bad — buy a real colorimeter.
