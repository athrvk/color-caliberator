import asyncio
import base64
import json
import logging
import sys
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

log = logging.getLogger("color-calibrator")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

MAX_RAW_UPLOAD_BYTES = 200 * 1024 * 1024  # 200 MB — bigger than any phone DNG

sys.path.insert(0, str(Path(__file__).parent.parent))

from display.videolut import apply_ramp_arrays, backend_available, backend_error, clear_ramp
from util.qr import generate_qr_png
from util.tls import detect_lan_ip

import io
import numpy as np
from PIL import Image
from calibration.ramp import apply_lut_to_image

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI()
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ---------- Session state (single session) ----------

class Endpoint:
    """One side of the session: WebSocket + inbound message queue."""

    def __init__(self, ws: WebSocket):
        self.ws = ws
        self.queue: asyncio.Queue[dict] = asyncio.Queue()
        self.closed = asyncio.Event()


class Session:
    def __init__(self):
        self.pc: Endpoint | None = None
        self.mobile: Endpoint | None = None
        self.calibration_task: asyncio.Task | None = None
        self.lock = asyncio.Lock()
        self.mode: str = "gamma"
        self.anchors: dict[int, bytes] = {}
        self.anchor_events: dict[int, asyncio.Event] = {
            0: asyncio.Event(), 1: asyncio.Event(),
            2: asyncio.Event(), 3: asyncio.Event(),
        }
        # Final corrected LUTs after calibration completes. Lets the PC result
        # screen flip the live VideoLUT between corrected/original for a
        # whole-desktop before/after preview (TruHu-style).
        self.final_luts: tuple = ()


session = Session()


async def _reader(endpoint: Endpoint) -> None:
    """Single task per WebSocket: pump parsed messages into the queue."""
    try:
        while True:
            text = await endpoint.ws.receive_text()
            try:
                msg = json.loads(text)
            except json.JSONDecodeError:
                log.warning("ignored non-JSON WS frame (%d bytes)", len(text))
                continue
            await endpoint.queue.put(msg)
    except WebSocketDisconnect:
        pass
    except Exception:
        # Catch-all so the reader task always sets `closed` and never escapes
        # silently into the asyncio task graveyard.
        log.exception("WS reader crashed")
    finally:
        endpoint.closed.set()


# ---------- HTTP routes ----------

@app.get("/")
async def pc_page():
    return FileResponse(STATIC_DIR / "pc.html")


@app.get("/mobile")
async def mobile_page():
    return FileResponse(STATIC_DIR / "mobile.html")


# ---------- WebSocket: PC ----------

@app.websocket("/ws/pc")
async def ws_pc(websocket: WebSocket):
    await websocket.accept()
    endpoint = Endpoint(websocket)
    async with session.lock:
        session.pc = endpoint

    if not backend_available():
        await _send(websocket, {
            "type": "error",
            "message": f"no VideoLUT backend available: {backend_error()}",
        })
        await websocket.close()
        session.pc = None
        return

    ip = detect_lan_ip()
    mobile_url = f"https://{ip}:8765/mobile"
    qr_png = generate_qr_png(mobile_url)
    qr_b64 = base64.b64encode(qr_png).decode()
    await _send(websocket, {"type": "qr_code", "png_b64": qr_b64, "url": mobile_url})

    # Single reader for this socket. All other code consumes from endpoint.queue.
    reader = asyncio.create_task(_reader(endpoint))
    control = asyncio.create_task(_pc_control_loop(endpoint))
    try:
        await endpoint.closed.wait()
    finally:
        reader.cancel()
        control.cancel()
        async with session.lock:
            if session.pc is endpoint:
                session.pc = None


async def _pc_control_loop(pc: "Endpoint") -> None:
    """Consume PC-originated messages: start_calibration + preview toggles."""
    while not pc.closed.is_set():
        msg = await pc.queue.get()
        t = msg.get("type")
        if t == "start_calibration":
            async with session.lock:
                mobile = session.mobile
                if mobile is None:
                    await _send(pc.ws, {"type": "error", "message": "Mobile not connected."})
                    continue
                if session.calibration_task and not session.calibration_task.done():
                    continue  # already running
                session.mode = msg.get("mode", "gamma")
                session.anchors.clear()
                session.final_luts = ()  # invalidate stale preview LUTs
                for ev in session.anchor_events.values():
                    ev.clear()
                session.calibration_task = asyncio.create_task(_run_calibration_task())
        elif t == "preview_corrected":
            # Reload the calibrated VideoLUT for whole-desktop before/after toggle.
            luts = session.final_luts
            if luts:
                try:
                    await asyncio.to_thread(apply_ramp_arrays, *luts)
                except Exception as exc:
                    await _send(pc.ws, {"type": "error", "message": f"preview failed: {exc}"})
        elif t == "preview_original":
            try:
                await asyncio.to_thread(clear_ramp)
            except Exception as exc:
                await _send(pc.ws, {"type": "error", "message": f"preview failed: {exc}"})


# ---------- WebSocket: Mobile ----------

@app.websocket("/ws/mobile")
async def ws_mobile(websocket: WebSocket):
    await websocket.accept()
    endpoint = Endpoint(websocket)
    async with session.lock:
        session.mobile = endpoint
        pc = session.pc

    if pc is not None:
        await _send(pc.ws, {"type": "mobile_connected"})
        # PC's setup screen now shows a Begin button; calibration starts when
        # the user clicks it (PC sends `start_calibration`).

    reader = asyncio.create_task(_reader(endpoint))
    try:
        await endpoint.closed.wait()
    finally:
        reader.cancel()
        async with session.lock:
            if session.mobile is endpoint:
                session.mobile = None
        if pc is not None and not pc.closed.is_set():
            await _send(pc.ws, {"type": "error", "message": "Mobile disconnected."})


@app.post("/upload/raw/{seq}")
async def upload_raw(seq: int, file: UploadFile = File(...)):
    if seq < 0 or seq > 3:
        raise HTTPException(status_code=400, detail="seq must be in [0, 3]")
    # Stream-read with a hard cap so a malicious POST can't OOM the server.
    chunks: list[bytes] = []
    size = 0
    while True:
        chunk = await file.read(1024 * 1024)
        if not chunk:
            break
        size += len(chunk)
        if size > MAX_RAW_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"upload exceeds {MAX_RAW_UPLOAD_BYTES // (1024 * 1024)} MB cap",
            )
        chunks.append(chunk)
    data = b"".join(chunks)
    if len(data) < 1024:
        raise HTTPException(status_code=400, detail="file too small to be a DNG")
    async with session.lock:
        session.anchors[seq] = data
        event = session.anchor_events.get(seq)
    if event is not None:
        event.set()
    log.info("RAW upload seq=%d bytes=%d", seq, len(data))
    return {"ok": True, "seq": seq, "bytes": len(data)}


# ---------- Helpers ----------

async def _send(ws: WebSocket, msg: dict) -> bool:
    """Send a JSON message; swallow dead-socket errors. Returns True on success."""
    try:
        await ws.send_text(json.dumps(msg))
        return True
    except (WebSocketDisconnect, RuntimeError) as exc:
        # RuntimeError "WebSocket is not connected" surfaces when the peer dropped
        # before send. Not fatal — caller usually has nothing better to do.
        log.info("WS send failed (%s): %s", type(exc).__name__, msg.get("type", "?"))
        return False
    except Exception:
        log.exception("WS send unexpected error")
        return False


def _ensure_test_chart(path: Path) -> None:
    if path.exists():
        return
    w, h = 640, 240
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for x in range(w):
        v = int(x / w * 255)
        img[: h // 2, x] = [v, v, v]
    colours = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (0, 255, 255), (255, 0, 255), (255, 165, 0), (128, 0, 128),
    ]
    strip_w = w // len(colours)
    for i, c in enumerate(colours):
        img[h // 2 :, i * strip_w : (i + 1) * strip_w] = c
    Image.fromarray(img).save(path)


def _build_comparison_b64(
    chart_path: Path,
    lut_r: np.ndarray,
    lut_g: np.ndarray,
    lut_b: np.ndarray,
) -> tuple[str, str]:
    """Return (before_b64, after_b64) as PNG base64 strings."""
    img = np.array(Image.open(chart_path).convert("RGB"))
    corrected = apply_lut_to_image(img, lut_r, lut_g, lut_b)

    def to_b64(arr: np.ndarray) -> str:
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    return to_b64(img), to_b64(corrected)


async def _run_calibration_task() -> None:
    """Wraps calibration loop; sends result or error to PC."""
    from calibration.iterate import run_calibration

    pc, mobile = session.pc, session.mobile
    if pc is None or mobile is None:
        return
    mode = session.mode

    async def pc_send(msg: dict) -> None:
        await _send(pc.ws, msg)

    async def mobile_send(msg: dict) -> None:
        await _send(mobile.ws, msg)

    async def mobile_recv() -> dict:
        # Race the queue read against mobile.closed so a disconnected mobile
        # surfaces immediately instead of relying on 10s/180s/300s timeouts
        # in iterate.py to fire one at a time.
        get_task = asyncio.create_task(mobile.queue.get())
        close_task = asyncio.create_task(mobile.closed.wait())
        done, pending = await asyncio.wait(
            {get_task, close_task}, return_when=asyncio.FIRST_COMPLETED
        )
        for t in pending:
            t.cancel()
        if get_task in done and not get_task.cancelled():
            return get_task.result()
        raise RuntimeError("Mobile disconnected during capture.")

    async def mobile_drain() -> None:
        """Discard any queued mobile messages (stale frames between patches)."""
        while not mobile.queue.empty():
            mobile.queue.get_nowait()

    async def wait_for_anchor(seq: int) -> bytes:
        # Fast-path: if the upload already landed, return immediately.
        async with session.lock:
            cached = session.anchors.get(seq)
        if cached is not None:
            return cached
        event = session.anchor_events[seq]
        try:
            await asyncio.wait_for(event.wait(), timeout=300.0)
        except asyncio.TimeoutError as exc:
            raise RuntimeError(f"Timed out waiting for RAW upload seq={seq}") from exc
        async with session.lock:
            data = session.anchors.get(seq)
        if data is None:
            raise RuntimeError(f"Anchor seq={seq} event set but data missing")
        return data

    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        try:
            icc_bytes, delta_e, lut_r, lut_g, lut_b = await run_calibration(
                pc_send, mobile_send, mobile_recv, mobile_drain, Path(tmp),
                mode=mode,
                wait_for_anchor=wait_for_anchor,
            )
            # Stash for the live before/after toggle on the result screen.
            async with session.lock:
                session.final_luts = (lut_r, lut_g, lut_b)
            # Restore the user's installed VideoLUT before delivering the
            # result. The "Show Corrected" button is opt-in — auto-applying
            # a bad/failed calibration would leave the display wrong-looking
            # until the user notices and clicks Show Original.
            try:
                await asyncio.to_thread(clear_ramp)
            except Exception:
                pass
            icc_b64 = base64.b64encode(icc_bytes).decode()
            import math
            delta_e_payload = None if math.isnan(delta_e) else delta_e
            chart_path = STATIC_DIR / "test_chart.png"
            _ensure_test_chart(chart_path)
            before_b64, after_b64 = _build_comparison_b64(chart_path, lut_r, lut_g, lut_b)
            await _send(pc.ws, {
                "type": "result",
                "mode": mode,
                "icc_b64": icc_b64,
                "delta_e": delta_e_payload,
                "before_b64": before_b64,
                "after_b64": after_b64,
                "live_preview": True,
            })
            await _send(mobile.ws, {"type": "all_done"})
        except Exception as exc:
            log.exception("calibration task failed")
            # _send already swallows dead-socket errors so this can't cascade.
            await _send(pc.ws, {"type": "error", "message": str(exc)})
