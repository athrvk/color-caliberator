import asyncio
import base64
import json
import socket
import sys
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, str(Path(__file__).parent.parent))

from display.dispwin import find_dispwin
from util.qr import generate_qr_png

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


session = Session()


async def _reader(endpoint: Endpoint) -> None:
    """Single task per WebSocket: pump parsed messages into the queue."""
    try:
        while True:
            text = await endpoint.ws.receive_text()
            try:
                msg = json.loads(text)
            except json.JSONDecodeError:
                continue
            await endpoint.queue.put(msg)
    except WebSocketDisconnect:
        pass
    finally:
        endpoint.closed.set()


def _local_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    finally:
        s.close()


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

    dispwin_path = find_dispwin()
    if dispwin_path is None:
        await _send(websocket, {
            "type": "error",
            "message": "dispwin not found — install ArgyllCMS and add it to PATH.",
        })
        await websocket.close()
        session.pc = None
        return

    ip = _local_ip()
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
    """Consume PC-originated messages. Currently: start_calibration trigger."""
    while not pc.closed.is_set():
        msg = await pc.queue.get()
        if msg.get("type") == "start_calibration":
            async with session.lock:
                mobile = session.mobile
                if mobile is None:
                    await _send(pc.ws, {"type": "error", "message": "Mobile not connected."})
                    continue
                if session.calibration_task and not session.calibration_task.done():
                    continue  # already running
                session.calibration_task = asyncio.create_task(_run_calibration_task())


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


# ---------- Helpers ----------

async def _send(ws: WebSocket, msg: dict) -> None:
    await ws.send_text(json.dumps(msg))


async def _run_calibration_task() -> None:
    """Wraps calibration loop; sends result or error to PC."""
    from calibration.iterate import run_calibration

    pc, mobile = session.pc, session.mobile
    if pc is None or mobile is None:
        return

    async def pc_send(msg: dict) -> None:
        await _send(pc.ws, msg)

    async def mobile_send(msg: dict) -> None:
        await _send(mobile.ws, msg)

    async def mobile_recv() -> dict:
        return await mobile.queue.get()

    async def mobile_drain() -> None:
        """Discard any queued mobile messages (stale frames between patches)."""
        while not mobile.queue.empty():
            mobile.queue.get_nowait()

    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        try:
            icc_bytes, delta_e = await run_calibration(
                pc_send, mobile_send, mobile_recv, mobile_drain, Path(tmp)
            )
            icc_b64 = base64.b64encode(icc_bytes).decode()
            await _send(pc.ws, {"type": "result", "icc_b64": icc_b64, "delta_e": delta_e})
            await _send(mobile.ws, {"type": "all_done"})
        except Exception as exc:
            await _send(pc.ws, {"type": "error", "message": str(exc)})
