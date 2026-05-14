import base64
import tempfile
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import calibration.iterate as iterate
from calibration.iterate import run_calibration


def _b64_jpeg_gray(level_0_1: float) -> str:
    v = int(round(np.clip(level_0_1, 0.0, 1.0) * 255))
    img = Image.new("RGB", (64, 64), (v, v, v))
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return base64.b64encode(buf.getvalue()).decode()


class FakeProtocol:
    """
    Simulates the mobile WebSocket. Produces JPEG frames whose grayscale level
    follows `display_gamma`: for input x, the camera 'sees' x ** gamma.
    """

    def __init__(self, display_gamma: float):
        self.gamma = display_gamma
        self.current_input = 1.0
        self.sent: list = []
        self.recv_queue: list[dict] = []

    async def pc_send(self, msg: dict) -> None:
        self.sent.append(("pc", msg))
        if msg.get("type") == "show_patch":
            r, g, b = msg["rgb"]
            self.current_input = (r + g + b) / 3.0 / 255.0

    async def mobile_send(self, msg: dict) -> None:
        self.sent.append(("mobile", msg))
        if msg.get("type") == "show_white_for_wb":
            self.recv_queue.append({"type": "ready"})

    async def mobile_recv(self) -> dict:
        if self.recv_queue:
            return self.recv_queue.pop(0)
        return {"type": "frame", "data": _b64_jpeg_gray(self.current_input ** self.gamma)}

    async def mobile_drain(self) -> None:
        self.recv_queue.clear()


@pytest.mark.asyncio
async def test_run_calibration_converges_on_gamma_1_8_display(monkeypatch):
    monkeypatch.setattr(iterate, "clear_ramp", lambda *a, **k: None)
    monkeypatch.setattr(iterate, "apply_ramp", lambda *a, **k: None)
    monkeypatch.setattr(iterate, "SETTLE_DELAY", 0.0)

    fake = FakeProtocol(display_gamma=1.8)

    with tempfile.TemporaryDirectory() as tmp:
        icc_bytes, delta_e = await run_calibration(
            fake.pc_send, fake.mobile_send, fake.mobile_recv, fake.mobile_drain, Path(tmp)
        )

    assert isinstance(icc_bytes, bytes) and len(icc_bytes) > 256
    assert icc_bytes[36:40] == b"acsp"
    assert np.isfinite(delta_e)
