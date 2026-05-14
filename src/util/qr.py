import io

import qrcode


def generate_qr_png(url: str) -> bytes:
    """Generate a QR code for `url` and return it as PNG bytes."""
    qr = qrcode.make(url)
    buf = io.BytesIO()
    qr.save(buf, format="PNG")
    return buf.getvalue()
