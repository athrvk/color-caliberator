import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from display.videolut import backend_available, backend_error, backend_name
from util.tls import detect_lan_ip, ensure_cert


def main():
    if not backend_available():
        print(
            f"\n[color-calibrator] ERROR: no VideoLUT backend available "
            f"({backend_error()}).\n"
            "  macOS / Windows: should work out of the box. File a bug.\n"
            "  Linux:           install ArgyllCMS (`sudo apt install argyll` or distro equivalent).\n"
            "  Wayland:         log into an X11 session — Wayland is unsupported.\n"
        )
        sys.exit(1)
    print(f"[color-calibrator] VideoLUT backend: {backend_name()}")

    cert_path, key_path = ensure_cert(Path(__file__).parent / ".cert")
    ip = detect_lan_ip()

    print(f"\n[color-calibrator] Server starting on https://{ip}:8765")
    print(f"[color-calibrator] Mobile URL: https://{ip}:8765/mobile")
    print("[color-calibrator] Self-signed cert: accept the browser warning once.\n")

    import uvicorn
    uvicorn.run(
        "web.server:app",
        host="0.0.0.0",
        port=8765,
        reload=False,
        ssl_keyfile=str(key_path),
        ssl_certfile=str(cert_path),
    )


if __name__ == "__main__":
    main()
