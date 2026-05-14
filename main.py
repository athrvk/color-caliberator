import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from display.dispwin import find_dispwin
from util.tls import detect_lan_ip, ensure_cert


def main():
    dispwin = find_dispwin()
    if dispwin is None:
        print(
            "\n[color-calibrator] ERROR: 'dispwin' not found on PATH.\n"
            "Install ArgyllCMS:\n"
            "  Windows: https://www.argyllcms.com/downloadwin.html\n"
            "  Mac:     brew install argyllcms\n"
            "  Linux:   sudo apt install argyll\n"
        )
        sys.exit(1)

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
