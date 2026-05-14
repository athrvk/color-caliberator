import ipaddress
import socket
from datetime import datetime, timedelta, timezone
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID


def detect_lan_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    finally:
        s.close()


def ensure_cert(cert_dir: Path) -> tuple[Path, Path]:
    """
    Generate (or reuse) a self-signed cert + key covering 127.0.0.1 and the
    current LAN IP. Returns (cert_path, key_path).

    Regenerates if the cert is missing, expired, or does not include the
    current LAN IP (handles laptop moving networks).
    """
    cert_dir.mkdir(parents=True, exist_ok=True)
    cert_path = cert_dir / "server.crt"
    key_path = cert_dir / "server.key"

    lan_ip = detect_lan_ip()
    if _cert_valid_for(cert_path, lan_ip):
        return cert_path, key_path

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    now = datetime.now(tz=timezone.utc)
    san = x509.SubjectAlternativeName([
        x509.IPAddress(ipaddress.IPv4Address("127.0.0.1")),
        x509.IPAddress(ipaddress.IPv4Address(lan_ip)),
        x509.DNSName("localhost"),
    ])
    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME, "color-calibrator"),
    ])
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=1))
        .not_valid_after(now + timedelta(days=825))
        .add_extension(san, critical=False)
        .sign(key, hashes.SHA256())
    )

    key_path.write_bytes(key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.TraditionalOpenSSL,
        encryption_algorithm=serialization.NoEncryption(),
    ))
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    return cert_path, key_path


def _cert_valid_for(cert_path: Path, lan_ip: str) -> bool:
    if not cert_path.exists():
        return False
    try:
        cert = x509.load_pem_x509_certificate(cert_path.read_bytes())
        if cert.not_valid_after_utc < datetime.now(tz=timezone.utc):
            return False
        san = cert.extensions.get_extension_for_class(
            x509.SubjectAlternativeName
        ).value
        ips = [str(v) for v in san.get_values_for_type(x509.IPAddress)]
        return lan_ip in ips
    except Exception:
        return False
