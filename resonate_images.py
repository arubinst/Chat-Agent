"""Safe access to Resonate map-overlay files mounted into the Chat-Agent."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import unquote, urlsplit


def _origin(url: str) -> tuple[str, str] | None:
    try:
        parsed = urlsplit(url)
        parsed.port
    except ValueError:
        return None
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        return None
    return parsed.scheme, parsed.netloc


def read_resonate_overlay(
    url: str,
    *,
    public_base_url: str | None,
    images_dir: str | Path | None,
    max_bytes: int,
) -> bytes | None:
    """Read a trusted Resonate overlay from a read-only shared volume.

    Return ``None`` for unrelated URLs so callers can use their normal remote
    downloader. URLs on the configured Resonate origin must be canonical
    ``/images/<filename>`` paths and never escape the mounted directory.
    """
    if not public_base_url or not images_dir:
        return None

    expected_origin = _origin(public_base_url)
    if expected_origin is None:
        raise ValueError("RESONATE_PUBLIC_BASE_URL must be an HTTP(S) origin")

    try:
        parsed = urlsplit(url)
        parsed.port
    except ValueError as exc:
        raise ValueError("Image URL is invalid") from exc

    if (parsed.scheme, parsed.netloc) != expected_origin:
        return None
    if parsed.query or parsed.fragment or not parsed.path.startswith("/images/"):
        raise ValueError("Resonate image URL must be a canonical /images/<filename> path")

    filename = unquote(parsed.path.removeprefix("/images/"))
    if not filename or "/" in filename or "\\" in filename or filename in {".", ".."}:
        raise ValueError("Resonate image URL contains an unsafe filename")

    root = Path(images_dir).resolve()
    candidate = (root / filename).resolve()
    if candidate.parent != root or not candidate.is_file():
        raise ValueError("Resonate overlay image is not available")

    if candidate.stat().st_size > max_bytes:
        raise ValueError("Resonate overlay image exceeds the export limit")
    return candidate.read_bytes()
