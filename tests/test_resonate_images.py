from pathlib import Path

import pytest

from resonate_images import read_resonate_overlay


def test_reads_a_validated_resonate_overlay_from_the_shared_volume(tmp_path: Path):
    image = tmp_path / "event_overlay.png"
    image.write_bytes(b"png-data")

    result = read_resonate_overlay(
        "https://resonate.example.com/images/event_overlay.png",
        public_base_url="https://resonate.example.com",
        images_dir=tmp_path,
        max_bytes=1024,
    )

    assert result == b"png-data"


@pytest.mark.parametrize(
    "url",
    [
        "https://resonate.example.com/images/../secret.png",
        "https://resonate.example.com/images/%2Fetc%2Fpasswd",
        "https://resonate.example.com/images/overlay.png?download=1",
        "https://resonate.example.com/images/overlay.png#fragment",
    ],
)
def test_rejects_unsafe_or_noncanonical_resonate_overlay_urls(tmp_path: Path, url: str):
    with pytest.raises(ValueError):
        read_resonate_overlay(
            url,
            public_base_url="https://resonate.example.com",
            images_dir=tmp_path,
            max_bytes=1024,
        )


def test_ignores_unrelated_remote_urls(tmp_path: Path):
    assert read_resonate_overlay(
        "https://example.com/image.png",
        public_base_url="https://resonate.example.com",
        images_dir=tmp_path,
        max_bytes=1024,
    ) is None


def test_rejects_missing_or_oversized_overlay_files(tmp_path: Path):
    with pytest.raises(ValueError, match="not available"):
        read_resonate_overlay(
            "https://resonate.example.com/images/missing.png",
            public_base_url="https://resonate.example.com",
            images_dir=tmp_path,
            max_bytes=1024,
        )

    (tmp_path / "large.png").write_bytes(b"x" * 5)
    with pytest.raises(ValueError, match="exceeds"):
        read_resonate_overlay(
            "https://resonate.example.com/images/large.png",
            public_base_url="https://resonate.example.com",
            images_dir=tmp_path,
            max_bytes=4,
        )
