"""Repository asset paths for the OmniContact training package."""

from __future__ import annotations

from pathlib import Path


ASSET_DIR = str(Path(__file__).resolve().parent)


def asset_path(*parts: str) -> str:
    """Return an absolute path below the repository's assets directory."""

    return str(Path(ASSET_DIR, *parts))


def existing_asset_path(*candidates: tuple[str, ...]) -> str:
    """Return the first existing asset path, or the first candidate if none exist."""

    paths = [asset_path(*parts) for parts in candidates]
    return next((path for path in paths if Path(path).exists()), paths[0])
