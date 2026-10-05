"""Shared asset labels for object spawning and reference motion sampling."""

from __future__ import annotations

from pathlib import Path
import re


_BOX_SIZE = r"\d+(?:-\d+){2}"
_CLIP_SUFFIX = r"(?:\d+_)*\d{3}"
_BOX_PATTERN = re.compile(
    rf"(?P<task>carry|push|slide)_box_(?P<size>{_BOX_SIZE})_{_CLIP_SUFFIX}\.npz", re.IGNORECASE
)
_LOCO_PATTERN = re.compile(rf"loco_box_(?P<size>{_BOX_SIZE})_.+\.npz", re.IGNORECASE)
_SOCCER_PATTERN = re.compile(rf"relocate_soccer_(?P<size>\d+)_{_CLIP_SUFFIX}\.npz", re.IGNORECASE)


def asset_label_from_motion_file(motion_file: str | Path) -> str | None:
    """Return the canonical asset label, or None for an unrecognized filename.

    Carry, slide and locomotion share box dimension labels. Push keeps its
    prefix because its environments use different material properties.
    """
    file_name = Path(motion_file).name
    match = _BOX_PATTERN.fullmatch(file_name)
    if match is not None:
        size = match.group("size")
        return f"push_{size}" if match.group("task").lower() == "push" else size

    match = _LOCO_PATTERN.fullmatch(file_name)
    if match is not None:
        return match.group("size")

    match = _SOCCER_PATTERN.fullmatch(file_name)
    if match is not None:
        return f"soccer_{match.group('size')}"

    return None
