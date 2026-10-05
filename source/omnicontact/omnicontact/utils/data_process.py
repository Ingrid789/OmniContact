"""Resample motion NPZs to the control frequency, without changing the source files.

Run from the repository root with: python -m omnicontact.utils.data_process
Only NumPy is required; this module does not launch Isaac Sim.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import tempfile
import zipfile

import numpy as np


# Explicit names keep static metadata (e.g. scale, repair windows) intact even
# when its first dimension happens to equal the number of motion frames.
LINEAR_FIELDS = {
    "joint_pos", "joint_vel",
    "body_pos_w", "body_lin_vel_w", "body_ang_vel_w",
    "ee_pos_w",
    "object_pos_w", "object_lin_vel_w", "object_ang_vel_w",
    "table1_pos_w", "table2_pos_w", "object_viewer_center_offset_w",
}
QUATERNION_FIELDS = {"body_quat_w", "ee_quat_w", "object_quat_w", "object_quat_input_wxyz"}
FRAME_FIELDS = LINEAR_FIELDS | QUATERNION_FIELDS | {"contact_info"}


def _positive_fps(value: np.ndarray | float, name: str) -> float:
    array = np.asarray(value)
    if array.size != 1 or array.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain one positive, finite number.")
    fps = float(array.reshape(-1)[0])
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"{name} must contain one positive, finite number.")
    return fps


def _slerp(values: np.ndarray, lower: np.ndarray, upper: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """Interpolate unit quaternions along the shortest arc (preserving wxyz order)."""
    quat = values.astype(np.float64)
    norms = np.linalg.norm(quat, axis=-1, keepdims=True)
    quat /= norms
    # q and -q describe the same rotation. Unwrap signs along each trajectory.
    signs = np.where(np.sum(quat[1:] * quat[:-1], axis=-1, keepdims=True) < 0, -1.0, 1.0)
    quat[1:] *= np.cumprod(signs, axis=0)
    q0, q1 = quat[lower], quat[upper]
    dot = np.clip(np.sum(q0 * q1, axis=-1, keepdims=True), 0.0, 1.0)
    angle = np.arccos(dot)
    # sin(a * angle) / sin(angle) = a * sinc(a * angle/pi) / sinc(angle/pi).
    # This is also stable at angle=0, so small rotations need no approximation.
    scaled_angle = angle / np.pi
    denominator = np.sinc(scaled_angle)
    w0 = (1.0 - weight) * np.sinc((1.0 - weight) * scaled_angle) / denominator
    w1 = weight * np.sinc(weight * scaled_angle) / denominator
    result = w0 * q0 + w1 * q1
    result /= np.linalg.norm(result, axis=-1, keepdims=True)
    return result.astype(values.dtype)


def resample_motion(data: dict[str, np.ndarray], target_fps: float = 50.0) -> dict[str, np.ndarray]:
    """Sample at k / target_fps, up to the last source timestamp.

    Continuous fields, including velocities in physical units, are interpolated
    on the original time axis. Contact labels use nearest-neighbor sampling.
    Static metadata is retained; any repair-frame indices still refer to the
    original clip. A tail shorter than one target interval is omitted.
    """
    source_fps = _positive_fps(data["fps"], "Source fps")
    target_fps = _positive_fps(target_fps, "Target fps")
    joint_pos = data["joint_pos"]
    if joint_pos.ndim != 2 or joint_pos.shape[0] == 0:
        raise ValueError("joint_pos must have shape (frames, joints) with at least one frame.")
    frames = joint_pos.shape[0]
    for name in FRAME_FIELDS & data.keys():
        values = data[name]
        if values.ndim < 1 or values.shape[0] != frames:
            raise ValueError(f"{name} must have {frames} frames; got shape {values.shape}.")
        if values.dtype.kind not in "biuf" or not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must contain finite numeric values.")
        if name != "contact_info" and values.dtype.kind != "f":
            raise ValueError(f"{name} must contain floating-point values.")
        if name in QUATERNION_FIELDS:
            if values.ndim < 2 or values.shape[-1] != 4:
                raise ValueError(f"{name} must have a final dimension of 4.")
            norms = np.linalg.norm(values.astype(np.float64), axis=-1)
            if not np.all(np.isfinite(norms)) or np.any(norms < 1e-12):
                raise ValueError(f"{name} contains zero-length or invalid quaternion norms.")

    result = dict(data)
    if source_fps == target_fps:
        return result

    last_target_index = int(np.floor((frames - 1) * target_fps / source_fps + 1e-9))
    source_indices = np.arange(last_target_index + 1, dtype=np.float64) * source_fps / target_fps
    source_indices = np.clip(source_indices, 0, frames - 1)
    lower = np.floor(source_indices).astype(np.int64)
    upper = np.minimum(lower + 1, frames - 1)
    nearest = np.floor(source_indices + 0.5).astype(np.int64)
    for name in FRAME_FIELDS & data.keys():
        values = data[name]
        weight = (source_indices - lower).reshape((-1,) + (1,) * (values.ndim - 1))
        if name == "contact_info":
            result[name] = values[nearest].copy()
        elif name in QUATERNION_FIELDS:
            result[name] = _slerp(values, lower, upper, weight)
        else:
            result[name] = ((1.0 - weight) * values[lower] + weight * values[upper]).astype(values.dtype)
    result["fps"] = np.full(np.asarray(data["fps"]).shape, target_fps, dtype=np.float64)
    return result


def process_file(source: Path, destination: Path, target_fps: float = 50.0, *, overwrite: bool = False) -> None:
    """Write one complete NPZ atomically; never overwrite its source."""
    if source.resolve() == destination.resolve():
        raise ValueError("Input and output files must be different.")
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {destination}. Use --overwrite to replace generated files.")
    try:
        with np.load(source, allow_pickle=False) as archive:
            data = {key: archive[key] for key in archive.files}
        result = resample_motion(data, target_fps)
    except (ValueError, KeyError, OSError, zipfile.BadZipFile) as exc:
        raise ValueError(f"{source}: {exc}") from exc

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp", delete=False
        ) as stream:
            temporary = Path(stream.name)
            np.savez_compressed(stream, **result)
        temporary.replace(destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Resample motion NPZs to a separate directory (default: 50 FPS).")
    parser.add_argument("--input-dir", type=Path, default=Path("assets/npz_clips"))
    parser.add_argument("--output-dir", type=Path, default=Path("assets/npz_clips_50fps"))
    parser.add_argument("--target-fps", type=float, default=50.0)
    parser.add_argument("--overwrite", action="store_true", help="Replace existing generated NPZs in the output directory.")
    args = parser.parse_args(argv)
    try:
        target_fps = _positive_fps(args.target_fps, "Target fps")
        source_root, output_root = args.input_dir.resolve(), args.output_dir.resolve()
        if source_root == output_root or source_root in output_root.parents or output_root in source_root.parents:
            raise ValueError("Input and output directories must be separate and must not contain each other.")
        if not source_root.is_dir():
            raise ValueError(f"Input directory does not exist: {source_root}")
        paths = sorted(source_root.rglob("*.npz"))
        if not paths:
            raise ValueError(f"No NPZ files found under {source_root}")
        if not args.overwrite:
            for path in paths:
                destination = output_root / path.relative_to(source_root)
                if destination.exists():
                    raise FileExistsError(f"Output already exists: {destination}. Use --overwrite to regenerate it.")
        for index, path in enumerate(paths, 1):
            process_file(path, output_root / path.relative_to(source_root), target_fps, overwrite=args.overwrite)
            if index == 1 or index % 100 == 0 or index == len(paths):
                print(f"[{index}/{len(paths)}] {path.relative_to(source_root)}", flush=True)
        print(f"Ready: {len(paths)} clips at {target_fps:g} FPS in {output_root}")
    except (ValueError, KeyError, OSError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
