"""Physical timing, rotations and file safety of offline motion resampling."""

import numpy as np
import pytest

from omnicontact.utils.data_process import main, process_file, resample_motion


def make_motion(frames=91, fps=90.0):
    time = np.arange(frames, dtype=np.float64) / fps
    xyz = np.stack((time, 2 * time, -time), axis=-1).astype(np.float32)
    # Constant angular speed, with deliberately alternating quaternion signs.
    angle = 0.8 * time
    quat = np.stack((np.cos(angle / 2), np.zeros(frames), np.zeros(frames), np.sin(angle / 2)), axis=-1)
    quat[1::2] *= -1
    quat = quat.astype(np.float32)
    return {
        "fps": np.array([fps]),
        "joint_pos": xyz.copy(),
        "joint_vel": np.tile(np.array([1, 2, -1], dtype=np.float32), (frames, 1)),
        "body_pos_w": xyz[:, None, :].copy(),
        "body_lin_vel_w": np.tile(np.array([1, 2, -1], dtype=np.float32), (frames, 1, 1)),
        "body_ang_vel_w": np.tile(np.array([0, 0, 0.8], dtype=np.float32), (frames, 1, 1)),
        "body_quat_w": quat[:, None, :].copy(),
        "ee_pos_w": xyz[:, None, :].copy(),
        "ee_quat_w": quat[:, None, :].copy(),
        "object_pos_w": xyz.copy(),
        "object_quat_w": quat.copy(),
        "object_quat_input_wxyz": quat.copy(),
        "object_lin_vel_w": np.tile(np.array([1, 2, -1], dtype=np.float32), (frames, 1)),
        "object_ang_vel_w": np.tile(np.array([0, 0, 0.8], dtype=np.float32), (frames, 1)),
        "object_viewer_center_offset_w": xyz.copy(),
        "table1_pos_w": xyz.copy(),
        "table2_pos_w": xyz.astype(np.float64),
        "contact_info": (time >= 0.5).astype(np.uint8).reshape(-1, 1, 1),
        "scale": np.array([0.7, 0.8, 0.9], dtype=np.float32),
        "object_geom_quat_wxyz": np.array([1, 0, 0, 0], dtype=np.float32),
        "object_xml": np.array("object_xml/box.xml"),
        "carry_newdof_windows": np.array([[0, 40, 50, 90]], dtype=np.int32),
    }


def test_resampling_preserves_physical_time_and_velocity():
    source = make_motion()
    output = resample_motion(source)
    assert output["fps"].item() == 50
    # 91 source samples span exactly 1 s; 51 target samples span that same 1 s.
    assert output["joint_pos"].shape == (51, 3)
    time = np.arange(51) / 50
    expected_xyz = np.stack((time, 2 * time, -time), axis=-1)
    for key in ("joint_pos", "object_pos_w", "table1_pos_w", "table2_pos_w", "object_viewer_center_offset_w"):
        np.testing.assert_allclose(output[key], expected_xyz, atol=1e-7)
        assert output[key].dtype == source[key].dtype
    for key in ("body_pos_w", "ee_pos_w"):
        np.testing.assert_allclose(output[key][:, 0], expected_xyz, atol=1e-7)
    np.testing.assert_allclose(np.diff(output["joint_pos"], axis=0) * 50, output["joint_vel"][:-1], atol=1e-5)
    np.testing.assert_allclose(output["body_lin_vel_w"][:, 0], output["joint_vel"])
    np.testing.assert_allclose(output["object_lin_vel_w"], output["joint_vel"])
    np.testing.assert_allclose(output["body_ang_vel_w"][:, 0, 2], 0.8)
    np.testing.assert_allclose(output["object_ang_vel_w"][:, 2], 0.8)
    for key in ("body_quat_w", "ee_quat_w", "object_quat_w", "object_quat_input_wxyz"):
        quat = output[key].reshape(51, 4)
        np.testing.assert_allclose(np.linalg.norm(quat, axis=-1), 1, atol=1e-7)
        np.testing.assert_allclose(2 * np.arctan2(quat[:, 3], quat[:, 0]), 0.8 * time, atol=1e-6)
    assert source["fps"].item() == 90
    assert source["joint_pos"].shape[0] == 91


@pytest.mark.parametrize("dtype", [np.uint8, np.float32, np.bool_])
def test_contact_is_discrete_and_uses_nearest_source_frame(dtype):
    source = make_motion(frames=10)
    source["contact_info"] = (np.arange(10) % 2).astype(dtype).reshape(-1, 1, 1)
    output = resample_motion(source)
    # Target times correspond to source indices 0, 1.8, 3.6, 5.4, 7.2, 9.
    np.testing.assert_array_equal(output["contact_info"][:, 0, 0], [0, 0, 0, 1, 1, 1])
    assert output["contact_info"].dtype == dtype


def test_slerp_takes_short_arc_across_rotation_wrap():
    source = make_motion(frames=2, fps=1)
    angle = np.deg2rad([170.0, -170.0])
    source["object_quat_w"] = np.stack((np.cos(angle / 2), np.zeros(2), np.zeros(2), np.sin(angle / 2)), axis=-1)
    midpoint = resample_motion(source, target_fps=2)["object_quat_w"][1]
    # Halfway through the short 20-degree arc is 180 degrees, not 0 degrees.
    np.testing.assert_allclose(np.abs(midpoint), [0, 0, 0, 1], atol=1e-7)


def test_slerp_handles_antipodal_quaternions():
    source = make_motion(frames=2, fps=1)
    source["object_quat_w"] = np.array([[1, 0, 0, 0], [-1, 0, 0, 0]], dtype=np.float32)
    output = resample_motion(source, target_fps=2)
    np.testing.assert_allclose(output["object_quat_w"], [[1, 0, 0, 0]] * 3)


def test_small_rotation_retains_constant_angular_speed():
    source = make_motion(frames=2, fps=1)
    angle = 0.06  # Small enough to exercise the former linear fallback.
    source["object_quat_w"] = np.array([[1, 0, 0, 0], [np.cos(angle / 2), 0, 0, np.sin(angle / 2)]])
    output = resample_motion(source)
    time = np.arange(51) / 50
    expected = np.column_stack((np.cos(angle * time / 2), np.zeros(51), np.zeros(51), np.sin(angle * time / 2)))
    np.testing.assert_allclose(output["object_quat_w"], expected, atol=1e-14, rtol=0)


def test_multibody_quaternions_match_independent_scipy_slerp():
    spatial = pytest.importorskip("scipy.spatial.transform")
    rng = np.random.default_rng(431)
    frames, bodies = 181, 6
    source = make_motion(frames)
    quat = spatial.Rotation.random(frames * bodies, random_state=rng).as_quat().reshape(frames, bodies, 4)
    quat *= rng.choice([-1, 1], size=(frames, bodies, 1))
    source["body_quat_w"] = quat[..., [3, 0, 1, 2]]
    output = resample_motion(source)
    source_times = np.arange(frames) / 90
    target_times = np.arange(len(output["joint_pos"])) / 50
    for body in range(bodies):
        expected = spatial.Slerp(source_times, spatial.Rotation.from_quat(quat[:, body]))(target_times)
        actual = spatial.Rotation.from_quat(output["body_quat_w"][:, body, [1, 2, 3, 0]])
        np.testing.assert_allclose((expected.inv() * actual).magnitude(), 0, atol=1e-12)


@pytest.mark.parametrize("frames", [1, 2, 3, 90, 91, 92])
def test_exact_target_grid_and_static_metadata(frames):
    source = make_motion(frames)
    output = resample_motion(source)
    source_duration = (frames - 1) / 90
    output_duration = (len(output["joint_pos"]) - 1) / 50
    assert -1e-12 <= source_duration - output_duration < 1 / 50
    for key in ("scale", "object_geom_quat_wxyz", "object_xml", "carry_newdof_windows"):
        np.testing.assert_array_equal(output[key], source[key])
    assert output.keys() == source.keys()


def test_already_50_fps_retains_every_array():
    source = make_motion(frames=51, fps=50)
    source["fps"] = np.array(50, dtype=np.int64)
    output = resample_motion(source)
    for key in source:
        np.testing.assert_array_equal(output[key], source[key])
        assert output[key].dtype == source[key].dtype


def test_already_50_fps_rejects_zero_quaternions():
    source = make_motion(frames=51, fps=50)
    source["object_quat_w"][0] = 0
    with pytest.raises(ValueError, match="zero-length"):
        resample_motion(source)


@pytest.mark.parametrize("fps", [0, -90, np.nan, np.inf, [50, 90]])
def test_rejects_invalid_source_fps(fps):
    source = make_motion()
    source["fps"] = np.asarray(fps)
    with pytest.raises(ValueError, match="Source fps"):
        resample_motion(source)


@pytest.mark.parametrize("fps", [0, -50, np.nan, np.inf])
def test_rejects_invalid_target_fps(fps):
    with pytest.raises(ValueError, match="Target fps"):
        resample_motion(make_motion(), fps)


def test_rejects_inconsistent_frames_and_bad_quaternions():
    source = make_motion()
    source["contact_info"] = source["contact_info"][:-1]
    with pytest.raises(ValueError, match="contact_info must have 91 frames"):
        resample_motion(source)
    source = make_motion()
    source["object_quat_w"][0] = 0
    with pytest.raises(ValueError, match="zero-length"):
        resample_motion(source)
    source["object_quat_w"][0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        resample_motion(source)


def test_recursive_cli_preserves_sources_and_requires_explicit_overwrite(tmp_path, capsys):
    source = tmp_path / "raw"
    output = tmp_path / "processed"
    (source / "box").mkdir(parents=True)
    np.savez(source / "box" / "motion.npz", **make_motion())
    np.savez(source / "already50.npz", **make_motion(51, 50))
    (source / "report.json").write_text('{"source_frames": 91}')
    original_bytes = {p: p.read_bytes() for p in source.rglob("*") if p.is_file()}
    args = ["--input-dir", str(source), "--output-dir", str(output)]
    main(args)
    assert "2 clips at 50 FPS" in capsys.readouterr().out
    assert len(list(output.rglob("*.npz"))) == 2
    assert not (output / "report.json").exists()
    with np.load(output / "box" / "motion.npz", allow_pickle=False) as data:
        assert data["joint_pos"].shape[0] == 51
        assert data["fps"].item() == 50
    with pytest.raises(SystemExit) as exc:
        main(args)
    assert exc.value.code == 1
    assert "--overwrite" in capsys.readouterr().err
    main(args + ["--overwrite"])
    assert all(p.read_bytes() == contents for p, contents in original_bytes.items())
    assert not list(output.rglob("*.tmp"))


def test_rejects_overlapping_directories_and_in_place_conversion(tmp_path):
    source = tmp_path / "raw"
    source.mkdir()
    for destination in (source, source / "processed", tmp_path):
        with pytest.raises(SystemExit) as exc:
            main(["--input-dir", str(source), "--output-dir", str(destination), "--overwrite"])
        assert exc.value.code == 1
    path = source / "motion.npz"
    np.savez(path, **make_motion())
    original_bytes = path.read_bytes()
    with pytest.raises(ValueError, match="must be different"):
        process_file(path, path, overwrite=True)
    assert path.read_bytes() == original_bytes


def test_invalid_input_leaves_existing_output_intact(tmp_path):
    source = tmp_path / "bad.npz"
    output = tmp_path / "output.npz"
    motion = make_motion()
    motion["fps"] = np.array([0])
    np.savez(source, **motion)
    output.write_bytes(b"previous output")
    with pytest.raises(ValueError, match="Source fps"):
        process_file(source, output, overwrite=True)
    assert output.read_bytes() == b"previous output"
    assert not list(tmp_path.glob("*.tmp"))
