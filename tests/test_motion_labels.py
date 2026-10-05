"""Asset identity across motion tasks, filename variants and box dimensions."""

from pathlib import Path

import pytest

from omnicontact.utils.motion_labels import asset_label_from_motion_file


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("push_box_37-42-56_001.npz", "push_37-42-56"),
        ("push_box_37-42-56_230_6135050_001.npz", "push_37-42-56"),
        ("PUSH_BOX_37-42-56_001.NPZ", "push_37-42-56"),
        ("carry_box_37-42-56_001.npz", "37-42-56"),
        ("slide_box_23-23-23_13_5871988_000.npz", "23-23-23"),
        ("loco_box_37-42-56_backward_motion48.npz", "37-42-56"),
        ("loco_box_37-42-56_loco_walkturnleft90_4_with_contact.npz", "37-42-56"),
        ("relocate_soccer_22_001.npz", "soccer_22"),
        ("/data/push_box_30-30-30/carry_box_37-42-56_001.npz", "37-42-56"),
        (Path("/data/push_box_37-42-56_001.npz"), "push_37-42-56"),
    ],
)
def test_canonical_asset_label(filename, expected):
    assert asset_label_from_motion_file(filename) == expected


def test_push_identity_is_distinct_from_other_tasks_and_similar_dimensions():
    reference = asset_label_from_motion_file("push_box_37-42-56_001.npz")
    for filename in (
        "carry_box_37-42-56_001.npz",
        "slide_box_37-42-56_001.npz",
        "loco_box_37-42-56_walk.npz",
        "push_box_137-42-56_001.npz",
        "push_box_37-42-560_001.npz",
    ):
        assert asset_label_from_motion_file(filename) != reference


@pytest.mark.parametrize(
    "filename",
    [
        "motion.npz",
        "push_37-42-56_001.npz",
        "push_box_37-42_001.npz",
        "push_box_37-42-56-78_001.npz",
        "push_box_37-42-56_001.npz.bak",
    ],
)
def test_unrecognized_filename_has_no_asset_label(filename):
    assert asset_label_from_motion_file(filename) is None
