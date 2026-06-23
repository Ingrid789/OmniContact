import argparse


TASK_ALIASES = {
    "pushbox": "pushbox-in",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SKILL_OmniContact directly in Mujoco.")
    parser.add_argument(
        "--xml-path",
        type=str,
        default="",
        help="Override Mujoco XML path. If omitted, the XML is selected from --task. Supports absolute paths or paths relative to PROJECT_ROOT.",
    )
    parser.add_argument(
        "--reference-source",
        type=str,
        default="CFgen",
        choices=["CFgen", "NPZmotion"],
        help="Reference generator used by OmniContact.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="carrybox",
        choices=["loco", "carrybox", "carryheart", "pushbox", "pushbox-two", "pushbox-in", "slidebox", "slidebox-left", "slidebox-right", "push-carry", "carry-push", "push-relocate", "carry*2", "carry*3", "carry-carry", "carry-carry-carry", "stackbox", "relocateball", "kickball", "kickbox"],
        help="Task preset used by CFgen. loco walks from init pelvis position to goal pelvis position. slidebox and kickball are upright behind-object walk-and-drive plans with hands kept free. stackbox carries three boxes to a fixed stack target.",
    )
    parser.add_argument(
        "--task-chaining",
        type=str,
        nargs="+",
        default=(),
        choices=["push", "carry", "carryheart", "push-carry", "carry-push", "push-relocate", "carry*2", "carry*3", "carry-carry", "carry-carry-carry"],
        metavar="SKILL",
        help="Meta-skill chain, for example: --task-chaining push-carry, --task-chaining carry-push, or the split form --task-chaining push carry.",
    )
    parser.add_argument(
        "--npz-dir",
        type=str,
        default="",
        help="For NPZmotion: path to .npz file or a directory containing .npz files.",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="For NPZmotion: inclusive start frame index after loading the npz.",
    )
    parser.add_argument(
        "--end-frame",
        type=int,
        default=-1,
        help="For NPZmotion: exclusive end frame index. -1 means use all remaining frames.",
    )
    parser.add_argument(
        "--box-half-dims",
        type=float,
        nargs=3,
        default=None,
        metavar=("HX", "HY", "HZ"),
        help="Optional object half-dimensions override. If omitted, dimensions are loaded from the selected Mujoco XML.",
    )
    parser.add_argument(
        "--goal-pos",
        type=float,
        nargs="+",
        default=(1.0, 1.0, 0.55),
        metavar="POS",
        help="Goal object position override as X Y or X Y Z. For --task loco, Z is always forced to default_pelvis_z.",
    )
    parser.add_argument(
        "--init-pos",
        type=float,
        nargs="+",
        default=(1.0, 0.0, 0.55),
        metavar="POS",
        help="Initial object position override as X Y or X Y Z. For --task loco, Z is always forced to default_pelvis_z.",
    )
    parser.add_argument(
        "--init-pos-extra",
        type=float,
        nargs="+",
        default=None,
        metavar="POS",
        help="Initial position for the extra object in chained tasks as X Y or X Y Z. If omitted, it is randomized.",
    )
    parser.add_argument(
        "--init-pos-extra-2",
        type=float,
        nargs="+",
        default=None,
        metavar="POS",
        help="Initial position for the second extra object in chained tasks as X Y or X Y Z. If omitted, it is randomized.",
    )
    parser.add_argument(
        "--policy",
        type=str,
        default="",
        help="Override ONNX policy path for OmniContact.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="Maximum simulation steps. <=0 means unlimited.",
    )
    parser.add_argument(
        "--no-reset-env",
        action="store_true",
        help="Disable startup environment reset for selected reference source.",
    )
    parser.add_argument(
        "--stop-when-done",
        action="store_true",
        help="Stop when OmniContact CFgen reaches the end and sets switch_to_loco.",
    )
    parser.add_argument(
        "--disable-replan",
        dest="replan",
        action="store_false",
        help="Disable replan support for CFgen runs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for startup reset sampling.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without opening the Mujoco GLFW viewer window.",
    )
    args = parser.parse_args()
    if args.reference_source == "NPZmotion" and not str(args.npz_dir).strip():
        parser.error("--npz-dir is required when --reference-source NPZmotion")
    args.task = TASK_ALIASES.get(args.task, args.task)
    args.reset_env = not args.no_reset_env
    return args
