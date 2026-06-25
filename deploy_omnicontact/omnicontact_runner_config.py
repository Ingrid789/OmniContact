from pathlib import Path

import yaml

from omnicontact_runner_utils import resolve_project_path


TASK_XML_PATHS = {
    # Single-skill tasks
    "loco": "g1_description/omnicontact_loco.xml",
    "carrybox": "g1_description/omnicontact_carry_box.xml",
    "pushbox": "g1_description/omnicontact_push_box.xml",
    "pushbox-two": "g1_description/omnicontact_push_box.xml",
    "pushbox-in": "g1_description/omnicontact_push_box.xml",
    "slidebox": "g1_description/omnicontact_slide_box.xml",
    "slidebox-left": "g1_description/omnicontact_slide_box.xml",
    "slidebox-right": "g1_description/omnicontact_slide_box.xml",
    "relocateball": "g1_description/omnicontact_relocate_ball.xml",
    "kickball": "g1_description/omnicontact_kick_ball.xml",
    # Meta-skill chaining tasks
    "push-carry": "g1_description/omnicontact_pushcarry_box.xml",
    "carry-push": "g1_description/omnicontact_pushcarry_box.xml",
    "push-relocate": "g1_description/omnicontact_pushrelocate_ball.xml",
    "carry-carry": "g1_description/omnicontact_stack_2box.xml",
    "carry-carry-carry": "g1_description/omnicontact_stack_3box.xml",
    "carryheart": "g1_description/omnicontact_heart_10box.xml",
}

NPZ_DIR_XML_PATHS = (
    ("data/loco", "g1_description/omnicontact_carry_box.xml"),
    ("data/carrybox", "g1_description/omnicontact_carry_box.xml"),
    ("data/pushbox", "g1_description/omnicontact_push_box_npz.xml"),
    ("data/slidebox", "g1_description/omnicontact_slide_box_npz.xml"),
    ("data/relocateball", "g1_description/omnicontact_relocate_ball.xml"),
    ("data/kickball", "g1_description/omnicontact_kick_ball_npz.xml"),
)

NPZ_DIR_POLICY_PATHS = (
    ("data/kickball", "kick_50k.onnx"),
)


class OmniContactConfigMixin:
    def _load_config(self):
        config_path = Path(__file__).resolve().parent / "config" / "mujoco.yaml"
        with open(config_path, "r") as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)
        self.simulation_dt = float(cfg["simulation_dt"])
        self.control_decimation = int(cfg["control_decimation"])
        self._resolve_task_chaining()
        task = str(getattr(self.args, "task", "")).strip()
        default_xml_path = TASK_XML_PATHS.get(task, str(cfg["xml_path"]))
        npz_xml_path = self._xml_path_from_npz_dir()
        if npz_xml_path:
            default_xml_path = npz_xml_path
            self.args.xml_path = npz_xml_path
        self.xml_path = resolve_project_path(getattr(self.args, "xml_path", ""), default_xml_path)
        print(f"[runner] xml_path: {self.xml_path}")

    def _xml_path_from_npz_dir(self) -> str:
        return self._path_from_npz_dir(NPZ_DIR_XML_PATHS)

    def _policy_path_from_npz_dir(self) -> str:
        return self._path_from_npz_dir(NPZ_DIR_POLICY_PATHS)

    def _path_from_npz_dir(self, mappings) -> str:
        if str(getattr(self.args, "reference_source", "")).strip() != "NPZmotion":
            return ""
        npz_dir = str(getattr(self.args, "npz_dir", "")).strip()
        if not npz_dir:
            return ""

        path_text = npz_dir.replace("\\", "/")
        path_no_glob = path_text.split("*", 1)[0].rstrip("/")
        project_root = Path(__file__).resolve().parent.parent
        path = Path(path_no_glob).expanduser()
        if not path.is_absolute():
            path = project_root / path

        try:
            relative = path.resolve().relative_to(project_root.resolve())
        except ValueError:
            return ""
        relative_text = relative.as_posix().rstrip("/")
        for npz_dir_prefix, mapped_path in mappings:
            if relative_text == npz_dir_prefix or relative_text.startswith(f"{npz_dir_prefix}/"):
                return mapped_path
        return ""

    def _resolve_task_chaining(self) -> None:
        direct_tasks = {"push-carry", "carry-push", "carryheart", "push-relocate", "carry-carry", "carry-carry-carry"}
        chain = self.args.task_chaining
        if not chain:
            return
        if chain[0] in direct_tasks:
            self.args.task = chain[0]
