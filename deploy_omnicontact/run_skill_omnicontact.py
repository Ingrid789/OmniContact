import argparse
import sys
import time
from pathlib import Path

try:
    import mujoco
    import mujoco.viewer
    _MUJOCO_IMPORT_ERROR = None
except ModuleNotFoundError as exc:
    mujoco = None
    _MUJOCO_IMPORT_ERROR = exc

import numpy as np

sys.path.append(str(Path(__file__).parent.parent.absolute()))

from common.ctrlcomp import PolicyOutput, StateAndCmd
from policy.omnicontact.CFgen_meta1_loco import DEFAULT_PELVIS_Z
from omnicontact_runner_args import parse_args
from omnicontact_runner_constants import (
    CONTACT_COLOR_OFF,
    CONTACT_COLOR_ON,
    LAB2MJ,
    TORQUE_LIMIT,
)
from omnicontact_runner_carryheart import OmniContactCarryheartMixin
from omnicontact_runner_config import OmniContactConfigMixin
from omnicontact_runner_metrics import OmniContactMetricsMixin
from omnicontact_runner_objects import OmniContactObjectsMixin
from omnicontact_replan import OmniContactCarryboxReplan
from omnicontact_runner_reset import OmniContactResetMixin
from omnicontact_runner_visualization import OmniContactVisualizationMixin


class OmniContactRunner(
    OmniContactConfigMixin,
    OmniContactCarryheartMixin,
    OmniContactVisualizationMixin,
    OmniContactObjectsMixin,
    OmniContactResetMixin,
    OmniContactMetricsMixin,
):
    def __init__(self, args: argparse.Namespace):
        if _MUJOCO_IMPORT_ERROR is not None:
            raise ModuleNotFoundError(
                "mujoco is required to run this script. Please activate the environment with mujoco installed."
            ) from _MUJOCO_IMPORT_ERROR

        self.args = args
        self._load_config()
        self.is_carryheart = str(getattr(self.args, "task", "")).strip() == "carryheart"
        self._init_mujoco()
        self._init_policy()
        self._cache_visual_ids()
        self.rng = np.random.default_rng(int(getattr(self.args, "seed", 0)))
        self._init_carryheart_state()

        self.sim_counter = 0
        self.policy_tick_counter = 0
        self.policy_action = np.zeros(self.num_joints, dtype=np.float32)
        self.policy_kps = np.zeros(self.num_joints, dtype=np.float32)
        self.policy_kds = np.zeros(self.num_joints, dtype=np.float32)
        self.contact_color_off = CONTACT_COLOR_OFF.copy()
        self.contact_color_on = CONTACT_COLOR_ON.copy()

        self.replan = OmniContactCarryboxReplan(self, enabled=bool(getattr(self.args, "replan", False)))
        self.last_episode_metrics = {}
        self._reset_episode_metrics()

    def _init_mujoco(self):
        self.m = mujoco.MjModel.from_xml_path(self.xml_path)
        self.d = mujoco.MjData(self.m)
        self.m.opt.timestep = self.simulation_dt
        self.num_joints = int(self.m.nu)

    def _geom_half_dims(self, *geom_names: str, default: tuple[float, float, float]) -> np.ndarray:
        for geom_name in geom_names:
            geom_id = self._name2id(mujoco.mjtObj.mjOBJ_GEOM, geom_name)
            if geom_id >= 0:
                size = np.asarray(self.m.geom_size[geom_id, :3], dtype=np.float32).copy()
                geom_type = int(self.m.geom_type[geom_id])
                if geom_type == int(mujoco.mjtGeom.mjGEOM_SPHERE):
                    radius = float(size[0])
                    return np.array([radius, radius, radius], dtype=np.float32)
                return size
        return np.asarray(default, dtype=np.float32).reshape(3).copy()

    def _configure_policy_dims_from_model(self) -> None:
        override = getattr(self.args, "box_half_dims", None)
        if override is not None:
            dims = np.asarray(override, dtype=np.float32).reshape(3)
            self.policy.box_dims = dims.copy()
            self.policy.ball_dims = dims.copy()
            self.policy.push_box_dims = dims.copy()
            self.policy.carry_box_dims = dims.copy()
        else:
            self.policy.box_dims = self._geom_half_dims(
                "ghost_box_geom",
                "box_geom",
                "ball_geom",
                default=(0.15, 0.15, 0.15),
            )
            self.policy.push_box_dims = self._geom_half_dims(
                "ghost_push_box_geom",
                "ghost_box_geom",
                "push_box_geom_top",
                "box_geom_top",
                default=(0.23, 0.25, 0.26),
            )
            self.policy.ball_dims = self._geom_half_dims(
                "ghost_ball_geom",
                "ball_geom",
                default=(0.10, 0.10, 0.10),
            )
            self.policy.carry_box_dims = self._geom_half_dims(
                "ghost_carry_box_geom",
                "carry_box_geom",
                default=(0.15, 0.15, 0.15),
            )

        stack_dims = []
        for geom_name, default in zip(
            ("stack_box_large_geom", "stack_box_mid_geom", "stack_box_small_geom"),
            ((0.20, 0.20, 0.15), (0.15, 0.15, 0.15), (0.10, 0.10, 0.10)),
        ):
            stack_dims.append(self._geom_half_dims(geom_name, default=default))
        self.policy.stack_box_dims = np.asarray(stack_dims, dtype=np.float32).reshape(3, 3)
        self.policy.bbox_scale = self.policy.box_dims * 2.0
        self.policy.bbox_offsets_scaled = self.policy.bbox_offsets * self.policy.bbox_scale.reshape(1, 3)

    def _init_policy(self):
        from policy.omnicontact.OmniContact import OmniContact

        def resolve_policy_path(policy_arg: str) -> str | None:
            if not str(policy_arg).strip():
                return None
            model_path = Path(policy_arg).expanduser()
            if model_path.is_absolute():
                return str(model_path.resolve())
            if model_path.exists():
                return str(model_path.resolve())
            candidate_path = Path(__file__).resolve().parent.parent / "policy" / "omnicontact" / "model" / model_path
            if candidate_path.exists():
                return str(candidate_path.resolve())
            return str(model_path.resolve())

        def ground_pos(value, default) -> np.ndarray:
            pos = np.asarray(value if value is not None else default, dtype=np.float32).reshape(-1)
            if len(pos) == 2:
                return np.array([pos[0], pos[1], self.policy.box_dims[2]], dtype=np.float32)
            if len(pos) == 3:
                return pos.astype(np.float32).copy()
            raise ValueError("--init-pos/--goal-pos must provide either X Y or X Y Z.")

        def ground_pos_with_half_z(value, default, half_z: float) -> np.ndarray:
            pos = np.asarray(value if value is not None else default, dtype=np.float32).reshape(-1)
            if len(pos) == 2:
                return np.array([pos[0], pos[1], half_z], dtype=np.float32)
            if len(pos) == 3:
                out = pos.astype(np.float32).copy()
                out[2] = half_z
                return out
            raise ValueError("--init-pos-extra must provide either X Y or X Y Z.")

        def pelvis_pos(value, default) -> np.ndarray:
            pos = np.asarray(value if value is not None else default, dtype=np.float32).reshape(-1)
            if len(pos) in {2, 3}:
                return np.array([pos[0], pos[1], DEFAULT_PELVIS_Z], dtype=np.float32)
            raise ValueError("--init-pos/--goal-pos must provide either X Y or X Y Z.")

        self.state_cmd = StateAndCmd(self.num_joints)
        self.policy_output = PolicyOutput(self.num_joints)
        task = "carrybox" if self.is_carryheart else self.args.task
        override_onnx_path = resolve_policy_path(getattr(self.args, "policy", ""))
        policy_label = "override policy onnx"
        task_policy_overrides = {
            "kickball": "kick_50k.onnx",
            "pushbox-two": "combine_50k.onnx",
        }
        if task in task_policy_overrides:
            override_onnx_path = resolve_policy_path(task_policy_overrides[task])
            policy_label = f"{task} policy onnx"
        npz_policy_path = self._policy_path_from_npz_dir()
        if npz_policy_path:
            override_onnx_path = resolve_policy_path(npz_policy_path)
            policy_label = "NPZ path policy onnx"
        self.policy = OmniContact(self.state_cmd, self.policy_output, onnx_path=override_onnx_path)
        if override_onnx_path is not None:
            print(f"[runner] {policy_label}: {override_onnx_path}")
        self.policy.task = task
        if self.policy.task == "carry-carry":
            self.policy.stackbox_stage_count = 2
        elif self.policy.task in {"stackbox", "carry-carry-carry"}:
            self.policy.stackbox_stage_count = 3
        self.policy.reference_source = self.args.reference_source
        self.policy.npz_dir = self.args.npz_dir
        self.policy.tracking_start_frame = int(self.args.start_frame)
        self.policy.tracking_end_frame = int(self.args.end_frame)
        self._configure_policy_dims_from_model()
        if self.policy.task == "push-carry":
            self.policy.active_object_name = "push_box"
            self.policy.box_dims = self.policy.push_box_dims.copy()
        elif self.policy.task == "push-relocate":
            self.policy.active_object_name = "push_box"
            self.policy.box_dims = self.policy.push_box_dims.copy()
        elif self.policy.task == "relocateball":
            self.policy.active_object_name = "ball"
            self.policy.box_dims = self.policy.ball_dims.copy()
        elif self.policy.task == "carry-push":
            self.policy.active_object_name = "carry_box"
            self.policy.box_dims = self.policy.carry_box_dims.copy()
        self.policy.bbox_scale = self.policy.box_dims * 2.0
        self.policy.bbox_offsets_scaled = self.policy.bbox_offsets * self.policy.bbox_scale.reshape(1, 3)
        if self.policy.task == "loco":
            self.policy.init_pos_override = pelvis_pos(getattr(self.args, "init_pos", None), (0.0, 0.0, DEFAULT_PELVIS_Z))
            self.policy.goal_pos_override = pelvis_pos(getattr(self.args, "goal_pos", None), (1.0, 0.0, DEFAULT_PELVIS_Z))
        else:
            self.policy.init_pos_override = ground_pos(getattr(self.args, "init_pos", None), (1.0, 0.0, 0.55))
            self.policy.goal_pos_override = ground_pos(getattr(self.args, "goal_pos", None), (1.0, 1.0, 0.55))
        init_pos_extra = getattr(self.args, "init_pos_extra", None)
        self.policy.carry_box_init_pos_override = None
        self.policy.ball_init_pos_override = None
        if init_pos_extra is not None:
            if self.policy.task == "push-relocate":
                self.policy.ball_init_pos_override = ground_pos_with_half_z(
                    init_pos_extra,
                    (2.2, -0.8, float(self.policy.ball_dims[2])),
                    float(self.policy.ball_dims[2]),
                )
            else:
                self.policy.carry_box_init_pos_override = ground_pos(init_pos_extra, (2.2, -0.8, 0.15))
        if self.policy.reference_source not in {"CFgen", "NPZmotion"}:
            raise ValueError(f"Unsupported reference_source: {self.policy.reference_source}")
        
        print(
            "[runner] task: "
            f"{self.policy.task} | box half dims: {self.policy.box_dims.tolist()} "
            f"| bbox full dims: {self.policy.bbox_scale.tolist()} "
            f"| replan: {bool(getattr(self.args, 'replan', False))}"
        )

    def _pd_control(self) -> np.ndarray:
        q = self.d.qpos[7 : 7 + self.num_joints]
        dq = self.d.qvel[6 : 6 + self.num_joints]
        tau = (self.policy_action - q) * self.policy_kps + (0.0 - dq) * self.policy_kds
        return np.clip(tau, -TORQUE_LIMIT[LAB2MJ], TORQUE_LIMIT[LAB2MJ])

    def _run_policy_tick(self):
        self._sync_state_cmd_from_mj()

        self.policy.run()
        current_action = self.policy_output.actions.copy()
        self._record_torso_tracking_metrics()
        self._record_limb_tracking_metrics()
        self.policy_action = current_action
        self.policy_kps = self.policy_output.kps.copy()
        self.policy_kds = self.policy_output.kds.copy()
        self.policy_tick_counter += 1

    def _handle_policy_tick_postprocess(self):
        if self._monitor_carryheart_done_boxes():
            return
        self._maybe_advance_carryheart()
        self.replan.maybe_replan()

    def _should_stop_when_done(self) -> bool:
        if not self.args.stop_when_done:
            return False
        if self.is_carryheart:
            return self.carryheart_all_done
        return bool(getattr(self.policy_output, "switch_to_loco", False))

    def run(self):
        self._prepare_episode()

        step_limit = int(self.args.max_steps) if self.args.max_steps > 0 else None
        print(
            f"[runner] start Omnicontact policy | reference_source={self.policy.reference_source}, "
            f"dt={self.simulation_dt}, decimation={self.control_decimation}"
        )

        if self.args.headless:
            while True:
                if step_limit is not None and self.sim_counter >= step_limit:
                    print(f"[runner] reached max_steps={step_limit}, exit.")
                    break

                if self.sim_counter % self.control_decimation == 0:
                    self._run_policy_tick()
                    self._handle_policy_tick_postprocess()

                    if self._should_stop_when_done():
                        result = getattr(self.policy_output, "success", "")
                        metrics = self._collect_episode_metrics()
                        print(
                            f"[runner] policy finished. success={result} | "
                            f"action_rate_mean={metrics['action_rate_mean']:.6f} rad/s | "
                            f"torso_tracking_error_mean={metrics['torso_tracking_error_mean']:.6f} m"
                        )
                        break

                tau = self._pd_control()
                self.d.ctrl[:] = tau
                mujoco.mj_step(self.m, self.d)
                self._update_visualization()
                self.sim_counter += 1
        else:
            with mujoco.viewer.launch_passive(self.m, self.d) as viewer:
                viewer.cam.lookat[:] = np.array([1.0, 0.0, 0.9], dtype=np.float32)
                viewer.cam.distance = 6
                viewer.cam.azimuth = 90.0
                viewer.cam.elevation = -20.0
                while viewer.is_running():
                    if step_limit is not None and self.sim_counter >= step_limit:
                        print(f"[runner] reached max_steps={step_limit}, exit.")
                        break

                    tic = time.time()
                    if self.sim_counter % self.control_decimation == 0:
                        self._run_policy_tick()
                        self._handle_policy_tick_postprocess()

                        if self._should_stop_when_done():
                            result = getattr(self.policy_output, "success", "")
                            metrics = self._collect_episode_metrics()
                            print(
                                f"[runner] policy finished. success={result} | "
                                f"action_rate_mean={metrics['action_rate_mean']:.6f} rad/s | "
                                f"torso_tracking_error_mean={metrics['torso_tracking_error_mean']:.6f} m"
                            )
                            break

                    tau = self._pd_control()
                    self.d.ctrl[:] = tau
                    mujoco.mj_step(self.m, self.d)
                    self._update_visualization()
                    self.sim_counter += 1

                    viewer.sync()
                    elapsed = time.time() - tic
                    if self.simulation_dt > elapsed:
                        time.sleep(self.simulation_dt - elapsed)
        if not self.last_episode_metrics:
            self._collect_episode_metrics()


def main() -> None:
    cli_args = parse_args()
    np.random.seed(cli_args.seed)
    runner = OmniContactRunner(cli_args)
    runner.run()


if __name__ == "__main__":
    main()
