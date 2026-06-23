import copy
import threading
import time
from types import SimpleNamespace

import numpy as np

from policy.omnicontact.CFgen_reference import initialize_cfgen_reference, init_cfgen_state


class OmniContactCarryboxReplan:
    _COOLDOWN_TICKS = 80
    _REFERENCE_DEVIATION_THRESHOLD = 1.5
    _CARRYBOX_FINAL_GOAL_TOLERANCE = 0.2
    _PUSHBOX_FINAL_GOAL_TOLERANCE = 0.5
    _OBJ_SPEED_THRESHOLD = 0.05
    _OBJ_ANG_SPEED_THRESHOLD = 0.05
    _PUSHBOX_TASKS = {"pushbox-two", "pushbox-in"}
    _BALL_TASKS = {"relocateball"}

    def __init__(self, runner, enabled: bool):
        self.runner = runner
        self.enabled = bool(enabled)
        self.counter = 0
        self.worker_thread = None
        self.pending_result = None
        self.worker_error = None
        self.worker_lock = threading.Lock()
        self.worker_generation = 0
        self.reset_detection()

    def reset_detection(self):
        self.cooldown_counter = 0

    def reset_episode(self):
        self.reset_detection()
        self.runner.policy.replan_active = False
        with self.worker_lock:
            self.worker_generation += 1
            self.pending_result = None
            self.worker_error = None
            self.worker_thread = None

    def supported(self) -> bool:
        return (
            self.enabled
            and self.runner.policy.reference_source == "CFgen"
            and self.runner.policy.task not in {"kickball", "loco"}
        )

    def _worker_running(self) -> bool:
        thread = self.worker_thread
        return thread is not None and thread.is_alive()

    def _current_goal_pos(self) -> np.ndarray:
        if hasattr(self.runner.policy, "goal_pos"):
            return np.asarray(self.runner.policy.goal_pos, dtype=np.float32).reshape(3).copy()
        return self.runner._goal_pos()

    def _final_goal_tolerance(self) -> float:
        if self.runner.policy.task in self._PUSHBOX_TASKS:
            return self._PUSHBOX_FINAL_GOAL_TOLERANCE
        return self._CARRYBOX_FINAL_GOAL_TOLERANCE

    def _current_ref_object_pos(self) -> np.ndarray | None:
        if not hasattr(self.runner.policy, "ref_object_pos"):
            return None
        ref_object_pos = np.asarray(self.runner.policy.ref_object_pos, dtype=np.float32)
        if ref_object_pos.ndim != 2 or ref_object_pos.shape[0] == 0:
            return None
        ref_idx = int(np.clip(int(getattr(self.runner.policy, "counter_step", 1)) - 1, 0, ref_object_pos.shape[0] - 1))
        return ref_object_pos[ref_idx].copy()

    def _current_object_pos(self) -> np.ndarray | None:
        active_body_id = self.runner._get_active_object_body_id()
        if active_body_id < 0:
            return None
        return self.runner.d.xpos[active_body_id].copy()

    def _object_reference_deviation(self) -> float:
        obj_pos = self._current_object_pos()
        ref_pos = self._current_ref_object_pos()
        if obj_pos is None or ref_pos is None:
            return 0.0
        return float(np.linalg.norm(obj_pos - ref_pos))

    def _object_goal_distance(self) -> float:
        obj_pos = self._current_object_pos()
        if obj_pos is None:
            return 0.0
        return float(np.linalg.norm(obj_pos - self._current_goal_pos()))

    def _current_obj_linear_speed(self) -> float:
        start = self.runner._get_active_object_qvel_adr()
        end = start + 3
        if start < 0 or end > self.runner.d.qvel.shape[0]:
            return 0.0
        return float(np.linalg.norm(self.runner.d.qvel[start:end]))

    def _current_obj_angular_speed(self) -> float:
        start = self.runner._get_active_object_qvel_adr()
        if start < 0:
            return 0.0
        start += 3
        end = start + 3
        if end > self.runner.d.qvel.shape[0]:
            return 0.0
        return float(np.linalg.norm(self.runner.d.qvel[start:end]))

    def _is_object_static(self, obj_lin_speed: float, obj_ang_speed: float) -> bool:
        if self.runner.policy.task in self._BALL_TASKS or self.runner.policy.active_object_name == "ball":
            return obj_lin_speed <= self._OBJ_SPEED_THRESHOLD
        return obj_lin_speed <= self._OBJ_SPEED_THRESHOLD and obj_ang_speed <= self._OBJ_ANG_SPEED_THRESHOLD

    def _snapshot_state_cmd(self):
        src = self.runner.state_cmd
        return SimpleNamespace(
            q=src.q.copy(),
            dq=src.dq.copy(),
            base_pos=src.base_pos.copy(),
            base_quat=src.base_quat.copy(),
            obj_pos=src.obj_pos.copy(),
            obj_quat=src.obj_quat.copy(),
            ball_pos=src.ball_pos.copy(),
            ball_quat=src.ball_quat.copy(),
            push_box_pos=src.push_box_pos.copy(),
            push_box_quat=src.push_box_quat.copy(),
            carry_box_pos=src.carry_box_pos.copy(),
            carry_box_quat=src.carry_box_quat.copy(),
            stack_box_pos=src.stack_box_pos.copy(),
            stack_box_quat=src.stack_box_quat.copy(),
        )

    def _snapshot_policy(self, state_cmd):
        policy = self.runner.policy
        snapshot = SimpleNamespace(
            state_cmd=state_cmd,
            kinematics=policy.kinematics,
            bbox_offsets=policy.bbox_offsets.copy(),
            task=policy.task,
            active_object_name=str(getattr(policy, "active_object_name", "box")),
            box_dims=np.asarray(policy.box_dims, dtype=np.float32).copy(),
            ball_dims=np.asarray(policy.ball_dims, dtype=np.float32).copy(),
            push_box_dims=np.asarray(policy.push_box_dims, dtype=np.float32).copy(),
            carry_box_dims=np.asarray(policy.carry_box_dims, dtype=np.float32).copy(),
            stack_box_names=tuple(policy.stack_box_names),
            stack_box_dims=np.asarray(policy.stack_box_dims, dtype=np.float32).copy(),
            stack_box_goal_pos=np.asarray(policy.stack_box_goal_pos, dtype=np.float32).copy(),
            goal_pos=np.asarray(policy.goal_pos, dtype=np.float32).copy(),
            push_carry_stage=copy.deepcopy(getattr(policy, "push_carry_stage", "idle")),
            push_relocate_stage=copy.deepcopy(getattr(policy, "push_relocate_stage", "idle")),
            stackbox_stage_idx=int(getattr(policy, "stackbox_stage_idx", 0)),
            stackbox_stage_count=int(getattr(policy, "stackbox_stage_count", 3)),
        )
        snapshot.bbox_scale = snapshot.box_dims * 2.0
        snapshot.bbox_offsets_scaled = snapshot.bbox_offsets * snapshot.bbox_scale.reshape(1, 3)
        init_cfgen_state(snapshot, pad=30)
        snapshot.push_carry_stage = copy.deepcopy(getattr(policy, "push_carry_stage", "idle"))
        snapshot.push_relocate_stage = copy.deepcopy(getattr(policy, "push_relocate_stage", "idle"))
        snapshot.stackbox_stage_idx = int(getattr(policy, "stackbox_stage_idx", 0))
        snapshot.stackbox_stage_count = int(getattr(policy, "stackbox_stage_count", 3))
        return snapshot

    @staticmethod
    def _collect_reference_result(policy, reason: str, obj_lin_speed: float, obj_ang_speed: float, start_time: float) -> dict:
        reference_attrs = (
            "target_yaw",
            "goal_pos",
            "active_object_name",
            "box_dims",
            "bbox_scale",
            "bbox_offsets_scaled",
            "push_carry_stage",
            "push_relocate_stage",
            "stackbox_stage_idx",
            "stackbox_stage_count",
            "traj_generator",
            "ref_left_wrist_pos",
            "ref_left_wrist_quat",
            "ref_right_wrist_pos",
            "ref_right_wrist_quat",
            "ref_object_pos",
            "ref_object_quat",
            "ref_contact",
            "ref_phase",
            "ref_torso_future_pos",
            "ref_torso_future_quat",
            "ref_left_ankle_future_pos",
            "ref_left_ankle_future_quat",
            "ref_right_ankle_future_pos",
            "ref_right_ankle_future_quat",
            "dof_pos",
            "ref_base_pos",
            "ref_base_quat",
        )
        return {
            "attrs": {name: copy.deepcopy(getattr(policy, name)) for name in reference_attrs if hasattr(policy, name)},
            "reason": reason,
            "obj_lin_speed": float(obj_lin_speed),
            "obj_ang_speed": float(obj_ang_speed),
            "elapsed": float(time.perf_counter() - start_time),
        }

    def _worker_entry(self, generation: int, policy_snapshot, fk_info, reason: str, obj_lin_speed: float, obj_ang_speed: float):
        start_time = time.perf_counter()
        try:
            initialize_cfgen_reference(policy_snapshot, fk_info)
            result = self._collect_reference_result(policy_snapshot, reason, obj_lin_speed, obj_ang_speed, start_time)
            with self.worker_lock:
                if generation != self.worker_generation:
                    return
                self.pending_result = result
                self.worker_error = None
        except Exception as exc:
            with self.worker_lock:
                if generation != self.worker_generation:
                    return
                self.worker_error = exc
                self.pending_result = None

    def _start_async_replan(self, reason: str, obj_lin_speed: float, obj_ang_speed: float):
        if not self.supported():
            return
        self.runner.policy.replan_active = True
        self.runner._sync_state_cmd_from_mj()
        state_snapshot = self._snapshot_state_cmd()
        policy_snapshot = self._snapshot_policy(state_snapshot)
        fk_info = self.runner.policy.kinematics.forward(state_snapshot.q, state_snapshot.base_pos, state_snapshot.base_quat)
        self.runner.policy_output.switch_to_loco = False
        self.runner.policy_output.success = ""
        self.reset_detection()
        self.cooldown_counter = self._COOLDOWN_TICKS
        with self.worker_lock:
            self.worker_generation += 1
            generation = self.worker_generation
        thread = threading.Thread(
            target=self._worker_entry,
            args=(generation, policy_snapshot, fk_info, reason, obj_lin_speed, obj_ang_speed),
            daemon=True,
        )
        with self.worker_lock:
            self.pending_result = None
            self.worker_error = None
            self.worker_thread = thread
        thread.start()
        print(
            f"[closed_loop] async replan requested | reason={reason}, "
            f"obj_lin_speed={obj_lin_speed:.3f} m/s, obj_ang_speed={obj_ang_speed:.3f} rad/s"
        )

    def _apply_pending_replan(self):
        with self.worker_lock:
            result = self.pending_result
            error = self.worker_error
            if result is None and error is None:
                return False
            self.pending_result = None
            self.worker_error = None
            self.worker_thread = None

        if error is not None:
            self.runner.policy.replan_active = False
            print(f"[closed_loop] async replan failed: {error}")
            return False

        for name, value in result["attrs"].items():
            setattr(self.runner.policy, name, value)
        self.runner.policy.counter_step = 0
        self.runner.policy.success = ""
        self.runner.policy.switch_to_loco = False
        self.runner.policy.replan_active = True
        self.runner.policy_output.switch_to_loco = False
        self.runner.policy_output.success = ""
        self.counter += 1
        self.cooldown_counter = self._COOLDOWN_TICKS
        print(
            f"[closed_loop] replan#{self.counter} applied | reason={result['reason']}, "
            f"elapsed={result['elapsed'] * 1000.0:.1f} ms, "
            f"obj_lin_speed={result['obj_lin_speed']:.3f} m/s, "
            f"obj_ang_speed={result['obj_ang_speed']:.3f} rad/s"
        )
        return True

    def maybe_replan(self):
        if not self.supported():
            return

        if self._apply_pending_replan():
            return

        if self._worker_running():
            return

        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            return

        obj_lin_speed = self._current_obj_linear_speed()
        obj_ang_speed = self._current_obj_angular_speed()
        if not self._is_object_static(obj_lin_speed, obj_ang_speed):
            return

        ref_deviation = self._object_reference_deviation()
        if ref_deviation > self._REFERENCE_DEVIATION_THRESHOLD:
            self._start_async_replan(
                reason=f"reference_deviation_{ref_deviation:.3f}",
                obj_lin_speed=obj_lin_speed,
                obj_ang_speed=obj_ang_speed,
            )
            return

        plan_done = bool(getattr(self.runner.policy_output, "switch_to_loco", False))
        if not plan_done:
            return

        goal_dist = self._object_goal_distance()
        if goal_dist > self._final_goal_tolerance():
            self._start_async_replan(
                reason=f"final_goal_error_{goal_dist:.3f}",
                obj_lin_speed=obj_lin_speed,
                obj_ang_speed=obj_ang_speed,
            )
