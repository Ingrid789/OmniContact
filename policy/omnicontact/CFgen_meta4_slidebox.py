import numpy as np

from common.utils import yaw_quat, yaw_to_quat
from policy.omnicontact.CFgen_base import CfGenBase
from policy.omnicontact.CFgen_meta1_loco import DEFAULT_PELVIS_Z, _append_contactloco_turn_walk_recover


class CfGenSlideBox(CfGenBase):
    """Standalone slide-box planner.

    High-level phases:
      11. Turn to face the box.
      12. Walk behind the box.
      13. Align to the goal direction, final behind-box stance, and settle.
      14. Stay upright and walk straight to the final goal while the box slides with a fixed base offset.

    This planner intentionally keeps the wrists away from the box and does not inject any
    extra contact flags into the reference.
    """

    def __init__(self, pad: int = 30, step_size_linear: float = 0.016, step_size_angular: float = 0.03):
        super().__init__()
        self.pad = int(pad)
        self.step_linear = float(step_size_linear)
        self.step_angular = float(step_size_angular)

        self.cfg = {
            "phase11_slide_standoff_dist": 0.3,
            "phase11_waypoint_trigger_margin": 0.05,
            "phase11_waypoint_trigger_distance": 0.18,
            "phase11_obstacle_margin": 0.2,
            "phase11_waypoint_clearance": 0.3,
            "slide_lateral_offset": 0.05,
        }

    def _slide_contact_for_task(self, task: str) -> tuple[np.ndarray, float]:
        if task not in {"slidebox", "slidebox-left", "slidebox-right"}:
            raise ValueError(f"Unknown slidebox task: {task}")

        if task == "slidebox-left":
            return self._contact_lfoot, -1.0

        # slidebox keeps the old behavior: right foot is the contact foot.
        return self._contact_rfoot, 1.0

    def generate(
        self,
        *,
        pelvis_pos: np.ndarray,
        pelvis_quat: np.ndarray,
        obj_pos: np.ndarray,
        obj_quat: np.ndarray,
        box_half_dims: np.ndarray = np.array([0.2, 0.3, 0.15]),
        target_obj_pos: np.ndarray = np.array([1.0, 1.0, 0.5]),
        task: str = "slidebox",
    ):
        pelvis_pos = np.asarray(pelvis_pos, dtype=np.float32).copy()
        pelvis_quat = np.asarray(pelvis_quat, dtype=np.float32).copy()
        obj_pos = np.asarray(obj_pos, dtype=np.float32).copy()
        obj_pos[2] = box_half_dims[2]  #  the box on the ground
        obj_quat = np.asarray(obj_quat, dtype=np.float32).copy()
        target_obj_pos = np.asarray(target_obj_pos, dtype=np.float32).copy()
        target_obj_pos[2] = obj_pos[2] #  target on the ground
        
        move_dir = (target_obj_pos - obj_pos).astype(np.float32)
        if float(np.linalg.norm(move_dir[:2])) < 1e-6:
            move_dir = (obj_pos - pelvis_pos).astype(np.float32)
        move_norm = float(np.linalg.norm(move_dir))
        move_dir = np.divide(move_dir, move_norm, out=np.array([1.0, 0.0, 0.0], dtype=np.float32), where=move_norm > 1e-6)
        target_yaw = float(np.arctan2(float(move_dir[1]), float(move_dir[0])))
        target_yaw_quat = yaw_to_quat(target_yaw).astype(np.float32)
        slide_contact, lateral_sign = self._slide_contact_for_task(task)
        slide_left_dir = np.array([-move_dir[1], move_dir[0]], dtype=np.float32)
        stance_lateral_offset = (slide_left_dir * lateral_sign * float(self.cfg["slide_lateral_offset"])).astype(np.float32)

        b = self._new_builder()
        final_stance_pos = np.array(
            [
                obj_pos[0] - move_dir[0] * float(self.cfg["phase11_slide_standoff_dist"]),
                obj_pos[1] - move_dir[1] * float(self.cfg["phase11_slide_standoff_dist"]),
                DEFAULT_PELVIS_Z,
            ],
            dtype=np.float32,
        )
        final_stance_pos[:2] += stance_lateral_offset

        # -------------------------
        # Phase11: turn toward the final standing point, walk there, then align with the slide direction and settle.
        # -------------------------
        pelvis_p11_target = np.array([final_stance_pos[0], final_stance_pos[1], DEFAULT_PELVIS_Z], dtype=np.float32)
        self._append_loco_approach_with_waypoints(
            b,
            pelvis_start=pelvis_pos,
            pelvis_target=pelvis_p11_target,
            yaw_start=yaw_quat(pelvis_quat),
            yaw_target=target_yaw_quat,
            step_linear=self.step_linear,
            step_angular=self.step_angular,
            object_pos=obj_pos,
            object_quat=obj_quat,
            obstacle_half_dims=box_half_dims,
        )
        b.pad(11, contact=self._contact0, count=self.pad)

        # -------------------------
        # Phase21: upright straight walk while the box slides with a fixed base offset.
        # -------------------------
        _append_contactloco_turn_walk_recover(
            b,
            phase_turn=None,
            phase_walk=21,
            object_target_pos=target_obj_pos,
            step_angular=self.step_angular,
            step_linear=self.step_linear * 0.5,
            contact=slide_contact,
            walk_pad=self.pad,
            keep_pelvis_z=True,
        )

        return b.finalize(), target_yaw
