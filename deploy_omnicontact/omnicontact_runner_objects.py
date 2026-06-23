import numpy as np

from policy.omnicontact.CFgen_meta1_loco import DEFAULT_PELVIS_Z


class OmniContactObjectsMixin:
    def _table_height_offset(self) -> float:
        return float(self.policy.box_dims[2])

    def _active_object_name(self) -> str:
        if self.is_carryheart:
            return self.carryheart_box_names[self.carryheart_active_idx]
        active_name = str(getattr(self.policy, "active_object_name", "")).strip()
        if self.policy.task == "push-carry" and active_name not in {"push_box", "carry_box"}:
            return "push_box"
        if self.policy.task == "carry-push" and active_name not in {"push_box", "carry_box"}:
            return "carry_box"
        if self.policy.task == "push-relocate" and active_name not in {"push_box", "ball"}:
            return "push_box"
        return active_name

    def _get_active_object_body_id(self) -> int:
        active_name = self._active_object_name()
        if self.is_carryheart and active_name in self.carryheart_box_names:
            idx = self.carryheart_box_names.index(active_name)
            body_id = int(self.carryheart_box_body_ids[idx])
            if body_id >= 0:
                return body_id
        if active_name in self.stack_box_names:
            idx = self.stack_box_names.index(active_name)
            body_id = int(self.stack_box_body_ids[idx])
            if body_id >= 0:
                return body_id
        if active_name == "carry_box" and self.carry_box_body_id >= 0:
            return self.carry_box_body_id
        if active_name == "push_box" and self.push_box_body_id >= 0:
            return self.push_box_body_id
        if active_name == "ball" and self.ball_body_id >= 0:
            return self.ball_body_id
        return self.object_body_id

    def _get_active_object_qpos_adr(self) -> int:
        active_name = self._active_object_name()
        if self.is_carryheart and active_name in self.carryheart_box_names:
            idx = self.carryheart_box_names.index(active_name)
            qpos_adr = int(self.carryheart_box_qpos_adrs[idx])
            if qpos_adr >= 0:
                return qpos_adr
        if active_name in self.stack_box_names:
            idx = self.stack_box_names.index(active_name)
            qpos_adr = int(self.stack_box_qpos_adrs[idx])
            if qpos_adr >= 0:
                return qpos_adr
        if active_name == "carry_box" and self.carry_box_qpos_adr >= 0:
            return self.carry_box_qpos_adr
        if active_name == "push_box" and self.push_box_joint_id >= 0:
            return int(self.m.jnt_qposadr[self.push_box_joint_id])
        if active_name == "ball" and self.ball_qpos_adr >= 0:
            return self.ball_qpos_adr
        if self.object_joint_id >= 0:
            return int(self.m.jnt_qposadr[self.object_joint_id])
        return -1

    def _get_active_object_qvel_adr(self) -> int:
        active_name = self._active_object_name()
        if self.is_carryheart and active_name in self.carryheart_box_names:
            idx = self.carryheart_box_names.index(active_name)
            qvel_adr = int(self.carryheart_box_qvel_adrs[idx])
            if qvel_adr >= 0:
                return qvel_adr
        if active_name in self.stack_box_names:
            idx = self.stack_box_names.index(active_name)
            qvel_adr = int(self.stack_box_qvel_adrs[idx])
            if qvel_adr >= 0:
                return qvel_adr
        if active_name == "carry_box" and self.carry_box_qvel_adr >= 0:
            return self.carry_box_qvel_adr
        if active_name == "push_box" and self.push_box_joint_id >= 0:
            return int(self.m.jnt_dofadr[self.push_box_joint_id])
        if active_name == "ball" and self.ball_qvel_adr >= 0:
            return self.ball_qvel_adr
        if self.object_joint_id >= 0:
            return int(self.m.jnt_dofadr[self.object_joint_id])
        return -1

    def _goal_pos(self) -> np.ndarray:
        if self.is_carryheart and self.carryheart_goal_positions:
            return self.carryheart_goal_positions[self.carryheart_active_idx].copy()
        goal_override = getattr(self.policy, "goal_pos_override", None)
        goal = np.asarray(goal_override, dtype=np.float32).reshape(3).copy()
        if self.policy.task == "loco":
            goal[2] = float(DEFAULT_PELVIS_Z)
            return goal
        if self.policy.task in {"pushbox-two", "pushbox-in", "slidebox", "slidebox-left", "slidebox-right", "kickball", "push-carry", "carry-push", "push-relocate"}:
            goal[2] = float(self.policy.box_dims[2])
        return goal

    def _carry_push_plane_z(self) -> float:
        push_dims = np.asarray(
            getattr(self.policy, "push_box_dims", np.array([0.23, 0.25, 0.26], dtype=np.float32)),
            dtype=np.float32,
        ).reshape(3)
        return float(push_dims[2] * 2.0 + 0.02)

    def _plane2_goal_pos(self) -> np.ndarray:
        goal_pos = self._goal_pos()
        if self.policy.task == "carry-push":
            carry_dims = np.asarray(
                getattr(self.policy, "carry_box_dims", np.array([0.15, 0.15, 0.15], dtype=np.float32)),
                dtype=np.float32,
            ).reshape(3)
            goal_pos[2] = float(self._carry_push_plane_z() + carry_dims[2])
        elif self.policy.task == "push-relocate":
            push_dims = np.asarray(
                getattr(self.policy, "push_box_dims", np.array([0.23, 0.25, 0.26], dtype=np.float32)),
                dtype=np.float32,
            ).reshape(3)
            goal_pos[2] = float(push_dims[2] * 2.0)
        return goal_pos

    def _init_pos(self) -> np.ndarray:
        init_override = getattr(self.policy, "init_pos_override", None)
        init_pos = np.asarray(init_override, dtype=np.float32).reshape(3).copy()
        if self.policy.task == "loco":
            init_pos[2] = float(DEFAULT_PELVIS_Z)
            return init_pos
        if self.policy.task in {"pushbox-two", "pushbox-in", "slidebox", "slidebox-left", "slidebox-right", "relocateball", "kickball", "push-carry", "carry-push", "push-relocate"}:
            init_pos[2] = float(self.policy.box_dims[2])
        return init_pos

    def _active_object_speed(self) -> tuple[float, float]:
        qvel_adr = self._get_active_object_qvel_adr()
        if qvel_adr < 0 or qvel_adr + 6 > self.d.qvel.shape[0]:
            return 0.0, 0.0
        lin_speed = float(np.linalg.norm(self.d.qvel[qvel_adr : qvel_adr + 3]))
        ang_speed = float(np.linalg.norm(self.d.qvel[qvel_adr + 3 : qvel_adr + 6]))
        return lin_speed, ang_speed
