# Copyright (c) 2024-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence
import numpy as np
import torch

import isaaclab.utils.math as PoseUtils
from isaaclab.envs import ManagerBasedRLMimicEnv


class FrankaPlaceCubeIntoBoxMimicEnv(ManagerBasedRLMimicEnv):

    def _get_eef_name(self):
        if self.cfg.subtask_configs:
            return list(self.cfg.subtask_configs.keys())[0]
        return "panda_hand"

    def get_robot_eef_pose(self, eef_name: str, env_ids: Sequence[int] | None = None) -> torch.Tensor:
        if env_ids is None:
            env_ids = slice(None)
        eef_pos = self.obs_buf["policy"]["eef_pos"][env_ids]
        eef_quat = self.obs_buf["policy"]["eef_quat"][env_ids]
        return PoseUtils.make_pose(eef_pos, PoseUtils.matrix_from_quat(eef_quat))

    def target_eef_pose_to_action(self, target_eef_pose_dict, gripper_action_dict, action_noise_dict=None, env_id=0):
        eef_name = self._get_eef_name()
        (target_eef_pose,) = target_eef_pose_dict.values()
        target_pos, target_rot = PoseUtils.unmake_pose(target_eef_pose)
        curr_pose = self.get_robot_eef_pose(eef_name, env_ids=[env_id])[0]
        curr_pos, curr_rot = PoseUtils.unmake_pose(curr_pose)
        delta_position = target_pos - curr_pos
        delta_rot_mat = target_rot.matmul(curr_rot.transpose(-1, -2))
        delta_quat = PoseUtils.quat_from_matrix(delta_rot_mat)
        delta_rotation = PoseUtils.axis_angle_from_quat(delta_quat)

        # The DIK controller applies deltas in the robot's ROOT (base) frame via
        # apply_delta_pose(ee_pos_base, ee_quat_base, command).
        # The deltas above are in the world frame.  Rotate them into the base frame.
        root_quat_w = self.scene["robot"].data.root_quat_w[env_id]
        root_rot_w = PoseUtils.matrix_from_quat(root_quat_w.unsqueeze(0))[0]
        R_base_from_world = root_rot_w.T
        delta_position = R_base_from_world @ delta_position
        delta_rotation = R_base_from_world @ delta_rotation

        (gripper_action,) = gripper_action_dict.values()
        pose_action = torch.cat([delta_position, delta_rotation], dim=0)
        if action_noise_dict is not None:
            noise = action_noise_dict.get(eef_name, 0.0) * torch.randn_like(pose_action)
            pose_action = torch.clamp(pose_action + noise, -1.0, 1.0)
        gripper_action = gripper_action.to(pose_action.device)
        return torch.cat([pose_action, gripper_action], dim=0)

    def action_to_target_eef_pose(self, action: torch.Tensor) -> dict[str, torch.Tensor]:
        eef_name = self._get_eef_name()
        delta_position_base = action[:, :3]
        delta_rotation_base = action[:, 3:6]

        # Recorded actions are base-frame deltas (produced by the DIK controller).
        # get_robot_eef_pose returns world-frame poses, so rotate deltas to world.
        root_quat_w = self.scene["robot"].data.root_quat_w  # (N, 4)
        R_world_from_base = PoseUtils.matrix_from_quat(root_quat_w)  # (N, 3, 3)
        delta_position = torch.bmm(R_world_from_base, delta_position_base.unsqueeze(-1)).squeeze(-1)
        delta_rotation = torch.bmm(R_world_from_base, delta_rotation_base.unsqueeze(-1)).squeeze(-1)

        curr_pose = self.get_robot_eef_pose(eef_name)
        curr_pos, curr_rot = PoseUtils.unmake_pose(curr_pose)
        target_pos = curr_pos + delta_position
        delta_rotation_angle = torch.linalg.norm(delta_rotation, dim=-1, keepdim=True)
        delta_rotation_axis = delta_rotation / (delta_rotation_angle + 1e-8)
        is_zero = torch.isclose(delta_rotation_angle, torch.zeros_like(delta_rotation_angle)).squeeze(1)
        delta_rotation_axis[is_zero] = 0.0
        delta_quat = PoseUtils.quat_from_angle_axis(delta_rotation_angle.squeeze(1), delta_rotation_axis).squeeze(0)
        delta_rot_mat = PoseUtils.matrix_from_quat(delta_quat)
        target_rot = torch.matmul(delta_rot_mat, curr_rot)
        return {eef_name: PoseUtils.make_pose(target_pos, target_rot).clone()}

    def actions_to_gripper_actions(self, actions) -> dict[str, torch.Tensor]:
        # Use type name check to handle cross-install torch tensor instances
        type_name = type(actions).__name__
        if type_name == 'Tensor' or hasattr(actions, 'detach'):
            actions = actions.detach().cpu().float()
        else:
            import numpy as np
            actions = torch.tensor(np.array(actions), dtype=torch.float32)
        return {self._get_eef_name(): actions[:, -1:]}

    def get_object_poses(self, env_ids: Sequence[int] | None = None) -> dict:
        if env_ids is None:
            env_ids = slice(None)
        cube_pos = self.obs_buf["policy"]["cube_pos"][env_ids]
        box_pos = self.obs_buf["policy"]["box_pos"][env_ids]
        n = cube_pos.shape[0]
        eye = torch.eye(3, device=cube_pos.device).unsqueeze(0).expand(n, -1, -1)
        return {
            "cube": PoseUtils.make_pose(cube_pos, eye),
            "box": PoseUtils.make_pose(box_pos, eye),
        }

    def get_subtask_term_signals(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        if env_ids is None:
            env_ids = slice(None)
        subtask_terms = self.obs_buf["subtask_terms"]
        return {
            "grasp": subtask_terms["grasp"][env_ids],
            "place": subtask_terms["place"][env_ids],
        }

    def get_subtask_start_signals(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
        subtask_terms = self.obs_buf["subtask_terms"]
        grasp = subtask_terms["grasp"][env_ids]
        
        # Grasp starts when the robot hand gets close to the cube (0 -> 1 transition)
        ee_frame = self.scene["ee_frame"]
        eef_pos = ee_frame.data.target_pos_w[env_ids, 0, :]
        cube_pos = self.scene["cube"].data.root_pos_w[env_ids, :3]
        dist = torch.norm(eef_pos - cube_pos, dim=-1, keepdim=True)
        grasp_start = (dist < 0.15).float()
        
        # Place starts exactly when grasp is achieved (0 -> 1 transition)
        place_start = grasp.clone()
        
        return {
            "grasp": grasp_start,
            "place": place_start,
        }

    def get_expected_attached_object(self, eef_name: str, subtask_index: int, cfg) -> str | None:
        """Tell skillgen/CuRobo what object should be attached during transitions.

        - Transition into grasp (subtask 0): nothing is attached yet.
        - Transition into place (subtask 1): cube should be grasped/carried.
        """
        if subtask_index >= 1:
            return "cube"
        return None

