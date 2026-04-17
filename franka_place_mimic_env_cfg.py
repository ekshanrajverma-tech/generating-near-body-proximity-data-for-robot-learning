# Copyright (c) 2024-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: Apache-2.0
import sys
import importlib.util as _ilu
from pathlib import Path

_spec = _ilu.spec_from_file_location(
    "prox_env_cfg",
    str(Path(__file__).resolve().with_name("env_cfg.py")),
)
_mod = _ilu.module_from_spec(_spec)
sys.modules["prox_env_cfg"] = _mod  # register BEFORE exec so @configclass can find the module
_spec.loader.exec_module(_mod)
FrankaPlaceCubeIntoBoxEnvCfg_PLAY = _mod.FrankaYCBEnvCfg_PLAY

from isaaclab.envs.mimic_env_cfg import MimicEnvCfg, SubTaskConfig
from isaaclab.utils import configclass


@configclass
class FrankaPlaceCubeIntoBoxMimicEnvCfg(FrankaPlaceCubeIntoBoxEnvCfg_PLAY, MimicEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.datagen_config.name = "demo_src_franka_place_cube_into_box"
        self.datagen_config.generation_guarantee = True
        self.datagen_config.generation_keep_failed = True
        self.datagen_config.generation_num_trials = 10
        self.datagen_config.generation_select_src_per_subtask = True
        self.datagen_config.generation_transform_first_robot_pose = False
        self.datagen_config.generation_interpolate_from_last_target_pose = True
        self.datagen_config.generation_relative = True
        self.datagen_config.max_num_failures = 25
        self.datagen_config.seed = 1
        # Datagen: clutter near task cube so skillgen / CuRobo sees obstacles (not empty counter).
        self.ycb_randomize_near_cube = True
        # Subtask timing / noise aligned with backward_il reference (panda_hand → fr3_hand for FR3).
        self.subtask_configs = {
            "fr3_hand": [
                SubTaskConfig(
                    object_ref="cube",
                    subtask_term_signal="grasp",
                    subtask_term_offset_range=(10, 20),
                    subtask_start_offset_range=(0, 0),
                    selection_strategy="nearest_neighbor_object",
                    selection_strategy_kwargs={"nn_k": 3},
                    action_noise=0.01,
                    num_interpolation_steps=5,
                    num_fixed_steps=0,
                    apply_noise_during_interpolation=False,
                    description="Reach and grasp the cube",
                    next_subtask_description="Carry cube to box and release",
                ),
                SubTaskConfig(
                    object_ref="box",
                    subtask_term_signal="place",
                    subtask_term_offset_range=(0, 0),
                    subtask_start_offset_range=(22, 22),
                    selection_strategy="nearest_neighbor_object",
                    selection_strategy_kwargs={"nn_k": 3},
                    action_noise=0.01,
                    num_interpolation_steps=5,
                    num_fixed_steps=0,
                    apply_noise_during_interpolation=False,
                    description="Place cube inside the box",
                    next_subtask_description=None,
                ),
            ]
        }

