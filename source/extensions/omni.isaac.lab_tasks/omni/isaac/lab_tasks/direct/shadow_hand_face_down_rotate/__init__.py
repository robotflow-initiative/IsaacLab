# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Shadow Hand environment.
"""

import gymnasium as gym

from . import agents
from .shadow_hand_face_down_rotate_lift_env_cfg import  ShadowHandDirectFaceDownRotateLiftEnvCfg
from .shadow_hand_face_down_rotate_lift_env import ShadowHandFaceDownRotateLiftEnv

from .shadow_hand_face_down_rotate_lift_env_cfg import ShadowHandDirectFaceDownReorientPCTactileEnvCfg
from .shadow_hand_face_down_reorientation_pc_tactile_env import ShadowHandDirectFaceDownReorientPCTactileSingleCamEnv

from .shadow_hand_face_down_rotate_lift_env_cfg import ShadowHandDirectFaceDownRotateLiftEnvDebugCfg
from .shadow_hand_face_down_rotate_lift_debug_env import ShadowHandFaceDownRotateLiftDebugEnv

from .shadow_hand_face_down_reorientation_pc_tactile_env import ShadowHandDirectFaceDownReorientPCTactileSingleCamMultiTargetEnv

##
# Register Gym environments.
##


gym.register(
    id="Isaac-Repose-Cube-Shadow-Direct-Face-Down-Rotate-Lift-v0",
    entry_point="omni.isaac.lab_tasks.direct.shadow_hand_face_down_rotate:ShadowHandFaceDownRotateLiftEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": ShadowHandDirectFaceDownRotateLiftEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ShadowHandPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Repose-Cube-Shadow-Direct-Face-Down-Reorient-Debug-v0",
    entry_point="omni.isaac.lab_tasks.direct.shadow_hand_face_down_rotate:ShadowHandFaceDownRotateLiftDebugEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": ShadowHandDirectFaceDownRotateLiftEnvDebugCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_debug_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ShadowHandPPORunnerCfg",
    },
)


gym.register(
    id="Isaac-Repose-Cube-Shadow-Direct-Face-Down-Reorient-PC-Tactile-v0",
    entry_point="omni.isaac.lab_tasks.direct.shadow_hand_face_down_rotate:ShadowHandDirectFaceDownReorientPCTactileSingleCamEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": ShadowHandDirectFaceDownReorientPCTactileEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ShadowHandPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Repose-Cube-Shadow-Direct-Face-Down-Reorient-PC-Tactile-Multi-Target-v0",
    entry_point="omni.isaac.lab_tasks.direct.shadow_hand_face_down_rotate:ShadowHandDirectFaceDownReorientPCTactileSingleCamMultiTargetEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": ShadowHandDirectFaceDownReorientPCTactileEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ShadowHandPPORunnerCfg",
    },
)







