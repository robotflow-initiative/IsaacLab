# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from omni.isaac.lab_assets.shadow_hand import SHADOW_HAND_CFG #  use modified version here
from omni.isaac.lab.actuators.actuator_cfg import ImplicitActuatorCfg


import omni.isaac.lab.envs.mdp as mdp
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg, RigidObjectCfg
from omni.isaac.lab.envs import DirectRLEnvCfg
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.markers import VisualizationMarkersCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim import PhysxCfg, SimulationCfg
from omni.isaac.lab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.lab.sensors import CameraCfg, ContactSensorCfg
from omni.isaac.lab.utils.noise import GaussianNoiseCfg, NoiseModelWithAdditiveBiasCfg

from dataclasses import MISSING

@configclass
class EventCfg:
    """Configuration for randomization."""

    # -- robot
    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        min_step_count_between_reset=720,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "static_friction_range": (2.5, 3.0),
            "dynamic_friction_range": (2.5, 3.0),
            "restitution_range": (1.0, 1.0),
            "num_buckets": 250,
        },
    )
    # -- object
    object_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        min_step_count_between_reset=720,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "static_friction_range": (2.5, 3.0),
            "dynamic_friction_range": (2.5, 3.0),
            "restitution_range": (1.0, 1.0),
            "num_buckets": 250,
        },
    )
    # -- scene
    reset_gravity = EventTerm(
        func=mdp.randomize_physics_scene_gravity,
        mode="interval",
        is_global_time=True,
        interval_range_s=(36.0, 36.0),  # time_s = num_steps * (decimation * dt)
        params={
            "gravity_distribution_params": ([0.0, 0.0, 0.0], [0.0, 0.0, 0.4]),
            "operation": "add",
            "distribution": "gaussian",
        },
    )



@configclass
class ShadowHandEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 10.0
    action_space = 20
    observation_space = 157  # (full)
    num_states = 0
    asymmetric_obs = False
    obs_type = "full"

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        physics_material=RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        physx=PhysxCfg(
            bounce_threshold_velocity=0.0,
        ),

    )
    # robot
    robot_cfg: ArticulationCfg = SHADOW_HAND_CFG.replace(prim_path="/World/envs/env_.*/Robot").replace(
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.5),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={".*": 0.0},
        )
    )

    actuated_joint_names = [
        "robot0_WRJ1",
        "robot0_WRJ0",
        "robot0_FFJ3",
        "robot0_FFJ2",
        "robot0_FFJ1",
        "robot0_MFJ3",
        "robot0_MFJ2",
        "robot0_MFJ1",
        "robot0_RFJ3",
        "robot0_RFJ2",
        "robot0_RFJ1",
        "robot0_LFJ4",
        "robot0_LFJ3",
        "robot0_LFJ2",
        "robot0_LFJ1",
        "robot0_THJ4",
        "robot0_THJ3",
        "robot0_THJ2",
        "robot0_THJ1",
        "robot0_THJ0",

    ]

    fingertip_body_names = [
        "robot0_ffdistal",
        "robot0_mfdistal",
        "robot0_rfdistal",
        "robot0_lfdistal",
        "robot0_thdistal",
    ]

    # in-hand object
    object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.0025,
                max_depenetration_velocity=1000.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(density=567.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -0.39, 0.6), rot=(1.0, 0.0, 0.0, 0.0)),
        # init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -0.42, 0), rot=(1.0, 0.0, 0.0, 0.0)),
    )
    # goal object
    goal_object_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/goal_marker",
        markers={
            "goal": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=(1.0, 1.0, 1.0),
            )
        },
    )
    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=8192, env_spacing=0.75, replicate_physics=True)

    # reset
    reset_position_noise = 0.01  # range of position at reset
    reset_dof_pos_noise = 0.2  # range of dof pos at reset
    reset_dof_vel_noise = 0.0  # range of dof vel at reset
    # reward scales
    dist_reward_scale = -10.0
    rot_reward_scale = 1.0
    rot_eps = 0.1
    action_penalty_scale = -0.0002
    reach_goal_bonus = 250
    fall_penalty = 0
    fall_dist = 0.24
    vel_obs_scale = 0.2
    success_tolerance = 0.1
    max_consecutive_success = 0
    av_factor = 0.1
    act_moving_average = 1.0
    force_torque_obs_scale = 10.0



@configclass
class ShadowHandDirectFaceDownRotateLiftEnvCfg(ShadowHandEnvCfg):
    # reward scales
    # fall_penalty = -150
    fall_penalty = -200
    
    success_tolerance = 0.3

    # some scales
    # _object_scale = (1.0, 1.0, 1.0)
    _object_scale = (0.7, 0.7, 0.7)

    # Modified
    # _object_scale = (0.85, 0.85, 0.85)

    table_scale = (0.5, 0.5, 0.24)
    table_pos = (0.0, -0.4, 0.35)
    table_top_pos = table_scale[2] / 2 + table_pos[2]
    # obj_pos = (0.0, -0.43, table_top_pos + 0.09)
    # obj_pos = (0.0, -0.43, table_top_pos + 0.1)
    
    # reward scales
    action_penalty_scale = -0.004
    # added scales
    obj_lin_vel_thresh = 0.04
    obj_ang_vel_thresh = 1
    ftip_reward_scale = -0.1
    dof_vel_thresh = 10
    energy_scale = 20
    hit_penalty = -150
    rot_reward_scale = 1.0  # may not change


    num_actions = 20  # (full)
    num_observations = 157  # (full)


    #######################################################
    ############# scales to overwrite between stages ##########: 
    stage = 3

    if stage == 0:    # train to learn to rotate
        obj_pos = (0.0, -0.43, table_top_pos + 0.09)
        fall_dist = 0.1
        penalize_table_contact = False
        hit_threshold = 0.02  # hit thresh for fingertips
        remove_table_after = -1 # < 0 means no remove




    elif stage == 1:  # train to learn to lift and rotate
        obj_pos = (0.0, -0.43, table_top_pos + 0.024)
        fall_dist = 0.1
        penalize_table_contact = False
        hit_threshold = 0.02  # hit thresh for fingertips
        remove_table_after = -1 # < 0 means no remove


    
    elif stage == 2:  # train to learn to rotate without the help of table
        obj_pos = (0.0, -0.43, table_top_pos + 0.024)
        fall_dist = 100.0  # won't truncate the "out of bound" situation
        penalize_table_contact = True    # False: stage-1, True: stage-2
        hit_threshold = -100.0
        remove_table_after = -1
    
    elif stage == 3:  # train to learn to lift and rotate in the midair
        obj_pos = (0.0, -0.43, table_top_pos + 0.024)
        fall_dist = 0.24  # won't truncate the "out of bound" situation
        penalize_table_contact = True    # False: stage-1, True: stage-2
        hit_threshold = -100.0
        remove_table_after = 20   # remove the table immediately

    # # overwrite to try to get a more natural behavior
    ######## testing for 10.27 #########
    # ftip_reward_scale = -0.5  # 10.27回复，感觉正好，这样表现比较natural
    # action_penalty_scale = -0.0004    # 10.27回复，感觉是小了，应该-0.004合适
    ####### end testing #########

    table_contact_force_scale = 10.0
    object_hit_thresh = 0.02
    reach_goal_bonus = 800
    events: EventCfg = EventCfg()
    
    ################ end of overwrite ##########################
    ############################################################

    ############## Property settings #############################
    ###############################################################
    # robot
    robot_cfg: ArticulationCfg = SHADOW_HAND_CFG.replace(prim_path="/World/envs/env_.*/Robot").replace(
        init_state=ArticulationCfg.InitialStateCfg(
            # pos=(0.0, 0.0, 0.6),
            pos=(0.0, 0.0, 0.58),       # use if the cube is small
            rot=(0.0, 0.0, 1.0, 0.0),
            joint_pos={".*": 0.0},
        )
    )

    # activate the contact sensors in the robot
    robot_cfg.spawn.activate_contact_sensors = True

    # _object_scale = (0.5, 0.5, 0.5) # use if the cube is small

    # in-hand object
    object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/object",
        spawn=sim_utils.UsdFileCfg(
            activate_contact_sensors=True,
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.0025,
                max_depenetration_velocity=1000.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(density=40.0),     # use to adjust the mass of the obj(not sure if it works)
            scale=_object_scale,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=obj_pos, rot=(1.0, 0.0, 0.0, 0.0)),
    )
    # goal object
    goal_object_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/goal_marker",
        markers={
            "goal": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=_object_scale,
            )
        },
    )

    # add configuration
    vis_goal_obj_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/goal_object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.0025,
                max_depenetration_velocity=1000.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(density=567.0),
            scale=_object_scale,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.2, -0.45, 0.73), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    

    table_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/table",
        spawn=sim_utils.CuboidCfg(
            size=table_scale,    # 0.25的意思是不是就是说上下各占0.125
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    kinematic_enabled=True,
                    disable_gravity=True,
                ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.5,
                dynamic_friction=0.5,
            ),
        ),

        init_state=RigidObjectCfg.InitialStateCfg(pos=table_pos, rot=(1.0, 0.0, 0.0, 0.0)),
        
    )

    table_sensor_cfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/table",
        history_length=1,
        debug_vis=False,
    )

    hand_sensor_cfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/robot0_(?!hand_mount|forearm).*",
        history_length=1,
        debug_vis=False,
    )

    hand_sensor_cfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/robot0_(ffdistal|mfdistal|rfdistal|lfdistal|thdistal).*",
        history_length=1,
        debug_vis=False,
        track_pose=True
    )

    wrist_sensor_cfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/robot0_wrist.*",
        history_length=1,
        debug_vis=False,
        track_pose=True
    )

    cube_sensor_cfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/object",
        history_length=1,
        debug_vis=False,
        track_pose=True
    )


    # config the ground
    ground_cfg = sim_utils.GroundPlaneCfg(
        visible=True,
    )
    glass_ground_cfg  = sim_utils.GlassMdlCfg(glass_ior=1.0, glass_color=(255 / 255, 100 / 255, 180 / 255),thin_walled=True)
    ground_prim_path = "/World/ground"

    # Domain randomnization and noises
    # at every time-step add gaussian noise + bias. The bias is a gaussian sampled at reset
    action_noise_model: NoiseModelWithAdditiveBiasCfg = NoiseModelWithAdditiveBiasCfg(
        noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.05, operation="add"),
        bias_noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.015, operation="abs"),
    )
    # at every time-step add gaussian noise + bias. The bias is a gaussian sampled at reset
    observation_noise_model: NoiseModelWithAdditiveBiasCfg = NoiseModelWithAdditiveBiasCfg(
        noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.002, operation="add"),
        bias_noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.0001, operation="abs"),
    )

@configclass
# configuration for debugging
class ShadowHandDirectFaceDownRotateLiftEnvDebugCfg(ShadowHandDirectFaceDownRotateLiftEnvCfg):
    @configclass
    class DebugEventCfg(EventCfg):  # 隔离开来，方便调试
        # -- robot
        robot_physics_material = EventTerm(
            func=mdp.randomize_rigid_body_material,
            mode="reset",
            min_step_count_between_reset=720,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                # "static_friction_range": (3.0, 3.5),
                # "dynamic_friction_range": (3.0, 3.5),
                "static_friction_range": (1.0, 1.5),
                "dynamic_friction_range": (1.0, 1.5),
                "restitution_range": (1.0, 1.0),
                "num_buckets": 250,
            },
        )
        # -- object
        object_physics_material = EventTerm(
            func=mdp.randomize_rigid_body_material,
            min_step_count_between_reset=720,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("object"),
                # "static_friction_range": (3.0, 3.5),
                # "dynamic_friction_range": (3.0, 3.5),
                "static_friction_range": (1.0, 1.5),
                "dynamic_friction_range": (1.0, 1.5),
                "restitution_range": (1.0, 1.0),
                "num_buckets": 250,
            },
        )
        # -- scene
        reset_gravity = EventTerm(
            func=mdp.randomize_physics_scene_gravity,
            mode="interval",
            is_global_time=True,
            interval_range_s=(36.0, 36.0),  # time_s = num_steps * (decimation * dt)
            params={
                "gravity_distribution_params": ([0.0, 0.0, 0.0], [0.0, 0.0, 0.4]),
                "operation": "add",
                "distribution": "gaussian",
            },
        )
    
    # overwrite some configurations
    events: DebugEventCfg = DebugEventCfg()
    # _object_scale = (0.85, 0.85, 0.85)
    _object_scale = (0.7, 0.7, 0.7)
    stage = 3
    if stage == 1: 
        fall_dist = 0.1
        penalize_table_contact = False
        hit_threshold = 0.02  # hit thresh for fingertips
        # hit_threshold = 0.01  # hit thresh for fingertips
        remove_table_after = -1 # < 0 means no remove

    elif stage == 2: 
        fall_dist = 100.0  # won't truncate the "out of bound" situation
        penalize_table_contact = True    # False: stage-1, True: stage-2
        hit_threshold = -100.0
        remove_table_after = -1
    
    elif stage == 3:
        fall_dist = 0.24  # won't truncate the "out of bound" situation
        penalize_table_contact = True    # False: stage-1, True: stage-2
        hit_threshold = -100.0
        remove_table_after = 30

    ####### use original settings #################
    ###############################################
    # ftip_reward_scale = -0.5
    # action_penalty_scale = -0.01
    # add dof velocity penalty and contact reward between hand and obj
    # fingertip_contact_sensor_cfg = ContactSensorCfg(
    #     prim_path="/World/envs/env_.*/Robot/robot0_({}|{}|{}|{}|{}|{}|{}|{}|{}|{}).*".format(
    #         "ffdistal", "mfdistal", "rfdistal", "lfdistal", "thdistal",
    #         "ffmiddle", "mfmiddle", "rfmiddle", "lfmiddle", "thmiddle"
    #     ),
    #     history_length=1,
    #     debug_vis=False,
    # )
    fingertip_contact_sensor_cfg = None
    # fingertip_contact_reward_scale = -0.02
    fingertip_contact_reward_scale = 0.0  # turn off fingertip reward
    # num_fingers = 5
    num_fingers = 10

    # Domain randomnization and noises
    # at every time-step add gaussian noise + bias. The bias is a gaussian sampled at reset
    action_noise_model: NoiseModelWithAdditiveBiasCfg = NoiseModelWithAdditiveBiasCfg(
        noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.05, operation="add"),
        bias_noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.015, operation="abs"),
    )
    # at every time-step add gaussian noise + bias. The bias is a gaussian sampled at reset
    observation_noise_model: NoiseModelWithAdditiveBiasCfg = NoiseModelWithAdditiveBiasCfg(
        noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.002, operation="add"),
        bias_noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.0001, operation="abs"),
    )




@configclass
class ShadowHandDirectFaceDownReorientPCTactileEnvCfg(ShadowHandDirectFaceDownRotateLiftEnvCfg):
# class ShadowHandDirectFaceDownReorientPCTactileEnvCfg(ShadowHandDirectFaceDownRotateLiftEnvDebugCfg):

    #########  For rollout ############
    table_scale = (0.5, 0.5, 0.24)
    table_pos = (0.0, -0.4, 0.35)
    table_top_pos = table_scale[2] / 2 + table_pos[2]

    # stage settings 
    stage = 3
    # obj_pos = (0.0, -0.43, table_top_pos + 0.09)
    obj_pos = (0.0, -0.43, table_top_pos + 0.024)
    fall_dist = 0.24  # won't truncate the "out of bound" situation
    penalize_table_contact = True    # False: stage-1, True: stage-2
    hit_threshold = -100.0
    remove_table_after = 20   # remove the table immediately

    ########  For Rollout for Imitation Model ##########
    # success_tolerance = 0.6
    # obj_lin_vel_thresh = 100.0
    # obj_ang_vel_thresh = 100.0
    ########## End For ##############################


    object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/object",
        spawn=sim_utils.UsdFileCfg(
            activate_contact_sensors=True,
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.0025,
                max_depenetration_velocity=1000.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(density=40.0),     # use to adjust the mass of the obj(not sure if it works)
            scale=(0.7, 0.7, 0.7),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=obj_pos, rot=(1.0, 0.0, 0.0, 0.0)),
    )

    #####################################

    # modify some configurations
    episode_length_s = 10000000.0
    max_consecutive_success = 0    # 防止unexpected behavior in _get_dones()
    table_pos = (0.0, -0.4, 0.35)


    # add camera
    # num_cameras = 2
    num_cameras = 1

    camera_config_00 = CameraCfg(
        prim_path="/World/envs/env_.*/Camera01",
        # height=168,
        # width=168,
        height=112,
        width=112,        
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            # focal_length=24.7, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1e4)
            focal_length=51.6, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1e4)
        ),
        # offset=CameraCfg.OffsetCfg(pos=(0.59024, -0.67162, 0.88707), rot=(0.67906, 0.44996, 0.32038, 0.4835), convention="opengl"),
        offset=CameraCfg.OffsetCfg(pos=(-0.3, 0.2, 0.15), rot=(-0.09916, -0.17147, 0.84852, 0.4907), convention="opengl"),
    )


    # camera_config_01 = CameraCfg(
    #     prim_path="/World/envs/env_.*/Camera02",
    #     # height=168,
    #     # width=168,
    #     height=112,
    #     width=112,
    #     data_types=["rgb", "distance_to_image_plane"],
    #     # spawn=sim_utils.PinholeCameraCfg(
    #     #     focal_length=20.9, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1e4)
    #     # ),
    #     spawn=sim_utils.PinholeCameraCfg(
    #         focal_length=38.8, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1e4)
    #     ),
    #     offset=CameraCfg.OffsetCfg(pos=(0.55, -0.8, 1.0), rot=(0.75221, 0.42114, 0.24708, 0.44161), convention="opengl"),
    # )


    # overwrite some configurations
    table_scale = (0.5, 0.5, 0.24)
    table_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/table",
        spawn=sim_utils.CuboidCfg(
            visible= False,   # not visible
            size=table_scale,    # 0.25的意思是不是就是说上下各占0.125
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    kinematic_enabled=True,
                    disable_gravity=True,
                ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.5,
                dynamic_friction=0.5,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=table_pos, rot=(1.0, 0.0, 0.0, 0.0)),

    )
    ground_cfg = sim_utils.GroundPlaneCfg(
        # visible=False,
    )

    contact_forces_cfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/robot0_(?!hand_mount|forearm).*", history_length=1, debug_vis=False
    )

    # Domain randomnization and noises
    # at every time-step add gaussian noise + bias. The bias is a gaussian sampled at reset
    action_noise_model: NoiseModelWithAdditiveBiasCfg = NoiseModelWithAdditiveBiasCfg(
        noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.05, operation="add"),
        bias_noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.015, operation="abs"),
    )
    # at every time-step add gaussian noise + bias. The bias is a gaussian sampled at reset
    observation_noise_model: NoiseModelWithAdditiveBiasCfg = NoiseModelWithAdditiveBiasCfg(
        noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.002, operation="add"),
        bias_noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.0001, operation="abs"),
    )


    


