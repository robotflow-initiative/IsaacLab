# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

'''
step里面的步骤顺序：add noise to act -> pre_step -> stepping -> get_dones(compute intermediat values) -> get_rewards -> reset_idx -> post_step -> get_observations -> add noise to obs -> return 
'''


from __future__ import annotations

import numpy as np
import torch
from collections.abc import Sequence

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation, RigidObject
from omni.isaac.lab.envs import DirectRLEnv
from omni.isaac.lab.markers import VisualizationMarkers
from omni.isaac.lab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from omni.isaac.lab.sensors import CameraCfg, Camera, ContactSensorCfg, ContactSensor

from omni.isaac.lab.utils.math import quat_conjugate, quat_from_angle_axis, quat_mul, sample_uniform, saturate, random_orientation

from omni.isaac.lab_tasks.direct.shadow_hand_face_down_rotate.shadow_hand_face_down_rotate_lift_env_cfg import ShadowHandDirectFaceDownRotateLiftEnvCfg

from termcolor import cprint


class ShadowHandFaceDownRotateLiftEnv(DirectRLEnv):
    cfg: ShadowHandDirectFaceDownRotateLiftEnvCfg

    def __init__(self, cfg: ShadowHandDirectFaceDownRotateLiftEnvCfg, render_mode: str | None = None, **kwargs):

        # cprint(f"cfg: {cfg}", "light_yellow")
        cfg.viewer.eye = (1.0, -0.3, 0.68)
        cfg.viewer.lookat = (0.0, -0.4, 0.5)


        super().__init__(cfg, render_mode, **kwargs)

        self.num_hand_dofs = self.hand.num_joints

        # buffers for position targets
        self.hand_dof_targets = torch.zeros((self.num_envs, self.num_hand_dofs), dtype=torch.float, device=self.device)
        self.prev_targets = torch.zeros((self.num_envs, self.num_hand_dofs), dtype=torch.float, device=self.device)
        self.cur_targets = torch.zeros((self.num_envs, self.num_hand_dofs), dtype=torch.float, device=self.device)

        # list of actuated joints
        self.actuated_dof_indices = list()
        for joint_name in cfg.actuated_joint_names:
            self.actuated_dof_indices.append(self.hand.joint_names.index(joint_name))
        self.actuated_dof_indices.sort()


        # finger bodies
        self.finger_bodies = list()
        for body_name in self.cfg.fingertip_body_names:
            self.finger_bodies.append(self.hand.body_names.index(body_name))
        self.finger_bodies.sort()
        self.num_fingertips = len(self.finger_bodies)

        # joint limits
        joint_pos_limits = self.hand.root_physx_view.get_dof_limits().to(self.device)
        self.hand_dof_lower_limits = joint_pos_limits[..., 0]
        self.hand_dof_upper_limits = joint_pos_limits[..., 1]

        # track goal resets
        self.reset_goal_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # used to compare object position
        self.in_hand_pos = self.object.data.default_root_state[:, 0:3].clone()
        self.in_hand_pos[:, 1] += 0.055 # offset to the center of the hand


        cprint(f"self.in_hand_pos: {self.in_hand_pos}", "light_yellow") # [ 0.0000, -0.3750,  0.0000]
        ''' default root state = the initial state inside the configuration'''
 
        self.goal_rot = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
        self.goal_rot[:, 0] = 1.0
        self.goal_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        # self.goal_pos[:, :] = torch.tensor([-0.2, -0.45, 0.73], device=self.device)

        #### test in 10.27 ###########
        self.goal_pos[:, :] = torch.tensor([-0.10, -0.52, 0.65], device=self.device)
        # self.goal_pos[:, :] = self.in_hand_pos
        # initialize goal marker
        self.goal_markers = VisualizationMarkers(self.cfg.goal_object_cfg)

        # track successes
        self.successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.consecutive_successes = torch.zeros(1, dtype=torch.float, device=self.device)

        # unit tensors
        self.x_unit_tensor = torch.tensor([1, 0, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.y_unit_tensor = torch.tensor([0, 1, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.z_unit_tensor = torch.tensor([0, 0, 1], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))

        self.reset_cnt = 0


    def _setup_scene(self):
        # add hand, in-hand object, and goal object
        self.hand = Articulation(self.cfg.robot_cfg)
        self.object = RigidObject(self.cfg.object_cfg)
        self.vis_goal_object = RigidObject(self.cfg.vis_goal_obj_cfg)
        
        # add ground plane
        self.cfg.ground_cfg.func(self.cfg.ground_prim_path, self.cfg.ground_cfg)

        # bound glass material to ground plane
        if self.cfg.glass_ground_cfg is not None:
            self.cfg.glass_ground_cfg.func("/World/Looks/glassMaterial", self.cfg.glass_ground_cfg)
            sim_utils.bind_visual_material(self.cfg.ground_prim_path, "/World/Looks/glassMaterial")
        
        # add tables
        self.table = RigidObject(self.cfg.table_cfg)

        # add sensors
        self.contact_sensors_table = ContactSensor(self.cfg.table_sensor_cfg)
        self.contact_sensors_hand = ContactSensor(self.cfg.hand_sensor_cfg)
        self.contact_sensors_wrist = ContactSensor(self.cfg.wrist_sensor_cfg)
        self.contact_sensors_cube = ContactSensor(self.cfg.cube_sensor_cfg)

        # clone and replicate (no need to filter for this environment)
        self.scene.clone_environments(copy_from_source=False)
        # add articultion to scene - we must register to scene to randomize with EventManager
        self.scene.articulations["robot"] = self.hand
        self.scene.rigid_objects["object"] = self.object
        self.scene.rigid_objects["vis_goal_obj"] = self.vis_goal_object
        self.scene.rigid_objects["table"] = self.table
        self.scene.sensors["contact_sensors_table"] = self.contact_sensors_table
        self.scene.sensors["contact_sensors_hand"] = self.contact_sensors_hand
        self.scene.sensors["contact_sensors_wrist"] = self.contact_sensors_wrist
        self.scene.sensors["contact_sensors_cube"] = self.contact_sensors_cube

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()

    def _apply_action(self) -> None:
        self.cur_targets[:, self.actuated_dof_indices] = scale(
            self.actions,
            self.hand_dof_lower_limits[:, self.actuated_dof_indices],
            self.hand_dof_upper_limits[:, self.actuated_dof_indices],
        )
        self.cur_targets[:, self.actuated_dof_indices] = (
            self.cfg.act_moving_average * self.cur_targets[:, self.actuated_dof_indices]
            + (1.0 - self.cfg.act_moving_average) * self.prev_targets[:, self.actuated_dof_indices]
        )
        self.cur_targets[:, self.actuated_dof_indices] = saturate(
            self.cur_targets[:, self.actuated_dof_indices],
            self.hand_dof_lower_limits[:, self.actuated_dof_indices],
            self.hand_dof_upper_limits[:, self.actuated_dof_indices],
        )

        self.prev_targets[:, self.actuated_dof_indices] = self.cur_targets[:, self.actuated_dof_indices]

        self.hand.set_joint_position_target(
            self.cur_targets[:, self.actuated_dof_indices], joint_ids=self.actuated_dof_indices
        )

    def _get_observations(self) -> dict:
        if self.cfg.asymmetric_obs:
            self.fingertip_force_sensors = self.hand.root_physx_view.get_link_incoming_joint_force()[
                :, self.finger_bodies
            ]

        if self.cfg.obs_type == "openai":
            obs = self.compute_reduced_observations()
        elif self.cfg.obs_type == "full":
            obs = self.compute_full_observations()
        else:
            print("Unknown observations type!")

        if self.cfg.asymmetric_obs:
            states = self.compute_full_state()

        observations = {"policy": obs}
        if self.cfg.asymmetric_obs:
            observations = {"policy": obs, "critic": states}

        return observations

    def _get_rewards(self) -> torch.Tensor:
        (
            total_reward,
            self.reset_goal_buf,
            self.successes[:],
            self.consecutive_successes[:],
        ) = compute_rewards(
            self.reset_buf,
            self.reset_goal_buf,
            self.successes,
            self.consecutive_successes,
            self.max_episode_length,
            self.object_pos,
            self.object_rot,
            self.in_hand_pos,
            self.goal_rot,
            self.cfg.dist_reward_scale,
            self.cfg.rot_reward_scale,
            self.cfg.rot_eps,
            self.actions,
            self.cfg.action_penalty_scale,
            self.cfg.success_tolerance,
            self.cfg.reach_goal_bonus,
            self.cfg.fall_dist,
            self.cfg.fall_penalty,
            self.cfg.av_factor,

            # add
            self.fingertip_pos,
            self.fingertip_pos[..., 2],
            self.cfg.ftip_reward_scale,
            self.cfg.table_contact_force_scale,
            self.cfg.penalize_table_contact,
            self.scene["contact_sensors_table"].data.net_forces_w,
            self.object_linvel,
            self.object_angvel,
            self.cfg.obj_lin_vel_thresh,
            self.cfg.obj_ang_vel_thresh,

            self.hand_dof_vel,
            self.cfg.dof_vel_thresh,
            
            self.cfg.object_hit_thresh + self.cfg.table_top_pos,
            self.cfg.hit_threshold + self.cfg.table_top_pos,
            self.cfg.hit_penalty,

        )

        # add to check when to uplift the table
        if self.cfg.remove_table_after > 0:
            # cprint(f"self.episode_length_buf: {self.episode_length_buf}", "cyan")
            remove_table_ids = torch.nonzero(self.episode_length_buf == self.cfg.remove_table_after, as_tuple=False).squeeze(-1)  # 只能 == 了，不然每一次到这里它都会lift阿... 反正step里面，每step一次只增加一次length_buf，并且在reset_idx的时候会清零，所以只能出此下策
            # cprint(f"remove_table_ids: {remove_table_ids}", "cyan")
            # cprint(f"len(remove_table_ids): {len(remove_table_ids)}", "cyan")
            if len(remove_table_ids) >= 0:
                self._remove_tables(remove_table_ids)


        if "log" not in self.extras:
            self.extras["log"] = dict()
        self.extras["log"]["consecutive_successes"] = self.consecutive_successes.mean()

        # reset goals if the goal has been reached
        goal_env_ids = self.reset_goal_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(goal_env_ids) > 0:
            self._reset_target_pose(goal_env_ids)

            if self.sim.has_rtx_sensors():
                self.sim.render()

        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:

        self._compute_intermediate_values()

        
        goal_dist = torch.norm(self.object_pos - self.in_hand_pos, p=2, dim=-1)
        out_of_reach = goal_dist >= self.cfg.fall_dist

        hit_ground = torch.any(self.fingertip_pos[..., 2] <= (self.cfg.hit_threshold + self.cfg.table_top_pos), dim=-1)

        terminated = out_of_reach | hit_ground

        if self.cfg.max_consecutive_success > 0:
            # Reset progress (episode length buf) on goal envs if max_consecutive_success > 0
            rot_dist = rotation_distance(self.object_rot, self.goal_rot)
            self.episode_length_buf = torch.where(
                torch.abs(rot_dist) <= self.cfg.success_tolerance,
                torch.zeros_like(self.episode_length_buf),
                self.episode_length_buf,
            )
            max_success_reached = self.successes >= self.cfg.max_consecutive_success

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        if self.cfg.max_consecutive_success > 0:
            time_out = time_out | max_success_reached
        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.hand._ALL_INDICES
        # resets articulation and rigid body attributes
        super()._reset_idx(env_ids)

        # reset goals
        self._reset_target_pose(env_ids)

        # reset object
        object_default_state = self.object.data.default_root_state.clone()[env_ids]
        '''Default root state ``[pos, quat, lin_vel, ang_vel]`` in local environment frame '''


        #################################### start of seperate line ###########################################

        pos_noise = sample_uniform(-1.0, 1.0, (len(env_ids), 3), device=self.device)
        # global object positions
        '''
        keep the xyz position relatively still, but add some noise to the it
        '''
        object_default_state[:, 0:3] = (
            object_default_state[:, 0:3] + self.cfg.reset_position_noise * pos_noise + self.scene.env_origins[env_ids]
        )

        rot_noise = sample_uniform(-1.0, 1.0, (len(env_ids), 2), device=self.device)  # noise for X and Y rotation
        object_default_state[:, 3:7] = randomize_rotation(
            rot_noise[:, 0], rot_noise[:, 1], self.x_unit_tensor[env_ids], self.y_unit_tensor[env_ids]
        )

        object_default_state[:, 7:] = torch.zeros_like(self.object.data.default_root_state[env_ids, 7:])
        self.object.write_root_state_to_sim(object_default_state, env_ids)

        # reset hand
        delta_max = self.hand_dof_upper_limits[env_ids] - self.hand.data.default_joint_pos[env_ids]
        delta_min = self.hand_dof_lower_limits[env_ids] - self.hand.data.default_joint_pos[env_ids]


        # turn on the noise
        dof_pos_noise = sample_uniform(-1.0, 1.0, (len(env_ids), self.num_hand_dofs), device=self.device)
        rand_delta = delta_min + (delta_max - delta_min) * 0.5 * dof_pos_noise
        dof_pos = self.hand.data.default_joint_pos[env_ids] + self.cfg.reset_dof_pos_noise * rand_delta

        dof_vel_noise = sample_uniform(-1.0, 1.0, (len(env_ids), self.num_hand_dofs), device=self.device)
        dof_vel = self.hand.data.default_joint_vel[env_ids] + self.cfg.reset_dof_vel_noise * dof_vel_noise


        # turn off the noise

        # dof_pos = self.hand.data.default_joint_pos[env_ids]
        # dof_vel = self.hand.data.default_joint_vel[env_ids]

        self.prev_targets[env_ids] = dof_pos
        self.cur_targets[env_ids] = dof_pos
        self.hand_dof_targets[env_ids] = dof_pos

        self.hand.set_joint_position_target(dof_pos, env_ids=env_ids)

        self.hand.write_joint_state_to_sim(dof_pos, dof_vel, env_ids=env_ids)

        # add, reset the table to its original position
        self._replace_tables(env_ids)

        self.successes[env_ids] = 0
        self._compute_intermediate_values()

        

    def _reset_target_pose(self, env_ids):
        
        # reset goal rotation
        rand_floats = sample_uniform(-1.0, 1.0, (len(env_ids), 2), device=self.device)


        ############ code to control the target pose ############
        new_rot = randomize_rotation(
            rand_floats[:, 0], rand_floats[:, 1], self.x_unit_tensor[env_ids], self.y_unit_tensor[env_ids]
        )

        # update goal pose and markers
        self.goal_rot[env_ids] = new_rot
        goal_pos = self.goal_pos + self.scene.env_origins
        self.goal_markers.visualize(goal_pos, self.goal_rot)

        # Add: reset the vis_goal_obj's pos and rot
        vis_root_state = torch.cat([goal_pos, self.goal_rot], dim=-1)
        self.vis_goal_object.write_root_pose_to_sim(root_pose=vis_root_state[env_ids], env_ids=env_ids)

        self.reset_goal_buf[env_ids] = 0
    
    def _remove_tables(self, env_ids):
        table_root_state = self.table.data.default_root_state.clone()[env_ids]
        # cprint(f"table_root_state: {table_root_state.shape}", "cyan")
        table_root_state[..., 0:3] = table_root_state[..., 0:3] + self.scene.env_origins[env_ids]
        table_root_state[..., 2] = 10.0 # height 丢上去
        table_root_state[..., 7:] = torch.zeros_like(table_root_state[..., 7:])
        self.table.write_root_state_to_sim(table_root_state, env_ids)

    def _replace_tables(self, env_ids):
        table_root_state = self.table.data.default_root_state.clone()[env_ids]
        # cprint(f"table_root_state: {table_root_state.shape}", "cyan")
        table_root_state[..., 0:3] = table_root_state[..., 0:3] + self.scene.env_origins[env_ids]
        # table_root_state[..., 2] = 10.0 # height 丢上去     # 不丢上去，就是原本的位置
        table_root_state[..., 7:] = torch.zeros_like(table_root_state[..., 7:])
        self.table.write_root_state_to_sim(table_root_state, env_ids)


    def _compute_intermediate_values(self):

        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()
        if is_rendering:
            self.sim.render()

        # data for hand
        self.fingertip_pos = self.hand.data.body_pos_w[:, self.finger_bodies]
        self.fingertip_rot = self.hand.data.body_quat_w[:, self.finger_bodies]
        self.fingertip_pos -= self.scene.env_origins.repeat((1, self.num_fingertips)).reshape(
            self.num_envs, self.num_fingertips, 3
        )

        self.fingertip_velocities = self.hand.data.body_vel_w[:, self.finger_bodies]

        self.hand_dof_pos = self.hand.data.joint_pos
        self.hand_dof_vel = self.hand.data.joint_vel

        # data for object
        self.object_pos = self.object.data.root_pos_w - self.scene.env_origins
        self.object_rot = self.object.data.root_quat_w
        self.object_velocities = self.object.data.root_vel_w
        self.object_linvel = self.object.data.root_lin_vel_w
        self.object_angvel = self.object.data.root_ang_vel_w



    def compute_reduced_observations(self):
        # Per https://arxiv.org/pdf/1808.00177.pdf Table 2
        #   Fingertip positions
        #   Object Position, but not orientation
        #   Relative target orientation

        obs = torch.cat(
            (
                self.fingertip_pos.view(self.num_envs, self.num_fingertips * 3),
                self.object_pos,
                quat_mul(self.object_rot, quat_conjugate(self.goal_rot)),
                self.actions,
            ),
            dim=-1,
        )

        return obs

    def compute_full_observations(self):
        #print(self.hand_dof_pos.shape, self.hand_dof_vel.shape)
        obs = torch.cat(
            (
                # hand
                unscale(self.hand_dof_pos, self.hand_dof_lower_limits, self.hand_dof_upper_limits),
                self.cfg.vel_obs_scale * self.hand_dof_vel,
                # object
                self.object_pos,
                self.object_rot,
                self.object_linvel,
                self.cfg.vel_obs_scale * self.object_angvel,
                # goal
                self.in_hand_pos,
                self.goal_rot,
                quat_mul(self.object_rot, quat_conjugate(self.goal_rot)),
                # fingertips
                self.fingertip_pos.view(self.num_envs, self.num_fingertips * 3),
                self.fingertip_rot.view(self.num_envs, self.num_fingertips * 4),
                self.fingertip_velocities.view(self.num_envs, self.num_fingertips * 6),
                # actions
                self.actions,
            ),
            dim=-1,
        )
        return obs

    def compute_full_state(self):
        
        states = torch.cat(
            (
                # hand
                unscale(self.hand_dof_pos, self.hand_dof_lower_limits, self.hand_dof_upper_limits), #[:24]
                self.cfg.vel_obs_scale * self.hand_dof_vel, #[24:31]
                # object
                self.object_pos, #[31:34]
                self.object_rot, #[34:38]
                self.object_linvel,
                self.cfg.vel_obs_scale * self.object_angvel,
                # goal
                self.in_hand_pos,
                self.goal_rot,
                quat_mul(self.object_rot, quat_conjugate(self.goal_rot)),
                # fingertips
                self.fingertip_pos.view(self.num_envs, self.num_fingertips * 3),
                self.fingertip_rot.view(self.num_envs, self.num_fingertips * 4),
                self.fingertip_velocities.view(self.num_envs, self.num_fingertips * 6),
                self.cfg.force_torque_obs_scale
                * self.fingertip_force_sensors.view(self.num_envs, self.num_fingertips * 6),
                # actions
                self.actions,
            ),
            dim=-1,
        )

        return states


@torch.jit.script
def scale(x, lower, upper):
    return 0.5 * (x + 1.0) * (upper - lower) + lower


@torch.jit.script
def unscale(x, lower, upper):
    return (2.0 * x - upper - lower) / (upper - lower)


@torch.jit.script
def randomize_rotation(rand0, rand1, x_unit_tensor, y_unit_tensor):
    return quat_mul(
        quat_from_angle_axis(rand0 * np.pi, x_unit_tensor), quat_from_angle_axis(rand1 * np.pi, y_unit_tensor)
    )


@torch.jit.script
def rotation_distance(object_rot, target_rot):
    # Orientation alignment for the cube in hand and goal cube
    quat_diff = quat_mul(object_rot, quat_conjugate(target_rot))
    return 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 1:4], p=2, dim=-1), max=1.0))  # changed quat convention


@torch.jit.script
def compute_rewards(
    reset_buf: torch.Tensor,
    reset_goal_buf: torch.Tensor,
    successes: torch.Tensor,
    consecutive_successes: torch.Tensor,
    max_episode_length: float,
    object_pos: torch.Tensor,
    object_rot: torch.Tensor,
    target_pos: torch.Tensor,
    target_rot: torch.Tensor,
    dist_reward_scale: float,
    rot_reward_scale: float,
    rot_eps: float,
    actions: torch.Tensor,
    action_penalty_scale: float,
    success_tolerance: float,
    reach_goal_bonus: float,
    fall_dist: float,
    fall_penalty: float,
    av_factor: float,

    # add
    fingertip_pos: torch.Tensor,
    fingertip_pos_z: torch.Tensor,
    ftip_reward_scale: float,
    table_contact_force_scale: float,
    penalize_table_contact: bool,
    table_contact_force: torch.Tensor,
    object_linvel: torch.Tensor,
    object_angvel: torch.Tensor,
    obj_lin_vel_thresh: float,
    obj_ang_vel_thresh: float,    
    dof_vel: torch.Tensor,
    dof_vel_thresh: float,
    object_hit_thresh: float,

    hit_threshold: float = 0.0, 
    hit_penalty: float = 0.0,
    
):
    
    num_envs = object_pos.shape[0]


    goal_dist = torch.norm(object_pos - target_pos, p=2, dim=-1)
    rot_dist = rotation_distance(object_rot, target_rot)

    dist_rew = goal_dist * dist_reward_scale
    rot_rew = 1.0 / (torch.abs(rot_dist) + rot_eps) * rot_reward_scale

    action_penalty = torch.sum(actions**2, dim=-1)    

    ################ computing additional rewards  ##########################
    reward_terms = dict()
    if ftip_reward_scale < 0:
        ftip_diff = (fingertip_pos.view(num_envs, -1, 3) - object_pos[:, None, :])
        ftip_dist = torch.linalg.norm(ftip_diff, dim=-1).view(num_envs, -1)
        ftip_dist_mean = ftip_dist.mean(dim=-1)
        ftip_reward = ftip_dist_mean * ftip_reward_scale
        reward_terms['ftip_reward'] = ftip_reward
    
    if penalize_table_contact:
        in_contact = (torch.abs(table_contact_force).sum(-1) > 0.0).view(num_envs,) | (object_pos[..., 2] <= object_hit_thresh)
        in_contact = in_contact
        reward_terms['tb_contact_reward'] = -in_contact.float() * table_contact_force_scale
    
    ################ end of computing additional rewards ####################
    # cprint(f"reward_terms: {reward_terms}", "red")
    # cprint(f"contact_force: {table_contact_force}", "red")


    # accumulate rewards
    reward = torch.sum(torch.stack(list(reward_terms.values())), dim=0)


    del reward_terms

    # Total reward is: position distance + orientation alignment + action regularization + success bonus + fall penalty + hit penalty(if threshold > 0.0)
    reward = reward + (dist_rew + rot_rew + action_penalty * action_penalty_scale)

    # Find out which envs hit the goal and update successes count
    # goal_resets = torch.where(torch.abs(rot_dist) <= success_tolerance, torch.ones_like(reset_goal_buf), reset_goal_buf)
    object_linvel_norm = torch.linalg.norm(object_linvel, dim=-1)
    object_angvel_norm = torch.linalg.norm(object_angvel, dim=-1)
    dof_vel_norm = torch.linalg.norm(dof_vel, dim=-1)

    goal_resets = torch.where(
        torch.abs(rot_dist) <= success_tolerance, torch.ones_like(reset_goal_buf) & (object_linvel_norm <= obj_lin_vel_thresh) & (object_angvel_norm <= obj_ang_vel_thresh) & (dof_vel_norm <= dof_vel_thresh) , reset_goal_buf
        )
    # goal_resets = torch.where(
    #     torch.abs(rot_dist) <= success_tolerance, torch.ones_like(reset_goal_buf) & (object_linvel_norm <= obj_lin_vel_thresh) & (object_angvel_norm <= obj_ang_vel_thresh) & (dof_vel_norm <= dof_vel_thresh) , reset_goal_buf
    #     )
    if penalize_table_contact:
        goal_resets = goal_resets & (torch.abs(table_contact_force).sum(-1).view(num_envs,) == 0.0) & (object_pos[..., 2] > object_hit_thresh)
    
    
    successes = successes + goal_resets
    # Success bonus: orientation is within `success_tolerance` of goal orientation
    reward = torch.where(goal_resets == 1, reward + reach_goal_bonus, reward)

    # Fall penalty: distance to the goal is larger than a threshold
    reward = torch.where(goal_dist >= fall_dist, reward + fall_penalty, reward)

    # add by STCZZZ: hit penalty: preventing the fingertips to hit the ground. 
    reward = torch.where(torch.any(fingertip_pos_z <= hit_threshold, dim=-1), reward + hit_penalty, reward)

    # Check env termination conditions, including maximum success number
    resets = torch.where(goal_dist >= fall_dist, torch.ones_like(reset_buf), reset_buf)

    resets = torch.where(torch.any(fingertip_pos_z <= hit_threshold, dim=-1), torch.ones_like(resets), resets)

    num_resets = torch.sum(resets)
    finished_cons_successes = torch.sum(successes * resets.float())

    cons_successes = torch.where(
        num_resets > 0,
        av_factor * finished_cons_successes / num_resets + (1.0 - av_factor) * consecutive_successes,
        consecutive_successes,
    )

    return reward, goal_resets, successes, cons_successes
