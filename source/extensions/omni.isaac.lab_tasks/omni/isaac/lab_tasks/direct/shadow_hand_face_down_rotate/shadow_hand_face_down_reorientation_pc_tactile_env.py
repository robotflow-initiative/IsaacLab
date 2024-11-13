from __future__ import annotations

import torch

from omni.isaac.lab_tasks.direct.shadow_hand_face_down_rotate import ShadowHandFaceDownRotateLiftEnv
from omni.isaac.lab_tasks.direct.shadow_hand_face_down_rotate import ShadowHandDirectFaceDownReorientPCTactileEnvCfg

from omni.isaac.lab_tasks.direct.shadow_hand_face_down_rotate.shadow_hand_face_down_rotate_lift_env import randomize_rotation, compute_rewards

# add imports
from omni.isaac.lab.sensors import  Camera, ContactSensor
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation, RigidObject
from omni.isaac.lab.envs.common import VecEnvStepReturn
from omni.isaac.lab.sensors.camera.utils import create_pointcloud_from_rgbd
from omni.isaac.lab.utils.math import sample_uniform


import open3d as o3d
import numpy as np
from collections.abc import Sequence


from termcolor import cprint

import omni.isaac.lab.sim as sim_utils




def get_pc_and_color(obs, env_id, camera_numbers):
    points_all = []
    colors_all = []
    for cam_id in range(camera_numbers):
        rgba_all = obs.get(f"rgba_img_0{cam_id}", None)
        depth_all = obs.get(f"depth_img_0{cam_id}", None)
        intrinsic_matrices_all = obs.get(f"intrinsic_matrices_0{cam_id}", None)
        pos_w_all = obs.get(f"pos_w_0{cam_id}", None)
        quat_w_ros_all = obs.get(f"quat_w_ros_0{cam_id}", None)

        rgba = rgba_all[env_id]
        depth = depth_all[env_id]
        intrinsic_matrix = intrinsic_matrices_all[env_id]
        pos_w = pos_w_all[env_id]
        quat_w_ros = quat_w_ros_all[env_id]

        # generate point cloud
        points_xyz, points_rgb = create_pointcloud_from_rgbd(
            intrinsic_matrix=intrinsic_matrix,
            depth=depth,
            rgb=rgba,
            normalize_rgb=False,  # normalize to get 0~1 pc, the same as dp3
            position=pos_w,
            orientation=quat_w_ros,
        )

        # add points and colors to list
        points_all.append(points_xyz)
        colors_all.append(points_rgb)

    # concatenate points and colors
    points_all = torch.cat(points_all, dim=0)
    colors_all = torch.cat(colors_all, dim=0)

    return points_all, colors_all


class ShadowHandDirectFaceDownReorientPCTactileSingleCamEnv(ShadowHandFaceDownRotateLiftEnv):
    cfg: ShadowHandDirectFaceDownReorientPCTactileEnvCfg
    
    def __init__(self, cfg: ShadowHandDirectFaceDownReorientPCTactileEnvCfg, render_mode: str | None = None, **kwargs):

        self.num_cameras = cfg.num_cameras
        self.camera_crop_max = 1024
        cprint(f"cfg: {cfg}", "magenta")
        # modify the configuration of the simulation env
        cprint(f"[ShadowHandDirectFaceDownReorientPCTactileSingleCamEnv]cfg.success_tolerance: {cfg.success_tolerance}", "green")

        # we should define the "camera_config" before calling the father class
        super().__init__(cfg, render_mode, **kwargs)
        cprint(f"self.episode_length_s: {self.max_episode_length_s}", "cyan")
        cprint(f"self.max_episode_length: {self.max_episode_length}", "cyan")
            

    ######### We don't want to modify the original "reset_target_pose" function ##########

    # Overwrite the "set_up_scene" function to add cameras, just cameras
    def _setup_scene(self):
        # add hand, in-hand object, and goal object
        self.hand = Articulation(self.cfg.robot_cfg)
        self.object = RigidObject(self.cfg.object_cfg)
        self.vis_goal_object = RigidObject(self.cfg.vis_goal_obj_cfg)
        # add ground plane
        # no ground plane
        # self.cfg.ground_cfg.func(self.cfg.ground_prim_path, self.cfg.ground_cfg)

        # # bound glass material to ground plane
        # if self.cfg.glass_ground_cfg is not None:
        #     self.cfg.glass_ground_cfg.func("/World/Looks/glassMaterial", self.cfg.glass_ground_cfg)
        #     sim_utils.bind_visual_material(self.cfg.ground_prim_path, "/World/Looks/glassMaterial")
        
        # add tables
        self.table = RigidObject(self.cfg.table_cfg)

        # add sensors
        self.contact_sensors_table = ContactSensor(self.cfg.table_sensor_cfg)
        self.contact_forces = ContactSensor(self.cfg.contact_forces_cfg)

        # add cameras
        self.camera_00 = Camera(self.cfg.camera_config_00)
        # self.camera_01 = Camera(self.cfg.camera_config_01)

        # clone and replicate (no need to filter for this environment)
        self.scene.clone_environments(copy_from_source=False)
        # add articultion to scene - we must register to scene to randomize with EventManager
        self.scene.articulations["robot"] = self.hand
        self.scene.rigid_objects["object"] = self.object
        self.scene.rigid_objects["vis_goal_obj"] = self.vis_goal_object
        self.scene.rigid_objects["table"] = self.table
        self.scene.sensors["contact_sensors_table"] = self.contact_sensors_table
        self.scene.sensors["camera_00"] = self.camera_00
        # self.scene.sensors["camera_01"] = self.camera_01
        self.scene.sensors["contact_forces"] = self.contact_forces
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
    
    # Overwrite the "compute_rewards" function to return gaol_env_ids additionally   
    def _get_rewards(self) -> tuple[torch.Tensor, torch.Tensor]:
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
        # cprint(f"self.episode_length_buf: {self.episode_length_buf}", "cyan")
        # add to check when to uplift the table
        if self.cfg.remove_table_after >= 0:
            # cprint(f"self.episode_length_buf: {self.episode_length_buf}", "cyan")
            remove_table_ids = torch.nonzero(self.episode_length_buf == self.cfg.remove_table_after, as_tuple=False).squeeze(-1)  # 只能 == 了，不然每一次到这里它都会lift阿... 反正step里面，每step一次只增加一次length_buf，并且在reset_idx的时候会清零，所以只能出此下策
            if len(remove_table_ids) > 0:
                self._remove_tables(remove_table_ids)

        if "log" not in self.extras:
            self.extras["log"] = dict()
        self.extras["log"]["consecutive_successes"] = self.consecutive_successes.mean()

        # reset goals if the goal has been reached
        goal_env_ids = self.reset_goal_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(goal_env_ids) > 0:
            # self._reset_target_pose(goal_env_ids)
            # just for testing, do nothing now
            if self.sim.has_rtx_sensors():
                self.sim.render()

        return total_reward, goal_env_ids

    def _get_observations(self) -> dict:

        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()
        # make sure the synchronization between the physics and the rendering
        if is_rendering:
            for i in range(3):
                self.sim.render()
        obs_origin = super()._get_observations()
        
        for cam_id in range(self.num_cameras):
        # get rgba, depth, intrinsic_matrices, pos_w, quat_w_ros from camera sensor
            obs_origin[f"rgba_img_0{cam_id}"] = self.scene[f"camera_0{cam_id}"].data.output["rgb"]
            obs_origin[f"depth_img_0{cam_id}"] = self.scene[f"camera_0{cam_id}"].data.output["distance_to_image_plane"]
            obs_origin[f"intrinsic_matrices_0{cam_id}"] = self.scene[f"camera_0{cam_id}"].data.intrinsic_matrices
            obs_origin[f"pos_w_0{cam_id}"] = self.scene[f"camera_0{cam_id}"].data.pos_w
            obs_origin[f"quat_w_ros_0{cam_id}"] = self.scene[f"camera_0{cam_id}"].data.quat_w_ros
        
        # add point_cloud here, will be used in the Imitation learning part. 
        for env_id in range(self.num_envs):
            point_cloud_list = []
            points_all, colors_all = get_pc_and_color(obs_origin, env_id, self.num_cameras)
            # cprint(f"points_all.shape: {points_all.shape}", "cyan")  # depends on the precision of the camera
            # cprint(f"self.num_cameras: {self.num_cameras}", "cyan")
            points_env = o3d.geometry.PointCloud()
            points_env.points = o3d.utility.Vector3dVector(points_all.detach().cpu().numpy())
            points_env.colors = o3d.utility.Vector3dVector(colors_all.detach().cpu().numpy())
            # farthest point sampling
            points_env = o3d.geometry.PointCloud.farthest_point_down_sample(points_env, self.camera_crop_max)
            # combine points and colors
            combined_points_colors = np.concatenate([np.asarray(points_env.points), np.asarray(points_env.colors)], axis=-1)
            point_cloud_list.append(torch.tensor(combined_points_colors))
            # cprint(f"pc.shape: {torch.tensor(combined_points_colors).shape}", "cyan")
        
        point_clout_tensor = torch.stack(point_cloud_list, dim=0)
        obs_origin["point_cloud"] = point_clout_tensor

        # check the output of the contact sensors
        contact_forces:ContactSensor = self.scene["contact_forces"]
        contact_data = contact_forces.data.net_forces_w

        # shape: [num_envs, num_contacts, 3], 3 means the xyz forces
        # I wanted to compute the 合力, I.E. X**2 + Y**2 + Z**2

        contact_data = torch.norm(contact_data, dim=-1)
        obs_origin["contact_forces"] = contact_data

        # add the agent pos
        obs_agent_pos = self.compute_full_observations()[..., :24]
        obs_origin["agent_pos"] = obs_agent_pos
        # add observation noise to agent_pos
        if self.cfg.observation_noise_model:
            obs_origin["agent_pos"] = self._observation_noise_model.apply(obs_origin["agent_pos"])

        return obs_origin
    
    # redefine _reset_idx, we cannot use super().reset_idx here
    ''' reset_idx: reset the [env, the obj, the target_obj], reset_target_pose: only reset the target_obj. '''
    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.hand._ALL_INDICES

        ######### From father.father  ###########
        self.scene.reset(env_ids)

        # apply events such as randomization for environments that need a reset
        if self.cfg.events:
            if "reset" in self.event_manager.available_modes:
                env_step_count = self._sim_step_counter // self.cfg.decimation
                self.event_manager.apply(mode="reset", env_ids=env_ids, global_env_step_count=env_step_count)

        # reset noise models
        if self.cfg.action_noise_model:
            self._action_noise_model.reset(env_ids)
        if self.cfg.observation_noise_model:
            self._observation_noise_model.reset(env_ids)

        # reset the episode length buffer
        self.episode_length_buf[env_ids] = 0

        ######### father.father ends ########

        # reset goals
        self._reset_target_pose(env_ids)

        # reset object
        object_default_state = self.object.data.default_root_state.clone()[env_ids]
        '''Default root state ``[pos, quat, lin_vel, ang_vel]`` in local environment frame '''

        pos_noise = sample_uniform(-1.0, 1.0, (len(env_ids), 3), device=self.device)
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

        dof_pos_noise = sample_uniform(-1.0, 1.0, (len(env_ids), self.num_hand_dofs), device=self.device)
        rand_delta = delta_min + (delta_max - delta_min) * 0.5 * dof_pos_noise
        dof_pos = self.hand.data.default_joint_pos[env_ids] + self.cfg.reset_dof_pos_noise * rand_delta

        dof_vel_noise = sample_uniform(-1.0, 1.0, (len(env_ids), self.num_hand_dofs), device=self.device)
        dof_vel = self.hand.data.default_joint_vel[env_ids] + self.cfg.reset_dof_vel_noise * dof_vel_noise

        self.prev_targets[env_ids] = dof_pos
        self.cur_targets[env_ids] = dof_pos
        self.hand_dof_targets[env_ids] = dof_pos

        self.hand.set_joint_position_target(dof_pos, env_ids=env_ids)
        self.hand.write_joint_state_to_sim(dof_pos, dof_vel, env_ids=env_ids)
        

        # add, reset the table to its original position
        self._replace_tables(env_ids)


        self.successes[env_ids] = 0
        self._compute_intermediate_values()

        # add by STCZZZ, apply phisics step, yet don't need to update dt, action, etc cause we don't want the obj to fall
        self.scene.write_data_to_sim()
        # simulate
        self.sim.step(render=False)
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()
        # make sure the synchronization between the physics and the rendering
        if is_rendering:
            for i in range(3):
                self.sim.render()
        
    # Inherit the original reset function so that "goal_env_ids" is added inside obs_dict every step
    def reset(self, *args, **kwargs):
        obs_dict, extra = super().reset(*args, **kwargs)
        _, goal_env_ids = self._get_rewards()
        obs_dict["goal_env_ids"] = goal_env_ids
        obs_dict["goal_reached"] = torch.tensor([len(goal_env_ids) > 0])
        return obs_dict, extra

    ''' overwrite the "step" function for the collection convenience, one-word and one-line difference'''
    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        """Execute one time-step of the environment's dynamics.

        The environment steps forward at a fixed time-step, while the physics simulation is decimated at a
        lower time-step. This is to ensure that the simulation is stable. These two time-steps can be configured
        independently using the :attr:`DirectRLEnvCfg.decimation` (number of simulation steps per environment step)
        and the :attr:`DirectRLEnvCfg.sim.physics_dt` (physics time-step). Based on these parameters, the environment
        time-step is computed as the product of the two.

        This function performs the following steps:

        1. Pre-process the actions before stepping through the physics.
        2. Apply the actions to the simulator and step through the physics in a decimated manner.
        3. Compute the reward and done signals.
        4. Reset environments that have terminated or reached the maximum episode length.
        5. Apply interval events if they are enabled.
        6. Compute observations.

        Args:
            action: The actions to apply on the environment. Shape is (num_envs, action_dim).

        Returns:
            A tuple containing the observations, rewards, resets (terminated and truncated) and extras.
        """
        action = action.to(self.device)
        # add action noise
        if self.cfg.action_noise_model:
            action = self._action_noise_model.apply(action)

        # process actions
        self._pre_physics_step(action)

        # check if we need to do rendering within the physics loop
        # note: checked here once to avoid multiple checks within the loop
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        # perform physics stepping
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            # set actions into buffers
            self._apply_action()
            # set actions into simulator
            self.scene.write_data_to_sim()
            # simulate
            self.sim.step(render=False)
            # render between steps only if the GUI or an RTX sensor needs it
            # note: we assume the render interval to be the shortest accepted rendering interval.
            #    If a camera needs rendering at a faster frequency, this will lead to unexpected behavior.
            
            # cprint(f"self._sim_step_counter: {self._sim_step_counter}", "yellow")
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                
                # cprint(f"rendering! ", "cyan")
                self.sim.render()
            # update buffers at sim dt
            self.scene.update(dt=self.physics_dt)

        # post-step:
        # -- update env counters (used for curriculum generation)
        self.episode_length_buf += 1  # step in current episode (per env)
        self.common_step_counter += 1  # total step (common for all envs)

        self.reset_terminated[:], self.reset_time_outs[:] = self._get_dones()
        self.reset_buf = self.reset_terminated | self.reset_time_outs

        # one-word difference here
        self.reward_buf, goal_env_ids = self._get_rewards()

        # -- reset envs that terminated/timed-out and log the episode information
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)  # SELF.RESETBUF是在get_rewards里面计算的

        if len(reset_env_ids) > 0:
            self._reset_idx(reset_env_ids)

        # post-step: step interval event
        if self.cfg.events:
            if "interval" in self.event_manager.available_modes:
                self.event_manager.apply(mode="interval", dt=self.step_dt)

        # update observations
        self.obs_buf = self._get_observations()

        # add observation noise
        # note: we apply no noise to the state space (since it is used for critic networks)
        if self.cfg.observation_noise_model:
            self.obs_buf["policy"] = self._observation_noise_model.apply(self.obs_buf["policy"])
        
        # one_line_difference_here
        self.obs_buf["goal_env_ids"] = goal_env_ids
        
        self.obs_buf["goal_reached"] = torch.tensor([len(goal_env_ids) > 0])

        # return observations, rewards, resets and extras
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras
    


class ShadowHandDirectFaceDownReorientPCTactileSingleCamMultiTargetEnv(ShadowHandDirectFaceDownReorientPCTactileSingleCamEnv):
    ''' One Line Difference'''
    # Overwrite the "compute_rewards" function to return gaol_env_ids additionally   
    def _get_rewards(self) -> tuple[torch.Tensor, torch.Tensor]:
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
        # cprint(f"self.episode_length_buf: {self.episode_length_buf}", "cyan")
        # add to check when to uplift the table
        if self.cfg.remove_table_after >= 0:
            # cprint(f"self.episode_length_buf: {self.episode_length_buf}", "cyan")
            remove_table_ids = torch.nonzero(self.episode_length_buf == self.cfg.remove_table_after, as_tuple=False).squeeze(-1)  # 只能 == 了，不然每一次到这里它都会lift阿... 反正step里面，每step一次只增加一次length_buf，并且在reset_idx的时候会清零，所以只能出此下策
            if len(remove_table_ids) > 0:
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

        return total_reward, goal_env_ids
    

        
    