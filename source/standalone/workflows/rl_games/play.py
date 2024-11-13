# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RL-Games."""

"""Launch Isaac Sim Simulator first."""

import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from RL-Games.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument(
    "--use_last_checkpoint",
    action="store_true",
    help="When no checkpoint provided, use the last saved model. Otherwise use the best saved model.",
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""


import gymnasium as gym
import math
import os
import torch

from rl_games.common import env_configurations, vecenv
from rl_games.common.player import BasePlayer
from rl_games.torch_runner import Runner

from omni.isaac.lab.envs import DirectMARLEnv, multi_agent_to_single_agent
from omni.isaac.lab.utils.assets import retrieve_file_path
from omni.isaac.lab.utils.dict import print_dict

import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils import get_checkpoint_path, load_cfg_from_registry, parse_env_cfg
from omni.isaac.lab_tasks.utils.wrappers.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper

def list2str(input):
    s="["
    for i in input:
        s+=str(i)+","
    s+="]"
    return s

def main():
    """Play with RL-Games agent."""
    # parse env configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg = load_cfg_from_registry(args_cli.task, "rl_games_cfg_entry_point")

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rl_games", agent_cfg["params"]["config"]["name"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    # find checkpoint
    if args_cli.checkpoint is None:
        # specify directory for logging runs
        run_dir = agent_cfg["params"]["config"].get("full_experiment_name", ".*")
        # specify name of checkpoint
        if args_cli.use_last_checkpoint:
            checkpoint_file = ".*"
        else:
            # this loads the best checkpoint
            checkpoint_file = f"{agent_cfg['params']['config']['name']}.pth"
        # get path to previous checkpoint
        resume_path = get_checkpoint_path(log_root_path, run_dir, checkpoint_file, other_dirs=["nn"])
    else:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    log_dir = os.path.dirname(os.path.dirname(resume_path))

    # wrap around environment for rl-games
    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_root_path, log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rl-games
    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions)

    # register the environment to rl-games registry
    # note: in agents configuration: environment name must be "rlgpu"
    vecenv.register(
        "IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

    # load previously trained model
    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = resume_path
    print(f"[INFO]: Loading model checkpoint from: {agent_cfg['params']['load_path']}")

    # set number of actors into agent config
    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs
    # create runner from rl-games
    runner = Runner()
    runner.load(agent_cfg)
    # obtain the agent from the runner
    agent: BasePlayer = runner.create_player()
    agent.restore(resume_path)
    agent.reset()

    # reset environment
    obs = env.reset()
    if isinstance(obs, dict):
        obs = obs["obs"]
    timestep = 0
    # required: enables the flag for batched observations
    _ = agent.get_batch_size(obs, 1)
    # initialize RNN states if used
    if agent.is_rnn:
        agent.init_rnn()
    # simulate environment
    # note: We simplified the logic in rl-games player.py (:func:`BasePlayer.run()`) function in an
    #   attempt to have complete control over environment stepping. However, this removes other
    #   operations such as masking that is used for multi-agent learning by RL-Games.
    print("-------------------------------")
    print(type(env.env.scene),type(agent.env.env))
    print("-------------------------------")
    from omni.isaac.core.objects import cuboid, sphere
    import numpy as np
    from pxr import UsdShade, Gf, UsdGeom, Usd
    from omni.isaac.core.simulation_context import SimulationContext

    spheres=None
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # convert obs to agent format
            obs = agent.obs_to_torch(obs)
            # agent stepping
            actions = agent.get_action(obs, is_deterministic=True)
            # env stepping
            obs, _, dones, _ = env.step(actions)

            # perform operations for terminated episodes
            if len(dones) > 0:
                # reset rnn state for terminated episodes
                if agent.is_rnn and agent.states is not None:
                    for s in agent.states:
                        s[:, dones, :] = 0.0

            print("-------------------------------")
            #print(env.env.scene["contact_sensors_hand"].data)
            sph_list = env.env.scene["contact_sensors_hand"].data.pos_w.cpu().numpy()
            force = torch.norm(env.env.scene["contact_sensors_hand"].data.net_forces_w,dim=-1)[0].cpu().numpy()
            force_sum = env.env.scene["contact_sensors_hand"].data.net_forces_w[0]
            
            force_sum = -torch.sum(force_sum,dim=0).cpu().numpy()
            wrist_pos = env.env.scene["contact_sensors_wrist"].data.pos_w.cpu().numpy()
            wrist_quat = env.env.scene["contact_sensors_wrist"].data.quat_w.cpu().numpy()
            obj_pos_quat = obs[0,48:55].cpu().numpy() # type: ignore

            cube_pos = list(env.env.scene["contact_sensors_cube"].data.pos_w.cpu().numpy()[0,0])
            cube_quat = list(env.env.scene["contact_sensors_cube"].data.quat_w.cpu().numpy()[0,0])
            cube_force = env.env.scene["contact_sensors_cube"].data.net_forces_w.cpu().numpy()[0,0]

        # wrist = env.env.scene.stage.GetPrimAtPath("/World/envs/env_0/Robot/robot0_wrist")
        # cube = env.env.scene.stage.GetPrimAtPath("/World/envs/env_0/object")
        # xformable = UsdGeom.Xformable(cube)
        # transform_matrix = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        # world_translation = transform_matrix.ExtractTranslation()
        # world_rotation = transform_matrix.ExtractRotationQuat()
        s="["
        for i in wrist_pos[0,0]:
            s+=str(i)+","
        for i in wrist_quat[0,0]:
            s+=str(i)+","
        s+="]"
        print(f"wrist_data={s}")

        #print(f"cube_data={list2str(obj_pos_quat)}")

        #print(f"force={list2str(force)}")
        #print(f"force_sum={list2str(force_sum)}")

        print(f"cube_pos={list2str(cube_pos+cube_quat)}")
        print(f"cube_force={list2str(cube_force)}")

        print(f"dt={SimulationContext.instance().get_physics_dt()}")

        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break
        #print(force)
        if spheres is None:
            spheres = []
            # create spheres:

            for si, pos in enumerate(sph_list[0]):
                if force[si]>0.0:
                    sp = sphere.VisualSphere(
                        prim_path="/World/robot_sphere_" + str(si),
                        position=np.ravel(pos),
                        radius=float(0.01),
                        color=np.array([0, 0.8, 0.2]),
                    )
                else:
                    sp = sphere.VisualSphere(
                        prim_path="/World/robot_sphere_" + str(si),
                        position=np.ravel(pos),
                        radius=float(0.01),
                        color=np.array([0.8, 0.2, 0.0]),
                    )
                spheres.append(sp)
        else:
            for si, pos in enumerate(sph_list[0]):
                if si is not 0:
                    shader_path = "/World/Looks/visual_material_" + str(si)+"/shader"
                else:
                    shader_path = "/World/Looks/visual_material/shader"
                shader = UsdShade.Shader.Get(env.env.scene.stage,shader_path)
                if force[si]>0.0:
                    shader.GetInput("diffuseColor").Set(Gf.Vec3f(1.0, 0.0, 0.0))
                else:
                    shader.GetInput("diffuseColor").Set(Gf.Vec3f(0.2, 0.2, 0.2))        
                #env.env.scene.stage.RemovePrim(shader_path)

                spheres[si].set_world_pose(position=np.ravel(pos))
        



    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
