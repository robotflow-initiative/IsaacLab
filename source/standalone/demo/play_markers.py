# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates how to create different types of markers in Orbit."""

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""


import argparse

from omni.isaac.orbit.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="This script demonstrates how to create different types of markers in Orbit."
)
parser.add_argument("--headless", action="store_true", default=False, help="Force display off at all times.")
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(headless=args_cli.headless)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
import traceback

import carb

import omni.isaac.orbit.sim as sim_utils
from omni.isaac.orbit.markers import VisualizationMarkers, VisualizationMarkersCfg
from omni.isaac.orbit.sim import SimulationContext
from omni.isaac.orbit.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.orbit.utils.math import quat_from_angle_axis


def main():
    """Spawns lights in the stage and sets the camera view."""

    # Load kit helper
    sim = SimulationContext(sim_utils.SimulationCfg(dt=0.01, substeps=1))
    # Set main camera
    sim.set_camera_view([0.0, 17.0, 12.0], [0.0, 2.0, 0.0])

    # Spawn things into stage
    # Lights-1
    cfg = sim_utils.SphereLightCfg(intensity=600.0, color=(0.75, 0.75, 0.75), radius=2.5)
    cfg.func("/World/Light/greyLight", cfg, translation=(4.5, 3.5, 10.0))
    # Lights-2
    cfg = sim_utils.SphereLightCfg(intensity=600.0, color=(1.0, 1.0, 1.0), radius=2.5)
    cfg.func("/World/Light/whiteSphere", cfg, translation=(-4.5, 3.5, 10.0))

    # Create markers with various different shapes
    marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/myMarkers",
        markers={
            "frame": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
                scale=(0.5, 0.5, 0.5),
            ),
            "arrow_x": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                scale=(1.0, 0.5, 0.5),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 1.0)),
            ),
            "cube": sim_utils.CuboidCfg(
                size=(1.0, 1.0, 1.0),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            ),
            "sphere": sim_utils.SphereCfg(
                radius=0.5,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
            ),
            "cylinder": sim_utils.CylinderCfg(
                radius=0.5,
                height=1.0,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
            ),
            "cone": sim_utils.ConeCfg(
                radius=0.5,
                height=1.0,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 0.0)),
            ),
            "mesh": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=(10.0, 10.0, 10.0),
            ),
            "mesh_recolored": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=(10.0, 10.0, 10.0),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.25, 0.0)),
            ),
        },
    )
    my_visualizer = VisualizationMarkers(marker_cfg)

    # marker locations
    num_marker_types = len(marker_cfg.markers)
    num_markers_per_type = 5
    grid_spacing = 2.0

    # Calculate the half-width and half-height
    half_width = (num_markers_per_type - 1) / 2.0
    half_height = (num_marker_types - 1) / 2.0
    # Create the x and y ranges centered around the origin
    x_range = torch.arange(-half_width * grid_spacing, (half_width + 1) * grid_spacing, grid_spacing)
    y_range = torch.arange(-half_height * grid_spacing, (half_height + 1) * grid_spacing, grid_spacing)
    # Create the grid
    x_grid, y_grid = torch.meshgrid(x_range, y_range, indexing="ij")
    x_grid = x_grid.reshape(-1)
    y_grid = y_grid.reshape(-1)
    z_grid = torch.zeros_like(x_grid)
    # marker locations
    marker_locations = torch.stack([x_grid, y_grid, z_grid], dim=1)
    marker_indices = torch.arange(num_marker_types).repeat(num_markers_per_type)

    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")

    # Yaw angle
    yaw = torch.zeros_like(marker_locations[:, 0])
    # Simulate physics
    while simulation_app.is_running():
        # rotate the markers around the z-axis for visualization
        marker_orientations = quat_from_angle_axis(yaw, torch.tensor([0.0, 0.0, 1.0]))
        # visualize
        my_visualizer.visualize(marker_locations, marker_orientations, marker_indices=marker_indices)
        # perform step
        sim.step()
        # increment yaw
        yaw += 0.01


if __name__ == "__main__":
    try:
        # run the main execution
        main()
    except Exception as err:
        carb.log_error(err)
        carb.log_error(traceback.format_exc())
        raise
    finally:
        # close sim app
        simulation_app.close()