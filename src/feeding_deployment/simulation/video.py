"""Make simulation videos."""

from __future__ import annotations

from pathlib import Path

import imageio.v2 as iio
from pybullet_helpers.camera import capture_superimposed_image

from feeding_deployment.simulation.simulator import (
    FeedingDeploymentPyBulletSimulator,
    FeedingDeploymentSimulatorState,
)


def make_simulation_video(
    sim: FeedingDeploymentPyBulletSimulator,
    traj: list[FeedingDeploymentSimulatorState],
    outfile: Path,
    fps: int = 20,
) -> None:
    """Make a video for a simulated drink manipulation plan."""
    imgs = []
    for state in traj:
        sim.sync(state)
        img = capture_superimposed_image(
            sim.physics_client_id, **sim.scene_description.camera_kwargs
        )
        imgs.append(img)
    iio.mimsave(outfile, imgs, fps=fps)  # type: ignore
    print(f"Wrote out to {outfile}")
