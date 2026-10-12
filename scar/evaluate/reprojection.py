"""Reprojection of fixed anchors: calibration quality on independent (test) data."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scar.core.camera import CameraModel
from scar.core.types import Sequence


@dataclass(frozen=True)
class ReprojectionStatistics:
    median: float
    mad: float
    mean: float
    count: int

    @classmethod
    def of(cls, errors: np.ndarray) -> ReprojectionStatistics:
        errors = np.asarray(errors, dtype=float)
        median = float(np.median(errors))
        return cls(median=median, mad=float(np.median(np.abs(errors - median))), mean=float(np.mean(errors)),
            count=len(errors))

    def __str__(self) -> str:
        return f"{self.median:.3f} ± {self.mad:.3f} px (mean {self.mean:.3f}, n = {self.count})"


def observation_arrays(sequence: Sequence) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per observation: frame index (into ``sequence.frames``), anchor position, pixel; plus body poses."""
    frame_index = {frame.frame_id: i for i, frame in enumerate(sequence.frames)}
    landmark_index = {int(i): k for k, i in enumerate(sequence.landmarks.ids)}
    frames = np.asarray([frame_index[int(i)] for i in sequence.observations.frame_ids], dtype=np.int64)
    points = sequence.landmarks.xyz[[landmark_index[int(i)] for i in sequence.observations.landmark_ids]]
    body_poses = np.asarray([frame.pose_prior.T_world_body for frame in sequence.frames])
    return frames, points, sequence.observations.uv, body_poses


def project_with_poses(camera: CameraModel, T_world_camera: np.ndarray, frames: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Project points (N, 3) into the cameras ``T_world_camera[frames]`` (vectorized)."""
    R = T_world_camera[frames, :3, :3]
    t = T_world_camera[frames, :3, 3]
    points_camera = np.einsum("nji,nj->ni", R, points - t)
    return camera.project_camera_points(points_camera)


def reprojection_errors(sequence: Sequence, camera: CameraModel, T_body_camera: np.ndarray) -> np.ndarray:
    """Errors [px] of all observations, using the sequence's pose priors and the given calibration."""
    frames, points, uv, body_poses = observation_arrays(sequence)
    projected = project_with_poses(camera, body_poses @ T_body_camera, frames, points)
    return np.linalg.norm(projected - uv, axis=1)
