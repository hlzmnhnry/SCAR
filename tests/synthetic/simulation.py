"""Synthetic nadir flights with known calibration, for testing the optimization without data."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scar.core import (
    CameraModel,
    Extrinsics,
    Frame,
    Landmarks,
    Observations,
    PosePrior,
    Sequence,
)
from scar.core.se3 import R_ENU_NED, euler_zyx_to_matrix, make_transform, so3_exp
from scar.geodesy import HeightReference, WorldFrame

TRUE_CAMERA = CameraModel(width=1600, height=1100, fx=1318.3, fy=1318.8, cx=801.1, cy=560.7,
    distortion=(-0.1446, 0.1249, 1.7e-4, 1.8e-5))
# camera looks down, image x to the right of the flight direction (like the LTC setup)
TRUE_T_BODY_CAMERA = make_transform(np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
    [0.016, 0.006, 0.181])
# rotation [rad] and position [m] standard deviation of the navigation poses
DEFAULT_NAVIGATION_SIGMA = (np.radians(0.05), 0.02)


@dataclass(frozen=True)
class SimulationTruth:
    camera: CameraModel
    T_body_camera: np.ndarray
    T_world_body: np.ndarray
    landmarks: np.ndarray


def terrain_height(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return 100.0 + 15.0 * np.sin(x / 60.0) * np.cos(y / 90.0)


def simulate_flight(seed: int, *, num_frames: int = 40, frame_spacing_m: float = 20.0, altitude_m: float = 600.0,
    num_landmarks: int = 1500, pixel_sigma: float = 0.5, anchor_sigma_m=(0.25, 0.25, 0.1),
    navigation_sigma=DEFAULT_NAVIGATION_SIGMA, camera: CameraModel = TRUE_CAMERA,
    T_body_camera: np.ndarray = TRUE_T_BODY_CAMERA, with_covariances: bool = False) -> tuple[Sequence, SimulationTruth]:
    """Straight flight northwards; returns a sequence with noisy priors (true intrinsics/extrinsics as guesses)."""
    rng = np.random.default_rng(seed)

    # true body poses: slight attitude variation, constant altitude above the mean terrain
    north = np.arange(num_frames) * frame_spacing_m
    roll, pitch = rng.normal(0.0, np.radians(1.0), (2, num_frames))
    yaw = rng.normal(0.0, np.radians(2.0), num_frames)
    R_world_body = R_ENU_NED @ euler_zyx_to_matrix(roll, pitch, yaw)
    T_world_body = make_transform(R_world_body, np.column_stack([np.zeros(num_frames), north,
        np.full(num_frames, 100.0 + altitude_m)]))

    ground_half_width = 0.6 * altitude_m
    landmarks_x = rng.uniform(-ground_half_width, ground_half_width, num_landmarks)
    landmarks_y = rng.uniform(-ground_half_width, north[-1] + ground_half_width, num_landmarks)
    landmarks = np.column_stack([landmarks_x, landmarks_y, terrain_height(landmarks_x, landmarks_y)])

    frame_ids, landmark_ids, uv = [], [], []
    for i in range(num_frames):
        T_world_camera = T_world_body[i] @ T_body_camera
        points_camera = (landmarks - T_world_camera[:3, 3]) @ T_world_camera[:3, :3]
        pixels = camera.project(landmarks, T_world_camera)
        visible = (points_camera[:, 2] > 0) & np.all((pixels >= 0) & (pixels < [camera.width, camera.height]), axis=1)
        frame_ids.extend([i] * int(visible.sum()))
        landmark_ids.extend(np.flatnonzero(visible).tolist())
        uv.extend(pixels[visible] + rng.normal(0.0, pixel_sigma, (int(visible.sum()), 2)))

    # observations grouped by landmark in time order, like collected tracks
    order = np.lexsort((np.asarray(frame_ids), np.asarray(landmark_ids)))
    observations = Observations(np.asarray(frame_ids)[order], np.asarray(landmark_ids)[order], np.asarray(uv)[order])

    anchor_sigma = np.asarray(anchor_sigma_m, dtype=float)
    anchors = landmarks + rng.normal(0.0, anchor_sigma, landmarks.shape)
    anchor_covariance = np.broadcast_to(np.diag(anchor_sigma**2), (num_landmarks, 3, 3)).copy()

    rotation_sigma, position_sigma = navigation_sigma
    nav_covariance = np.diag([rotation_sigma**2] * 3 + [position_sigma**2] * 3)
    frames = []
    for i in range(num_frames):
        R_noisy = so3_exp(rng.normal(0.0, rotation_sigma, 3)) @ T_world_body[i, :3, :3]
        t_noisy = T_world_body[i, :3, 3] + rng.normal(0.0, position_sigma, 3)
        prior = PosePrior(make_transform(R_noisy, t_noisy), nav_covariance if with_covariances else None)
        frames.append(Frame(frame_id=i, timestamp_ns=i * 200_000_000, image_path=f"{i}.png", pose_prior=prior))

    sequence = Sequence(
        camera=camera, extrinsics=Extrinsics(T_body_camera), frames=tuple(frames),
        world_frame=WorldFrame(crs="EPSG:25832", height=HeightReference.ellipsoidal()),
        landmarks=Landmarks(np.arange(num_landmarks), anchors, anchor_covariance if with_covariances else None),
        observations=observations,
    )
    return sequence, SimulationTruth(camera, T_body_camera, T_world_body, landmarks)
