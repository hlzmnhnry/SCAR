"""Systematic offsets between navigation poses and the reference data.

With a fixed calibration, estimate a constant world-frame translation ``delta`` of all camera
positions and a time offset ``dt`` (poses taken at ``t + dt``, to first order ``p + v dt``) that
minimize the robust reprojection error of anchors. A datum mismatch (e.g. ITRF vs. ETRS89 positions)
shows up in ``delta``; a synchronization error along the flight direction in ``dt``.

For near-nadir views a small tilt of the camera is almost indistinguishable from a horizontal
translation (0.1 deg at 500 m ~ 0.9 m). With ``estimate_rotation`` a camera-frame rotation
correction ``omega`` is estimated as well: it turns with the heading while a datum offset stays
constant in the world frame, so both are separable if the sequence has different headings.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from scar.core.camera import CameraModel
from scar.core.types import Sequence

from .reprojection import ReprojectionStatistics, observation_arrays, project_with_poses


@dataclass(frozen=True)
class PoseOffset:
    delta: np.ndarray  # (3,) world frame [m]
    dt: float  # [s]
    before: ReprojectionStatistics
    after: ReprojectionStatistics
    delta_std: np.ndarray  # (3,) approximate standard deviations
    dt_std: float
    omega_deg: np.ndarray | None = None  # (3,) camera-frame rotation correction
    omega_std_deg: np.ndarray | None = None


def _velocities(sequence: Sequence) -> np.ndarray:
    times = np.asarray([frame.timestamp_ns for frame in sequence.frames], dtype=float) * 1e-9
    positions = np.asarray([frame.pose_prior.T_world_body[:3, 3] for frame in sequence.frames])
    return np.gradient(positions, times, axis=0)


def estimate_pose_offset(sequence: Sequence, camera: CameraModel, T_body_camera: np.ndarray,
    estimate_time_offset: bool = True, estimate_rotation: bool = False, huber_px: float = 3.0) -> PoseOffset:
    from scar.core.se3 import so3_exp

    frames, points, uv, body_poses = observation_arrays(sequence)
    velocities = _velocities(sequence)
    active = [True, True, True, estimate_time_offset, *([estimate_rotation] * 3)]

    def full(x_active: np.ndarray) -> np.ndarray:
        x = np.zeros(7)
        x[np.flatnonzero(active)] = x_active
        return x

    def cameras(x: np.ndarray) -> np.ndarray:
        T_bc = T_body_camera.copy()
        T_bc[:3, :3] = T_bc[:3, :3] @ so3_exp(x[4:7])
        T = body_poses @ T_bc
        T[:, :3, 3] += x[:3] + velocities * x[3]
        return T

    def residuals(x_active: np.ndarray) -> np.ndarray:
        return (project_with_poses(camera, cameras(full(x_active)), frames, points) - uv).ravel()

    x0 = np.zeros(sum(active))
    before = ReprojectionStatistics.of(np.linalg.norm(residuals(x0).reshape(-1, 2), axis=1))
    scales = np.array([1.0, 1.0, 1.0, 0.01, 1e-3, 1e-3, 1e-3])[np.flatnonzero(active)]
    result = least_squares(residuals, x0, loss="huber", f_scale=huber_px, x_scale=scales, diff_step=1e-6)
    after = ReprojectionStatistics.of(np.linalg.norm(result.fun.reshape(-1, 2), axis=1))

    # Gauss-Newton covariance with the robust residual scale (approximate)
    scale = 1.4826 * np.median(np.abs(result.fun))
    std = full(np.sqrt(np.diag(np.linalg.pinv(result.jac.T @ result.jac))) * scale)
    x = full(result.x)
    return PoseOffset(delta=x[:3], dt=float(x[3]), before=before, after=after, delta_std=std[:3], dt_std=float(std[3]),
        omega_deg=np.degrees(x[4:7]) if estimate_rotation else None,
        omega_std_deg=np.degrees(std[4:7]) if estimate_rotation else None)
