"""Rigid-body transform utilities.

Conventions used throughout SCAR:

- ``T_a_b`` maps coordinates from frame ``b`` into frame ``a`` (``p_a = T_a_b @ p_b``).
  ``T_world_body`` is therefore the pose of the body in the world frame.
- Rotation uncertainty is expressed as a *left* (world-frame) perturbation,
  ``R = Exp(delta) @ R_hat``. A 6x6 pose covariance is ordered ``[rotation, translation]``
  with both blocks in world-frame axes.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

# rotation from a north-east-down to an east-north-up frame (R_enu_ned)
R_ENU_NED = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])


def make_transform(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Build (..., 4, 4) homogeneous transforms from rotations (..., 3, 3) and translations (..., 3)."""
    R = np.asarray(R, dtype=float)
    t = np.asarray(t, dtype=float)
    T = np.zeros((*R.shape[:-2], 4, 4))
    T[..., :3, :3] = R
    T[..., :3, 3] = t
    T[..., 3, 3] = 1.0
    return T


def invert(T: np.ndarray) -> np.ndarray:
    """Invert (..., 4, 4) rigid transforms."""
    T = np.asarray(T, dtype=float)
    R_inv = np.swapaxes(T[..., :3, :3], -1, -2)
    t_inv = -np.einsum("...ij,...j->...i", R_inv, T[..., :3, 3])
    return make_transform(R_inv, t_inv)


def is_rigid_transform(T: np.ndarray, atol: float = 1e-6) -> bool:
    T = np.asarray(T, dtype=float)
    if T.shape[-2:] != (4, 4):
        return False
    R = T[..., :3, :3]
    RtR = np.einsum("...ji,...jk->...ik", R, R)
    return (
        np.allclose(T[..., 3, :], [0.0, 0.0, 0.0, 1.0], atol=atol)
        and np.allclose(RtR, np.eye(3), atol=atol)
        and bool(np.all(np.linalg.det(R) > 0.0))
    )


def rot_z(angle: np.ndarray | float) -> np.ndarray:
    """Counter-clockwise rotation(s) about +z by ``angle`` [rad]; shape (..., 3, 3)."""
    angle = np.asarray(angle, dtype=float)
    c, s = np.cos(angle), np.sin(angle)
    R = np.zeros((*angle.shape, 3, 3))
    R[..., 0, 0], R[..., 0, 1] = c, -s
    R[..., 1, 0], R[..., 1, 1] = s, c
    R[..., 2, 2] = 1.0
    return R


def quat_xyzw_to_matrix(quat_xyzw: np.ndarray) -> np.ndarray:
    """Convert (..., 4) quaternions in (x, y, z, w) order to rotation matrices (normalizing them)."""
    quat = np.asarray(quat_xyzw, dtype=float)
    return Rotation.from_quat(quat.reshape(-1, 4)).as_matrix().reshape((*quat.shape[:-1], 3, 3))


def so3_exp(rotvec: np.ndarray) -> np.ndarray:
    rotvec = np.asarray(rotvec, dtype=float)
    return Rotation.from_rotvec(rotvec.reshape(-1, 3)).as_matrix().reshape((*rotvec.shape[:-1], 3, 3))


def so3_log(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=float)
    return Rotation.from_matrix(R.reshape(-1, 3, 3)).as_rotvec().reshape((*R.shape[:-2], 3))


def rotation_angle(R_a: np.ndarray, R_b: np.ndarray) -> np.ndarray:
    """Geodesic angle(s) [rad] between rotations."""
    R_rel = np.einsum("...ji,...jk->...ik", np.asarray(R_a, dtype=float), np.asarray(R_b, dtype=float))
    return np.linalg.norm(so3_log(R_rel), axis=-1)


def euler_zyx_to_matrix(roll: np.ndarray, pitch: np.ndarray, yaw: np.ndarray) -> np.ndarray:
    """R = Rz(yaw) @ Ry(pitch) @ Rx(roll) (aerospace convention, angles in rad)."""
    angles = np.stack(np.broadcast_arrays(yaw, pitch, roll), axis=-1).astype(float)
    return Rotation.from_euler("ZYX", angles.reshape(-1, 3)).as_matrix().reshape((*angles.shape[:-1], 3, 3))


def euler_zyx_rotation_covariance(roll: np.ndarray, pitch: np.ndarray, yaw: np.ndarray,
    std_roll: np.ndarray, std_pitch: np.ndarray, std_yaw: np.ndarray) -> np.ndarray:
    """Propagate independent roll/pitch/yaw standard deviations to a rotation covariance.

    For ``R = Rz(yaw) Ry(pitch) Rx(roll)`` a small change of the angles corresponds to the
    left perturbation ``delta = z * d_yaw + Rz(yaw) y * d_pitch + Rz(yaw) Ry(pitch) x * d_roll``.
    The result is expressed in the frame the Euler angles refer to (e.g. NED); shape (..., 3, 3).
    """
    roll, pitch, yaw = np.broadcast_arrays(*(np.asarray(v, dtype=float) for v in (roll, pitch, yaw)))
    Rz = rot_z(yaw)
    Ry = Rotation.from_euler("y", pitch.reshape(-1)).as_matrix().reshape((*pitch.shape, 3, 3))

    J = np.zeros((*roll.shape, 3, 3))
    J[..., :, 0] = (Rz @ Ry)[..., :, 0]
    J[..., :, 1] = Rz[..., :, 1]
    J[..., 2, 2] = 1.0

    stds = np.stack(np.broadcast_arrays(std_roll, std_pitch, std_yaw), axis=-1).astype(float)
    return np.einsum("...ij,...j,...kj->...ik", J, stds**2, J)


def rotate_pose_covariance(covariance: np.ndarray, R_new_old: np.ndarray) -> np.ndarray:
    """Express (..., 6, 6) [rotation, translation] covariances in a rotated frame."""
    R = np.asarray(R_new_old, dtype=float)
    A = np.zeros((*R.shape[:-2], 6, 6))
    A[..., :3, :3] = R
    A[..., 3:, 3:] = R
    return A @ np.asarray(covariance, dtype=float) @ np.swapaxes(A, -1, -2)
