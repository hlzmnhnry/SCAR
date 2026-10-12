import numpy as np
import pytest

from scar.core.se3 import (
    euler_zyx_rotation_covariance,
    euler_zyx_to_matrix,
    invert,
    is_rigid_transform,
    make_transform,
    rot_z,
    rotate_pose_covariance,
    rotation_angle,
    so3_exp,
    so3_log,
)


def random_transforms(rng, n):
    return make_transform(so3_exp(rng.normal(size=(n, 3))), rng.normal(scale=100.0, size=(n, 3)))


def test_invert_composes_to_identity():
    T = random_transforms(np.random.default_rng(0), 5)
    np.testing.assert_allclose(T @ invert(T), np.broadcast_to(np.eye(4), T.shape), atol=1e-12)
    assert is_rigid_transform(T)


def test_so3_exp_log_roundtrip():
    rotvec = np.random.default_rng(1).normal(scale=0.5, size=(10, 3))
    np.testing.assert_allclose(so3_log(so3_exp(rotvec)), rotvec, atol=1e-12)


def test_rot_z_is_counter_clockwise():
    np.testing.assert_allclose(rot_z(np.pi / 2) @ [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], atol=1e-15)


def test_rotation_angle():
    assert rotation_angle(np.eye(3), rot_z(0.3)) == pytest.approx(0.3)


@pytest.mark.parametrize("angles", [(0.0, 0.0, 0.0), (0.1, -0.2, 2.5), (-0.3, 0.4, -1.0)])
def test_euler_covariance_matches_numerical_jacobian(angles):
    roll, pitch, yaw = angles
    R = euler_zyx_to_matrix(roll, pitch, yaw)

    eps = 1e-7
    J = np.zeros((3, 3))
    for k in range(3):
        perturbed = np.array(angles)
        perturbed[k] += eps
        J[:, k] = so3_log(euler_zyx_to_matrix(*perturbed) @ R.T) / eps

    stds = np.array([0.01, 0.02, 0.03])
    expected = J @ np.diag(stds**2) @ J.T
    np.testing.assert_allclose(euler_zyx_rotation_covariance(roll, pitch, yaw, *stds), expected, atol=1e-10)


def test_rotate_pose_covariance_swaps_axes():
    covariance = np.diag([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    R_enu_ned = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
    rotated = rotate_pose_covariance(covariance, R_enu_ned)
    np.testing.assert_allclose(np.diag(rotated), [2.0, 1.0, 3.0, 5.0, 4.0, 6.0])
