import numpy as np
import pytest

from scar.core.se3 import R_ENU_NED, make_transform, rot_z, rotation_angle
from scar.geodesy import HeightReference, WorldFrame
from scar.sources.poses import (
    PoseTable,
    image_pose_timestamps,
    ins_accuracy_covariance,
    to_nanoseconds,
)


@pytest.fixture
def table():
    timestamps = np.array([0, 10, 20], dtype=np.int64)
    rotations = rot_z(np.array([0.0, 0.2, 0.4]))
    positions = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 1.0], [20.0, 0.0, 2.0]])
    covariance = np.stack([np.eye(6) * (i + 1) for i in range(3)])
    return PoseTable(timestamps, make_transform(rotations, positions), covariance)


def test_exact_timestamp_returns_sample(table):
    np.testing.assert_array_equal(table.prior_at(10).T_world_body, table.T_world_body[1])


def test_linear_interpolation(table):
    prior = table.prior_at(13)
    np.testing.assert_allclose(prior.T_world_body[:3, 3], [13.0, 0.0, 1.3])
    assert rotation_angle(prior.T_world_body[:3, :3], rot_z(0.26)) == pytest.approx(0.0, abs=1e-12)
    np.testing.assert_array_equal(prior.covariance, table.covariance[1])  # nearer sample


def test_nearest_mode_prefers_earlier_sample_on_ties(table):
    np.testing.assert_array_equal(table.prior_at(15, mode="nearest").T_world_body, table.T_world_body[1])
    np.testing.assert_array_equal(table.prior_at(16, mode="nearest").T_world_body, table.T_world_body[2])


def test_out_of_range_and_gaps_are_rejected(table):
    with pytest.raises(ValueError):
        table.prior_at(25)
    with pytest.raises(ValueError):
        table.prior_at(25, max_gap_ns=4)
    np.testing.assert_array_equal(table.prior_at(25, max_gap_ns=5).T_world_body, table.T_world_body[2])
    with pytest.raises(ValueError):
        table.prior_at(13, max_gap_ns=5)
    with pytest.raises(ValueError):
        table.prior_at(40, mode="nearest", max_gap_ns=5)


def test_table_validation():
    with pytest.raises(ValueError):
        PoseTable(np.array([0, 0]), np.stack([np.eye(4)] * 2))


def test_to_nanoseconds():
    np.testing.assert_array_equal(to_nanoseconds(np.array([1666183007634260]), "us"), [1666183007634260000])
    np.testing.assert_array_equal(to_nanoseconds(np.array(["1776855480.070", "12.5"]), "s"), [1776855480070000000, 12500000000])
    # float seconds are only as precise as float64 allows
    assert abs(to_nanoseconds(np.array([1776855480.07]), "s")[0] - 1776855480070000000) < 1000
    with pytest.raises(ValueError):
        to_nanoseconds([1], "min")


def test_image_pose_timestamps():
    t = image_pose_timestamps([1000, 2000], time_offset_ns=5, exposure_ns=[100, 300], exposure_mid=True)
    np.testing.assert_array_equal(t, [1055, 2155])


def test_from_geodetic_composes_navigation_rotation():
    frame = WorldFrame(crs="EPSG:25832", height=HeightReference.ellipsoidal(), apply_meridian_convergence=False)
    quat_identity = np.array([[0.0, 0.0, 0.0, 1.0]])
    table = PoseTable.from_geodetic([0], [50.5], [7.2], [300.0], quat_identity, world=frame,
        height_reference=HeightReference.ellipsoidal(), covariance_nav=np.diag([1, 2, 3, 4, 5, 6.0])[None])

    np.testing.assert_allclose(table.T_world_body[0, :3, :3], R_ENU_NED)
    # NED covariance -> ENU axes: north/east swapped
    np.testing.assert_allclose(np.diag(table.covariance[0]), [2, 1, 3, 5, 4, 6])


def test_ins_accuracy_covariance_for_level_attitude():
    covariance = ins_accuracy_covariance(0.0, 0.0, 0.0, 0.01, 0.02, 0.03, 0.1, 0.2, 0.3)
    np.testing.assert_allclose(np.diag(covariance), [1e-4, 4e-4, 9e-4, 0.01, 0.04, 0.09])
