import numpy as np
import pytest

from scar.core import (
    CameraModel,
    Extrinsics,
    Frame,
    Landmarks,
    Observations,
    PosePrior,
    Sequence,
)
from scar.core.se3 import make_transform, so3_exp
from scar.evaluate import estimate_pose_offset, reprojection_errors
from scar.geodesy import HeightReference, WorldFrame

CAMERA = CameraModel(width=640, height=480, fx=520.0, fy=520.0, cx=320.0, cy=240.0, distortion=(-0.05, 0.01, 0.0, 0.0))
T_BC = make_transform(np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]), [0.02, 0.01, 0.15])


def circle_flight(delta=(0.0, 0.0, 0.0), dt=0.0, omega=(0.0, 0.0, 0.0), seed=0):
    """Frames on a circle (all headings) 400 m above flat ground; priors carry the given errors."""
    rng = np.random.default_rng(seed)
    radius, speed, n = 400.0, 30.0, 120
    times = np.arange(n) * 0.5
    angles = speed * times / radius
    positions = np.column_stack([radius * np.cos(angles), radius * np.sin(angles), np.full(n, 400.0)])
    velocities = speed * np.column_stack([-np.sin(angles), np.cos(angles), np.zeros(n)])
    # body x along the velocity, z down (NED-like body in an ENU world)
    R_true = []
    for v in velocities:
        x = v / np.linalg.norm(v)
        z = np.array([0.0, 0.0, -1.0])
        R_true.append(np.column_stack([x, np.cross(z, x), z]))
    T_true = make_transform(np.asarray(R_true), positions)

    landmarks = np.column_stack([rng.uniform(-700, 700, 3000), rng.uniform(-700, 700, 3000), rng.normal(0, 2, 3000)])
    T_bc_true = T_BC @ make_transform(so3_exp(np.radians(np.asarray(omega, dtype=float))), np.zeros(3))
    frame_ids, landmark_ids, uv = [], [], []
    for i, T in enumerate(T_true):
        pixels = CAMERA.project(landmarks, T @ T_bc_true)
        depth = ((landmarks - (T @ T_bc_true)[:3, 3]) @ (T @ T_bc_true)[:3, :3])[:, 2]
        visible = (depth > 0) & np.all((pixels > 0) & (pixels < [640, 480]), axis=1)
        frame_ids += [i] * int(visible.sum())
        landmark_ids += np.flatnonzero(visible).tolist()
        uv.append(pixels[visible] + rng.normal(0, 0.3, (int(visible.sum()), 2)))

    # priors: true pose shifted by -delta and sampled dt too late (camera sees the pose at t + dt)
    frames = tuple(Frame(frame_id=i, timestamp_ns=int(t * 1e9), image_path="",
        pose_prior=PosePrior(make_transform(R, p - np.asarray(delta) - v * dt)))
        for i, (t, R, p, v) in enumerate(zip(times, R_true, positions, velocities, strict=True)))
    return Sequence(camera=CAMERA, extrinsics=Extrinsics(T_BC), frames=frames,
        world_frame=WorldFrame(crs="EPSG:25832", height=HeightReference.ellipsoidal()),
        landmarks=Landmarks(np.arange(len(landmarks)), landmarks),
        observations=Observations(frame_ids, landmark_ids, np.vstack(uv)))


def test_reprojection_errors_of_a_consistent_sequence_are_the_noise():
    errors = reprojection_errors(circle_flight(), CAMERA, T_BC)
    assert np.median(errors) == pytest.approx(0.3 * 1.1774, rel=0.1)  # median of a 2D Gaussian norm


def test_offset_estimation_separates_datum_time_and_tilt():
    delta, dt, omega = np.array([0.6, -0.4, 0.3]), -0.008, np.array([0.08, -0.05, 0.1])
    sequence = circle_flight(delta, dt, omega)

    offset = estimate_pose_offset(sequence, CAMERA, T_BC, estimate_rotation=True)
    np.testing.assert_allclose(offset.delta, delta, atol=0.03)
    assert offset.dt == pytest.approx(dt, abs=0.002)
    np.testing.assert_allclose(offset.omega_deg, omega, atol=0.005)
    assert offset.after.median < 0.5 < offset.before.median


def test_localization_recovers_poses_and_reports_extrinsic_errors():
    from scar.evaluate import LocalizationConfig, localize

    sequence = circle_flight(omega=(0.0, 0.0, 0.0))
    config = LocalizationConfig(min_points=30, min_inliers=20)
    result = localize(sequence, CAMERA, T_BC, config)
    assert len(result.frame_ids) == len(sequence.frames)
    assert np.median(result.rotation_error_deg) < 0.02
    assert np.median(result.translation_error_m) < 0.2
    assert result.accuracy(0.5, 2.0) == 1.0

    # a wrong extrinsic rotation shows up as rotation error of the localized poses
    tilted = T_BC @ make_transform(so3_exp(np.radians([0.0, 0.0, 1.0])), np.zeros(3))
    assert np.median(localize(sequence, CAMERA, tilted, config).rotation_error_deg) == pytest.approx(1.0, abs=0.05)
