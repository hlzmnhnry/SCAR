import gtsam
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
from scar.geodesy import HeightReference, WorldFrame
from scar.optimize import CalibrationProblem, PosePriorSpec, estimate_extrinsics
from scar.optimize.extrinsics import se3_exp, se3_log
from scar.optimize.graph import camera_prior_covariance_local


def filter_points_with_few_observations(pixel_coords, world_points, minimum_observations):
    """Copy of scar/sequence.py at tag v1.0.0-cvprw2026."""
    filtered_pixel_coords, filtered_world_points = {}, {}
    for point_id, observations in pixel_coords.items():
        if len(observations) >= minimum_observations and point_id in world_points:
            filtered_pixel_coords[point_id] = observations
            filtered_world_points[point_id] = world_points[point_id]
    return filtered_pixel_coords, filtered_world_points


def filter_frames_with_few_features(pixel_coords, world_points, minimum_features_per_frame):
    """Copy of scar/sequence.py at tag v1.0.0-cvprw2026."""
    frame_counts = {}
    for observations in pixel_coords.values():
        for appearance in observations:
            frame_counts[appearance[0]] = frame_counts.get(appearance[0], 0) + 1
    valid_frames = {frame for frame, count in frame_counts.items() if count >= minimum_features_per_frame}
    filtered_pixel_coords, filtered_world_points = {}, {}
    for point_id, observations in pixel_coords.items():
        kept = [obs for obs in observations if obs[0] in valid_frames]
        if kept:
            filtered_pixel_coords[point_id] = kept
            filtered_world_points[point_id] = world_points[point_id]
    return filtered_pixel_coords, filtered_world_points, sorted(set(frame_counts) - valid_frames)


CAMERA = CameraModel(width=100, height=100, fx=100.0, fy=100.0, cx=50.0, cy=50.0)


def random_sequence(rng, num_frames=12, num_landmarks=40, num_observations=300):
    frames = tuple(Frame(frame_id=10 + i, timestamp_ns=1000 * i, image_path=f"{i}.png",
        pose_prior=PosePrior(np.eye(4))) for i in range(num_frames))
    landmark_ids = np.arange(num_landmarks) * 3 + 7
    frame_ids = rng.integers(10, 10 + num_frames, num_observations)
    observed = rng.choice(landmark_ids, num_observations)
    return Sequence(camera=CAMERA, extrinsics=Extrinsics(np.eye(4)), frames=frames,
        world_frame=WorldFrame(crs="EPSG:25832", height=HeightReference.ellipsoidal()),
        landmarks=Landmarks(landmark_ids, rng.normal(size=(num_landmarks, 3))),
        observations=Observations(frame_ids, observed, rng.normal(size=(num_observations, 2))))


@pytest.mark.parametrize("seed", range(5))
def test_problem_filtering_matches_legacy(seed):
    rng = np.random.default_rng(seed)
    sequence = random_sequence(rng)
    problem = CalibrationProblem.from_sequence(sequence, min_point_observations=7, min_frame_features=20)

    # the same data as legacy dicts: point id -> [(frame key, uv)], keys in id order
    tracks, points = {}, {}
    order = np.argsort(sequence.observations.landmark_ids, kind="stable")
    for index in order:
        landmark = str(sequence.observations.landmark_ids[index])
        tracks.setdefault(landmark, []).append((int(sequence.observations.frame_ids[index]),
            sequence.observations.uv[index].tolist()))
    for landmark_id, xyz in zip(sequence.landmarks.ids, sequence.landmarks.xyz, strict=True):
        points[str(landmark_id)] = xyz.tolist()
    tracks = dict(sorted(tracks.items(), key=lambda item: int(item[0])))

    tracks, points = filter_points_with_few_observations(tracks, points, 7)
    tracks, points, _ = filter_frames_with_few_features(tracks, points, 20)

    # the case must exercise both filters and keep something
    assert 0 < len(tracks) < sequence.landmarks.ids.size
    assert 0 < problem.num_frames < len(sequence.frames)
    assert problem.landmark_ids.tolist() == [int(key) for key in tracks]
    np.testing.assert_array_equal(problem.landmark_xyz, np.reshape([points[key] for key in tracks], (-1, 3)))
    expected = [(frame, uv) for track in tracks.values() for frame, uv in track]
    got = [(int(problem.frame_ids[f]), uv.tolist()) for f, uv in zip(problem.observation_frame,
        problem.observation_uv, strict=True)]
    assert got == expected
    assert problem.frame_ids.tolist() == sorted({frame for frame, _ in expected})


def test_camera_prior_covariance_matches_numerical_propagation():
    rng = np.random.default_rng(3)
    T_world_body = make_transform(so3_exp(rng.normal(size=3)), rng.normal(size=3) * 100)
    T_body_camera = make_transform(so3_exp(rng.normal(size=3)), [0.3, -0.2, 0.5])
    T_world_camera = T_world_body @ T_body_camera
    navigation = np.diag(rng.uniform(1e-4, 1e-2, 6))

    # d(local camera coordinates) / d(world-axes body perturbation) by finite differences
    eps = 1e-7
    J = np.zeros((6, 6))
    for k in range(6):
        delta = np.zeros(6)
        delta[k] = eps
        body = make_transform(so3_exp(delta[:3]) @ T_world_body[:3, :3], T_world_body[:3, 3] + delta[3:])
        J[:, k] = gtsam.Pose3.Logmap(gtsam.Pose3(T_world_camera).between(gtsam.Pose3(body @ T_body_camera))) / eps

    spec = PosePriorSpec(rotation_sigma_deg=(0.0,) * 3, position_sigma_m=(0.0,) * 3, add_navigation_covariance=True)
    covariance = camera_prior_covariance_local(T_world_camera, T_world_body, spec, navigation)
    np.testing.assert_allclose(covariance, J @ navigation @ J.T, rtol=1e-5, atol=1e-10)


def test_world_axes_sigmas_are_rotated_into_camera_axes():
    T_world_camera = make_transform(so3_exp(np.array([np.pi, 0.0, 0.0])), np.zeros(3))  # looking down
    spec = PosePriorSpec(rotation_sigma_deg=(1.0, 2.0, 3.0), position_sigma_m=(0.1, 0.2, 0.3), axes="world")
    covariance = camera_prior_covariance_local(T_world_camera, T_world_camera, spec, None)
    # a 180 deg rotation about x keeps the axes' variances
    np.testing.assert_allclose(np.sqrt(np.diag(covariance))[3:], [0.1, 0.2, 0.3])


def test_se3_exp_log_roundtrip():
    xi = np.array([0.3, -0.2, 0.1, 0.05, -0.02, 0.1])
    np.testing.assert_allclose(se3_log(se3_exp(xi)), xi, atol=1e-12)


def test_extrinsics_recovered_from_consistent_poses():
    rng = np.random.default_rng(0)
    T_true = make_transform(so3_exp(np.array([0.01, -0.02, 1.57])), [0.02, 0.01, 0.18])
    bodies = np.asarray([make_transform(so3_exp(rng.normal(size=3)), rng.normal(size=3) * 50) for _ in range(20)])
    cameras = bodies @ T_true
    guess = T_true @ make_transform(so3_exp(np.radians([2.0, -1.0, 1.5])), [0.05, 0.0, -0.03])

    estimate = estimate_extrinsics(bodies, cameras, guess)
    assert estimate.success
    np.testing.assert_allclose(estimate.T_body_camera, T_true, atol=1e-8)
