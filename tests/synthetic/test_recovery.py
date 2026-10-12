"""Calibration recovery on simulated flights with known ground truth (no data needed)."""

import numpy as np
import pytest

from scar.core import Extrinsics
from scar.core.se3 import make_transform, rotation_angle, so3_exp
from scar.optimize import (
    LandmarkPriorSpec,
    OptimizationPreset,
    PosePriorSpec,
    RobustKernel,
    StageSpec,
    load_preset,
    run_calibration,
)

from .simulation import simulate_flight

CAL3DS2_WITHOUT_SKEW = [0, 1, 3, 4, 5, 6, 7, 8]


def intrinsics_error(camera, truth_camera):
    return np.array([camera.fx - truth_camera.fx, camera.fy - truth_camera.fy, camera.cx - truth_camera.cx,
        camera.cy - truth_camera.cy, *np.subtract(camera.distortion, truth_camera.distortion)])


@pytest.mark.parametrize("seed", [0, 1])
def test_paper_preset_recovers_extrinsics(seed):
    sequence, truth = simulate_flight(seed)
    T_guess = truth.T_body_camera @ make_transform(so3_exp(np.radians([1.0, -0.8, 1.5])), np.zeros(3))
    result = run_calibration(sequence.replace(extrinsics=Extrinsics(T_guess)), load_preset("cvprw2026"))

    assert np.degrees(rotation_angle(T_guess[:3, :3], truth.T_body_camera[:3, :3])) > 1.9
    assert np.degrees(rotation_angle(result.T_body_camera[:3, :3], truth.T_body_camera[:3, :3])) < 0.05
    assert np.linalg.norm(result.T_body_camera[:3, 3] - truth.T_body_camera[:3, 3]) < 0.05
    assert np.all(np.abs(intrinsics_error(result.camera, truth.camera)[:4]) < 0.1)


def joint_preset():
    loose = {"fx": 1e3, "fy": 1e3, "skew": 1e-9, "cx": 1e3, "cy": 1e3, "k1": 1.0, "k2": 1.0, "p1": 1.0, "p2": 1.0}
    stage = StageSpec(
        name="joint", kernel=RobustKernel(None), reprojection_sigma_px=(0.5, 0.5),
        pose_prior=PosePriorSpec(rotation_sigma_deg=(1e-4,) * 3, position_sigma_m=(1e-4,) * 3, axes="world",
            add_navigation_covariance=True),
        landmark_prior=LandmarkPriorSpec(use_landmark_covariance=True),
        calibration_prior=loose, compute_calibration_covariance=True,
    )
    return OptimizationPreset(name="joint", stages=(stage,), extrinsics=None)


def test_joint_stage_estimates_intrinsics_with_consistent_covariance():
    """With the true noise models, errors must be consistent with the marginal covariance (NEES ~ chi2(8))."""
    nees = []
    for seed in range(20):
        sequence, truth = simulate_flight(seed, num_frames=25, num_landmarks=800, with_covariances=True)
        start = truth.camera.replace(fx=truth.camera.fx + 3, fy=truth.camera.fy - 2, cx=truth.camera.cx + 2,
            cy=truth.camera.cy - 2)
        result = run_calibration(sequence.replace(camera=start), joint_preset())

        error = intrinsics_error(result.camera, truth.camera)
        covariance = result.stages[-1].calibration_covariance[np.ix_(CAL3DS2_WITHOUT_SKEW, CAL3DS2_WITHOUT_SKEW)]
        assert np.all(np.abs(error) < 5 * np.sqrt(np.diag(covariance)))
        nees.append(error @ np.linalg.solve(covariance, error))

    # mean of 20 chi2(8) samples: 8 +- 0.9 (1 sigma)
    assert 5.5 < np.mean(nees) < 10.5


def test_results_are_written(tmp_path):
    from scar.io import load_camera, load_extrinsics
    from scar.io.results import save_result

    sequence, _ = simulate_flight(0, num_frames=10, num_landmarks=300)
    result = run_calibration(sequence, load_preset("cvprw2026"))
    save_result(result, tmp_path, preset_name="cvprw2026")

    assert load_camera(tmp_path / "calibration.yaml") == result.camera
    np.testing.assert_array_equal(load_extrinsics(tmp_path / "calibration.yaml").T_body_camera, result.T_body_camera)
    assert (tmp_path / "stages.json").exists()
    assert np.load(tmp_path / "estimates.npz")["T_world_camera"].shape == (result.problem.num_frames, 4, 4)


def test_joint_preset_recovers_intrinsics_and_extrinsics():
    """The paper schedule keeps intrinsic errors; the joint stage (extrinsics in the graph) removes them."""
    preset = load_preset("joint")
    assert [stage.name for stage in preset.stages][-1] == "joint"
    assert len(preset.stages) == len(load_preset("cvprw2026").stages) + 1

    sequence, truth = simulate_flight(0)
    start = truth.camera.replace(fx=truth.camera.fx + 3, fy=truth.camera.fy - 2, cx=truth.camera.cx + 2,
        cy=truth.camera.cy - 2)
    T_guess = truth.T_body_camera @ make_transform(so3_exp(np.radians([1.0, -0.8, 1.5])), np.zeros(3))
    result = run_calibration(sequence.replace(camera=start, extrinsics=Extrinsics(T_guess)), preset)

    error = intrinsics_error(result.camera, truth.camera)
    assert np.all(np.abs(error[:4]) < 0.6)
    rotation_error = np.degrees(rotation_angle(result.T_body_camera[:3, :3], truth.T_body_camera[:3, :3]))
    assert rotation_error < 0.05

    covariance = result.stages[-1].extrinsics_covariance
    assert covariance.shape == (6, 6)
    assert rotation_error < 4 * np.degrees(np.sqrt(np.trace(covariance[:3, :3])))
