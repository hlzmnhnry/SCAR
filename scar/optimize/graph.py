"""GTSAM factor graph of one optimization stage."""

from __future__ import annotations

from dataclasses import dataclass, field

import gtsam
import numpy as np

from scar.core.camera import CameraModel

from .problem import CalibrationProblem, CalibrationState
from .stages import CALIBRATION_PARAMETERS, PosePriorSpec, RobustKernel, StageSpec

KEY_CALIBRATION = gtsam.symbol("K", 0)
KEY_EXTRINSICS = gtsam.symbol("E", 0)


def camera_key(frame_index: int) -> int:
    return gtsam.symbol("C", int(frame_index))


def landmark_key(landmark_index: int) -> int:
    return gtsam.symbol("W", int(landmark_index))


def body_key(frame_index: int) -> int:
    return gtsam.symbol("B", int(frame_index))


def _chain_error(C: gtsam.Pose3, B: gtsam.Pose3, E: gtsam.Pose3) -> np.ndarray:
    return gtsam.Pose3.Logmap(C.between(B.compose(E)))


def chain_factor(noise, camera: int, body: int, extrinsics: int) -> gtsam.CustomFactor:
    """Soft constraint C = B E between camera pose, body pose and extrinsics (central-difference Jacobians)."""

    def error(this, values, jacobians):
        poses = [values.atPose3(camera), values.atPose3(body), values.atPose3(extrinsics)]
        e = _chain_error(*poses)
        if jacobians is not None:
            eps = 1e-6
            for slot in range(3):
                J = np.zeros((6, 6))
                for k in range(6):
                    step = np.zeros(6)
                    step[k] = eps
                    plus, minus = list(poses), list(poses)
                    plus[slot] = poses[slot].retract(step)
                    minus[slot] = poses[slot].retract(-step)
                    J[:, k] = (_chain_error(*plus) - _chain_error(*minus)) / (2 * eps)
                jacobians[slot] = J
        return e

    return gtsam.CustomFactor(noise, [camera, body, extrinsics], error)


def _skew(v: np.ndarray) -> np.ndarray:
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def camera_prior_covariance_local(T_world_camera: np.ndarray, T_world_body: np.ndarray,
    spec: PosePriorSpec, navigation_covariance: np.ndarray | None) -> np.ndarray:
    """6x6 covariance of a camera pose prior in GTSAM's local pose coordinates ``[rotation, translation]``.

    GTSAM perturbs poses as ``T = T_hat @ Exp(xi)``, i.e. in camera axes. World-axes covariances are
    rotated accordingly. A navigation covariance of the body pose (world axes) is propagated through
    the lever arm between body and camera origin.
    """
    R_c = T_world_camera[:3, :3]
    A = np.zeros((6, 6))
    A[:3, :3] = R_c.T
    A[3:, 3:] = R_c.T

    sigmas = np.concatenate([np.deg2rad(spec.rotation_sigma_deg), spec.position_sigma_m])
    covariance = np.diag(sigmas**2)
    if spec.axes == "world":
        covariance = A @ covariance @ A.T

    if spec.add_navigation_covariance:
        if navigation_covariance is None:
            raise ValueError("add_navigation_covariance needs pose covariances in the sequence")
        # camera origin p_c = p_b + R_b t_bc: a world rotation delta moves it by -[l]x delta
        lever = T_world_camera[:3, 3] - T_world_body[:3, 3]
        M = np.eye(6)
        M[3:, :3] = -_skew(lever)
        covariance = covariance + A @ M @ navigation_covariance @ M.T @ A.T
    return covariance


def _robust_noise(kernel: RobustKernel, base):
    if kernel.type is None:
        return base
    estimator = {
        "huber": gtsam.noiseModel.mEstimator.Huber,
        "cauchy": gtsam.noiseModel.mEstimator.Cauchy,
        "geman_mcclure": gtsam.noiseModel.mEstimator.GemanMcClure,
    }[kernel.type].Create(kernel.param)
    return gtsam.noiseModel.Robust.Create(estimator, base)


def _cal3ds2(camera: CameraModel) -> gtsam.Cal3DS2:
    if len(camera.distortion) == 5 and camera.distortion[4] != 0.0:
        raise NotImplementedError("Optimization supports four distortion coefficients (k1, k2, p1, p2)")
    k1, k2, p1, p2 = camera.dist[:4]
    return gtsam.Cal3DS2(fx=camera.fx, fy=camera.fy, s=0.0, u0=camera.cx, v0=camera.cy, k1=k1, k2=k2, p1=p1, p2=p2)


def build_stage_graph(problem: CalibrationProblem, state: CalibrationState,
    stage: StageSpec) -> tuple[gtsam.NonlinearFactorGraph, gtsam.Values]:
    """Factor graph and initial values of a stage; priors are centered at ``state``.

    As in the legacy pipeline, the intrinsics restart without skew in every stage.
    """
    graph = gtsam.NonlinearFactorGraph()

    if stage.joint_extrinsics:
        _add_joint_factors(graph, problem, state, stage)
    elif stage.pose_prior is not None:
        spec = stage.pose_prior
        shared_noise = None
        if spec.axes == "camera" and not spec.add_navigation_covariance:
            shared_noise = gtsam.noiseModel.Diagonal.Sigmas([*np.deg2rad(spec.rotation_sigma_deg), *spec.position_sigma_m])

        for i in range(problem.num_frames):
            noise = shared_noise
            if noise is None:
                navigation = None if problem.navigation_covariance is None else problem.navigation_covariance[i]
                noise = gtsam.noiseModel.Gaussian.Covariance(camera_prior_covariance_local(
                    state.T_world_camera[i], problem.T_world_body[i], spec, navigation))
            graph.add(gtsam.PriorFactorPose3(camera_key(i), gtsam.Pose3(state.T_world_camera[i]), noise))

    if stage.landmark_prior is not None:
        spec = stage.landmark_prior
        shared_noise = None if spec.use_landmark_covariance else gtsam.noiseModel.Diagonal.Sigmas([*spec.sigma_m])
        if spec.use_landmark_covariance and problem.landmark_covariance is None:
            raise ValueError("use_landmark_covariance needs landmark covariances in the sequence")

        centers = problem.landmark_xyz if spec.center == "anchors" else state.landmark_xyz
        for j in range(problem.num_landmarks):
            noise = shared_noise or gtsam.noiseModel.Gaussian.Covariance(problem.landmark_covariance[j])
            graph.add(gtsam.PriorFactorPoint3(landmark_key(j), gtsam.Point3(*centers[j]), noise))

    calibration = _cal3ds2(state.camera)
    if stage.calibration_prior is not None:
        sigmas = [stage.calibration_prior[name] for name in CALIBRATION_PARAMETERS]
        graph.add(gtsam.PriorFactorCal3DS2(KEY_CALIBRATION, calibration, gtsam.noiseModel.Diagonal.Sigmas(sigmas)))

    reprojection_noise = _robust_noise(stage.kernel, gtsam.noiseModel.Diagonal.Sigmas(list(stage.reprojection_sigma_px)))
    for frame_index, landmark_index, (u, v) in zip(problem.observation_frame, problem.observation_landmark,
        problem.observation_uv, strict=True):
        graph.add(gtsam.GeneralSFMFactor2Cal3DS2(gtsam.Point2(float(u), float(v)), reprojection_noise,
            camera_key(frame_index), landmark_key(landmark_index), KEY_CALIBRATION))

    values = gtsam.Values()
    values.insert(KEY_CALIBRATION, calibration)
    for i in range(problem.num_frames):
        values.insert(camera_key(i), gtsam.Pose3(state.T_world_camera[i]))
    for j in range(problem.num_landmarks):
        values.insert(landmark_key(j), gtsam.Point3(*state.landmark_xyz[j]))
    if stage.joint_extrinsics:
        values.insert(KEY_EXTRINSICS, gtsam.Pose3(state.T_body_camera))
        for i in range(problem.num_frames):
            values.insert(body_key(i), gtsam.Pose3(problem.T_world_body[i]))

    return graph, values


def _add_joint_factors(graph, problem: CalibrationProblem, state: CalibrationState, stage: StageSpec) -> None:
    """Navigation priors on body poses, a prior on the extrinsics and C_t = B_t E links."""
    if state.T_body_camera is None:
        raise ValueError(f"Joint stage '{stage.name}' needs initial extrinsics in the state")

    spec = stage.pose_prior
    for i in range(problem.num_frames):
        navigation = None if problem.navigation_covariance is None else problem.navigation_covariance[i]
        # body pose prior: lever arm zero, so the camera helper applies with T_world_body for both
        covariance = camera_prior_covariance_local(problem.T_world_body[i], problem.T_world_body[i], spec, navigation)
        graph.add(gtsam.PriorFactorPose3(body_key(i), gtsam.Pose3(problem.T_world_body[i]),
            gtsam.noiseModel.Gaussian.Covariance(covariance)))

    extrinsics = stage.extrinsics_prior
    sigmas = np.concatenate([np.deg2rad(extrinsics.rotation_sigma_deg), extrinsics.position_sigma_m])
    graph.add(gtsam.PriorFactorPose3(KEY_EXTRINSICS, gtsam.Pose3(state.T_body_camera),
        gtsam.noiseModel.Diagonal.Sigmas(sigmas)))

    link_noise = gtsam.noiseModel.Isotropic.Sigma(6, 1e-5)
    for i in range(problem.num_frames):
        graph.add(chain_factor(link_noise, camera_key(i), body_key(i), KEY_EXTRINSICS))


@dataclass
class SolverReport:
    errors: list[float] = field(default_factory=list)

    @property
    def initial_error(self) -> float:
        return self.errors[0] if self.errors else float("nan")

    @property
    def final_error(self) -> float:
        return self.errors[-1] if self.errors else float("nan")

    @property
    def iterations(self) -> int:
        return max(len(self.errors) - 1, 0)


def solve(graph: gtsam.NonlinearFactorGraph, values: gtsam.Values, solver: str = "dogleg") -> tuple[gtsam.Values, SolverReport]:
    report = SolverReport()

    def record(iteration, error_before, error_after):
        if iteration == 1:
            report.errors.append(error_before)
        report.errors.append(error_after)

    params, optimizer_type = {
        "dogleg": (gtsam.DoglegParams, gtsam.DoglegOptimizer),
        "lm": (gtsam.LevenbergMarquardtParams, gtsam.LevenbergMarquardtOptimizer),
        "gn": (gtsam.GaussNewtonParams, gtsam.GaussNewtonOptimizer),
    }[solver]
    params = params()
    params.iterationHook = record
    result = optimizer_type(graph, values, params).optimize()

    if not report.errors:
        report.errors = [graph.error(values), graph.error(result)]
    return result, report


def extract_state(problem: CalibrationProblem, result: gtsam.Values, template: CameraModel,
    joint: bool = False) -> CalibrationState:
    calibration = result.atCal3DS2(KEY_CALIBRATION)
    K, k = calibration.K(), calibration.k()
    camera = template.replace(fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2], distortion=tuple(float(c) for c in k[:4]))
    poses = np.asarray([result.atPose3(camera_key(i)).matrix() for i in range(problem.num_frames)]).reshape(-1, 4, 4)
    landmarks = np.asarray([result.atPoint3(landmark_key(j)) for j in range(problem.num_landmarks)]).reshape(-1, 3)
    T_body_camera = result.atPose3(KEY_EXTRINSICS).matrix() if joint else None
    return CalibrationState(camera=camera, T_world_camera=poses, landmark_xyz=landmarks, skew=float(K[0, 1]),
        T_body_camera=T_body_camera)


def calibration_covariance(graph: gtsam.NonlinearFactorGraph, result: gtsam.Values,
    joint: bool = False) -> tuple[np.ndarray, np.ndarray | None]:
    """Marginal covariances of the calibration.

    Intrinsics (9x9, :data:`CALIBRATION_PARAMETERS` order) and, for joint stages, extrinsics
    (6x6, [rotation, translation] in camera axes).
    """
    marginals = gtsam.Marginals(graph, result)
    extrinsics = marginals.marginalCovariance(KEY_EXTRINSICS) if joint else None
    return marginals.marginalCovariance(KEY_CALIBRATION), extrinsics


__all__ = [
    "KEY_CALIBRATION",
    "KEY_EXTRINSICS",
    "SolverReport",
    "body_key",
    "build_stage_graph",
    "calibration_covariance",
    "camera_key",
    "camera_prior_covariance_local",
    "chain_factor",
    "extract_state",
    "landmark_key",
    "solve",
]
