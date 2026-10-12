"""Recovery of the camera-body extrinsics from optimized camera poses.

Solves ``min_T sum_t rho(|| Log(C_t^-1 @ B_t @ T) ||^2)`` for ``T = T_body_camera`` over body poses
``B_t`` from navigation and optimized camera poses ``C_t``, like the CVPRW 2026 pipeline. The SE(3)
parametrization ``xi = [v, w]`` and the update ``T = T0 @ exp(xi)`` are kept exactly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from .stages import ExtrinsicsSpec


def _invert(T: np.ndarray) -> np.ndarray:
    # same arithmetic as the legacy code, so residuals match bit for bit
    R, t = T[:3, :3], T[:3, 3].reshape(3, 1)
    return np.vstack([np.hstack([R.T, -R.T @ t]), np.array([0.0, 0.0, 0.0, 1.0])])


def _hat(w: np.ndarray) -> np.ndarray:
    return np.array([[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]], dtype=np.float64)


def _vee(W: np.ndarray) -> np.ndarray:
    return np.array([W[2, 1], W[0, 2], W[1, 0]], dtype=np.float64)


def se3_exp(xi: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Exponential map of ``xi = [v, w]`` (translation part first)."""
    v, w = xi[:3], xi[3:]
    theta = np.linalg.norm(w)
    W = _hat(w)
    W2 = W @ W

    if theta < eps:
        R = np.eye(3) + W + 0.5 * W2
        V = np.eye(3) + 0.5 * W + (1.0 / 6.0) * W2
    else:
        th2 = theta * theta
        A = np.sin(theta) / theta
        B = (1.0 - np.cos(theta)) / th2
        C = (theta - np.sin(theta)) / (theta * th2)
        R = np.eye(3) + A * W + B * W2
        V = np.eye(3) + B * W + C * W2

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = V @ v
    return T


def se3_log(T: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Logarithm map returning ``[v, w]``; the rotation is re-orthonormalized first."""
    U, _, Vt = np.linalg.svd(T[:3, :3])
    R = U @ Vt
    t = T[:3, 3]

    theta = np.arccos(np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0))
    if theta < eps:
        W = 0.5 * (R - R.T)
        V_inv = np.eye(3) - 0.5 * W + (1.0 / 12.0) * (W @ W)
    else:
        W = (theta / (2.0 * np.sin(theta))) * (R - R.T)
        th2 = theta * theta
        beta = (1.0 / th2) - (1.0 + np.cos(theta)) / (2.0 * theta * np.sin(theta))
        V_inv = np.eye(3) - 0.5 * W + beta * (W @ W)

    return np.hstack([V_inv @ t, _vee(W)])


@dataclass(frozen=True)
class ExtrinsicsEstimate:
    T_body_camera: np.ndarray
    residuals: np.ndarray  # (N, 6) [v, w] per frame after the fit
    cost: float
    success: bool
    message: str


def estimate_extrinsics(T_world_body: np.ndarray, T_world_camera: np.ndarray, T_body_camera_guess: np.ndarray,
    spec: ExtrinsicsSpec | None = None) -> ExtrinsicsEstimate:
    spec = spec or ExtrinsicsSpec()
    solver = dict(spec.solver)
    if solver.get("method") == "lm" and solver.get("loss", "linear") != "linear":
        raise ValueError("Levenberg-Marquardt supports only the linear loss")

    T0 = np.asarray(T_body_camera_guess, dtype=float).copy()
    T_camera_world = [_invert(T) for T in np.asarray(T_world_camera, dtype=float)]
    T_world_body = np.asarray(T_world_body, dtype=float)

    def frame_residuals(delta_xi: np.ndarray) -> np.ndarray:
        T = T0 @ se3_exp(delta_xi)
        errors = []
        for B, C_inv in zip(T_world_body, T_camera_world, strict=True):
            error = se3_log(C_inv @ (B @ T))
            error[:3] *= spec.weight_translation
            error[3:] *= spec.weight_rotation
            errors.append(error)
        return np.asarray(errors)

    result = least_squares(lambda x: frame_residuals(x).ravel(), np.zeros(6), **solver)
    return ExtrinsicsEstimate(
        T_body_camera=T0 @ se3_exp(result.x),
        residuals=frame_residuals(result.x),
        cost=float(result.cost),
        success=bool(result.success),
        message=str(result.message),
    )


def extrinsics_fit(T_world_body: np.ndarray, T_world_camera: np.ndarray, T_body_camera: np.ndarray) -> ExtrinsicsEstimate:
    """Residuals of given extrinsics w.r.t. body and camera poses (no estimation)."""
    residuals = np.asarray([se3_log(_invert(C) @ (B @ T_body_camera))
        for B, C in zip(np.asarray(T_world_body), np.asarray(T_world_camera), strict=True)])
    return ExtrinsicsEstimate(T_body_camera=np.asarray(T_body_camera), residuals=residuals,
        cost=float(0.5 * np.sum(residuals**2)), success=True, message="estimated in the factor graph")
