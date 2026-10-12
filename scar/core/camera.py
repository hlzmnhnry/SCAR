"""Pinhole camera with OpenCV radial-tangential distortion."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import cv2
import numpy as np

from .se3 import invert

SUPPORTED_MODELS = ("pinhole-radtan",)


@dataclass(frozen=True)
class CameraModel:
    """Pinhole intrinsics with OpenCV distortion coefficients ``(k1, k2, p1, p2[, k3])``.

    This matches GTSAM's ``Cal3DS2`` for four coefficients (no skew is modelled).
    """

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    distortion: tuple[float, ...] = ()
    model: str = "pinhole-radtan"
    metadata: dict[str, Any] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        if self.model not in SUPPORTED_MODELS:
            raise ValueError(f"Unsupported camera model {self.model!r}, expected one of {SUPPORTED_MODELS}")
        if int(self.width) <= 0 or int(self.height) <= 0:
            raise ValueError("Camera width and height must be positive")
        if float(self.fx) <= 0.0 or float(self.fy) <= 0.0:
            raise ValueError("Focal lengths must be positive")
        if len(self.distortion) not in (0, 4, 5):
            raise ValueError("Distortion must have 0, 4 (k1, k2, p1, p2) or 5 (.., k3) coefficients")

        for name in ("width", "height"):
            object.__setattr__(self, name, int(getattr(self, name)))
        for name in ("fx", "fy", "cx", "cy"):
            object.__setattr__(self, name, float(getattr(self, name)))
        object.__setattr__(self, "distortion", tuple(float(c) for c in self.distortion))

    @classmethod
    def from_matrix(cls, K: np.ndarray, width: int, height: int, distortion=(), **kwargs) -> CameraModel:
        K = np.asarray(K, dtype=float)
        if K.shape != (3, 3):
            raise ValueError("K must be a 3x3 matrix")
        if abs(K[0, 1]) > 1e-9:
            raise ValueError("Skewed camera matrices are not supported")
        return cls(width=width, height=height, fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2],
            distortion=tuple(np.ravel(distortion)), **kwargs)

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]])

    @property
    def dist(self) -> np.ndarray:
        """Distortion coefficients as OpenCV expects them (always 5 entries)."""
        coefficients = np.zeros(5)
        coefficients[: len(self.distortion)] = self.distortion
        return coefficients

    @property
    def image_size(self) -> tuple[int, int]:
        return self.width, self.height

    def replace(self, **changes) -> CameraModel:
        return replace(self, **changes)

    def distort_normalized(self, xy: np.ndarray) -> np.ndarray:
        """Apply the distortion model to normalized image coordinates (N, 2)."""
        k1, k2, p1, p2, k3 = self.dist
        x, y = xy[..., 0], xy[..., 1]
        r2 = x * x + y * y
        radial = 1.0 + r2 * (k1 + r2 * (k2 + r2 * k3))
        x_d = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
        y_d = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
        return np.stack([x_d, y_d], axis=-1)

    def project_camera_points(self, points_camera: np.ndarray) -> np.ndarray:
        """Project points (N, 3) given in the camera frame (+z forward) to pixels (N, 2)."""
        points_camera = np.asarray(points_camera, dtype=float)
        xy = points_camera[..., :2] / points_camera[..., 2:3]
        xy_d = self.distort_normalized(xy)
        return np.stack([self.fx * xy_d[..., 0] + self.cx, self.fy * xy_d[..., 1] + self.cy], axis=-1)

    def project(self, points_world: np.ndarray, T_world_camera: np.ndarray) -> np.ndarray:
        """Project world points (N, 3) into a camera with pose ``T_world_camera``."""
        T_camera_world = invert(T_world_camera)
        points_camera = np.asarray(points_world, dtype=float) @ T_camera_world[:3, :3].T + T_camera_world[:3, 3]
        return self.project_camera_points(points_camera)

    def undistort_pixels(self, pixels: np.ndarray) -> np.ndarray:
        """Remove distortion from pixels (N, 2), returning ideal pinhole pixels (N, 2)."""
        pixels = np.asarray(pixels, dtype=np.float64).reshape(-1, 1, 2)
        undistorted = cv2.undistortPoints(pixels, cameraMatrix=self.K, distCoeffs=self.dist, P=self.K)
        return undistorted.reshape(-1, 2)
