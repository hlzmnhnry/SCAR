"""Frame-to-frame feature tracking (pyramidal Lucas-Kanade) with epipolar outlier rejection."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from scar.core.camera import CameraModel


@dataclass(frozen=True)
class TrackingConfig:
    window_size: int = 51
    pyramid_levels: int = 4
    max_iterations: int = 50
    epsilon: float = 0.001
    min_eigen_threshold: float = 1e-4
    fundamental_filter: bool = True
    fundamental_threshold_px: float = 1.0
    fundamental_confidence: float = 0.95
    fundamental_max_iterations: int = 2000

    @property
    def lk_parameters(self) -> dict:
        return {
            "winSize": (self.window_size, self.window_size),
            "maxLevel": self.pyramid_levels,
            "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, self.max_iterations, self.epsilon),
            "minEigThreshold": self.min_eigen_threshold,
        }


def track(previous: np.ndarray, current: np.ndarray, points: np.ndarray, camera: CameraModel,
    config: TrackingConfig) -> tuple[np.ndarray, np.ndarray]:
    """Track ``points`` (N, 2) from ``previous`` to ``current``.

    Returns the new positions (N, 2) and a mask of points that survived tracking and, if enabled,
    RANSAC on the fundamental matrix (computed on points undistorted with ``camera``).
    """
    points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if len(points) == 0:
        return points.astype(float), np.zeros(0, dtype=bool)

    new_points, status, _ = cv2.calcOpticalFlowPyrLK(previous, current, points, None, **config.lk_parameters)
    if new_points is None or status is None:
        raise RuntimeError("Optical flow failed")
    keep = status.ravel() == 1

    if config.fundamental_filter and keep.sum() >= 8:
        p0 = camera.undistort_pixels(points[keep])
        p1 = camera.undistort_pixels(new_points[keep])
        _, inliers = cv2.findFundamentalMat(p0.astype(np.float32), p1.astype(np.float32), cv2.FM_RANSAC,
            ransacReprojThreshold=config.fundamental_threshold_px, confidence=config.fundamental_confidence,
            maxIters=config.fundamental_max_iterations)
        if inliers is None:
            raise RuntimeError("Outlier rejection for tracking failed")
        tracked = np.flatnonzero(keep)
        keep[tracked[inliers.ravel() == 0]] = False

    return new_points.astype(float), keep
