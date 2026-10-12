"""Visual localization (PnP) against reference anchors, as in the CVPRW 2026 evaluation.

Each frame's 2D-3D anchor correspondences (collected per frame without tracking) give a camera pose
by PnP with the calibration's intrinsics. It is compared with the navigation pose composed with the
calibration's extrinsics. Both intrinsics (PnP) and extrinsics (reference pose) enter the result.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from scar.core.camera import CameraModel
from scar.core.se3 import invert, make_transform, rotation_angle
from scar.core.types import Sequence

# (translation [m], rotation [deg]) thresholds of the paper; the first three follow common visual
# localization benchmarks, the others suit high-altitude flights
DEFAULT_THRESHOLDS = ((0.25, 2.0), (0.5, 5.0), (5.0, 10.0), (2.0, 2.0), (5.0, 5.0), (10.0, 10.0))


@dataclass(frozen=True)
class LocalizationConfig:
    max_error_px: float = 4.0
    confidence: float = 0.99
    iterations: int = 10000
    min_points: int = 200
    min_inliers: int = 50
    thresholds: tuple[tuple[float, float], ...] = DEFAULT_THRESHOLDS


@dataclass(frozen=True)
class LocalizationResult:
    frame_ids: np.ndarray
    rotation_error_deg: np.ndarray
    translation_error_m: np.ndarray
    inliers: np.ndarray
    frames_total: int
    skipped_few_points: int
    skipped_few_inliers: int
    thresholds: tuple[tuple[float, float], ...] = field(default=DEFAULT_THRESHOLDS)

    @property
    def frames_evaluated(self) -> int:
        """Frames with enough points (the denominator of the accuracies, as in the paper)."""
        return self.frames_total - self.skipped_few_points

    def accuracy(self, translation_m: float, rotation_deg: float) -> float:
        good = (self.translation_error_m < translation_m) & (self.rotation_error_deg < rotation_deg)
        return float(good.sum()) / max(self.frames_evaluated, 1)

    def summary(self) -> dict:
        return {
            "frames": self.frames_total,
            "localized": len(self.frame_ids),
            "median_rotation_error_deg": float(np.median(self.rotation_error_deg)) if len(self.frame_ids) else None,
            "median_translation_error_m": float(np.median(self.translation_error_m)) if len(self.frame_ids) else None,
            "accuracy": {f"{t}m/{r}deg": self.accuracy(t, r) for t, r in self.thresholds},
        }


def localize(sequence: Sequence, camera: CameraModel, T_body_camera: np.ndarray,
    config: LocalizationConfig | None = None) -> LocalizationResult:
    """PnP per frame of ``sequence`` (anchors and observations per frame) and errors to the prior poses."""
    config = config or LocalizationConfig()
    landmark_index = {int(i): k for k, i in enumerate(sequence.landmarks.ids)}
    observations = sequence.observations

    ids, rotation_errors, translation_errors, inliers = [], [], [], []
    few_points = few_inliers = 0
    for frame in sequence.frames:
        rows = np.flatnonzero(observations.frame_ids == frame.frame_id)
        if len(rows) < config.min_points:
            few_points += 1
            continue
        points = sequence.landmarks.xyz[[landmark_index[int(i)] for i in observations.landmark_ids[rows]]]
        pixels = observations.uv[rows]

        # object points relative to their centroid for numerical stability (UTM coordinates are large)
        centre = points.mean(axis=0)
        ok, rvec, tvec, inlier_index = cv2.solvePnPRansac(points - centre, pixels, camera.K, camera.dist,
            reprojectionError=config.max_error_px, confidence=config.confidence, iterationsCount=config.iterations,
            flags=cv2.SOLVEPNP_SQPNP)
        if not ok or inlier_index is None or len(inlier_index) < config.min_inliers:
            few_inliers += 1
            continue
        inlier_index = inlier_index.ravel()
        rvec, tvec = cv2.solvePnPRefineLM(points[inlier_index] - centre, pixels[inlier_index], camera.K, camera.dist,
            rvec, tvec)

        T_camera_world = make_transform(cv2.Rodrigues(rvec)[0], tvec.ravel())
        T_world_camera = invert(T_camera_world)
        T_world_camera[:3, 3] += centre
        reference = frame.pose_prior.T_world_body @ T_body_camera

        ids.append(frame.frame_id)
        rotation_errors.append(float(np.degrees(rotation_angle(T_world_camera[:3, :3], reference[:3, :3]))))
        translation_errors.append(float(np.linalg.norm(T_world_camera[:3, 3] - reference[:3, 3])))
        inliers.append(len(inlier_index))

    return LocalizationResult(frame_ids=np.asarray(ids, dtype=np.int64), rotation_error_deg=np.asarray(rotation_errors),
        translation_error_m=np.asarray(translation_errors), inliers=np.asarray(inliers, dtype=np.int64),
        frames_total=len(sequence.frames), skipped_few_points=few_points, skipped_few_inliers=few_inliers,
        thresholds=config.thresholds)
