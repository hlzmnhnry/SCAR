"""Array representation of a calibration problem and the estimates optimized for it."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scar.core.camera import CameraModel
from scar.core.types import Extrinsics, Sequence


def _index_of(values: np.ndarray, queries: np.ndarray) -> np.ndarray:
    """Positions of ``queries`` in the (unsorted, unique) array ``values``."""
    if len(queries) == 0:
        return np.zeros(0, dtype=np.int64)
    sorter = np.argsort(values)
    return sorter[np.searchsorted(values, queries, sorter=sorter)].astype(np.int64)


@dataclass(frozen=True)
class CalibrationProblem:
    """Frames, anchors and observations that take part in the optimization.

    Observations reference frames and landmarks by their index in this problem. They are
    ordered by landmark and keep the order of each track, which fixes the factor order.
    """

    camera: CameraModel
    extrinsics: Extrinsics
    frame_ids: np.ndarray
    timestamps_ns: np.ndarray
    T_world_body: np.ndarray
    navigation_covariance: np.ndarray | None
    landmark_ids: np.ndarray
    landmark_xyz: np.ndarray
    landmark_covariance: np.ndarray | None
    observation_frame: np.ndarray
    observation_landmark: np.ndarray
    observation_uv: np.ndarray

    @property
    def num_frames(self) -> int:
        return len(self.frame_ids)

    @property
    def num_landmarks(self) -> int:
        return len(self.landmark_ids)

    @property
    def num_observations(self) -> int:
        return len(self.observation_uv)

    @property
    def T_world_camera(self) -> np.ndarray:
        return self.T_world_body @ self.extrinsics.T_body_camera

    @classmethod
    def from_sequence(cls, sequence: Sequence, *, min_point_observations: int = 1,
        min_frame_features: int = 1) -> CalibrationProblem:
        """Select the anchors and frames to optimize, like the legacy pipeline did.

        1. drop landmarks with fewer than ``min_point_observations`` observations,
        2. drop observations in frames that keep fewer than ``min_frame_features`` of them,
           and landmarks left without observations (no second pass of step 1),
        3. keep only frames that are still observed.
        """
        observations, landmarks = sequence.observations, sequence.landmarks

        # stable grouping by landmark id preserves the order within each track
        order = np.argsort(observations.landmark_ids, kind="stable")
        obs_landmark = observations.landmark_ids[order]
        obs_frame = observations.frame_ids[order]
        obs_uv = observations.uv[order]

        ids, counts = np.unique(obs_landmark, return_counts=True)
        keep = np.isin(obs_landmark, ids[counts >= min_point_observations])
        obs_landmark, obs_frame, obs_uv = obs_landmark[keep], obs_frame[keep], obs_uv[keep]

        frame_values, frame_counts = np.unique(obs_frame, return_counts=True)
        keep = np.isin(obs_frame, frame_values[frame_counts >= min_frame_features])
        obs_landmark, obs_frame, obs_uv = obs_landmark[keep], obs_frame[keep], obs_uv[keep]

        # frames in time order, landmarks in id order
        observed_frames = set(obs_frame.tolist())
        frames = [frame for frame in sequence.frames if frame.frame_id in observed_frames]
        frame_ids = np.asarray([frame.frame_id for frame in frames], dtype=np.int64)
        landmark_ids = np.unique(obs_landmark)

        landmark_index = _index_of(landmarks.ids, landmark_ids)

        navigation_covariance = None
        if frames and all(frame.pose_prior.covariance is not None for frame in frames):
            navigation_covariance = np.asarray([frame.pose_prior.covariance for frame in frames])

        return cls(
            camera=sequence.camera,
            extrinsics=sequence.extrinsics,
            frame_ids=frame_ids,
            timestamps_ns=np.asarray([frame.timestamp_ns for frame in frames], dtype=np.int64),
            T_world_body=np.asarray([frame.pose_prior.T_world_body for frame in frames]).reshape(-1, 4, 4),
            navigation_covariance=navigation_covariance,
            landmark_ids=landmark_ids,
            landmark_xyz=landmarks.xyz[landmark_index],
            landmark_covariance=None if landmarks.covariance is None else landmarks.covariance[landmark_index],
            observation_frame=_index_of(frame_ids, obs_frame),
            observation_landmark=_index_of(landmark_ids, obs_landmark),
            observation_uv=obs_uv,
        )


@dataclass(frozen=True)
class CalibrationState:
    """Estimates of one optimization stage: intrinsics, camera poses and landmarks."""

    camera: CameraModel
    T_world_camera: np.ndarray
    landmark_xyz: np.ndarray
    skew: float = 0.0
    T_body_camera: np.ndarray | None = None  # estimated by joint stages

    @classmethod
    def initial(cls, problem: CalibrationProblem) -> CalibrationState:
        return cls(camera=problem.camera, T_world_camera=problem.T_world_camera, landmark_xyz=problem.landmark_xyz)
