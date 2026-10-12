"""Core data model shared by all SCAR stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from scar.geodesy.world_frame import WorldFrame

from .camera import CameraModel
from .se3 import is_rigid_transform

SEQUENCE_SCHEMA_VERSION = "2.0"


def _rigid_transform(value: Any, name: str) -> np.ndarray:
    transform = np.asarray(value, dtype=float)
    if transform.shape != (4, 4) or not is_rigid_transform(transform):
        raise ValueError(f"{name} must be a 4x4 rigid transform")
    return transform


def _covariance(value: Any, dim: int, name: str) -> np.ndarray:
    covariance = np.asarray(value, dtype=float)
    if covariance.shape[-2:] != (dim, dim):
        raise ValueError(f"{name} must have shape (..., {dim}, {dim}), got {covariance.shape}")
    if not np.allclose(covariance, np.swapaxes(covariance, -1, -2)):
        raise ValueError(f"{name} must be symmetric")
    return covariance


@dataclass(frozen=True)
class Extrinsics:
    """Rigid transform from the camera frame into the body (INS) frame: ``p_body = T_body_camera @ p_camera``."""

    T_body_camera: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "T_body_camera", _rigid_transform(self.T_body_camera, "T_body_camera"))


@dataclass(frozen=True)
class PosePrior:
    """Body pose from navigation data with optional 6x6 covariance.

    The covariance is ordered ``[rotation, translation]`` in world-frame axes, with the
    rotation as left perturbation (see :mod:`scar.core.se3`).
    """

    T_world_body: np.ndarray
    covariance: np.ndarray | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "T_world_body", _rigid_transform(self.T_world_body, "T_world_body"))
        if self.covariance is not None:
            object.__setattr__(self, "covariance", _covariance(self.covariance, 6, "covariance"))

    def T_world_camera(self, extrinsics: Extrinsics) -> np.ndarray:
        return self.T_world_body @ extrinsics.T_body_camera


@dataclass(frozen=True)
class Frame:
    """One image with its timestamp [ns] and navigation pose prior."""

    frame_id: int
    timestamp_ns: int
    image_path: str
    pose_prior: PosePrior
    metadata: dict[str, Any] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        if int(self.frame_id) < 0:
            raise ValueError("frame_id must be non-negative")
        object.__setattr__(self, "frame_id", int(self.frame_id))
        object.__setattr__(self, "timestamp_ns", int(self.timestamp_ns))
        object.__setattr__(self, "image_path", str(self.image_path))


@dataclass(frozen=True)
class Landmarks:
    """Georeferenced anchor points in local world coordinates, with optional 3x3 covariances."""

    ids: np.ndarray
    xyz: np.ndarray
    covariance: np.ndarray | None = None

    def __post_init__(self) -> None:
        ids = np.asarray(self.ids, dtype=np.int64).reshape(-1)
        xyz = np.asarray(self.xyz, dtype=float).reshape(-1, 3)
        if len(ids) != len(xyz):
            raise ValueError("Landmark ids and positions differ in length")
        if len(np.unique(ids)) != len(ids):
            raise ValueError("Landmark ids must be unique")
        object.__setattr__(self, "ids", ids)
        object.__setattr__(self, "xyz", xyz)
        if self.covariance is not None:
            covariance = _covariance(self.covariance, 3, "landmark covariance")
            if covariance.shape != (len(ids), 3, 3):
                raise ValueError("Landmark covariance must have shape (M, 3, 3)")
            object.__setattr__(self, "covariance", covariance)

    @classmethod
    def empty(cls) -> Landmarks:
        return cls(ids=np.zeros(0, dtype=np.int64), xyz=np.zeros((0, 3)))

    def __len__(self) -> int:
        return len(self.ids)


@dataclass(frozen=True)
class Observations:
    """Pixel observations ``uv`` (N, 2) of landmarks in frames (struct of arrays)."""

    frame_ids: np.ndarray
    landmark_ids: np.ndarray
    uv: np.ndarray

    def __post_init__(self) -> None:
        frame_ids = np.asarray(self.frame_ids, dtype=np.int64).reshape(-1)
        landmark_ids = np.asarray(self.landmark_ids, dtype=np.int64).reshape(-1)
        uv = np.asarray(self.uv, dtype=float).reshape(-1, 2)
        if not len(frame_ids) == len(landmark_ids) == len(uv):
            raise ValueError("Observation arrays differ in length")
        object.__setattr__(self, "frame_ids", frame_ids)
        object.__setattr__(self, "landmark_ids", landmark_ids)
        object.__setattr__(self, "uv", uv)

    @classmethod
    def empty(cls) -> Observations:
        return cls(frame_ids=np.zeros(0, dtype=np.int64), landmark_ids=np.zeros(0, dtype=np.int64), uv=np.zeros((0, 2)))

    def __len__(self) -> int:
        return len(self.frame_ids)


@dataclass(frozen=True)
class Sequence:
    """Everything SCAR knows about one recording: sensors, frames and (collected) anchors."""

    camera: CameraModel
    extrinsics: Extrinsics
    frames: tuple[Frame, ...]
    world_frame: WorldFrame
    landmarks: Landmarks = field(default_factory=Landmarks.empty)
    observations: Observations = field(default_factory=Observations.empty)
    metadata: dict[str, Any] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        frames = tuple(sorted(self.frames, key=lambda frame: frame.timestamp_ns))
        object.__setattr__(self, "frames", frames)

        frame_ids = np.asarray([frame.frame_id for frame in frames], dtype=np.int64)
        if len(np.unique(frame_ids)) != len(frame_ids):
            raise ValueError("Frame ids must be unique")
        timestamps = np.asarray([frame.timestamp_ns for frame in frames], dtype=np.int64)
        if len(np.unique(timestamps)) != len(timestamps):
            raise ValueError("Frame timestamps must be unique")

        unknown_frames = ~np.isin(self.observations.frame_ids, frame_ids)
        if np.any(unknown_frames):
            raise ValueError(f"Observations reference unknown frame ids, e.g. {self.observations.frame_ids[unknown_frames][0]}")
        unknown_landmarks = ~np.isin(self.observations.landmark_ids, self.landmarks.ids)
        if np.any(unknown_landmarks):
            raise ValueError(f"Observations reference unknown landmark ids, e.g. {self.observations.landmark_ids[unknown_landmarks][0]}")

    @property
    def frame_by_id(self) -> dict[int, Frame]:
        return {frame.frame_id: frame for frame in self.frames}

    @property
    def frame_by_timestamp(self) -> dict[int, Frame]:
        return {frame.timestamp_ns: frame for frame in self.frames}

    def T_world_camera(self) -> np.ndarray:
        """Camera pose priors of all frames (F, 4, 4), in frame order."""
        return np.asarray([frame.pose_prior.T_world_camera(self.extrinsics) for frame in self.frames])

    def replace(self, **changes) -> Sequence:
        values = {name: getattr(self, name) for name in self.__dataclass_fields__}
        values.update(changes)
        return Sequence(**values)
