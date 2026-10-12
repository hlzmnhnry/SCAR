"""Import of data produced by the legacy (CVPRW 2026) scripts.

Two things live here:

- readers for legacy anchor files (``PT_*.json`` pixel tracks, ``WP_*.json`` world points) and
  stage results (``<stage>_<datetime>.json``), keyed by image timestamps in microseconds;
- :func:`legacy_ltc_body_poses`, which reproduces *exactly* how the legacy pipeline turned LTC
  ``ins.csv`` rows into body poses via pygeon 3c5aae8. That includes its known deficiencies:
  latitude/longitude/altitude/quaternions read as float32, no meridian convergence, INS altitude
  used as world height and nearest-sample association. It exists only to reproduce published
  results; new code should use :class:`scar.sources.poses.PoseTable`.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from scar.core.camera import CameraModel
from scar.core.se3 import R_ENU_NED
from scar.core.types import (
    Extrinsics,
    Frame,
    Landmarks,
    Observations,
    PosePrior,
    Sequence,
)
from scar.geodesy.world_frame import HeightReference, WorldFrame
from scar.sources.poses import PoseTable, to_nanoseconds

# the legacy scripts hard-coded the LTC image size when reading stage results
LEGACY_IMAGE_SIZE = (1600, 1100)

_INS_COLUMNS = ("latitude", "longitude", "altitude", "quat.x", "quat.y", "quat.z", "quat.w")


@dataclass(frozen=True)
class LegacyAnchors:
    """Anchors from legacy files; observations reference images by timestamp [ns]."""

    landmarks: Landmarks
    observation_timestamps_ns: np.ndarray
    observation_landmark_ids: np.ndarray
    observation_uv: np.ndarray

    @property
    def image_timestamps_ns(self) -> np.ndarray:
        return np.unique(self.observation_timestamps_ns)

    @property
    def observation_timestamps_us(self) -> np.ndarray:
        return self.observation_timestamps_ns // 1000

    @property
    def image_timestamps_us(self) -> np.ndarray:
        return self.image_timestamps_ns // 1000


@dataclass(frozen=True)
class LegacyResult:
    """One legacy optimization stage result."""

    camera: CameraModel
    image_timestamps_ns: np.ndarray
    T_world_camera: np.ndarray
    anchors: LegacyAnchors
    T_ins_cam: np.ndarray | None = None


def _timestamps_to_ns(values: list, time_unit: str) -> np.ndarray:
    # JSON numbers are converted through their shortest decimal representation, so e.g.
    # 1776855480.0249462 s becomes 1776855480024946200 ns exactly
    text = np.asarray([repr(v) if isinstance(v, float) else str(v) for v in values])
    return to_nanoseconds(text, time_unit)


def _anchors_from_dicts(pixel_tracks: dict, world_points: dict, time_unit: str = "us") -> LegacyAnchors:
    landmark_ids = np.asarray(sorted(int(key) for key in world_points), dtype=np.int64)
    landmarks = Landmarks(ids=landmark_ids, xyz=[world_points[str(i)] for i in landmark_ids])

    timestamps, ids, uv = [], [], []
    for key, observations in pixel_tracks.items():
        for timestamp, (u, v) in observations:
            timestamps.append(timestamp)
            ids.append(int(key))
            uv.append((float(u), float(v)))

    return LegacyAnchors(
        landmarks=landmarks,
        observation_timestamps_ns=_timestamps_to_ns(timestamps, time_unit),
        observation_landmark_ids=np.asarray(ids, dtype=np.int64),
        observation_uv=np.asarray(uv, dtype=float).reshape(-1, 2),
    )


def load_legacy_anchors(pixel_tracks_path: str | Path, world_points_path: str | Path,
    time_unit: str = "us") -> LegacyAnchors:
    """Legacy anchor files; ``time_unit`` of their image timestamps ('us' for LTC, 's' for other exports)."""
    with Path(pixel_tracks_path).open(encoding="utf-8") as file:
        pixel_tracks = json.load(file)
    with Path(world_points_path).open(encoding="utf-8") as file:
        world_points = json.load(file)
    return _anchors_from_dicts(pixel_tracks, world_points, time_unit)


def load_legacy_result(path: str | Path, time_unit: str = "us") -> LegacyResult:
    with Path(path).open(encoding="utf-8") as file:
        result = json.load(file)

    calibration = result["camera_calibration"]
    width = result.get("image_width", LEGACY_IMAGE_SIZE[0])
    height = result.get("image_height", LEGACY_IMAGE_SIZE[1])
    camera = CameraModel(width=width, height=height, fx=calibration["fx"], fy=calibration["fy"],
        cx=calibration["cx"], cy=calibration["cy"],
        distortion=(calibration["k1"], calibration["k2"], calibration["p1"], calibration["p2"]))

    poses = np.asarray([result["camera_poses"][str(t)] for t in result["image_indices"]], dtype=float)
    T_ins_cam = np.asarray(result["T_ins_cam"], dtype=float) if "T_ins_cam" in result else None

    return LegacyResult(camera=camera, image_timestamps_ns=_timestamps_to_ns(result["image_indices"], time_unit),
        T_world_camera=poses, anchors=_anchors_from_dicts(result["pixel_tracks"], result["world_points"], time_unit),
        T_ins_cam=T_ins_cam)


def read_legacy_ins(path: str | Path) -> dict[str, np.ndarray]:
    """Read an LTC ``ins.csv`` the way pygeon 3c5aae8 did (pose columns as float32, sorted by time)."""
    with Path(path).open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    data = {"timestamp": np.asarray([int(row["timestamp"]) for row in rows], dtype=np.int64)}
    for column in _INS_COLUMNS:
        data[column] = np.asarray([float(row[column]) for row in rows], dtype=np.float64).astype(np.float32)

    if np.any(np.diff(data["timestamp"]) < 0):
        order = np.argsort(data["timestamp"], kind="stable")
        data = {key: value[order] for key, value in data.items()}
    return data


def _legacy_nearest(sorted_timestamps: np.ndarray, t: int) -> int:
    i = int(np.searchsorted(sorted_timestamps, t))
    if i == 0:
        return 0
    if i == len(sorted_timestamps):
        return i - 1
    return i - 1 if (t - sorted_timestamps[i - 1]) <= (sorted_timestamps[i] - t) else i


def legacy_ltc_body_poses(ins: dict[str, np.ndarray], timestamps_us) -> np.ndarray:
    """Body poses (N, 4, 4) as the legacy pipeline computed them for the given image timestamps."""
    import utm  # legacy dependency (pinned version), see the 'legacy' extra

    poses = []
    for t in np.asarray(timestamps_us, dtype=np.int64):
        j = _legacy_nearest(ins["timestamp"], int(t))
        # scalar float32 inputs on purpose: utm then computes partly in float32, like the legacy code
        x, y = utm.from_latlon(ins["latitude"][j], ins["longitude"][j])[:2]
        quat = np.array([ins["quat.x"][j], ins["quat.y"][j], ins["quat.z"][j], ins["quat.w"][j]])
        R = R_ENU_NED @ Rotation.from_quat(quat, scalar_first=False).as_matrix()

        T = np.hstack([R, np.array([x, y, ins["altitude"][j]]).reshape((-1, 1))])
        poses.append(np.vstack([T, np.array([0.0, 0.0, 0.0, 1.0])]))
    return np.asarray(poses)


def legacy_ltc_sequence(anchors: LegacyAnchors, ins_csv: str | Path, camera: CameraModel,
    extrinsics: Extrinsics, world_frame: WorldFrame | None = None) -> Sequence:
    """Sequence of the images observed in legacy anchors, with legacy LTC pose priors.

    Frames get ids in time order; their ``image_path`` is the legacy image index (timestamp [us]).
    The default world frame is EPSG:25832 with DHHN2016 heights and zero origin, like the legacy data.
    """
    world_frame = world_frame or WorldFrame(crs="EPSG:25832", height=HeightReference.geoid("GCG2016", "DHHN2016"))
    timestamps_us = anchors.image_timestamps_us
    body_poses = legacy_ltc_body_poses(read_legacy_ins(ins_csv), timestamps_us)

    frames = tuple(
        Frame(frame_id=i, timestamp_ns=int(t) * 1000, image_path=f"{int(t)}.png", pose_prior=PosePrior(T),
            metadata={"legacy_image_index": int(t)})
        for i, (t, T) in enumerate(zip(timestamps_us, body_poses, strict=True))
    )
    frame_of_timestamp = {int(t): i for i, t in enumerate(timestamps_us)}
    observations = Observations(
        frame_ids=[frame_of_timestamp[int(t)] for t in anchors.observation_timestamps_us],
        landmark_ids=anchors.observation_landmark_ids,
        uv=anchors.observation_uv,
    )
    return Sequence(camera=camera, extrinsics=extrinsics, frames=frames, world_frame=world_frame,
        landmarks=anchors.landmarks, observations=observations, metadata={"source": "legacy"})


def sequence_from_anchors(anchors: LegacyAnchors, poses: PoseTable | None, camera: CameraModel,
    extrinsics: Extrinsics, world_frame: WorldFrame, interpolation: str = "linear",
    max_gap_ns: int | None = 50_000_000, priors: list[PosePrior] | None = None) -> Sequence:
    """Sequence of the images observed in legacy anchors.

    Pose priors come from a pose table, or from ``priors`` (one per image in time order).
    """
    timestamps = anchors.image_timestamps_ns
    if priors is None:
        priors = [poses.prior_at(int(t), mode=interpolation, max_gap_ns=max_gap_ns) for t in timestamps]
    frames = tuple(
        Frame(frame_id=i, timestamp_ns=int(t), image_path="", metadata={"legacy_timestamp_ns": int(t)}, pose_prior=prior)
        for i, (t, prior) in enumerate(zip(timestamps, priors, strict=True))
    )
    frame_of_timestamp = {int(t): i for i, t in enumerate(timestamps)}
    observations = Observations(
        frame_ids=[frame_of_timestamp[int(t)] for t in anchors.observation_timestamps_ns],
        landmark_ids=anchors.observation_landmark_ids,
        uv=anchors.observation_uv,
    )
    return Sequence(camera=camera, extrinsics=extrinsics, frames=frames, world_frame=world_frame,
        landmarks=anchors.landmarks, observations=observations, metadata={"source": "legacy"})
