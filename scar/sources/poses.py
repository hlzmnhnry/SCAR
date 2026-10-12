"""Time series of navigation poses and their association with image timestamps."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Literal

import numpy as np

from scar.core.se3 import (
    euler_zyx_rotation_covariance,
    make_transform,
    quat_xyzw_to_matrix,
    rotate_pose_covariance,
    so3_exp,
    so3_log,
)
from scar.core.types import PosePrior
from scar.geodesy import (
    HeightReference,
    WorldFrame,
    geodetic_to_world,
    world_from_navigation_rotation,
)

InterpolationMode = Literal["linear", "nearest"]

_NS_PER_UNIT = {"s": 1_000_000_000, "ms": 1_000_000, "us": 1_000, "ns": 1}


def to_nanoseconds(values, unit: str) -> np.ndarray:
    """Convert timestamps in ``unit`` ('s', 'ms', 'us', 'ns') to int64 nanoseconds.

    Integers and decimal strings are converted exactly. Floats are limited by float64: epoch
    seconds (~1.7e9) resolve only ~240 ns, so read timestamps from files as text.
    """
    if unit not in _NS_PER_UNIT:
        raise ValueError(f"Unknown time unit {unit!r}, expected one of {sorted(_NS_PER_UNIT)}")
    scale = _NS_PER_UNIT[unit]
    values = np.asarray(values)
    if np.issubdtype(values.dtype, np.integer):
        return values.astype(np.int64) * scale
    if values.dtype.kind in "USO":
        return np.asarray([int((Decimal(str(v).strip()) * scale).to_integral_value()) for v in values.ravel()],
            dtype=np.int64).reshape(values.shape)
    return np.round(values.astype(np.float64) * scale).astype(np.int64)


@dataclass(frozen=True)
class PoseTable:
    """Body poses ``T_world_body`` (N, 4, 4) at strictly increasing timestamps [ns].

    ``covariance`` (N, 6, 6) follows the :class:`~scar.core.types.PosePrior` convention.
    """

    timestamps_ns: np.ndarray
    T_world_body: np.ndarray
    covariance: np.ndarray | None = None

    def __post_init__(self) -> None:
        timestamps = np.asarray(self.timestamps_ns, dtype=np.int64).reshape(-1)
        T = np.asarray(self.T_world_body, dtype=float)
        if T.shape != (len(timestamps), 4, 4):
            raise ValueError("T_world_body must have shape (N, 4, 4) matching the timestamps")
        if len(timestamps) == 0:
            raise ValueError("A pose table needs at least one pose")
        if np.any(np.diff(timestamps) <= 0):
            raise ValueError("Pose timestamps must be strictly increasing")
        object.__setattr__(self, "timestamps_ns", timestamps)
        object.__setattr__(self, "T_world_body", T)
        if self.covariance is not None:
            covariance = np.asarray(self.covariance, dtype=float)
            if covariance.shape != (len(timestamps), 6, 6):
                raise ValueError("covariance must have shape (N, 6, 6)")
            object.__setattr__(self, "covariance", covariance)

    def __len__(self) -> int:
        return len(self.timestamps_ns)

    @classmethod
    def from_geodetic(
        cls,
        timestamps_ns,
        latitude,
        longitude,
        height,
        quat_xyzw,
        *,
        world: WorldFrame,
        height_reference: HeightReference,
        source_crs: str = "EPSG:4937",
        navigation_frame: str = "NED",
        undulation=None,
        covariance_nav=None,
    ) -> PoseTable:
        """Build a pose table from geodetic positions and body-to-navigation-frame quaternions.

        ``quat_xyzw`` rotates body vectors into the local navigation frame (``R_nav_body``).
        ``covariance_nav`` (N, 6, 6) is given in navigation-frame axes and rotated into the world frame.
        """
        latitude = np.asarray(latitude, dtype=np.float64)
        longitude = np.asarray(longitude, dtype=np.float64)

        positions = geodetic_to_world(latitude, longitude, height, world=world, height_reference=height_reference,
            source_crs=source_crs, undulation=undulation)
        R_world_nav = world_from_navigation_rotation(latitude, longitude, world, navigation_frame)
        R_world_body = R_world_nav @ quat_xyzw_to_matrix(quat_xyzw)

        covariance = None
        if covariance_nav is not None:
            covariance = rotate_pose_covariance(covariance_nav, R_world_nav)

        return cls(timestamps_ns=timestamps_ns, T_world_body=make_transform(R_world_body, positions), covariance=covariance)

    def prior_at(self, timestamp_ns: int, *, mode: InterpolationMode = "linear", max_gap_ns: int | None = None) -> PosePrior:
        """Pose prior at ``timestamp_ns``.

        ``linear`` interpolates positions linearly and rotations on SO(3) (slerp) between the
        enclosing samples and uses the covariance of the nearer sample. ``nearest`` returns the
        closest sample (ties go to the earlier one). ``max_gap_ns`` bounds the distance to the
        used samples; in ``linear`` mode timestamps outside the table use the boundary sample if it
        is within ``max_gap_ns``, otherwise they are rejected.
        """
        t = int(timestamp_ns)
        ts = self.timestamps_ns
        i = int(np.searchsorted(ts, t, side="left"))

        if i < len(ts) and ts[i] == t:
            return self._prior_from_index(i)

        if mode == "nearest":
            if i == 0:
                j = 0
            elif i == len(ts):
                j = len(ts) - 1
            else:
                j = i - 1 if (t - ts[i - 1]) <= (ts[i] - t) else i
            if max_gap_ns is not None and abs(int(ts[j]) - t) > max_gap_ns:
                raise ValueError(f"Nearest pose to t={t} ns is {abs(int(ts[j]) - t)} ns away")
            return self._prior_from_index(j)

        if mode != "linear":
            raise ValueError(f"Unknown interpolation mode {mode!r}")
        if i == 0 or i == len(ts):
            # just outside the table: use the boundary sample if it is close enough
            j = 0 if i == 0 else len(ts) - 1
            if max_gap_ns is not None and abs(int(ts[j]) - t) <= max_gap_ns:
                return self._prior_from_index(j)
            raise ValueError(f"Timestamp {t} ns lies outside the pose table [{ts[0]}, {ts[-1]}]")

        t0, t1 = int(ts[i - 1]), int(ts[i])
        if max_gap_ns is not None and max(t - t0, t1 - t) > max_gap_ns:
            raise ValueError(f"Poses around t={t} ns are more than {max_gap_ns} ns apart")

        alpha = (t - t0) / (t1 - t0)
        T0, T1 = self.T_world_body[i - 1], self.T_world_body[i]
        R = T0[:3, :3] @ so3_exp(alpha * so3_log(T0[:3, :3].T @ T1[:3, :3]))
        p = (1.0 - alpha) * T0[:3, 3] + alpha * T1[:3, 3]

        covariance = None
        if self.covariance is not None:
            covariance = self.covariance[i - 1] if alpha <= 0.5 else self.covariance[i]
        return PosePrior(T_world_body=make_transform(R, p), covariance=covariance)

    def _prior_from_index(self, index: int) -> PosePrior:
        covariance = None if self.covariance is None else self.covariance[index]
        return PosePrior(T_world_body=self.T_world_body[index], covariance=covariance)


def image_pose_timestamps(image_timestamps_ns, *, time_offset_ns: int = 0, exposure_ns=None,
    exposure_mid: bool = False) -> np.ndarray:
    """Navigation-clock timestamps at which to evaluate the pose of each image.

    ``t_pose = t_image + time_offset`` and, if ``exposure_mid`` is set and the image timestamp
    marks the start of the exposure, ``+ exposure / 2``.
    """
    t = np.asarray(image_timestamps_ns, dtype=np.int64) + int(time_offset_ns)
    if exposure_mid:
        if exposure_ns is None:
            raise ValueError("exposure_mid requires exposure durations")
        t = t + np.asarray(exposure_ns, dtype=np.int64) // 2
    return t


def ins_accuracy_covariance(roll, pitch, yaw, std_roll, std_pitch, std_yaw, std_north, std_east, std_down) -> np.ndarray:
    """6x6 pose covariances in NED axes from per-axis INS accuracies (independent errors assumed)."""
    rotation = euler_zyx_rotation_covariance(roll, pitch, yaw, std_roll, std_pitch, std_yaw)
    position_std = np.stack(np.broadcast_arrays(std_north, std_east, std_down), axis=-1).astype(float)

    covariance = np.zeros((*rotation.shape[:-2], 6, 6))
    covariance[..., :3, :3] = rotation
    covariance[..., 3:, 3:] = np.einsum("...i,ij->...ij", position_std**2, np.eye(3))
    return covariance
