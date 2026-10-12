"""The LTC ("Leaving the City") dataset: https://hlzmnhnry.github.io/publications/ltc.

A campaign folder contains ``camera.csv`` (``index,timestamp,exposure``; microseconds),
``ins.csv`` (SBG INS solution with SAPOS RTK corrections, i.e. ETRS89 latitude/longitude,
``altitude`` above SBG's geoid with ``undulation``, body-to-NED quaternions) and the images as
``images/<index>.png``.
"""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np

from scar.core.camera import CameraModel
from scar.core.types import Extrinsics, Frame, PosePrior, Sequence
from scar.geodesy.world_frame import HeightReference, WorldFrame
from scar.io.legacy import legacy_ltc_body_poses, read_legacy_ins
from scar.sources.poses import PoseTable

LTC_CRS = "EPSG:25832"


def ltc_world_frame(geoid_grid: str | None = None, apply_meridian_convergence: bool = True) -> WorldFrame:
    """ETRS89 / UTM 32N with DHHN2016 heights (GCG2016 quasigeoid), like the German reference data."""
    return WorldFrame(crs=LTC_CRS, height=HeightReference.geoid(geoid_grid or "GCG2016", "DHHN2016"),
        apply_meridian_convergence=apply_meridian_convergence)


def select_images(camera_csv: str | Path, start_us: int | None = None, end_us: int | None = None,
    image_step: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Image indices and timestamps [us] in ``[start_us, end_us]``, every ``image_step``-th image."""
    with Path(camera_csv).open(newline="") as file:
        rows = [(int(row["index"]), int(row["timestamp"])) for row in csv.DictReader(file)]
    rows.sort(key=lambda row: row[1])
    indices = np.asarray([row[0] for row in rows], dtype=np.int64)
    timestamps = np.asarray([row[1] for row in rows], dtype=np.int64)

    first = 0 if start_us is None else int(np.searchsorted(timestamps, start_us, side="left"))
    last = len(timestamps) if end_us is None else int(np.searchsorted(timestamps, end_us, side="right"))
    selection = np.arange(first, last, max(1, int(image_step)))
    return indices[selection], timestamps[selection]


def read_ins_pose_table(ins_csv: str | Path, world: WorldFrame, height: str = "undulation") -> PoseTable:
    """INS poses in the world frame, read in float64.

    ``height``: ``undulation`` treats ``altitude + undulation`` as ellipsoidal height (SBG output
    relative to its geoid model), ``world`` uses ``altitude`` unchanged as world height (legacy).
    """
    with Path(ins_csv).open(newline="") as file:
        rows = list(csv.DictReader(file))

    def column(name, dtype=np.float64):
        return np.asarray([row[name] for row in rows], dtype=dtype)

    reference = {"undulation": HeightReference.undulation("SBG"), "world": HeightReference.world()}[height]
    quat = np.column_stack([column("quat.x"), column("quat.y"), column("quat.z"), column("quat.w")])
    return PoseTable.from_geodetic(column("timestamp", np.int64) * 1000, column("latitude"), column("longitude"),
        column("altitude"), quat, world=world, height_reference=reference, source_crs="EPSG:4937",
        navigation_frame="NED", undulation=column("undulation"))


def ltc_sequence(campaign_directory: str | Path, camera: CameraModel, extrinsics: Extrinsics, *,
    start_us: int | None = None, end_us: int | None = None, image_step: int = 1, poses: str = "legacy",
    world_frame: WorldFrame | None = None, interpolation: str = "nearest", height: str = "undulation") -> Sequence:
    """Frames of an LTC campaign with INS pose priors.

    ``poses='legacy'`` reproduces the CVPRW 2026 pose computation exactly (see
    :func:`scar.io.legacy.legacy_ltc_body_poses`); ``poses='geodetic'`` converts the INS data with
    :mod:`scar.geodesy` into ``world_frame`` (float64, optional meridian convergence, explicit heights).
    """
    campaign_directory = Path(campaign_directory)
    world_frame = world_frame or ltc_world_frame()
    indices, timestamps = select_images(campaign_directory / "camera.csv", start_us, end_us, image_step)

    if poses == "legacy":
        body_poses = legacy_ltc_body_poses(read_legacy_ins(campaign_directory / "ins.csv"), timestamps)
        priors = [PosePrior(T) for T in body_poses]
    elif poses == "geodetic":
        table = read_ins_pose_table(campaign_directory / "ins.csv", world_frame, height)
        priors = [table.prior_at(int(t) * 1000, mode=interpolation, max_gap_ns=50_000_000) for t in timestamps]
    else:
        raise ValueError(f"Unknown pose mode {poses!r}")

    frames = tuple(
        Frame(frame_id=i, timestamp_ns=int(t) * 1000, image_path=f"{index}.png", pose_prior=prior,
            metadata={"image_index": int(index)})
        for i, (index, t, prior) in enumerate(zip(indices, timestamps, priors, strict=True))
    )
    return Sequence(camera=camera, extrinsics=extrinsics, frames=frames, world_frame=world_frame,
        metadata={"dataset": "ltc", "campaign": campaign_directory.name, "poses": poses})


def image_reader(image_directory: str | Path):
    """Reader for frames whose ``image_path`` is relative to ``image_directory``."""
    image_directory = Path(image_directory)

    def read(frame: Frame) -> np.ndarray:
        path = image_directory / frame.image_path
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Could not read image '{path}'")
        return image

    return read
