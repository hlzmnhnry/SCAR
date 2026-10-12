"""Versioned JSON serialization of :class:`~scar.core.types.Sequence`."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from scar.core.types import (
    SEQUENCE_SCHEMA_VERSION,
    Extrinsics,
    Frame,
    Landmarks,
    Observations,
    PosePrior,
    Sequence,
)
from scar.geodesy.world_frame import WorldFrame

from .calibration import camera_from_dict, camera_to_dict


def sequence_to_dict(sequence: Sequence) -> dict[str, Any]:
    landmarks, observations = sequence.landmarks, sequence.observations
    return {
        "schema_version": SEQUENCE_SCHEMA_VERSION,
        "world_frame": sequence.world_frame.to_dict(),
        "camera": camera_to_dict(sequence.camera),
        "extrinsics": {"T_body_camera": sequence.extrinsics.T_body_camera.tolist()},
        "frames": [
            {
                "frame_id": frame.frame_id,
                "timestamp_ns": frame.timestamp_ns,
                "image_path": frame.image_path,
                "T_world_body": frame.pose_prior.T_world_body.tolist(),
                "covariance": None if frame.pose_prior.covariance is None else frame.pose_prior.covariance.tolist(),
                "metadata": frame.metadata,
            }
            for frame in sequence.frames
        ],
        # anchors are stored column-wise, sequences easily contain 10^5 observations
        "landmarks": {
            "ids": landmarks.ids.tolist(),
            "xyz": landmarks.xyz.tolist(),
            "covariance": None if landmarks.covariance is None else landmarks.covariance.tolist(),
        },
        "observations": {
            "frame_ids": observations.frame_ids.tolist(),
            "landmark_ids": observations.landmark_ids.tolist(),
            "uv": observations.uv.tolist(),
        },
        "metadata": sequence.metadata,
    }


def sequence_from_dict(data: dict[str, Any]) -> Sequence:
    version = data.get("schema_version")
    if version != SEQUENCE_SCHEMA_VERSION:
        raise ValueError(f"Unsupported sequence schema version {version!r}, expected {SEQUENCE_SCHEMA_VERSION!r}")

    landmarks, observations = data["landmarks"], data["observations"]
    return Sequence(
        camera=camera_from_dict(data["camera"]),
        extrinsics=Extrinsics(T_body_camera=data["extrinsics"]["T_body_camera"]),
        world_frame=WorldFrame.from_dict(data["world_frame"]),
        frames=tuple(
            Frame(
                frame_id=frame["frame_id"],
                timestamp_ns=frame["timestamp_ns"],
                image_path=frame["image_path"],
                pose_prior=PosePrior(T_world_body=frame["T_world_body"], covariance=frame.get("covariance")),
                metadata=frame.get("metadata", {}),
            )
            for frame in data["frames"]
        ),
        landmarks=Landmarks(
            ids=np.asarray(landmarks["ids"], dtype=np.int64),
            xyz=np.asarray(landmarks["xyz"], dtype=float).reshape(-1, 3),
            covariance=landmarks.get("covariance"),
        ),
        observations=Observations(
            frame_ids=np.asarray(observations["frame_ids"], dtype=np.int64),
            landmark_ids=np.asarray(observations["landmark_ids"], dtype=np.int64),
            uv=np.asarray(observations["uv"], dtype=float).reshape(-1, 2),
        ),
        metadata=data.get("metadata", {}),
    )


def save_sequence(sequence: Sequence, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(sequence_to_dict(sequence), file)


def load_sequence(path: str | Path) -> Sequence:
    with Path(path).open(encoding="utf-8") as file:
        return sequence_from_dict(json.load(file))
