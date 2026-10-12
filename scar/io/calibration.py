"""Reading and writing camera intrinsics and camera-body extrinsics.

Supported inputs (JSON or YAML):

- SCAR: ``{camera: {model, width, height, intrinsics: [fx, fy, cx, cy], distortion}, extrinsics: {T_body_camera}}``;
  either section may also be a file of its own.
- LTC / pygeon: ``camera.json`` with ``{width, height, K, dist}`` and ``extrinsics.json`` with ``{T_bc, T_cb}``.
- Kalibr camchain: ``{cam0: {camera_model: pinhole, intrinsics, distortion_model: radtan, distortion_coeffs,
  resolution, T_cam_imu}}``. The IMU frame is taken as the body frame.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml

from scar.core.camera import CameraModel
from scar.core.se3 import invert
from scar.core.types import Extrinsics


def _read(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict):
        raise ValueError(f"'{path}' does not contain a mapping")  # noqa: TRY004 (file content, not argument type)
    return data


def _kalibr_camera_section(data: dict[str, Any], camera_key: str | None) -> dict[str, Any] | None:
    cameras = {key: value for key, value in data.items() if isinstance(value, dict) and "camera_model" in value}
    if not cameras:
        return None
    if camera_key is None:
        if len(cameras) > 1:
            raise ValueError(f"Kalibr file contains several cameras {sorted(cameras)}, choose one with camera_key")
        camera_key = next(iter(cameras))
    if camera_key not in cameras:
        raise KeyError(f"Camera '{camera_key}' not in Kalibr file (available: {sorted(cameras)})")
    return cameras[camera_key]


def camera_from_dict(data: dict[str, Any], camera_key: str | None = None) -> CameraModel:
    kalibr = _kalibr_camera_section(data, camera_key)
    if kalibr is not None:
        if kalibr["camera_model"] != "pinhole" or kalibr.get("distortion_model", "radtan") != "radtan":
            raise ValueError(
                f"Only Kalibr pinhole-radtan cameras are supported, got "
                f"{kalibr['camera_model']}-{kalibr.get('distortion_model')}"
            )
        fx, fy, cx, cy = kalibr["intrinsics"]
        width, height = kalibr["resolution"]
        return CameraModel(width=width, height=height, fx=fx, fy=fy, cx=cx, cy=cy,
            distortion=tuple(kalibr.get("distortion_coeffs", ())))

    data = data.get("camera", data)
    if "intrinsics" in data:
        fx, fy, cx, cy = data["intrinsics"]
    elif "K" in data:
        return CameraModel.from_matrix(data["K"], width=data["width"], height=data["height"],
            distortion=data.get("dist", data.get("distortion", ())))
    else:
        fx, fy, cx, cy = (data[key] for key in ("fx", "fy", "cx", "cy"))

    return CameraModel(width=data["width"], height=data["height"], fx=fx, fy=fy, cx=cx, cy=cy,
        distortion=tuple(data.get("distortion", data.get("dist", ()))), model=data.get("model", "pinhole-radtan"))


def extrinsics_from_dict(data: dict[str, Any], camera_key: str | None = None) -> Extrinsics:
    kalibr = _kalibr_camera_section(data, camera_key)
    if kalibr is not None:
        if "T_cam_imu" not in kalibr:
            raise ValueError("Kalibr camera section has no T_cam_imu")
        return Extrinsics(T_body_camera=invert(np.asarray(kalibr["T_cam_imu"], dtype=float)))

    data = data.get("extrinsics", data)
    for key in ("T_body_camera", "T_bc"):
        if key in data:
            return Extrinsics(T_body_camera=np.asarray(data[key], dtype=float))
    raise ValueError("No extrinsics found (expected T_body_camera, T_bc or a Kalibr T_cam_imu)")


def load_camera(path: str | Path, camera_key: str | None = None) -> CameraModel:
    return camera_from_dict(_read(path), camera_key)


def load_extrinsics(path: str | Path, camera_key: str | None = None) -> Extrinsics:
    return extrinsics_from_dict(_read(path), camera_key)


def camera_to_dict(camera: CameraModel) -> dict[str, Any]:
    return {
        "model": camera.model,
        "width": camera.width,
        "height": camera.height,
        "intrinsics": [camera.fx, camera.fy, camera.cx, camera.cy],
        "distortion": list(camera.distortion),
    }


def extrinsics_to_dict(extrinsics: Extrinsics) -> dict[str, Any]:
    return {"T_body_camera": extrinsics.T_body_camera.tolist()}


def save_calibration(path: str | Path, camera: CameraModel | None = None, extrinsics: Extrinsics | None = None,
    metadata: dict[str, Any] | None = None) -> None:
    """Write a SCAR calibration YAML file with the given sections."""
    data: dict[str, Any] = {}
    if camera is not None:
        data["camera"] = camera_to_dict(camera)
    if extrinsics is not None:
        data["extrinsics"] = extrinsics_to_dict(extrinsics)
    if metadata:
        data["metadata"] = metadata
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(data, file, sort_keys=False, default_flow_style=None)


def export_calibration(path: str | Path, camera: CameraModel, extrinsics: Extrinsics, format: str) -> list[Path]:
    """Write a calibration for other tools; returns the written files.

    ``kalibr``: camchain YAML (``T_cam_imu`` = inverse of ``T_body_camera``); ``opencv``: FileStorage
    YAML with ``camera_matrix``, ``distortion_coefficients`` and ``T_body_camera``; ``ltc``: the LTC
    ``camera.json`` and ``extrinsics.json`` (``T_bc``/``T_cb``) into the directory ``path``.
    """
    path = Path(path)
    T_bc = extrinsics.T_body_camera
    if format == "kalibr":
        data = {"cam0": {
            "camera_model": "pinhole",
            "intrinsics": [camera.fx, camera.fy, camera.cx, camera.cy],
            "distortion_model": "radtan",
            "distortion_coeffs": [float(c) for c in camera.dist[:4]],
            "resolution": [camera.width, camera.height],
            "T_cam_imu": invert(T_bc).tolist(),
        }}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(data, sort_keys=False, default_flow_style=None))
        return [path]
    if format == "opencv":
        import cv2

        path.parent.mkdir(parents=True, exist_ok=True)
        storage = cv2.FileStorage(str(path), cv2.FILE_STORAGE_WRITE)
        storage.write("image_width", camera.width)
        storage.write("image_height", camera.height)
        storage.write("camera_matrix", camera.K)
        storage.write("distortion_coefficients", camera.dist[: max(len(camera.distortion), 4)].reshape(1, -1))
        storage.write("T_body_camera", T_bc)
        storage.release()
        return [path]
    if format == "ltc":
        import json

        path.mkdir(parents=True, exist_ok=True)
        (path / "camera.json").write_text(json.dumps({"width": camera.width, "height": camera.height,
            "K": camera.K.tolist(), "dist": [float(c) for c in camera.dist[:4]]}, indent=4))
        (path / "extrinsics.json").write_text(json.dumps({"T_bc": T_bc.tolist(), "T_cb": invert(T_bc).tolist()},
            indent=4))
        return [path / "camera.json", path / "extrinsics.json"]
    raise ValueError(f"Unknown export format {format!r}; use kalibr, opencv or ltc")
