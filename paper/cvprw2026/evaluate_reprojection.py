"""Reprojection error of fixed anchors under different calibrations (CVPRW 2026 metric).

Anchors (legacy ``PT_``/``WP_`` files of a test sequence) are projected with INS camera poses
``T_world_body @ T_body_camera`` and each calibration's intrinsics; reported is the median and
MAD of all observation errors. ``float32`` reproduces the legacy evaluation, which passed world
points to OpenCV as float32 (0.5 m resolution in UTM northing).

    pixi run python paper/cvprw2026/evaluate_reprojection.py 2022-10-19-14-14-59 \
        output/2022-10-19-14-14-59/geofeatures/PT_..._test.json \
        kalibr=config kalibr_guess golden=tests/regression/golden/ltc_2022-10-19-14-14-59_train \
        scar=output/scar/2022-10-19-14-14-59/train_legacy/calibration_cvprw2026/calibration.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from scar.core.camera import CameraModel
from scar.io import load_camera, load_extrinsics, load_sequence
from scar.io.legacy import LegacyAnchors, legacy_ltc_body_poses, load_legacy_anchors, read_legacy_ins

REPO = Path(__file__).resolve().parents[2]


def load_calibration(spec: str) -> tuple[CameraModel, np.ndarray]:
    if spec == "kalibr_guess":
        return (load_camera(REPO / "paper" / "cvprw2026" / "calibration" / "kalibr_camera.json"),
            load_extrinsics(REPO / "paper" / "cvprw2026" / "calibration" / "kalibr_extrinsics.json").T_body_camera)
    path = Path(spec)
    if path.is_dir():  # LTC calibration folder (camera.json + extrinsics.json)
        return load_camera(path / "camera.json"), load_extrinsics(path / "extrinsics.json").T_body_camera
    if path.with_suffix(".json").exists() and path.with_suffix(".npz").exists():  # golden reference
        golden = json.loads(path.with_suffix(".json").read_text())
        c = golden["stages"]["full"]["camera_calibration"]
        camera = CameraModel(1600, 1100, c["fx"], c["fy"], c["cx"], c["cy"], (c["k1"], c["k2"], c["p1"], c["p2"]))
        return camera, np.asarray(golden["T_ins_cam"])
    return load_camera(path), load_extrinsics(path).T_body_camera


def load_test_anchors(path: Path) -> LegacyAnchors:
    """Legacy PT_*.json (with its WP_ file) or a scar anchors.json."""
    if path.name.startswith("PT_"):
        return load_legacy_anchors(path, str(path).replace("/PT_", "/WP_"))
    sequence = load_sequence(path)
    timestamp_ns = {frame.frame_id: frame.timestamp_ns for frame in sequence.frames}
    return LegacyAnchors(landmarks=sequence.landmarks,
        observation_timestamps_ns=np.asarray([timestamp_ns[int(i)] for i in sequence.observations.frame_ids]),
        observation_landmark_ids=sequence.observations.landmark_ids, observation_uv=sequence.observations.uv)


def errors_for(camera: CameraModel, T_body_camera: np.ndarray, anchors, body_poses: dict, float32: bool) -> np.ndarray:
    xyz = {int(i): p for i, p in zip(anchors.landmarks.ids, anchors.landmarks.xyz, strict=True)}
    errors = []
    for t in np.unique(anchors.observation_timestamps_us):
        rows = np.flatnonzero(anchors.observation_timestamps_us == t)
        points = np.asarray([xyz[int(i)] for i in anchors.observation_landmark_ids[rows]])
        T_camera_world = np.linalg.inv(body_poses[int(t)] @ T_body_camera)
        if float32:
            rvec, _ = cv2.Rodrigues(T_camera_world[:3, :3])
            projected = cv2.projectPoints(points.astype(np.float32), rvec, T_camera_world[:3, 3], camera.K,
                camera.dist[:4])[0].reshape(-1, 2)
        else:
            projected = camera.project_camera_points(points @ T_camera_world[:3, :3].T + T_camera_world[:3, 3])
        errors.append(np.linalg.norm(projected - anchors.observation_uv[rows], axis=1))
    return np.concatenate(errors)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("campaign")
    parser.add_argument("anchors", type=Path, help="legacy PT_*.json or scar anchors.json of a test sequence")
    parser.add_argument("calibrations", nargs="+", help="name=path (calibration.yaml, golden prefix or kalibr_guess)")
    parser.add_argument("--data", type=Path, default=REPO / "data")
    args = parser.parse_args()

    anchors = load_test_anchors(args.anchors)
    timestamps = anchors.image_timestamps_us
    poses = dict(zip(timestamps.tolist(), legacy_ltc_body_poses(read_legacy_ins(args.data / args.campaign / "ins.csv"),
        timestamps), strict=True))

    print(f"{len(anchors.landmarks)} anchors, {len(anchors.observation_uv)} observations in {len(timestamps)} frames")
    for spec in args.calibrations:
        name, path = spec.split("=", 1)
        camera, T = load_calibration(path)
        for float32 in (False, True):
            e = errors_for(camera, T, anchors, poses, float32)
            median = np.median(e)
            print(f"{name:12s} {'float32' if float32 else 'float64'}: median {median:6.3f} px, "
                  f"MAD {np.median(np.abs(e - median)):6.3f} px, mean {e.mean():7.3f} px")


if __name__ == "__main__":
    main()
