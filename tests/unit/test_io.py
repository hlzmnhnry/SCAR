import json

import numpy as np
import pytest
import yaml

from scar.core import (
    CameraModel,
    Extrinsics,
    Frame,
    Landmarks,
    Observations,
    PosePrior,
    Sequence,
)
from scar.core.se3 import invert, make_transform, rot_z
from scar.geodesy import HeightReference, WorldFrame
from scar.io import (
    load_camera,
    load_extrinsics,
    load_sequence,
    save_calibration,
    save_sequence,
)

T_BC = make_transform(rot_z(np.pi / 2), [0.016, 0.005, 0.18])
CAMERA = CameraModel(width=1600, height=1100, fx=1318.3, fy=1318.8, cx=801.1, cy=560.7,
    distortion=(-0.1446, 0.1249, 1.7e-4, 1.8e-5))


def test_ltc_calibration_files(tmp_path):
    (tmp_path / "camera.json").write_text(json.dumps(
        {"width": 1600, "height": 1100, "K": CAMERA.K.tolist(), "dist": list(CAMERA.distortion)}))
    (tmp_path / "extrinsics.json").write_text(json.dumps({"T_bc": T_BC.tolist(), "T_cb": invert(T_BC).tolist()}))

    assert load_camera(tmp_path / "camera.json") == CAMERA
    np.testing.assert_allclose(load_extrinsics(tmp_path / "extrinsics.json").T_body_camera, T_BC)


def test_kalibr_camchain(tmp_path):
    camchain = {"cam0": {
        "camera_model": "pinhole", "intrinsics": [CAMERA.fx, CAMERA.fy, CAMERA.cx, CAMERA.cy],
        "distortion_model": "radtan", "distortion_coeffs": list(CAMERA.distortion),
        "resolution": [1600, 1100], "T_cam_imu": invert(T_BC).tolist(),
    }}
    path = tmp_path / "camchain.yaml"
    path.write_text(yaml.safe_dump(camchain))

    assert load_camera(path) == CAMERA
    np.testing.assert_allclose(load_extrinsics(path).T_body_camera, T_BC, atol=1e-15)

    camchain["cam0"]["distortion_model"] = "equidistant"
    path.write_text(yaml.safe_dump(camchain))
    with pytest.raises(ValueError):
        load_camera(path)


def test_scar_calibration_roundtrip(tmp_path):
    path = tmp_path / "calibration.yaml"
    save_calibration(path, camera=CAMERA, extrinsics=Extrinsics(T_BC))
    assert load_camera(path) == CAMERA
    np.testing.assert_array_equal(load_extrinsics(path).T_body_camera, T_BC)


def make_sequence():
    world = WorldFrame(crs="EPSG:25832", height=HeightReference.ellipsoidal(), origin=(374000.0, 5605000.0, 0.0))
    frames = tuple(
        Frame(frame_id=i, timestamp_ns=1_666_183_007_634_260_000 + i * 200_000_000, image_path=f"{i}.png",
            pose_prior=PosePrior(make_transform(rot_z(0.1 * i), [i, 2.0 * i, 500.0]),
                covariance=np.eye(6) * 1e-4 if i else None))
        for i in range(3)
    )
    landmarks = Landmarks(ids=[4, 7], xyz=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], covariance=np.stack([np.eye(3)] * 2))
    observations = Observations(frame_ids=[0, 1, 2], landmark_ids=[4, 4, 7], uv=[[1.5, 2.5], [3.5, 4.5], [5.5, 6.5]])
    return Sequence(camera=CAMERA, extrinsics=Extrinsics(T_BC), frames=frames, world_frame=world,
        landmarks=landmarks, observations=observations, metadata={"source": "test"})


def test_sequence_roundtrip_is_exact(tmp_path):
    sequence = make_sequence()
    save_sequence(sequence, tmp_path / "sequence.json")
    loaded = load_sequence(tmp_path / "sequence.json")

    assert loaded.camera == sequence.camera
    assert loaded.world_frame == sequence.world_frame
    assert [f.timestamp_ns for f in loaded.frames] == [f.timestamp_ns for f in sequence.frames]
    for a, b in zip(loaded.frames, sequence.frames, strict=True):
        np.testing.assert_array_equal(a.pose_prior.T_world_body, b.pose_prior.T_world_body)
    assert loaded.frames[0].pose_prior.covariance is None
    np.testing.assert_array_equal(loaded.landmarks.xyz, sequence.landmarks.xyz)
    np.testing.assert_array_equal(loaded.observations.uv, sequence.observations.uv)
    assert loaded.metadata == {"source": "test"}


def test_sequence_rejects_dangling_references():
    sequence = make_sequence()
    with pytest.raises(ValueError, match="unknown landmark"):
        sequence.replace(observations=Observations(frame_ids=[0], landmark_ids=[99], uv=[[0.0, 0.0]]))
    with pytest.raises(ValueError, match="unknown frame"):
        sequence.replace(observations=Observations(frame_ids=[9], landmark_ids=[4], uv=[[0.0, 0.0]]))


def test_file_list_image_source(tmp_path):
    import cv2

    from scar.sources import FileListImageSource

    (tmp_path / "images").mkdir()
    cv2.imwrite(str(tmp_path / "images" / "a.png"), np.full((4, 6), 7, dtype=np.uint8))
    (tmp_path / "frames.csv").write_text("timestamp,path\n1776855480.170,images/a.png\n1776855480.070,images/b.png\n")

    source = FileListImageSource.from_csv(tmp_path / "frames.csv", time_unit="s")
    np.testing.assert_array_equal(source.timestamps_ns, [1776855480070000000, 1776855480170000000])
    assert source.paths == ("images/b.png", "images/a.png")
    assert source.read("images/a.png").shape == (4, 6)
    with pytest.raises(FileNotFoundError):
        source.read("images/b.png")
