import cv2
import numpy as np
import pytest

from scar.core.camera import CameraModel
from scar.core.se3 import invert, make_transform, so3_exp

LTC_LIKE = {"width": 1600, "height": 1100, "fx": 1318.3, "fy": 1318.8, "cx": 801.1, "cy": 560.7}


@pytest.mark.parametrize("distortion", [(), (-0.1446, 0.1249, 1.7e-4, 1.8e-5), (-0.2, 0.1, 1e-3, -2e-3, 0.05)])
def test_projection_matches_opencv(distortion):
    camera = CameraModel(**LTC_LIKE, distortion=distortion)
    rng = np.random.default_rng(0)
    points_world = rng.uniform([-300, -200, -5], [300, 200, 40], size=(200, 3))
    T_world_camera = make_transform(so3_exp(np.array([np.pi, 0.02, -0.01])), [0.0, 0.0, 500.0])

    T_camera_world = invert(T_world_camera)
    rvec, _ = cv2.Rodrigues(T_camera_world[:3, :3])
    expected, _ = cv2.projectPoints(points_world, rvec, T_camera_world[:3, 3], camera.K, camera.dist)

    np.testing.assert_allclose(camera.project(points_world, T_world_camera), expected.reshape(-1, 2), atol=1e-8)


def test_undistort_inverts_distortion():
    camera = CameraModel(**LTC_LIKE, distortion=(-0.1446, 0.1249, 1.7e-4, 1.8e-5))
    ideal = np.random.default_rng(1).uniform([100, 100], [1500, 1000], size=(50, 2))
    normalized = (ideal - [camera.cx, camera.cy]) / [camera.fx, camera.fy]
    distorted = camera.project_camera_points(np.column_stack([normalized, np.ones(len(normalized))]))
    np.testing.assert_allclose(camera.undistort_pixels(distorted), ideal, atol=1e-4)


def test_from_matrix_roundtrip_and_validation():
    camera = CameraModel(**LTC_LIKE, distortion=(0.1, 0.0, 0.0, 0.0))
    assert CameraModel.from_matrix(camera.K, 1600, 1100, distortion=camera.distortion) == camera
    assert camera.dist.shape == (5,)

    skewed = camera.K.copy()
    skewed[0, 1] = 0.5
    with pytest.raises(ValueError):
        CameraModel.from_matrix(skewed, 1600, 1100)
    with pytest.raises(ValueError):
        CameraModel(**LTC_LIKE, distortion=(0.1, 0.2))
