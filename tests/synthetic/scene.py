"""Synthetic orthophoto scene: aerial images rendered exactly from a textured flat terrain."""

from __future__ import annotations

import cv2
import numpy as np

from scar.core import CameraModel, Extrinsics, Frame, PosePrior, Sequence
from scar.core.se3 import make_transform
from scar.geodesy import HeightReference, WorldFrame
from scar.references.base import ReferenceView

GSD = 0.5  # orthophoto metres per pixel (2 px/m)
EXTENT = (1000.0, 1500.0)  # orthophoto covers x in [0, 1000), y in [0, 1500)
GROUND = 100.0
CAMERA = CameraModel(width=640, height=480, fx=520.0, fy=520.0, cx=320.0, cy=240.0, distortion=(-0.05, 0.01, 0.0, 0.0))
R_NADIR = np.diag([1.0, -1.0, -1.0])  # image x = east, image y = south, optical axis down


def texture(seed: int = 0) -> np.ndarray:
    """Multi-scale random texture (RGB, uint8) with corners SIFT can localize."""
    rng = np.random.default_rng(seed)
    height, width = int(EXTENT[1] / GSD), int(EXTENT[0] / GSD)
    image = np.zeros((height, width), dtype=np.float32)
    for scale, weight in ((64, 1.0), (16, 0.6), (4, 0.3)):
        noise = rng.normal(size=(height // scale + 2, width // scale + 2)).astype(np.float32)
        image += weight * cv2.resize(noise, (width, height), interpolation=cv2.INTER_CUBIC)
    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return np.dstack([image, image, (255 - image // 2)])


def ground_points(camera: CameraModel, T_world_camera: np.ndarray, pixels: np.ndarray) -> np.ndarray:
    """Intersect the rays of ``pixels`` (OpenCV convention) with the ground plane."""
    undistorted = camera.undistort_pixels(pixels)
    rays = np.column_stack([(undistorted[:, 0] - camera.cx) / camera.fx, (undistorted[:, 1] - camera.cy) / camera.fy,
        np.ones(len(pixels))]) @ T_world_camera[:3, :3].T
    t = (GROUND - T_world_camera[2, 3]) / rays[:, 2]
    return T_world_camera[:3, 3] + rays * t[:, None]


def render_aerial(ortho: np.ndarray, camera: CameraModel, T_world_camera: np.ndarray, noise: float = 2.0,
    seed: int = 0) -> np.ndarray:
    u, v = np.meshgrid(np.arange(camera.width, dtype=float), np.arange(camera.height, dtype=float))
    ground = ground_points(camera, T_world_camera, np.column_stack([u.ravel(), v.ravel()]))
    # continuous array index of the orthophoto (pixel centres at integers, as cv2.remap expects)
    col = (ground[:, 0] / GSD - 0.5).reshape(u.shape).astype(np.float32)
    row = ((EXTENT[1] - ground[:, 1]) / GSD - 0.5).reshape(u.shape).astype(np.float32)
    gray = cv2.cvtColor(ortho, cv2.COLOR_RGB2GRAY)
    image = cv2.remap(gray, col, row, cv2.INTER_LINEAR).astype(np.float32)
    image += np.random.default_rng(seed).normal(0.0, noise, image.shape)
    return np.clip(image, 0, 255).astype(np.uint8)


class OrthoReferenceProvider:
    """Axis-aligned orthophoto/DSM cutouts around the camera footprint (area pixel convention)."""

    def __init__(self, ortho: np.ndarray, margin_m: float = 40.0) -> None:
        self.ortho = ortho
        self.margin_m = margin_m

    def render(self, T_world_camera, camera, *, pixels_per_meter=None, timestamp_ns=None) -> ReferenceView:
        corners = np.array([[0, 0], [camera.width, 0], [camera.width, camera.height], [0, camera.height]], dtype=float)
        ground = ground_points(camera, T_world_camera, corners)
        c0 = max(int((ground[:, 0].min() - self.margin_m) / GSD), 0)
        c1 = min(int((ground[:, 0].max() + self.margin_m) / GSD), self.ortho.shape[1])
        r0 = max(int((EXTENT[1] - ground[:, 1].max() - self.margin_m) / GSD), 0)
        r1 = min(int((EXTENT[1] - ground[:, 1].min() + self.margin_m) / GSD), self.ortho.shape[0])
        image = self.ortho[r0:r1, c0:c1]
        pixel_to_world = np.array([[GSD, 0.0, c0 * GSD], [0.0, -GSD, EXTENT[1] - r0 * GSD], [0.0, 0.0, 1.0]])
        return ReferenceView(image=image, dsm=np.full(image.shape[:2], GROUND), pixel_to_world=pixel_to_world)


def flight(num_frames: int = 6, spacing_m: float = 25.0, altitude_m: float = 300.0) -> Sequence:
    frames = tuple(
        Frame(frame_id=i, timestamp_ns=i * 200_000_000, image_path=f"{i}",
            pose_prior=PosePrior(make_transform(R_NADIR, [500.0, 550.0 + spacing_m * i, GROUND + altitude_m])))
        for i in range(num_frames)
    )
    return Sequence(camera=CAMERA, extrinsics=Extrinsics(np.eye(4)), frames=frames,
        world_frame=WorldFrame(crs="EPSG:25832", height=HeightReference.ellipsoidal()))
