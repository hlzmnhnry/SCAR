"""Create a synthetic flight over your "own" region (UTM 17N, USA) to try SCAR end to end.

Writes an orthophoto and a DSM (GeoTIFF), aerial images rendered from them, an INS log with WGS84
positions and body-to-NED quaternions, an initial calibration with a 1.5 deg extrinsic error and
the true calibration:

    python examples/custom_region_geotiff/make_demo_data.py examples/custom_region_geotiff/demo
    scar run examples/custom_region_geotiff/config.yaml
    python examples/custom_region_geotiff/make_demo_data.py --compare examples/custom_region_geotiff/demo

The terrain is flat, which keeps the image rendering exact and simple.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from pyproj import Transformer
from scipy.spatial.transform import Rotation

from scar.core import CameraModel, Extrinsics
from scar.core.se3 import euler_zyx_to_matrix, make_transform, rotation_angle, so3_exp
from scar.geodesy import HeightReference, WorldFrame, world_from_navigation_rotation
from scar.io import load_camera, load_extrinsics, save_calibration

CRS = "EPSG:32617"
X0, Y0 = 500_000.0, 4_200_000.0  # lower left corner of the orthophoto
EXTENT = (1500.0, 2200.0)
GSD = 0.5
GROUND = 200.0
TRUE_CAMERA = CameraModel(width=640, height=480, fx=520.0, fy=521.0, cx=322.0, cy=238.0,
    distortion=(-0.06, 0.012, 0.0004, -0.0003))
# camera looks down, image x to the right of the flight direction
TRUE_T_BODY_CAMERA = make_transform(np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]), [0.02, 0.01, 0.15])


def texture(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    height, width = int(EXTENT[1] / GSD), int(EXTENT[0] / GSD)
    image = np.zeros((height, width), dtype=np.float32)
    for scale, weight in ((96, 1.0), (24, 0.6), (6, 0.3)):
        noise = rng.normal(size=(height // scale + 2, width // scale + 2)).astype(np.float32)
        image += weight * cv2.resize(noise, (width, height), interpolation=cv2.INTER_CUBIC)
    gray = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return np.dstack([gray, (gray * 0.8 + 30).astype(np.uint8), 255 - gray // 2])


def write_geotiff(path: Path, data: np.ndarray, nodata=None) -> None:
    import rasterio
    from rasterio.transform import from_origin

    data = data[None] if data.ndim == 2 else np.moveaxis(data, -1, 0)
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", driver="GTiff", width=data.shape[2], height=data.shape[1], count=data.shape[0],
        dtype=data.dtype, crs=CRS, transform=from_origin(X0, Y0 + EXTENT[1], GSD, GSD), nodata=nodata) as out:
        out.write(data)


def render(ortho_gray: np.ndarray, T_world_camera: np.ndarray, rng) -> np.ndarray:
    camera = TRUE_CAMERA
    u, v = np.meshgrid(np.arange(camera.width, dtype=float), np.arange(camera.height, dtype=float))
    undistorted = camera.undistort_pixels(np.column_stack([u.ravel(), v.ravel()]))
    rays = np.column_stack([(undistorted[:, 0] - camera.cx) / camera.fx, (undistorted[:, 1] - camera.cy) / camera.fy,
        np.ones(len(undistorted))]) @ T_world_camera[:3, :3].T
    ground = T_world_camera[:3, 3] + rays * ((GROUND - T_world_camera[2, 3]) / rays[:, 2])[:, None]
    col = ((ground[:, 0] - X0) / GSD - 0.5).reshape(u.shape).astype(np.float32)
    row = ((Y0 + EXTENT[1] - ground[:, 1]) / GSD - 0.5).reshape(u.shape).astype(np.float32)
    image = cv2.remap(ortho_gray, col, row, cv2.INTER_LINEAR).astype(np.float32)
    return np.clip(image + rng.normal(0.0, 2.0, image.shape), 0, 255).astype(np.uint8)


def make_demo_data(directory: str | Path, seed: int = 0, num_images: int = 40) -> Path:
    directory = Path(directory)
    rng = np.random.default_rng(seed)
    ortho = texture(seed)
    write_geotiff(directory / "reference" / "orthophoto.tif", ortho)
    write_geotiff(directory / "reference" / "dsm.tif", np.full(ortho.shape[:2], GROUND, dtype=np.float32), nodata=-9999.0)

    # INS at 50 Hz: straight line north at 30 m/s, 400 m above ground, gentle attitude changes
    world = WorldFrame(crs=CRS, height=HeightReference.ellipsoidal())
    t = np.arange(0.0, num_images * 0.5 + 1.0, 0.02)
    x = X0 + 750.0 + 5.0 * np.sin(t / 3.0)
    y = Y0 + 350.0 + 30.0 * t
    z = np.full_like(t, GROUND + 400.0)
    roll, pitch = np.radians(1.5) * np.sin(t / 2.0), np.radians(1.0) * np.cos(t / 2.5)
    yaw = 0.02 * np.sin(t / 4.0)  # heading north; body frame forward-right-down
    R_ned_body = euler_zyx_to_matrix(roll, pitch, yaw)
    longitude, latitude = Transformer.from_crs(CRS, "EPSG:4979", always_xy=True).transform(x, y)
    R_world_body = world_from_navigation_rotation(latitude, longitude, world, "NED") @ R_ned_body
    quat = Rotation.from_matrix(R_ned_body).as_quat()
    with (directory / "ins.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["timestamp", "latitude", "longitude", "height", "qx", "qy", "qz", "qw"])
        for row in zip(t, latitude, longitude, z, *quat.T, strict=True):
            writer.writerow([f"{row[0]:.3f}", f"{row[1]:.10f}", f"{row[2]:.10f}", f"{row[3]:.4f}",
                *(f"{q:.12f}" for q in row[4:])])

    # images at 2 Hz, rendered with the true calibration
    gray = cv2.cvtColor(ortho, cv2.COLOR_RGB2GRAY)
    (directory / "images").mkdir(exist_ok=True)
    with (directory / "images.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["timestamp", "path"])
        for i in range(num_images):
            k = round((0.5 + 0.5 * i) / 0.02)
            T_world_body = make_transform(R_world_body[k], [x[k], y[k], z[k]])
            path = f"images/{i:04d}.png"
            cv2.imwrite(str(directory / path), render(gray, T_world_body @ TRUE_T_BODY_CAMERA, rng))
            writer.writerow([f"{t[k]:.3f}", path])

    save_calibration(directory / "calibration" / "truth.yaml", TRUE_CAMERA, Extrinsics(TRUE_T_BODY_CAMERA))
    initial = TRUE_T_BODY_CAMERA @ make_transform(so3_exp(np.radians([0.8, -0.6, 1.1])), [0.0, 0.0, 0.0])
    save_calibration(directory / "calibration" / "initial.yaml", TRUE_CAMERA, Extrinsics(initial))
    return directory


def compare(directory: str | Path, calibration: str | Path) -> dict:
    directory = Path(directory)
    truth_T = load_extrinsics(directory / "calibration" / "truth.yaml").T_body_camera
    initial_T = load_extrinsics(directory / "calibration" / "initial.yaml").T_body_camera
    result_T = load_extrinsics(calibration).T_body_camera
    camera, truth = load_camera(calibration), load_camera(directory / "calibration" / "truth.yaml")
    return {
        "initial_extrinsic_error_deg": float(np.degrees(rotation_angle(initial_T[:3, :3], truth_T[:3, :3]))),
        "extrinsic_error_deg": float(np.degrees(rotation_angle(result_T[:3, :3], truth_T[:3, :3]))),
        "focal_error_px": [camera.fx - truth.fx, camera.fy - truth.fy],
        "principal_point_error_px": [camera.cx - truth.cx, camera.cy - truth.cy],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("directory")
    parser.add_argument("--compare", action="store_true", help="compare runs/demo/calibration with the truth")
    parser.add_argument("--calibration", default=str(Path(__file__).parent / "runs" / "demo" / "calibration" / "calibration.yaml"))
    args = parser.parse_args()
    if args.compare:
        for key, value in compare(args.directory, args.calibration).items():
            print(f"{key}: {value}")
    else:
        print(f"wrote demo data to {make_demo_data(args.directory)}")


if __name__ == "__main__":
    main()
