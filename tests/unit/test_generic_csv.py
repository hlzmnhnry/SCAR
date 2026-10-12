import csv

import numpy as np
import pytest
from pyproj import Transformer

from scar.core.se3 import R_ENU_NED, euler_zyx_to_matrix, rotation_angle
from scar.datasets.generic_csv import PoseCsvSpec, read_pose_csv
from scar.geodesy import HeightReference, WorldFrame
from scar.sources import FileListImageSource

LAT, LON = 50.5921, 7.2240


def write_csv(path, header, rows):
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(rows)
    return path


def world(**kwargs):
    return WorldFrame(crs="EPSG:25832", height=HeightReference.ellipsoidal(), **kwargs)


def test_geodetic_quaternion_ned(tmp_path):
    path = write_csv(tmp_path / "ins.csv", ["t", "lat", "lon", "h", "qx", "qy", "qz", "qw"],
        [["0.0", LAT, LON, 300.0, 0, 0, 0, 1], ["0.1", LAT + 1e-5, LON, 301.0, 0, 0, 0, 1]])
    spec = PoseCsvSpec(timestamp_column="t", time_unit="s", position_columns=("lat", "lon", "h"),
        orientation_columns=("qx", "qy", "qz", "qw"))
    table = read_pose_csv(path, spec, world(apply_meridian_convergence=False))

    easting, northing = Transformer.from_crs("EPSG:4979", "EPSG:25832", always_xy=True).transform(LON, LAT)
    np.testing.assert_allclose(table.T_world_body[0, :3, 3], [easting, northing, 300.0], atol=1e-6)
    np.testing.assert_allclose(table.T_world_body[0, :3, :3], R_ENU_NED)
    np.testing.assert_array_equal(table.timestamps_ns, [0, 100_000_000])


def test_projected_euler_enu_in_other_crs(tmp_path):
    x, y = Transformer.from_crs("EPSG:4326", "EPSG:32632", always_xy=True).transform(LON, LAT)
    path = write_csv(tmp_path / "ins.csv", ["timestamp", "x", "y", "z", "roll", "pitch", "yaw"],
        [[1, x, y, 250.0, 1.0, -2.0, 30.0], [2, x + 1, y, 250.0, 1.0, -2.0, 30.0]])
    spec = PoseCsvSpec(time_unit="s", position="projected", crs="EPSG:32632", position_columns=("x", "y", "z"),
        orientation="euler", orientation_columns=("roll", "pitch", "yaw"), degrees=True, navigation_frame="ENU")
    table = read_pose_csv(path, spec, world(apply_meridian_convergence=False))
    # WGS84 and ETRS89 UTM coordinates coincide under the registered null transformation
    np.testing.assert_allclose(table.T_world_body[0, :3, 3], [x, y, 250.0], atol=1e-3)
    expected = euler_zyx_to_matrix(np.radians(1.0), np.radians(-2.0), np.radians(30.0))
    assert rotation_angle(table.T_world_body[0, :3, :3], expected) < 1e-9


def test_accuracy_columns_become_covariances(tmp_path):
    path = write_csv(tmp_path / "ins.csv", ["t", "lat", "lon", "h", "qx", "qy", "qz", "qw", "sn", "se", "sd", "sr", "sp", "sy"],
        [[0, LAT, LON, 300.0, 0, 0, 0, 1, 0.1, 0.2, 0.3, 0.01, 0.02, 0.03]])
    spec = PoseCsvSpec(timestamp_column="t", position_columns=("lat", "lon", "h"),
        orientation_columns=("qx", "qy", "qz", "qw"), position_std_columns=("sn", "se", "sd"),
        rotation_std_columns=("sr", "sp", "sy"))
    table = read_pose_csv(path, spec, world(apply_meridian_convergence=False))
    # NED -> ENU: north/east swap, level attitude: roll/pitch/yaw map onto x/y/z of the navigation frame
    np.testing.assert_allclose(np.sqrt(np.diag(table.covariance[0]))[3:], [0.2, 0.1, 0.3])
    np.testing.assert_allclose(np.sqrt(np.diag(table.covariance[0]))[:3], [0.02, 0.01, 0.03])


def test_zero_accuracies_are_rejected(tmp_path):
    # as in early internal LTC exports, where position_accuracy.* is 0 in every row
    path = write_csv(tmp_path / "ins.csv", ["t", "lat", "lon", "h", "qx", "qy", "qz", "qw", "sn", "se", "sd", "sr", "sp", "sy"],
        [[0, LAT, LON, 300.0, 0, 0, 0, 1, 0.0, 0.0, 0.0, 0.01, 0.02, 0.03]])
    spec = PoseCsvSpec(timestamp_column="t", position_columns=("lat", "lon", "h"),
        orientation_columns=("qx", "qy", "qz", "qw"), position_std_columns=("sn", "se", "sd"),
        rotation_std_columns=("sr", "sp", "sy"))
    with pytest.raises(ValueError, match="standard deviations"):
        read_pose_csv(path, spec, world())


def test_missing_column_is_reported(tmp_path):
    path = write_csv(tmp_path / "ins.csv", ["timestamp", "lat"], [[0, LAT]])
    with pytest.raises(KeyError, match="latitude"):
        read_pose_csv(path, PoseCsvSpec(), world())


def test_image_selection_on_a_time_grid(tmp_path):
    for name in ["10.000", "10.005", "10.240", "10.250", "10.495", "10.765", "11.000"]:
        (tmp_path / f"{name}.png").touch()
    source = FileListImageSource.from_directory(tmp_path)
    chosen = source.select(interval_ns=250_000_000, tolerance_ns=10_000_000)
    assert [source.paths[i] for i in chosen] == ["10.000.png", "10.250.png", "10.495.png", "11.000.png"]
    assert len(source.select(start_ns=10_200_000_000, step=2)) == 3
