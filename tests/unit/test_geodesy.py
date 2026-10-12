import numpy as np
import pytest
from pyproj import Transformer

from scar.core.se3 import R_ENU_NED
from scar.geodesy import (
    HeightReference,
    WorldFrame,
    convert_height,
    describe_horizontal_transformation,
    geodetic_to_world,
    geoid_undulation,
    meridian_convergence,
    orientation_to_world,
)

# near the LTC airfield (Bad Breisig), UTM zone 32N
LAT, LON = 50.5921, 7.2240


def write_gtx(path, lat0, lon0, dlat, dlon, values):
    """Minimal NOAA .gtx vertical grid (big endian): header, then rows from south to north."""
    values = np.asarray(values, dtype=">f4")
    rows, cols = values.shape
    with open(path, "wb") as file:
        file.write(np.array([lat0, lon0, dlat, dlon], dtype=">f8").tobytes())
        file.write(np.array([rows, cols], dtype=">i4").tobytes())
        file.write(values.tobytes())


@pytest.fixture
def linear_geoid(tmp_path):
    """Geoid with N = 40 + 2 * (lat - 50) + 1 * (lon - 7) over 50..51 N, 7..8 E."""
    lats = np.linspace(50.0, 51.0, 11)
    lons = np.linspace(7.0, 8.0, 11)
    values = 40.0 + 2.0 * (lats[:, None] - 50.0) + 1.0 * (lons[None, :] - 7.0)
    path = tmp_path / "linear_geoid.gtx"
    write_gtx(path, 50.0, 7.0, 0.1, 0.1, values)
    return str(path)


def world(height=None, **kwargs):
    return WorldFrame(crs="EPSG:25832", height=height or HeightReference.ellipsoidal(), **kwargs)


def test_horizontal_conversion_matches_pyproj():
    reference = Transformer.from_crs("EPSG:4258", "EPSG:25832", always_xy=True).transform(LON, LAT)
    xyz = geodetic_to_world(LAT, LON, 300.0, world=world(), height_reference=HeightReference.ellipsoidal())
    np.testing.assert_allclose(xyz[0], [*reference, 300.0], atol=1e-6)


def test_origin_is_subtracted():
    frame = world(origin=(374000.0, 5605000.0, 100.0))
    absolute = geodetic_to_world(LAT, LON, 300.0, world=world(), height_reference=HeightReference.ellipsoidal())
    local = geodetic_to_world(LAT, LON, 300.0, world=frame, height_reference=HeightReference.ellipsoidal())
    np.testing.assert_allclose(frame.to_absolute(local), absolute, atol=1e-9)


def test_float32_latitudes_are_not_used():
    # float64 input must not be truncated anywhere on the way (0.42 m northing steps otherwise)
    lat = LAT + 1e-7  # ~1 cm north
    a = geodetic_to_world(LAT, LON, 0.0, world=world(), height_reference=HeightReference.ellipsoidal())
    b = geodetic_to_world(lat, LON, 0.0, world=world(), height_reference=HeightReference.ellipsoidal())
    assert 0.005 < b[0, 1] - a[0, 1] < 0.02


def test_transformation_is_described():
    info = describe_horizontal_transformation("EPSG:4937", world())
    assert "UTM zone 32N" in info["description"]


def test_geoid_undulation_is_interpolated_bilinearly(linear_geoid):
    N = geoid_undulation(np.array([50.25, 50.55]), np.array([7.35, 7.75]), linear_geoid)
    np.testing.assert_allclose(N, [40.0 + 0.5 + 0.35, 40.0 + 1.1 + 0.75], atol=1e-4)


def test_missing_or_non_covering_geoid_fails_loudly(linear_geoid, tmp_path):
    with pytest.raises(ValueError, match="does not cover"):
        geoid_undulation(np.array([52.0]), np.array([7.5]), linear_geoid)
    with pytest.raises(ValueError, match="could not be loaded"):
        geoid_undulation(np.array([50.5]), np.array([7.5]), str(tmp_path / "does_not_exist.gtx"))


def test_height_conversions(linear_geoid):
    geoid = HeightReference.geoid(linear_geoid, name="test")
    N = 40.0 + 2.0 * (LAT - 50.0) + 1.0 * (LON - 7.0)

    # normal height -> ellipsoidal world
    h = convert_height(LAT, LON, np.array([100.0]), geoid, world())
    np.testing.assert_allclose(h, [100.0 + N], atol=1e-4)

    # ellipsoidal -> geoid world, and the identity for equal references
    H = convert_height(LAT, LON, np.array([100.0 + N]), HeightReference.ellipsoidal(), world(geoid))
    np.testing.assert_allclose(H, [100.0], atol=1e-4)
    np.testing.assert_array_equal(convert_height(LAT, LON, np.array([5.0]), geoid, world(geoid)), [5.0])

    # per-sample undulation (e.g. SBG INS output) against a geoid world frame
    H = convert_height(LAT, LON, np.array([100.0]), HeightReference.undulation(), world(geoid), undulation=np.array([47.8]))
    np.testing.assert_allclose(H, [100.0 + 47.8 - N], atol=1e-4)

    # 'world' heights are passed through unchanged
    np.testing.assert_array_equal(convert_height(LAT, LON, np.array([7.0]), HeightReference.world(), world(geoid)), [7.0])


def test_meridian_convergence_rotates_true_north_onto_its_grid_direction():
    frame = world()
    gamma = meridian_convergence(LAT, LON, frame)
    assert np.degrees(gamma) == pytest.approx(-1.3724, abs=1e-3)

    # direction of true north in grid coordinates by finite differences
    a = geodetic_to_world(LAT, LON, 0.0, world=frame, height_reference=HeightReference.ellipsoidal())[0]
    b = geodetic_to_world(LAT + 1e-5, LON, 0.0, world=frame, height_reference=HeightReference.ellipsoidal())[0]
    north_grid = (b - a) / np.linalg.norm(b - a)

    # a body looking north (NED identity) must have its x axis along that direction
    R_world_body = orientation_to_world(np.eye(3)[None], LAT, LON, frame, navigation_frame="NED")[0]
    np.testing.assert_allclose(R_world_body[:, 0], north_grid, atol=1e-7)


def test_convergence_is_zero_on_central_meridian():
    assert abs(meridian_convergence(LAT, 9.0, world())) < 1e-12


def test_convergence_can_be_disabled_for_legacy_reproduction():
    R = orientation_to_world(np.eye(3)[None], LAT, LON, world(apply_meridian_convergence=False))[0]
    np.testing.assert_allclose(R, R_ENU_NED)


def test_world_frame_validation_and_roundtrip(linear_geoid):
    frame = world(HeightReference.geoid(linear_geoid, "test"), origin=(1.0, 2.0, 3.0))
    assert WorldFrame.from_dict(frame.to_dict()) == frame
    with pytest.raises(ValueError):
        WorldFrame(crs="EPSG:4326", height=HeightReference.ellipsoidal())
    with pytest.raises(ValueError):
        WorldFrame(crs="EPSG:25832", height=HeightReference.world())
