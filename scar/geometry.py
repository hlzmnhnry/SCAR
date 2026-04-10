import numpy as np

from .utilities import round_if_close_array

def round_to_step(value, step: float = 0.1, offset: float = 0.05, mode: str = "round"):
    
    if mode == "round":
        return np.round((value - offset) / step) * step + offset
    elif mode == "floor":
        return np.floor((value - offset) / step) * step + offset
    elif mode == "ceil":
        return np.ceil((value - offset) / step) * step + offset
    else:
        raise ValueError(f"Unknown mode: {mode}, must be 'round', 'floor', or 'ceil'")

def grid_pixel_to_utm(pixel_grid: np.ndarray, grid_offset_utm_x: float, grid_offset_utm_y: float,
    grid_height_pixels: int, resolution: float, convert_to_continuous_space: bool = False):
    
    assert pixel_grid.ndim == 1, "'pixel_grid' should be a simple pixel"
    assert 0 <= pixel_grid[1] < grid_height_pixels, "Pixel y value out of range"
    assert grid_offset_utm_x > 0 and grid_offset_utm_y, "Grid offset should be positive"
    
    if convert_to_continuous_space:
        # pixel coordinate/index (0, 0) in discrete space is (0.5, 0.5) in cont. pixel space
        pixel_grid = pixel_grid.astype(float) + 0.5
    
    utm_x = grid_offset_utm_x + ((pixel_grid[0]) / resolution)
    utm_y = grid_offset_utm_y + ((grid_height_pixels - pixel_grid[1]) / resolution)
    
    return utm_x, utm_y    

def cutout_pixel_to_utm(inverse_transform: np.ndarray, pixel_cutout: np.ndarray,
    grid_offset_utm_x: float, grid_offset_utm_y: float, grid_height_pixels: int,
    resolution: float):

    assert pixel_cutout.flatten().shape == (2,), "'cutout_pixel_to_utm' only accepts one pixel as input"
    assert grid_height_pixels >= pixel_cutout[1], "Pixel height grid less than cutout pixel y coordinate"

    # convert points in rotated rectangle to coords in large grid
    pixel_grid = apply_homography_to_points(H=inverse_transform, pts=pixel_cutout.reshape((-1, 2))).flatten()
    
    return grid_pixel_to_utm(pixel_grid=pixel_grid, grid_offset_utm_x=grid_offset_utm_x,
        grid_offset_utm_y=grid_offset_utm_y, grid_height_pixels=grid_height_pixels,
        resolution=resolution)

def utm_bbox_to_pixels_continuous(x_floor_min: float, y_floor_min: float, grid_width_utm: float,
    grid_height_utm: float, resolution: float, upper_left: np.ndarray, upper_right: np.ndarray,
    lower_right: np.ndarray, lower_left: np.ndarray):
    
    COORDS = [upper_left, upper_right, lower_right, lower_left]
    
    for utmx, utmy in COORDS:
        assert x_floor_min <= utmx <= (x_floor_min + grid_width_utm), "UTM coordinate (x) is outside of tile"
        assert y_floor_min <= utmy <= (y_floor_min + grid_height_utm), "UTM coordinate (y) is outside of tile"
    
    def utm_to_pixel_continuous(utm_x: float, utm_y: float, tile_x0: float, tile_y0: float,
        grid_height_pixel: int, pixel_resolution: float):
    
        assert pixel_resolution > 0, "Only positive pixel resolutions allowed"

        pixel_x = (utm_x - tile_x0) * pixel_resolution
        pixel_y = (utm_y - tile_y0) * pixel_resolution

        # e.g. (resized) tile has height 2000 pixel then pixel (cont.) 0.01 becomes 1999.99
        # these coordinates are in continuous space, i.e. pixel center is (*.5, *.5)
        pixel_y = float(grid_height_pixel) - pixel_y

        return np.array([[pixel_x, pixel_y]])
    
    # coordinate of first pixel center (lower left) in continuous space
    tile_x0, tile_y0 = x_floor_min, y_floor_min
    
    grid_height_pixel = grid_height_utm * resolution
    assert isinstance(grid_height_pixel, int) or grid_height_pixel.is_integer(), \
        "Pixel height of grid must be an integer"
    
    grid_height_pixel = int(grid_height_pixel)
    
    bbox_pixel = np.vstack([utm_to_pixel_continuous(utm_x=utmx, utm_y=utmy, tile_x0=tile_x0, tile_y0=tile_y0,
        grid_height_pixel=grid_height_pixel, pixel_resolution=resolution) \
        for utmx, utmy, in COORDS], dtype=np.float32)
    
    return bbox_pixel

def rotation_2D(alpha: float) -> np.ndarray:
    """ Rotates points in the xy plane counterclockwise through an angle

    Args:
        alpha (float): Rotation angle in radians

    Returns:
        np.ndarray: 2x2 Rotation matrix
    """

    return np.array([
        [np.cos(alpha), -np.sin(alpha)],
        [np.sin(alpha), np.cos(alpha)]
    ])

def apply_homography_to_points(H: np.ndarray, pts: np.ndarray, sanity_check: bool = True,
    round_on_numerical_closeness: bool = True, tol_numerical_closeness: float = 1e-8):
    
    assert (pts.ndim == 1 and len(pts) == 2) or pts.shape[1] == 2, "'pts' is expected to have shape (N, 2)"
    assert H.shape == (3, 3), "'H' is expected to be a homography"
    
    pts = pts.reshape((-1, 2))
    
    pts_homogeneous = np.hstack([pts, np.ones((pts.shape[0], 1))])
    pts_transformed = (H @ pts_homogeneous.T).T

    z = pts_transformed[:, 2:3]

    assert np.all(np.isfinite(z)), "z of homography projection contains NaNs or Infs"
    assert np.all(np.abs(z) > 1e-6), "z values are too small (unstable homography)"

    pts_transformed = pts_transformed[:, :2] / z
    
    if round_on_numerical_closeness:
        pts_transformed = round_if_close_array(pts_transformed, tol=tol_numerical_closeness)
    
    if sanity_check:
        
        # inverse transform
        H_inv = np.linalg.inv(H)
        pts_transformed_homogeneous = np.hstack([pts_transformed.copy(), np.ones((pts.shape[0], 1))])
        pts_recovered = (H_inv @ pts_transformed_homogeneous.T).T
        pts_recovered = pts_recovered[:, :2] / pts_recovered[:, 2:3]

        # compare original with recovered
        if not np.allclose(pts, pts_recovered, atol=1e-6, rtol=0.):
            max_err = np.max(np.linalg.norm(pts - pts_recovered, axis=1))
            raise ValueError(f"Sanity check failed: max reprojection error = {max_err:.2e}")

    return pts_transformed

def nadir_camera_footprint(center: np.ndarray, fov_x: float, fov_y: float,
    altitude: float, yaw: float, footprint_enlargement: float = 1.):
    """ Calculate camera footprint from a nadir perspective

    Args:
        fov_x (float): FOV of camera in x axis of camera
        fov_y (float): FOV of camera in y axis of camera
        altitude (float): Altitude above ground level
        yaw (float): Heading in radians (clockwise, 0° when pointing north)
        footprint_enlargement (float): Percentage by how much the footprint should be enlargened

    Returns:
        np.ndarray: Array of camera footprint coordinates in xy-plane
    """
    
    assert footprint_enlargement > 0, "Footprint enlargement must be positive"
    assert fov_x > 0 and fov_y > 0, "FOVs must be positive"
    assert altitude > 0, "Altitude must be positive"
    
    # simple trigonometry
    delta_x = (np.tan(fov_x / 2) * altitude) * footprint_enlargement
    delta_y = (np.tan(fov_y / 2) * altitude) * footprint_enlargement

    # yaw shall be provided in NED frame
    # i.e., clockwise rotation and 0° when pointing north
    # 'rotation_2D' rotates counterclockwise -> add neg. sign
    R = rotation_2D(-yaw)
    
    center = center.reshape((2, 1))
    
    # calculate corner points of rectangle
    upper_left = center + R @ np.array([[-delta_x], [delta_y]])
    upper_right = center + R @ np.array([[delta_x], [delta_y]])
    lower_right = center + R @ np.array([[delta_x], [-delta_y]])
    lower_left = center + R @ np.array([[-delta_x], [-delta_y]])

    return upper_left, upper_right, lower_right, lower_left
