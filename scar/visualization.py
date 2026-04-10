import io
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from os import makedirs
from matplotlib import cm
from typing import Tuple, List, Optional

from .geometry import apply_homography_to_points

def color_for_id(idx: int):
    
    # 'idx' is used as seed
    r = random.Random(idx)
    
    return (r.randint(0, 255), r.randint(0, 255), r.randint(0, 255))

def resize_with_padding(image: np.ndarray, target_size: Tuple[int, int], white=False):

    target_width, target_height = target_size
    h, w = image.shape[:2]

    scale = min(target_width / w, target_height / h)
    
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    result = 255 * np.ones((target_height, target_width, 3), dtype=np.uint8) if white else \
        np.zeros((target_height, target_width, 3), dtype=np.uint8)

    x_offset = (target_width - new_w) // 2
    y_offset = (target_height - new_h) // 2

    result[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

    return result

def draw_3d_map(elevation_map_dop_resolution: np.ndarray, tracked_ids: List,
    world_points: np.ndarray, x_floor_min: int, y_floor_min: int, grid_height_utm: int,
    sampled_dop_resolution: int, default_height=900, default_width=1600, print_info_texts=True,
    print_ids=False, figure_basis_length=20, padding_utm=10):
    
    # 'arrayize' points that are currently tracked
    tracked_ids_arr = np.array(tracked_ids)
    visible_world_coordinates = []
    
    for idx in tracked_ids_arr:
        visible_world_coordinates.append(world_points[idx])

    visible_world_coordinates = np.array(visible_world_coordinates, dtype=float)
    COLORS = [color_for_id(int(idx)) for idx in tracked_ids_arr]
    COLORS = [(r/255, g/255, b/255) for (r, g, b) in COLORS]

    if len(visible_world_coordinates) > 0:
        
        assert visible_world_coordinates.shape == (tracked_ids_arr.shape[0], 3)

        xmin = np.floor(np.min(visible_world_coordinates[:, 0])).astype(int) - padding_utm
        xmax = np.ceil(np.max(visible_world_coordinates[:, 0])).astype(int) + padding_utm
        ymin = np.floor(np.min(visible_world_coordinates[:, 1])).astype(int) - padding_utm
        ymax = np.ceil(np.max(visible_world_coordinates[:, 1])).astype(int) + padding_utm

        ratio = (ymax - ymin) / (xmax - xmin)
        
        _, ax = plt.subplots(1, 1, figsize=(figure_basis_length, ratio * figure_basis_length))
        
        cutout_x0 = (xmin - x_floor_min) * sampled_dop_resolution
        cutout_y0 = (grid_height_utm - (ymax - y_floor_min)) * sampled_dop_resolution
        cutout_x1 = (xmax - x_floor_min) * sampled_dop_resolution
        cutout_y1 = (grid_height_utm - (ymin - y_floor_min)) * sampled_dop_resolution

        small_map = elevation_map_dop_resolution[cutout_y0:cutout_y1, cutout_x0:cutout_x1]
    
        ax.imshow(small_map, extent=(xmin, xmax, ymin, ymax), cmap="viridis", aspect="auto")
        ax.scatter(visible_world_coordinates[:, 0], visible_world_coordinates[:, 1], c=COLORS)
        ax.set_aspect("equal")
        ax.axis("off")

        if print_info_texts:
            for idx in tracked_ids_arr:
                x, y, z = world_points[idx]
                id_string = f"({idx}), " if print_ids else ""
                ax.text(x=x, y=y+2, s=f"{id_string}{z:.2f}m", ha="center", va="bottom")
            
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        
        buf.seek(0)
        plt.close()
        
        image_map = Image.open(buf)
        image_map = np.array(image_map)[:, :, :3]
    
        buf.close()
        
    else:
        
        image_map = np.zeros((default_height, default_width, 3), dtype=np.uint8)

        text = "NO POINTS"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_thickness = 2
        text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
        
        text_x = (default_width - text_size[0]) // 2
        text_y = (default_height + text_size[1]) // 2
        
        image_map = cv2.putText(image_map, text, (text_x, text_y), font, font_scale,
            (255, 255, 255), font_thickness, cv2.LINE_AA)
        
    return image_map

def draw_2d_3d_correspondences(image: np.ndarray, tracked_points: List,
    tracked_ids: List, world_points: np.ndarray, grid_offset_utm_x: float, grid_offset_utm_y: float,
    grid_height_pixels: int, satellite_image: np.ndarray, N_new: int, image_index: int,
    transform: np.ndarray, sampled_dop_resolution: int, satellite_scale_vis: float = 0.3):
    
    # 'arrayize' points that are currently tracked
    tracked_ids_arr = np.array(tracked_ids)
    tracked_points_arr = np.array(tracked_points)
    
    assert tracked_ids_arr.shape[0] == tracked_points_arr.shape[0], \
        "Number of point IDs must match number of points"
    
    # convert images to color to draw on them
    current_image = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2RGB)
    satellite_image = cv2.cvtColor(satellite_image.copy(), cv2.COLOR_GRAY2RGB)
    satellite_image = cv2.resize(satellite_image, (int(np.round(satellite_image.shape[1] * satellite_scale_vis)),
        int(np.round(satellite_image.shape[0] * satellite_scale_vis))))
    
    ### draw feature positions on image capture ###
    
    # unique colors based on ID
    COLORS = [color_for_id(int(idx)) for idx in tracked_ids_arr]
    visible_world_coordinates = []
    
    for i, (idx, point) in enumerate(zip(tracked_ids_arr, tracked_points_arr)):
        
        ### plot on aerial image ###
    
        # NOTE: 'floor' because (5.8, 4.7) is in pixel (5, 4) not in (6, 5)
        draw_x, draw_y = int(np.floor(point[0])), int(np.floor(point[1]))
        current_image = cv2.putText(current_image, str(idx), (draw_x + 5, draw_y),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        current_image = cv2.circle(current_image, (draw_x, draw_y), 5,
            COLORS[i], -1)
        
        ### plot on satellite image ###
        
        utm_x_vis, utm_y_vis = world_points[idx][:2]
        
        # position in grid
        draw_x_grid = (utm_x_vis - grid_offset_utm_x) * sampled_dop_resolution
        draw_y_grid = grid_height_pixels - ((utm_y_vis - grid_offset_utm_y) * sampled_dop_resolution)

        # transform from grid to cutout    
        draw_x, draw_y = apply_homography_to_points(H=transform, pts=np.array([[draw_x_grid, draw_y_grid]])).flatten()
        # NOTE: 'floor' because (5.8, 4.7) is in pixel (5, 4) not in (6, 5)
        draw_x, draw_y = np.floor(satellite_scale_vis * np.array([draw_x, draw_y])).astype(int)

        satellite_image = cv2.putText(satellite_image, str(idx), (draw_x + 5, draw_y),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        satellite_image = cv2.circle(satellite_image, (draw_x, draw_y), 5,
            COLORS[i], -1)

        visible_world_coordinates.append(world_points[idx])
    
    visible_world_coordinates = np.array(visible_world_coordinates, dtype=float)
    
    ### combine both images ###
    
    height1, width1 = current_image.shape[:2]
    height2, width2 = satellite_image.shape[:2]
    combined_height = max(height1, height2)
    combined_width = width1 + width2

    combined_img = np.zeros((combined_height, combined_width, 3), dtype=np.uint8)

    y_offset1 = (combined_height - height1) // 2
    combined_img[y_offset1:y_offset1 + height1, 0:width1] = current_image
    combined_img[:height2, width1:width1 + width2] = satellite_image

    ### draw lines for new matches ###

    for i in range(1, N_new + 1):
        
        index = tracked_points_arr.shape[0] - i
        
        # NOTE: 'floor' because (5.8, 4.7) is in pixel (5, 4) not in (6, 5)
        pt1 = (int(np.floor(tracked_points_arr[index][0])),
            int(np.floor(tracked_points_arr[index][1])) + y_offset1)
        
        utm_x_vis = visible_world_coordinates[index][0]
        utm_y_vis = visible_world_coordinates[index][1]
        
        draw_x_grid = (utm_x_vis - grid_offset_utm_x) * sampled_dop_resolution
        draw_y_grid = grid_height_pixels - ((utm_y_vis - grid_offset_utm_y) * sampled_dop_resolution)

        assert draw_y_grid >= 0, f"Pixel y coordinate {draw_y_grid} for drawing is out of scope"

        # transform from grid to cutout 
        draw_x, draw_y = apply_homography_to_points(H=transform, pts=np.array([[draw_x_grid, draw_y_grid]])).flatten()
        # NOTE: 'floor' because (5.8, 4.7) is in pixel (5, 4) not in (6, 5)
        draw_x, draw_y = np.floor(satellite_scale_vis * np.array([draw_x, draw_y])).astype(int)

        pt2 = (draw_x + width1, draw_y)

        cv2.line(combined_img, pt1, pt2, COLORS[index], 4)

    ### additional info texts ###
    
    info_texts = [f"Frame: {image_index}", f"Num. of currently tracked features: {len(tracked_ids_arr)}",
        f"Num. of distinct world points: {len(world_points)}", f"Num. new points: {N_new}"]
    
    line_height = 20
    
    if y_offset1 >= line_height * len(info_texts) + 10:
        start_y = (y_offset1 - line_height * len(info_texts)) // 2
    else:
        start_y = max(10, y_offset1 // 2)

    for i, text in enumerate(info_texts):
        y = start_y + i * line_height
        cv2.putText(combined_img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
            (255, 255, 255), 2, cv2.LINE_AA)
    
    return combined_img

def draw_reprojection_errors(image: np.ndarray, points: np.ndarray, projected_points: np.ndarray,
    point_ids: List[str], font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.5, font_thickness=1):
    
    _image = image.copy()
    
    # write IDs
    for i, point in enumerate(points):
        point = np.floor(point).astype(int)
        cv2.putText(_image, point_ids[i], (point[0], point[1] - 3), font, font_scale,
            (255, 255, 0), font_thickness, cv2.LINE_AA)
    
    # draw lines
    for point1, point2 in zip(points, projected_points):
        point1 = np.floor(point1).astype(int)
        point2 = np.floor(point2).astype(int)
        cv2.line(_image, point1, point2, (255, 0, 0), 2)

    # draw groundtruth
    for i, point in enumerate(points):
        point = np.floor(point).astype(int)
        cv2.circle(_image, point, 2, (0, 255, 0), 1)
     
    # draw projections
    for point in projected_points:
        point = np.floor(point).astype(int)
        cv2.circle(_image, tuple(np.round(point).astype(int)), 2, (0, 0, 255), 1)

    return cv2.cvtColor(_image, cv2.COLOR_BGR2RGB)

def create_paper_reprojection_image(
    image: np.ndarray,
    points: np.ndarray,
    projected_points: np.ndarray,
    image_index: int,
    dpi: int = 300,
    cmap=cm.Reds,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
):
    H, W = image.shape[:2]

    valid_mask = (
        (points[:, 0] >= 0) & (points[:, 0] < W) &
        (points[:, 1] >= 0) & (points[:, 1] < H) &
        (projected_points[:, 0] >= 0) & (projected_points[:, 0] < W) &
        (projected_points[:, 1] >= 0) & (projected_points[:, 1] < H)
    )

    points = points[valid_mask]
    projected = projected_points[valid_mask]

    if len(points) == 0:
        raise ValueError("No valid reprojection points remain after filtering.")

    err = np.linalg.norm(projected - points, axis=1)
    med = float(np.median(err))
    mad = float(np.median(np.abs(err - med)))

    if vmin is None:
        vmin = np.percentile(err, 5)
    if vmax is None:
        vmax = np.percentile(err, 95)

    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    colors = cmap(norm(err))

    fig = plt.figure(figsize=(4.2, 4.2), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])  # fills the full figure, no margins
    ax.set_axis_off()

    # grayscale image
    if image.ndim == 3:
        gray = np.dot(image[..., :3], [0.299, 0.587, 0.114])
    else:
        gray = image

    im = ax.imshow(gray, cmap="gray", interpolation="nearest", origin="upper")

    # force axes to exactly match image extent
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)

    dx = projected[:, 0] - points[:, 0]
    dy = projected[:, 1] - points[:, 1]

    # clip all overlays to the image patch
    clip_path = im.get_clip_path()
    clip_box = im.get_extent()

    quivers = []
    for x, y, dx_i, dy_i, c in zip(points[:, 0], points[:, 1], dx, dy, colors):
        q = ax.quiver(
            x, y, dx_i, dy_i,
            angles="xy",
            scale_units="xy",
            scale=1,
            width=0.007,
            headwidth=3.2,
            headlength=4.2,
            headaxislength=3.6,
            color=c,
            zorder=3,
            clip_on=True,
        )
        q.set_clip_path(ax.patch)
        quivers.append(q)

    sc = ax.scatter(
        points[:, 0],
        points[:, 1],
        s=10,
        c=colors,
        marker="o",
        edgecolors="w",
        linewidths=0.6,
        zorder=2,
        clip_on=True,
    )
    sc.set_clip_path(ax.patch)

    # draw once so bbox is computed correctly
    fig.canvas.draw()

    # export only the axes/image area, ignoring arrows outside
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    
    from os.path import join

    out_dir = join("output", "reprojection")
    makedirs(out_dir, exist_ok=True)
    
    out_path = join(out_dir, f"visualization_image_{image_index}_{med:.2f}_{mad:.2f}.png")
    fig.savefig(out_path, dpi=dpi, bbox_inches=bbox, pad_inches=0)
    
    plt.close(fig)
    