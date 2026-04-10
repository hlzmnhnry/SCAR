import os
import re
import cv2
import sys
import json
import torch
import signal
import argparse
import threading
import numpy as np

from glob import glob
from tqdm import tqdm
from pathlib import Path
from functools import partial
from os import remove, makedirs
from scipy.spatial import KDTree
from collections import defaultdict
from torchvision.transforms import ToTensor
from os.path import join, isfile, exists, isdir
from typing import List, Dict, Optional, Callable, Tuple, Any, DefaultDict
from matching import get_matcher # type: ignore

from pygeon.references.reference_campaign_strategy import ClosestReferenceCampaign, RankingOrder
from pygeon.references.footprint_strategy import FootprintStrategy, YawAlignedEnclosingRectFootprint, NorthAlignedFootprint
from pygeon.datasets.vgl_iterator import VGLIterator, FrameType
from pygeon.references.orthophoto import OrthoProvider
from pygeon.io.calibrations import load_camera_model
from pygeon.types import CameraModel
from pygeon.utils import normalize_minmax_to_uint8, round_125
from pygeon.enums import CampaignDate, is_valid_campaign_date_str
from pygeon.visualization.matching import draw_matches
from pygeon.visualization.georeferencation import draw_image_world_correspondences, draw_world_point_map
from pygeon.visualization.utils import make_na_image

os.environ.setdefault("MPLBACKEND", "Agg")

LK_PARAMS = dict(
    winSize = (51, 51),     # window size around each feature point
    maxLevel = 4,           # layers of subsampling pyramid 
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        50,         # stop after 50 iterations (TERM_CRITERIA_COUNT)
        0.001       # stop if estimate changes less than 0.001 (TERM_CRITERIA_EPS)
    ),
    minEigThreshold = 1e-4  # criteria for output features (reject bad features)
)

# cv2 window names for visualization
WINDOW_IMAGE_CAPTURE = "IMAGE CAPTURE"
WINDOW_ORTHOPHOTO = "ORTHOPHOTO"
WINDOW_DSM = "DSM"
WINDOW_MATCHING = "MATCHING"
WINDOW_SLOPE = "SLOPE"
WINDOW_OVERVIEW = "OVERVIEW"
WINDOW_3D_MAP = "ELEVATION MAP"

# global variables for threaded GUI
GUI_FRAMES = {}
GUI_THREAD = None
GUI_RUNNING = False

# global superpoint extractor for backmatching
SUPERPOINT_EXTRACTOR = None
# global lightglue matcher for backmatching
LIGHTGLUE_MATCHER = None
# global image matcher instance
MATCHER = None
# put tensors on GPU (if available)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

###############################
### load state for resuming ###
###############################

def _parse_id_string_from_filename(path: str) -> str:
    """ extract id string from result filename """
    
    # format is: PT_<id_string>.json / WP_<id_string>.json
    name = Path(path).name
    m = re.match(r"^(PT|WP)_(.*)\.json$", name)
    
    if not m:
        raise ValueError(f"Unexpected filename format: {name}")
    
    return m.group(2)

def _extract_last_timestamp_from_id_string(id_string: str) -> int:
    """ extract last timestamp from result filename """
    
    # format is: {campaign}_{start_time}_{sample_ts}_{image_step}_{matcher_type_str}{suffix}
    parts = id_string.split("_")
    if len(parts) < 4:
        raise ValueError(f"Cannot parse id_string: {id_string}")
    
    # parts[0] campaign, parts[1] start_time, parts[2] sample_ts
    return int(parts[2])

def find_latest_state_files(output_directory: Path) -> Tuple[str, str]:
    """ in output directory, search latest PT_*.json file (result file) """
    
    pt_files = sorted(glob(str(output_directory / "PT_*.json")))
    if not pt_files:
        raise FileNotFoundError(f"No PT_*.json found in {output_directory}")

    # pick by timestamp parsed from filename (not mtime)
    best_pt = None
    best_ts = -1
    
    for f in pt_files:
        id_string = _parse_id_string_from_filename(f)
        ts = _extract_last_timestamp_from_id_string(id_string)
        if ts > best_ts:
            best_ts = ts
            best_pt = f

    if best_pt is None:
        raise ValueError("Could not load last file from PT_* files")

    id_string = _parse_id_string_from_filename(best_pt)
    wp = str(output_directory / f"WP_{id_string}.json")
    if not exists(wp):
        raise FileNotFoundError(f"Matching WP file not found for {best_pt}: expected {wp}")

    return best_pt, wp

def load_state(pixel_tracks_file: str, world_points_file: str) -> Tuple[Dict[int, List], Dict[int, np.ndarray]]:
    """ load pixel tracks and world points from files for resuming """
    
    with open(pixel_tracks_file, "r") as f:
        pt_raw = json.load(f)
    with open(world_points_file, "r") as f:
        wp_raw = json.load(f)

    # JSON keys come back as strings -> convert to int
    pixel_tracks: Dict[int, List] = {int(k): v for k, v in pt_raw.items()}
    world_points: Dict[int, np.ndarray] = {int(k): np.array(v, dtype=float) for k, v in wp_raw.items()}

    return pixel_tracks, world_points

def reconstruct_tracked_points_at_last_timestamp(pixel_tracks: Dict[int, List]) -> Tuple[int, List, List]:
    """ from state loaded for resuming, construct tracked points and ids """

    if not pixel_tracks:
        return -1, [], []

    last_ts = -1
    for _id, obs in pixel_tracks.items():
        if len(obs) == 0:
            continue
        ts = int(obs[-1][0])
        if ts > last_ts:
            last_ts = ts

    if last_ts < 0:
        return -1, [], []

    tracked_points = []
    tracked_ids = []
    for _id, obs in pixel_tracks.items():
        if len(obs) == 0:
            continue
        ts = int(obs[-1][0])
        if ts == last_ts:
            x, y = obs[-1][1]
            tracked_points.append([float(x), float(y)])
            tracked_ids.append(int(_id))

    # keep deterministic ordering (important for reproducibility)
    order = np.argsort(np.array(tracked_ids))
    tracked_ids = [tracked_ids[i] for i in order]
    tracked_points = [tracked_points[i] for i in order]

    return last_ts, tracked_points, tracked_ids

########################
### helper functions ###
########################

def deep_update(base, update):
    """overwrite base config entries"""
    
    for k, v in update.items():
        if isinstance(v, dict) and k in base:
            deep_update(base[k], v)
        else:
            base[k] = v

    return base

def is_config_valid(config: Dict):
    """check fields of config dictionary"""
    
    if "path_camera_parameters_guess" not in config:
        raise ValueError("'path_camera_parameters_guess' must be passed for data collection")
    
    if "dataset" not in config:
        raise ValueError("'dataset' section must be defined for data collection")
    
    ### dataset config ###
    
    dataset_config = config["dataset"]
    
    if not is_valid_campaign_date_str(dataset_config["campaign"]):
        raise ValueError(f"'{dataset_config['campaign']}' is not a supported campaign date")
    
    if "start_time" not in dataset_config or "end_time" not in dataset_config:
        raise ValueError("'start_time' and 'end_time' must be set in dataset config")
    
    if "image_directory" in dataset_config:
        if not isdir(dataset_config["image_directory"]):
            raise ValueError(f"Image directory '{dataset_config['image_directory']}' not found")
    
    if "pixels_per_meter" in dataset_config:
        if dataset_config["pixels_per_meter"] <= 0:
            raise ValueError("'pixels_per_meter' must be positive")
        if 10 % dataset_config["pixels_per_meter"] != 0:
            raise ValueError("'pixels_per_meter' must be an integer divisor of 10")
    
    if "footprint_enlargement" in dataset_config:
        if dataset_config["footprint_enlargement"] < 0:
            raise ValueError("'footprint_enlargement' must be positive or zero")
        
    if "image_step" in dataset_config:
        if dataset_config["image_step"] <= 0:
            raise ValueError("'image_step' must be positive")
    
    ### image matching config ###
    
    image_matching_config: Dict[str, Any] = config.get("image_matching", {})
    
    if "matcher_type" not in image_matching_config:
        raise ValueError("'matcher_type' must be set in image matching config")

    return True

def add_to_path(path: str | Path, insert: int | None = None) -> None:
    """Add input path to sys.path ($PATH), allowing for imports from the specified path

    Args:
        path (str | Path): path to add to sys.path
        insert (int | None, optional): insert location / order. Defaults to None, which inserts at end of sys.path
    """
    
    path = str(path)
    if path in sys.path:
        sys.path.remove(path)
    if insert is None:
        sys.path.append(path)
    else:
        sys.path.insert(insert, path)

##############################
### GUI / Thread functions ###
##############################

def shutdown(_signum=None, _frame=None):
    """function to be called to stop programm + threaded GUI"""
    
    global GUI_RUNNING, GUI_THREAD
    GUI_RUNNING = False
    
    if isinstance(GUI_THREAD, threading.Thread):
        GUI_THREAD.join(timeout=1.0)

    sys.exit(0)

def should_enable_gui(config: Dict):
    """check if any window is set to visible"""
    
    visualization = config.get("visualization", {})
    
    keys = [
        "show_image_capture",
        "show_orthophoto",
        "show_dsm",
        "show_matching",
        "show_slope",
        "show_overview",
        "show_3d_map"
    ]

    return any(bool(visualization.get(k, False)) for k in keys)

def GUI_loop(config: Dict):
    """loop for GUI to be run in its own thread (show images/windows and wait)"""
    
    cv2.startWindowThread()
    created_windows = []
    
    for name, flag in [
        (WINDOW_IMAGE_CAPTURE, config.get("show_image_capture", False)),
        (WINDOW_ORTHOPHOTO, config.get("show_orthophoto", False)),
        (WINDOW_DSM, config.get("show_dsm", False)),
        (WINDOW_MATCHING, config.get("show_matching", False)),
        (WINDOW_SLOPE, config.get("show_slope", False)),
        (WINDOW_OVERVIEW, config.get("show_overview", False)),
        (WINDOW_3D_MAP, config.get("show_3d_map", False)),
    ]:
        if flag:
            cv2.namedWindow(name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)
            created_windows.append(name)
            GUI_FRAMES[name] = make_na_image(shape=(480, 640))

    while GUI_RUNNING:
        for name, img in list(GUI_FRAMES.items()):
            if name in created_windows:
                cv2.imshow(name, img)
        if cv2.waitKey(30) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()

def create_named_window(name, debug: bool, flags: int = cv2.WINDOW_GUI_EXPANDED | cv2.WINDOW_NORMAL):
    """create window if flag set"""
    if debug: cv2.namedWindow(name, flags=flags) # noqa: E701

def show_window(name: str, img: np.ndarray, debug: bool, resize=None):
    """show window if flag set"""
    
    if not debug:
        return
    
    global GUI_FRAMES

    disp = cv2.resize(img, dsize=resize) if resize else img
    GUI_FRAMES[name] = disp

##############################
### Georeferenced features ###
##############################

def filter_by_min_distance(kp0: np.ndarray, kp1: np.ndarray, matches: np.ndarray,
    matches_confidence: np.ndarray, existing_points: np.ndarray, d_min: float,
    max_features: int, minimum_confidence: float = 0.0):
    """filter kp0 based on distance to existing_points, min_conf and max_features"""
    
    if kp0.shape[1] != 2 or kp1.shape[1] != 2:
        raise ValueError("'kp*' must be an array of N pixels (2D)")
    
    if matches.shape[0] != matches_confidence.shape[0]:
        raise ValueError("'matches' and 'matches_confidence' must have same length")
    
    if kp0.shape[0] != matches.shape[0]:
        raise ValueError("Each keypoint from kp0 must have one match and vice versa")
        
    num_existing = existing_points.shape[0]
    
    # don't collect new points if there are already enough
    if max_features is not None and num_existing >= max_features:
        return np.empty((0, 2)), np.empty((0, 2))

    # only take matches with a certain confidence
    valid_idx = np.where(matches_confidence >= minimum_confidence)[0] ### [ 1,  52, 100, ...] idx of matches_conf
    print(f"{len(matches_confidence) - valid_idx.size} matches have confidence less than {minimum_confidence}")
    
    if valid_idx.size == 0:
        return np.empty((0, 2)), np.empty((0, 2))

    tree = KDTree(existing_points) if num_existing > 0 else None
    idx_sorted = valid_idx[np.argsort(-matches_confidence[valid_idx])]
    new_kp0, new_kp1 = [], []

    for idx in idx_sorted:
        
        # do we have enough points?
        if max_features is not None and (num_existing + len(new_kp0)) >= max_features:
            break

        # NOTE: kp1 must be used, this is aerial image
        pt = kp1[matches[idx]]
        
        if tree is not None:
            dist, _ = tree.query(pt, k=1)
            if dist < d_min: # only keep points that have min dist to all others
                continue

        new_kp0.append(kp0[idx])
        new_kp1.append(pt)

        num_existing += 1
        existing_points = np.vstack([existing_points, pt.reshape(1, 2)])
        tree = KDTree(existing_points)

    print(f"{valid_idx.size - len(new_kp0)} keypoints have been removed due to distance threshold")

    return np.array(new_kp0), np.array(new_kp1)

def collect_georeferenced_points(config: Dict, capture: np.ndarray, capture_timestamp_us: int, orthophoto: np.ndarray,
    pixel_tracks: dict, world_points: Dict[int, np.ndarray], current_points: List, current_ids: List,
    dsm: np.ndarray, pixel_to_utm: Callable, last_provided_id: int, camera_model: CameraModel):
    """collect new geofeatures by matching capture to orthophoto and lifting with DSM"""

    image_matching_config: Dict[str, Any] = config.get("image_matching", {})
    visualization_config: Dict[str, Any] = config.get("visualization", {})
    tracking_config: Dict[str, Any] = config.get("tracking", {})
    
    if image_matching_config.get("minimum_detections", 50) <= image_matching_config.get("minimum_inliers", 20):
        raise ValueError("Min. #detections must be greater than min. #inliers")
    
    global MATCHER # make global image matching method accessible

    print(">>> Satellite-to-Aerial-Matching")
    
    if MATCHER is None: # load MATCHER once
        matcher_kwargs = image_matching_config.get("matcher_kwargs", {})
        MATCHER = get_matcher(image_matching_config["matcher_type"], device="cuda", **matcher_kwargs)
        
    # region run image matching
    
    if image_matching_config["matcher_type"].startswith("roma"):
        # *0 is satellite, *1 is aerial image capture
        img0_pth = join("data", "temp", "img0.png")
        img1_pth = join("data", "temp", "img1.png")
        cv2.imwrite(img0_pth, orthophoto)
        cv2.imwrite(img1_pth, capture)
        img0 = MATCHER.load_image(img0_pth, resize=None)
        img1 = MATCHER.load_image(img1_pth, resize=None)
    else:
        # *0 is satellite, *1 is aerial image capture
        img0 = ToTensor()(orthophoto.copy())
        img1 = ToTensor()(capture.copy())

    result: Dict[str, np.ndarray] = MATCHER(img0, img1)
    kp0 = result["matched_kpts0"].copy()
    kp1 = result["matched_kpts1"].copy()
    matches = np.arange(len(kp0), dtype=int)
    # NOTE: this is legacy, replace with actual confidences if available
    matches_confidence = np.ones_like(matches, dtype=float)

    # endregion

    if visualization_config.get("show_matching", False):
        
        matching_image = draw_matches(image_0=capture, image_1=orthophoto, kp_0=kp1, kp_1=kp0)

        show_window(WINDOW_MATCHING, matching_image, True,
            resize=(
                int(round(matching_image.shape[1] * visualization_config.get("scale_matching", 1.0))),
                int(round(matching_image.shape[0] * visualization_config.get("scale_matching", 1.0)))
            )
        )

    print(f"Received {len(matches)} matches from {config['image_matching']['matcher_type']}")
    
    minimum_required_detections = image_matching_config.get('minimum_detections', 50)
    if matches.shape[0] < minimum_required_detections:
        
        print(f"{minimum_required_detections} detections required for collecting new point, got only {matches.shape[0]}...")
        
        return {
            "pixel_tracks": pixel_tracks, 
            "world_points": world_points,
            "tracked_points": current_points,
            "tracked_ids": current_ids,
            "n_new": 0, 
            "last_id": last_provided_id
        }
    
    if image_matching_config.get("filter_aerial_satellite_matches", True):

        # keypoints used in geometric filtering 
        kp0_filter, kp1_filter = kp0.copy(), kp1.copy()
        
        # undistort points with calibration guess
        kp1_filter = cv2.undistortPoints(kp1.reshape(-1, 1, 2), cameraMatrix=camera_model.K,
            distCoeffs=camera_model.dist, P=camera_model.K).reshape(-1, 2)

        if kp0_filter.shape[0] != kp1_filter[matches].shape[0]:
            raise ValueError("Expected matched keypoints to be of equal shape")
        
        _, mask = cv2.findHomography(kp0_filter, kp1_filter[matches], method=cv2.USAC_MAGSAC,
            ransacReprojThreshold=image_matching_config.get("hom_magsac_reprojection_threshold", 3.0),
            confidence=image_matching_config.get("hom_magsac_confidence", 0.95),
            maxIters=image_matching_config.get("hom_magsac_maximum_iterations", 2000))
        
        if mask is None:
            
            print("Outlier rejection in satellite-aerial matching failed, neglecting new points...")
            
            return {
                "pixel_tracks": pixel_tracks,
                "world_points": world_points,
                "tracked_points": current_points,
                "tracked_ids": current_ids,
                "n_new": 0,
                "last_id": last_provided_id
            }
        
        if kp0.shape[0] != mask.shape[0] or kp0.shape[0] != matches.shape[0] or kp0.shape[0] != matches_confidence.shape[0]:
            raise ValueError("Number of keypoints must match dimension of filter mask")
    
        # NOTE: don't update kp1 here, this is incorporated in updating 'matches'
        kp0 = kp0[mask.ravel() == 1]
        matches = matches[mask.ravel() == 1]
        matches_confidence = matches_confidence[mask.ravel() == 1]
        
        num_inliers = np.sum(mask)
        num_outliers = np.sum((mask.ravel() == 0))
        
        if num_inliers < image_matching_config.get("minimum_inliers", 20):
            
            print(f"{image_matching_config.get('minimum_inliers', 20)} inliers required for collecting new point, got only {num_inliers}...")
            
            return {
                "pixel_tracks": pixel_tracks,
                "world_points": world_points,
                "tracked_points": current_points,
                "tracked_ids": current_ids,
                "n_new": 0,
                "last_id": last_provided_id
            }

        print(f"Removed {num_outliers} outliers from aerial to satellite matching")
    
    if image_matching_config.get("filter_by_slope", True):
        
        slope_window = image_matching_config.get("slope_window", 15)
        kernel = np.ones((slope_window, slope_window), dtype=np.float32)
        maximum_neighbor = cv2.dilate(dsm.astype(np.float32), kernel, iterations=1)
        minimum_neighbor = cv2.erode(dsm.astype(np.float32), kernel, iterations=1)
        slope = np.maximum(maximum_neighbor - dsm, dsm - minimum_neighbor)

        if slope.shape[:2] != dsm.shape[:2]:
            raise ValueError("'slope' shape does not match shape of 'dsm'")

        # NOTE: 'floor' because (5.8, 4.7) is in pixel (5, 4) not in (6, 5)
        rows = np.floor(kp0[:, 1].copy()).astype(int)  # shape = (n,)
        cols = np.floor(kp0[:, 0].copy()).astype(int)  # shape = (m,)
        kp_slopes = slope[rows, cols]

        # get keypoints where slope is within range
        mask = kp_slopes <= image_matching_config.get("slope_threshold", 5)
        
        # ignore keypoints that have to high slope in local region
        kp0 = kp0[mask.ravel() == 1]
        matches = matches[mask.ravel() == 1]
        matches_confidence = matches_confidence[mask.ravel() == 1]
        
        num_high_slope = np.sum((mask.ravel() == 0))
        
        # region visualize slope
        
        slope_u8 = normalize_minmax_to_uint8(slope.copy())
        slope_colored = cv2.applyColorMap(slope_u8, cv2.COLORMAP_JET) if visualization_config.get("show_slope", False) else None
        
        if slope_colored is not None:
            show_window(WINDOW_SLOPE, slope_colored,
                visualization_config.get("show_slope", False), 
                resize=(
                    int(round(slope_colored.shape[1] * visualization_config.get("scale_slope", 1.0))),
                    int(round(slope_colored.shape[0] * visualization_config.get("scale_slope", 1.0)))
                )
            )
        
        # endregion
        
        print(f"Removed {num_high_slope} correspondences due to high slope in surrounding")
    
    # 'arrayize' points that are currently tracked
    current_points_arr = np.array(current_points).reshape((-1, 2))
    
    collected_kp0, collected_kp1 = filter_by_min_distance(
        kp0=kp0, kp1=kp1, matches=matches, matches_confidence=matches_confidence,
        existing_points=current_points_arr,
        d_min=tracking_config.get("minimum_pixel_distance", 30.0),
        max_features=tracking_config.get("maximum_features", 500),
        minimum_confidence=image_matching_config.get("minimum_confidence", 0.5)
    )
    
    if collected_kp0.shape != collected_kp1.shape:
        raise RuntimeError("Keypoint sets have a different shape")

    N = len(collected_kp0)
    
    def dop_pixel_to_3d(dop_pixel: np.ndarray):
        
        # use mapping provided from orthophoto provider
        utm_x, utm_y = pixel_to_utm(pixel=dop_pixel)
        
        # NOTE: 'floor' because (5.8, 4.7) is in pixel (5, 4) not in (6, 5)
        pixel_dsm = np.floor(dop_pixel).astype(int)
        elevation = dsm[pixel_dsm[1], pixel_dsm[0]]
        
        return np.array([utm_x, utm_y, elevation], dtype=float)
    
    for point_dop, point_capture in zip(collected_kp0, collected_kp1):
        
        new_id = last_provided_id + 1
        
        if new_id in pixel_tracks:
            raise RuntimeError("IDs are not unique")
        
        # initialize tracking list for newly added point
        pixel_tracks[new_id] = [
            (capture_timestamp_us, (float(point_capture[0]), float(point_capture[1])))
        ]

        # add new point + id
        current_points.append(point_capture)
        current_ids.append(new_id)

        new_world_point = dop_pixel_to_3d(dop_pixel=point_dop).tolist()
        
        if len(new_world_point) != 3:
            raise RuntimeError(f"New world point with ID={new_id} has more than 3 entries (expected 3D point)")

        world_points[new_id] = new_world_point
        last_provided_id = new_id

    print(f"Collected {N} new world points from aerial to satellite matching")

    return {"pixel_tracks": pixel_tracks, "world_points": world_points, "tracked_points": current_points,
        "tracked_ids": current_ids, "n_new": N, "last_id": last_provided_id}

def match_back(capture: np.ndarray, orthophoto: np.ndarray, dsm: np.ndarray,
    tracked_features: np.ndarray | List, tracked_ids: np.ndarray | List, pixel_to_utm: Callable):
    
    import matching
    THIRD_PARTY_DIR = Path(matching.__file__).resolve().parent / "third_party"
    add_to_path(THIRD_PARTY_DIR.joinpath("LightGlue"))

    from lightglue import SuperPoint, LightGlue # type: ignore  # noqa: E402, F401
    from lightglue.utils import rbd  # type: ignore  # noqa: E402, F401

     # make global superpoint extractor and lightglue matcher accessible
    global SUPERPOINT_EXTRACTOR, LIGHTGLUE_MATCHER
    LIGHTGLUE_MATCHER = LightGlue(features="superpoint").eval().cuda()
    
    if SUPERPOINT_EXTRACTOR is None: # load SUPERPOINT_EXTRACTOR once
        SUPERPOINT_EXTRACTOR = SuperPoint(max_num_keypoints=2048).eval().cuda()
        
    if isinstance(tracked_features, list):
        tracked_features = np.array(tracked_features, dtype=np.float32)
    
    if isinstance(tracked_ids, list):
        tracked_ids = np.array(tracked_ids, dtype=int)
    
    tracked_keypoints = torch.from_numpy(tracked_features).to(DEVICE).float()
    feature_orthophoto = SUPERPOINT_EXTRACTOR.extract(ToTensor()(orthophoto.copy()).to(DEVICE))
    feature_capture = SUPERPOINT_EXTRACTOR.extract(ToTensor()(capture.copy()).to(DEVICE), keypoints=tracked_keypoints)
    
    # keep convention: *0 is satellite, *1 is aerial image capture
    matches01 = LIGHTGLUE_MATCHER({"image0": feature_orthophoto, "image1": feature_capture})
    feature_orthophoto, _, matches01 = [rbd(x) for x in [feature_orthophoto, feature_capture, matches01]]
    points_orthophoto = feature_orthophoto["keypoints"][matches01["matches"][..., 0]].cpu().numpy() 
    
    def dop_pixel_to_3d(dop_pixel: np.ndarray):
        
        # use mapping provided from orthophoto provider
        utm_x, utm_y = pixel_to_utm(pixel=dop_pixel)
        
        # NOTE: 'floor' because (5.8, 4.7) is in pixel (5, 4) not in (6, 5)
        pixel_dsm = np.floor(dop_pixel).astype(int)
        elevation = dsm[pixel_dsm[1], pixel_dsm[0]]
        
        return np.array([utm_x, utm_y, elevation], dtype=float)
    
    match_indices_ortho = matches01["matches"][..., 0].cpu().numpy()
    match_indices_capture = matches01["matches"][..., 1].cpu().numpy()

    points_orthophoto = feature_orthophoto["keypoints"][match_indices_ortho].cpu().numpy()
    matched_tracked_ids = tracked_ids[match_indices_capture]
    
    rematched_points: Dict[int, np.ndarray] = {}
    
    if matched_tracked_ids.shape[0] != points_orthophoto.shape[0]:
        raise ValueError("There must an equal number of rematched IDs and DOP points")
    
    for id, point_dop in zip(matched_tracked_ids, points_orthophoto):
        rematched_points[id] = dop_pixel_to_3d(dop_pixel=point_dop)
    
    return rematched_points

def main():
    
    # region config / IO / setup 
    
    global GUI_THREAD, GUI_RUNNING
    cv2.setLogLevel(0)
    
    parser = argparse.ArgumentParser(description="SCAR data pre-collection", add_help=True)
    parser.add_argument("--base-config-file", type=str, required=True, help="File to read base configuration parameters from")
    parser.add_argument("--config-file", type=str, required=True, help="File to read campaign-specific configuration parameters from")
    
    args = parser.parse_args()
    
    if not isfile(args.base_config_file):
        raise ValueError("Provided base config file does not exist")
    
    if not isfile(args.config_file):
        raise ValueError("Provided config file does not exist")

    config: Dict = json.load(open(args.base_config_file))
    override: Dict = json.load(open(args.config_file))

    config = deep_update(config, override)
    
    if not is_config_valid(config=config):
        raise ValueError("Provided config file is not valid")

    visualization_config: Dict[str, Any] = config.get("visualization", {})

    if should_enable_gui(config):
        GUI_RUNNING = True
        GUI_THREAD = threading.Thread(target=GUI_loop, daemon=True, kwargs={"config": visualization_config})
        GUI_THREAD.start()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    
    camera_calib_dir = Path(config["path_camera_parameters_guess"]).parent
    camera_file_basename = Path(config["path_camera_parameters_guess"]).name
    camera_model_guess = load_camera_model(calib_dir=camera_calib_dir, camera_file_basename=camera_file_basename)
    
    extrinsics_calib_dir = Path(config["path_extrinsic_parameters_guess"]).parent
    extrinsics_file_basename = Path(config["path_extrinsic_parameters_guess"]).name
    
    dataset_config: Dict[str, Any] = config["dataset"]
    campaign = CampaignDate(dataset_config["campaign"])
    
    output_directory = Path(__file__).parent.parent.joinpath("output", campaign.value)
    output_directory = output_directory / "geofeatures"
    makedirs(output_directory, exist_ok=True)
    
    footprint_strategy: FootprintStrategy | None = None
    footprint_type = dataset_config.get("footprint_type", "yaw_enclosing")
    
    if footprint_type == "yaw_enclosing":
        footprint_strategy = YawAlignedEnclosingRectFootprint(
            should_enforce_aspect_ratio=False,
            enlargement=dataset_config.get("footprint_enlargement", 0.)
        )
    elif footprint_type == "north_square":
        footprint_strategy = NorthAlignedFootprint(
            should_round_to_full_meters=True,
            convert_to_square=True,
            square_size=dataset_config.get("footprint_square_size", 1000.)
        )
    else:
        raise ValueError(f"Unknown footprint type '{footprint_type}'")
    
    reference_campaign_strategy = ClosestReferenceCampaign(
        ranking_order=RankingOrder.SEASON_YEAR_STATE
    )
    
    pixels_per_meter = dataset_config.get("pixels_per_meter", 5)
    image_directory = Path(dataset_config["image_directory"]) \
        if "image_directory" in dataset_config else None
    data_directory = Path(dataset_config["data_directory"]) \
        if "data_directory" in dataset_config else None
    
    provider = OrthoProvider(
        campaign=campaign,
        data_directory=data_directory,
        footprint_strategy=footprint_strategy,
        reference_campaign_strategy=reference_campaign_strategy,
        prioritize_nrw_tiles=True,
        pixels_per_meter=pixels_per_meter,
        provide_dsm=True,
    )
    
    # endregion
    
    #################
    ### main loop ###
    #################
    
    # region result and helper variables
    
    # collect feature tracks and georeferenced points
    pixel_tracks: Dict[int, List] = {}
    world_points: Dict[int, List] = {}
    rematched_world_points: DefaultDict[int, List[np.ndarray]] = defaultdict(list)
    
    # temporary lists for currently tracked features
    tracked_points = []
    tracked_ids = []
    
    # last image capture (for feature tracking)
    last_img_capture = None
    # last geodata (for backmatching)
    last_pixel_to_utm = None
    last_orthophoto = None
    last_dsm = None
    # last provided feature ID
    last_provided_id = -1
    
    # output is saved every time (and deleted again)
    # '-> keep track of file names to only keep latest file
    output_file_pixel_tracks = None
    output_file_world_points = None
    output_file_backmatching = None
    
    # endregion
    
    # region resume if set 
    
    resume_enabled = bool(config.get("resume", False))
    resume_last_ts_us: Optional[int] = None
    
    if resume_enabled:
        
        mode = config.get("resume_mode", "latest")

        if mode == "latest":
            pt_file, wp_file = find_latest_state_files(output_directory=output_directory)
        elif mode == "manual":
            pt_file = config.get("resume_pixel_tracks_file", "")
            wp_file = config.get("resume_world_points_file", "")
            if not pt_file or not wp_file:
                raise ValueError("resume_mode='manual' requires resume_pixel_tracks_file and resume_world_points_file")
            if not exists(pt_file) or not exists(wp_file):
                raise FileNotFoundError(f"Resume files not found: {pt_file}, {wp_file}")
        else:
            raise ValueError(f"Unknown resume_mode: {mode}")

        print(f"\nResuming, loading state:\n\tPT: {pt_file}\n\tWP: {wp_file}")

        pixel_tracks, world_points = load_state(pt_file, wp_file)
        resume_last_ts_us, tracked_points, tracked_ids = reconstruct_tracked_points_at_last_timestamp(pixel_tracks)

        if resume_last_ts_us is None or resume_last_ts_us < 0:
            raise RuntimeError("Resuming failed to reconstruct last timestamp from pixel_tracks")

        last_provided_id = max(pixel_tracks.keys()) if pixel_tracks else -1

        print(f"\nResuming: last_ts_us={resume_last_ts_us}, alive_points={len(tracked_ids)}, last_id={last_provided_id}")
    
    # endregion
    
    t_start_us = int(dataset_config["start_time"])
    
    if resume_enabled and resume_last_ts_us is not None:
        t_start_us = int(resume_last_ts_us)
    
    t_end_us = int(dataset_config["end_time"])
    image_step = dataset_config.get("image_step", 10)

    it = VGLIterator(
        campaign=campaign,
        data_directory=data_directory,
        undistort=False,
        frame=FrameType.CAMERA,
        image_directory=image_directory,
        image_indexed_naming=True,
        with_landcover=False,
        image_step=image_step,
        ortho_provider=provider,
        camera_calibration_directory=camera_calib_dir,
        camera_file_basename=camera_file_basename,
        extrinsics_calibration_directory=extrinsics_calib_dir,
        extrinsics_file_basename=extrinsics_file_basename,
        t_start_us=t_start_us,
        t_end_us=t_end_us
    )
    
    # clahe for both image capture and orthophoto
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    first_iteration = True
    
    for i, sample in tqdm(enumerate(it), total=len(it)):
        
        print("\n>>> Status information")
        print(f"Total of {len(world_points)} distinct world points")
        print(f"Total of {sum([len(obs) for obs in pixel_tracks.values()])} observations")
        print(f"Start processing for image ts = {sample['timestamp']}")
        
        ############################################
        ######## load data for current image #######
        ############################################
        
        # region show image capture 

        img_capture: np.ndarray = sample.get("image", None)

        if img_capture is None:
            raise ValueError("Failed to load image capture")
        
        img_capture = clahe.apply(img_capture)
        
        show_window(WINDOW_IMAGE_CAPTURE, cv2.cvtColor(img_capture, cv2.COLOR_GRAY2BGR),
            visualization_config.get("show_image_capture", False),
            resize=(
                int(round(img_capture.shape[1] * visualization_config.get("scale_image_capture", 1.0))),
                int(round(img_capture.shape[0] * visualization_config.get("scale_image_capture", 1.0)))
            )
        )
        
        # endregion
        
        # region resume
        
        if resume_enabled and first_iteration:
            
            first_iteration = False
            
            if not isinstance(resume_last_ts_us, int):
                raise ValueError("Expected 'resume_last_ts_us' to be integer")

            if int(sample["timestamp"]) != int(resume_last_ts_us):
                raise RuntimeError(f"Resume mismatch: expected ts={resume_last_ts_us}, got ts={sample['timestamp']}")

            print("Resuming: priming last_img_capture and skipping this frame")
            last_img_capture = img_capture.copy()
            continue

        first_iteration = False
        
        if dataset_config.get("update_to_real_ppm", True):
            footprint_diagonal = np.linalg.norm(sample["orthophoto_raw_footprint"][2] - sample["orthophoto_raw_footprint"][0])
            real_ppm = np.linalg.norm(img_capture.shape[:2]) / footprint_diagonal # noqa: F841
            new_resolution = round_125(x=real_ppm, mode="nearest")
            provider.update(pixels_per_meter=new_resolution, clear_cache=False)
        
        # endregion
        
        # region show orthophoto

        orthophotos = sample.get("orthophoto", [])
        orthophoto = None
        
        if orthophotos is not None:
            
            if len(orthophotos) > 1:
                raise ValueError("Expected only one orthophoto reference")
            
            orthophoto: np.ndarray = cv2.cvtColor(orthophotos[0], cv2.COLOR_RGB2GRAY)
            orthophoto = clahe.apply(orthophoto)

            show_window(WINDOW_ORTHOPHOTO, orthophoto,
                visualization_config.get("show_orthophoto", False),
                resize=(
                    int(round(orthophoto.shape[1] * visualization_config.get("scale_orthophoto", 1.0))),
                    int(round(orthophoto.shape[0] * visualization_config.get("scale_orthophoto", 1.0)))
                )
            )
        
        # endregion
        
        # region show digital elevation model
        
        dsm: np.ndarray | None = sample.get("dsm", None)
        
        if dsm is not None:

            if orthophoto is not None:
                if not dsm.shape == orthophoto.shape[:2]:
                    raise ValueError("Orthophoto and DSM must be of equal size/resolution")
            
            dsm_u8 = normalize_minmax_to_uint8(dsm.copy())
            dsm_colored = cv2.applyColorMap(dsm_u8, cv2.COLORMAP_JET) if visualization_config.get("show_dsm", False) else None

            if dsm_colored is not None:
                show_window(WINDOW_DSM, dsm_colored,
                    visualization_config.get("show_dsm", False),
                    resize=(
                        int(round(dsm_colored.shape[1] * visualization_config.get("scale_dsm", 1.0))),
                        int(round(dsm_colored.shape[0] * visualization_config.get("scale_dsm", 1.0)))
                    )
                )
        
        # endregion
 
        # region perform tracking
        
        tracking_config: Dict[str, Any] = config.get("tracking", {})
        
        if tracking_config.get("perform_tracking", True) and last_img_capture is not None \
            and len(tracked_points) > 0:
            
            print(">>> Feature tracking")
            
            if len(tracked_points) != len(tracked_ids):
                raise RuntimeError("There should be a one-to-one matching b/w tracked points and IDs")
            
            # 'arrayize' list of current points
            p0 = np.array(tracked_points, dtype=np.float32)
            
            print(f"Tracking {p0.shape[0]} current features...")
            
            p1, status, _ = cv2.calcOpticalFlowPyrLK(last_img_capture, img_capture, p0, None, **LK_PARAMS)
            
            if p1 is None or status is None:
                raise ValueError("Calculating optical flow failed")
            
            good_idx: np.ndarray = np.array(tracked_ids)[status.flatten() == 1]
            good_old: np.ndarray = p0[status.flatten() == 1]
            good_new: np.ndarray = p1[status.flatten() == 1]
            
            ### remove outlier based on fundamental matrix and RANSAC ###
            
            if tracking_config.get("perform_fm_outlier_rejection", True) and good_new.shape[0] >= 8:
                
                pts1, pts2 = good_old.copy(), good_new.copy()
                
                if camera_model_guess is None or not isinstance(camera_model_guess, CameraModel):
                    raise ValueError("Guess for camera matrix and distortion are needed for point undistortion")

                pts1 = cv2.undistortPoints(pts1.reshape(-1, 1, 2), cameraMatrix=camera_model_guess.K,
                    distCoeffs=camera_model_guess.dist, P=camera_model_guess.K).reshape(-1, 2)
                pts2 = cv2.undistortPoints(pts2.reshape(-1, 1, 2), cameraMatrix=camera_model_guess.K,
                    distCoeffs=camera_model_guess.dist, P=camera_model_guess.K).reshape(-1, 2)

                _, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC,
                    ransacReprojThreshold=tracking_config.get("fm_ransac_reprojection_threshold", 1.0),
                    confidence=tracking_config.get("fm_ransac_confidence", 0.95),
                    maxIters=tracking_config.get("fm_ransac_maximum_iterations", 2000))
                
                if mask is not None:
                    good_idx = good_idx[mask.ravel() == 1]
                    good_old = good_old[mask.ravel() == 1]
                    good_new = good_new[mask.ravel() == 1]
                    num_outliers = np.sum((mask.ravel() == 0))
                    print(f"Removed {num_outliers} outliers (RANSAC + FM) within feature tracking")
                else:
                    raise RuntimeError("Not able to perform outlier rejection for tracking")

            ### save pixels ###
            
            if good_idx.shape[0] != good_new.shape[0]:
                raise ValueError("Point correspondence and ID array do not match in length")
            
            for ID, (x, y) in zip(good_idx, good_new):
                pixel_tracks[ID].append((int(sample["timestamp"]), (float(x), float(y))))
            
            tracked_points = good_new.tolist()
            tracked_ids = good_idx.tolist()
            
            print(f"{len(tracked_points)} features survived feature tracking + outlier removal")
        
            # region backmatching
            
            backmatching_config: Dict[str, Any]  = config.get("backmatching", {})
            
            if backmatching_config.get("perform_backmatching", True):
                
                raise NotImplementedError("Backmatching is not fully supported yet")
            
                # take features tracked from last frame, create Superpoint descriptors for new frame
                # and match these back with last orthophoto; get UTM position for new matches
                # this yields rematched 3D coordiantes for features, i.e. point spread in world frame
                # NOTE: it would also be possible to match these tracked features with new DOP
                rematched_points = match_back(capture=img_capture, orthophoto=last_orthophoto, dsm=last_dsm,
                    tracked_features=tracked_points, tracked_ids=tracked_ids, pixel_to_utm=last_pixel_to_utm)
                
                # save rematched points in separate dict to not confuse structures
                for id, world_point in rematched_points.items():
                    rematched_world_points[id].append(world_point)
            
            # endregion
        
        else:
            # reset tracking if only single-shot matching
            tracked_points = []
            tracked_ids = []
            
        # endregion
        
        # region collect new features
        
        image_matching_config: Dict[str, Any] = config.get("image_matching", {})
        
        # number of newly collected features
        num_new_geofeatures = 0
        
        if orthophoto is None:
            raise RuntimeError(f"No orthophoto available for image with ts={sample['timestamp']}")
        
        if dsm is None:
            raise RuntimeError(f"No DSM available for image with ts={sample['timestamp']}")

        if len(tracked_points) < tracking_config.get("maximum_features", 500) and \
            i % image_matching_config.get("collect_new_geofeatures_every_nth_image", 1) == 0:
            
            last_provided_id_backup = last_provided_id
            
            result = collect_georeferenced_points(config=config, capture=img_capture, capture_timestamp_us=sample["timestamp"],
                orthophoto=orthophoto, pixel_tracks=pixel_tracks, world_points=world_points,
                current_points=tracked_points, current_ids=tracked_ids, dsm=dsm, pixel_to_utm=sample["pixel_to_utm"],
                last_provided_id=last_provided_id, camera_model=camera_model_guess)

            # unpack return dict to variables
            pixel_tracks = result["pixel_tracks"]
            world_points = result["world_points"]
            tracked_points = result["tracked_points"]
            tracked_ids = result["tracked_ids"]
            last_provided_id = result["last_id"]
            num_new_geofeatures = result["n_new"]
            
            if (last_provided_id - last_provided_id_backup) != num_new_geofeatures:
                raise ValueError("ID increment not compatible with #new")
            
            if len(np.unique(tracked_ids)) != len(tracked_ids):
                raise ValueError("ID duplication for tracked points")

        # endregion
            
        if num_new_geofeatures > 0:
            
            world_points_x = np.array(list(world_points.values()))[-num_new_geofeatures:, 0]
            world_points_y = np.array(list(world_points.values()))[-num_new_geofeatures:, 1]
            
            # min/max UTM range from sample's orthophoto footprint
            fp_min_x = np.min(sample["orthophoto_footprint"][:, 0])
            fp_max_x = np.max(sample["orthophoto_footprint"][:, 0])
            fp_min_y = np.min(sample["orthophoto_footprint"][:, 1])
            fp_max_y = np.max(sample["orthophoto_footprint"][:, 1])
            
            min_x, max_x = np.ceil(world_points_x.min()), np.floor(world_points_x.max())
            min_y, max_y = np.ceil(world_points_y.min()), np.floor(world_points_y.max())
            
            if min_x < fp_min_x or max_x > fp_max_x:
                raise RuntimeError("Collected world points are out of range in X")
            
            if min_y < fp_min_y or max_y > fp_max_y:
                raise RuntimeError("Collected world points are out of range in y")

        # region show primary overview
        
        if visualization_config.get("show_overview", True):

            world_point_ids = np.array(list(map(int, world_points.keys())))
            new_world_point_ids = world_point_ids[-num_new_geofeatures:] \
                if num_new_geofeatures > 0 else []
            world_point_values = np.array(list(world_points.values()))
            
            overview_image = draw_image_world_correspondences(image=img_capture,
                orthophoto=orthophoto, image_points=tracked_points, world_points=world_point_values,
                image_identifier=sample["timestamp"], utm_to_pixel=sample["utm_to_pixel"],
                image_point_ids=tracked_ids, world_point_ids=world_point_ids,
                new_world_point_ids=new_world_point_ids, plot_identifiers=False)
                        
            show_window(WINDOW_OVERVIEW, overview_image, True,
                resize=(
                    int(round(overview_image.shape[1] * visualization_config.get("scale_overview", 1.0))),
                    int(round(overview_image.shape[0] * visualization_config.get("scale_overview", 1.0)))
                )
            )

        # endregion
        
        # region show 3d map
        
        if visualization_config.get("show_3d_map", False):

            world_point_ids = list(map(int, world_points.keys()))
            world_point_values = list(world_points.values())
            
            for _id, _world_points in rematched_world_points.items():
                world_point_ids.extend([_id] * len(_world_points))
                world_point_values.extend(_world_points)
            
            world_point_ids = np.asarray(world_point_ids)
            world_point_values = np.asarray(world_point_values)

            world_map = draw_world_point_map(dsm=dsm, world_points=world_point_values,
                world_point_ids=world_point_ids, utm_to_pixel=sample["utm_to_pixel"],
                plot_world_point_ids=tracked_ids, plot_info=False,
                cv_color_map=cv2.COLORMAP_BONE)

            show_window(WINDOW_3D_MAP, world_map, True,
                resize=(
                    int(round(world_map.shape[1] * visualization_config.get("scale_3d_map", 1.0))),
                    int(round(world_map.shape[0] * visualization_config.get("scale_3d_map", 1.0)))
                )
            )
        
        # endregion
        
        last_pixel_to_utm = partial(sample["pixel_to_utm"].func,
            *sample["pixel_to_utm"].args, **sample["pixel_to_utm"].keywords)
        last_img_capture = img_capture.copy()
        last_orthophoto = orthophoto.copy()
        last_dsm = dsm.copy()
        
        # region save results
        
        if i % config.get("save_results_every_nth_iteration", 100) == 0 and i > 0 or i == len(it) - 1:
        
            # delete last 'backup' files (created after last iteration)
            if output_file_pixel_tracks is not None and exists(output_file_pixel_tracks):
                remove(output_file_pixel_tracks)

            # delete last 'backup' files (created after last iteration)
            if output_file_world_points is not None and exists(output_file_world_points):
                remove(output_file_world_points)
            
            # delete last 'backup' files (created after last iteration)
            if output_file_backmatching is not None and exists(output_file_backmatching):
                remove(output_file_backmatching)

            # construct name basis for output files
            
            matcher_type_str = image_matching_config["matcher_type"].replace("-", "_")
            suffix = config.get("manual_output_suffix", "")
            suffix = f"_{suffix}" if len(suffix) > 0 else ""
            
            id_string = f"{campaign.value}_{dataset_config['start_time']}_{sample['timestamp']}_{dataset_config['image_step']}_{matcher_type_str}{suffix}"

            output_file_pixel_tracks = join(output_directory, f"PT_{id_string}.json")
            output_file_world_points = join(output_directory, f"WP_{id_string}.json")
            output_file_backmatching = join(output_directory, f"BM_{id_string}.json")

            with open(output_file_pixel_tracks, "w+") as fout:
                json.dump(pixel_tracks, fout, indent=4)

            with open(output_file_world_points, "w+") as fout:
                json.dump({int(k): [float(x) for x in v] for k, v in world_points.items()}, fout, indent=4)
            
            with open(output_file_backmatching, "w+") as fout:
                json.dump({int(k): [[float(y) for y in x] for x in v] for k, v in rematched_world_points.items()}, fout, indent=4)
        
        # endregion
            
    shutdown()

if __name__ == "__main__":
    main()
