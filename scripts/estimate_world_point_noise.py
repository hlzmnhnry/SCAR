import sys
import cv2
import json
import torch
import argparse
import numpy as np

from tqdm import tqdm
from os import makedirs
from pathlib import Path
from os.path import join, isfile
from collections import defaultdict
from pygeon.types import CameraModel
from typing import List, Dict, Callable
from torchvision.transforms import ToTensor
from pygeon.io.calibrations import load_camera_model
from pygeon.datasets.vgl_iterator import VGLIterator    # type: ignore
from pygeon.references.orthophoto import OrthoProvider  # type: ignore
from pygeon.enums import CampaignDate, FrameType, RankingOrder      # type: ignore
from pygeon.references.reference_campaign_strategy import ClosestReferenceCampaign  # type: ignore
from pygeon.references.footprint_strategy import YawAlignedEnclosingRectFootprint    # type: ignore

# global superpoint extractor for backmatching
SUPERPOINT_EXTRACTOR = None
# global lightglue matcher for backmatching
LIGHTGLUE_MATCHER = None
# put tensors on GPU (if available)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def deep_update(base, update):
    """overwrite base config entries"""
    
    for k, v in update.items():
        if isinstance(v, dict) and k in base:
            deep_update(base[k], v)
        else:
            base[k] = v

    return base

def match_back(capture: np.ndarray, orthophoto: np.ndarray, dsm: np.ndarray,
    tracked_features: np.ndarray | List, tracked_ids: np.ndarray | List, pixel_to_utm: Callable, cam_model: CameraModel):
    
    import matching
    THIRD_PARTY_DIR = Path(matching.__file__).resolve().parent / "third_party"
    add_to_path(THIRD_PARTY_DIR.joinpath("LightGlue"))

    from lightglue import SuperPoint, LightGlue # type: ignore  # noqa: E402, F401
    from lightglue.utils import rbd  # type: ignore  # noqa: E402, F401

    # make global superpoint extractor and lightglue matcher accessible
    global SUPERPOINT_EXTRACTOR, LIGHTGLUE_MATCHER
    if LIGHTGLUE_MATCHER is None:
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
    
    new_tracked_ids = []
    new_tracked_features = []
    actual_extracted = feature_capture["keypoints"].cpu().numpy().copy()[0]
    
    for kp in actual_extracted:
        diff = np.linalg.norm(tracked_features - kp, axis=1)
        closest = np.argmin(diff)
        closes_diff = diff[closest]
        if closes_diff < 0.1:
            new_tracked_ids.append(tracked_ids[closest])
            new_tracked_features.append(tracked_features[closest])
    
    assert len(new_tracked_ids) == actual_extracted.shape[0]
    tracked_ids = np.array(new_tracked_ids)
    tracked_features = np.array(new_tracked_features)

    assert len(tracked_ids) == actual_extracted.shape[0]
    assert len(tracked_features) == actual_extracted.shape[0]
    
    # keep convention: *0 is satellite, *1 is aerial image capture
    matches01 = LIGHTGLUE_MATCHER({"image0": feature_orthophoto, "image1": feature_capture})
    feature_orthophoto, feature_capture, matches01 = [rbd(x) for x in [feature_orthophoto, feature_capture, matches01]]
    
    def dop_pixel_to_3d(dop_pixel: np.ndarray):
        
        # use mapping provided from orthophoto provider
        utm_x, utm_y = pixel_to_utm(pixel=dop_pixel)
        
        # NOTE: 'floor' because (5.8, 4.7) is in pixel (5, 4) not in (6, 5)
        pixel_dsm = np.floor(dop_pixel).astype(int)
        elevation = dsm[pixel_dsm[1], pixel_dsm[0]]
        
        return np.array([utm_x, utm_y, elevation], dtype=float)
    
    match_indices_ortho = matches01["matches"][..., 0].cpu().numpy()
    match_indices_capture = matches01["matches"][..., 1].cpu().numpy()

    points_orthophoto = feature_orthophoto["keypoints"].cpu().numpy()[match_indices_ortho]
    points_capture = feature_capture["keypoints"].cpu().numpy()[match_indices_capture]
    matched_tracked_ids = tracked_ids[match_indices_capture]
    
    assert points_orthophoto.shape[0] == points_capture.shape[0]
    assert matched_tracked_ids.shape[0] == points_capture.shape[0]
    
    rematched_points: Dict[int, np.ndarray] = {}
    
    if matched_tracked_ids.shape[0] != points_orthophoto.shape[0]:
        raise ValueError("There must an equal number of rematched IDs and DOP points")
    
    ##### filter out ######

    # keypoints used in geometric filtering 
    kp0_filter, kp1_filter = points_orthophoto.copy(), points_capture.copy()
    
    if kp0_filter.shape[0] == 0 or kp1_filter.shape[0] == 0:
        return rematched_points
    
    # undistort points with calibration guess
    kp1_filter =  cv2.undistortPoints(kp1_filter.reshape(-1, 1, 2), cameraMatrix=cam_model.K,
        distCoeffs=cam_model.dist, P=cam_model.K).reshape(-1, 2)
    
    if kp0_filter.shape[0] < 8:
        return rematched_points
    
    _, mask = cv2.findHomography(kp0_filter, kp1_filter, method=cv2.USAC_MAGSAC,
        ransacReprojThreshold=3.0, confidence=0.95, maxIters=2000)
    
    if mask is None: raise ValueError  # noqa: E701
    
    # NOTE: don't update kp1 here, this is incorporated in updating 'matches'
    points_orthophoto = points_orthophoto[mask.ravel() == 1]
    points_capture = points_capture[mask.ravel() == 1]
    matched_tracked_ids = matched_tracked_ids[mask.ravel() == 1]
    
    assert points_orthophoto.shape[0] == points_capture.shape[0]
    assert matched_tracked_ids.shape[0] == points_capture.shape[0]
    
    num_outliers = np.sum((mask.ravel() == 0))
    print(f"Removed {num_outliers} outliers from aerial to satellite matching")
    
    for id, point_dop in zip(matched_tracked_ids, points_orthophoto):
        rematched_points[id] = dop_pixel_to_3d(dop_pixel=point_dop)
    
    return rematched_points

def main():
    
    parser = argparse.ArgumentParser(description="Estimate world point noise", add_help=True)
    parser.add_argument("--base-config-file", type=str, default="config/world_point_noise.json", help="File to read base configuration parameters from")
    parser.add_argument("--config-file", type=str, required=True, help="File to read campaign-specific configuration parameters from")
    
    args = parser.parse_args()
    
    if not isfile(args.base_config_file):
        raise ValueError("Provided base config file does not exist")

    if not isfile(args.config_file):
        raise ValueError("Provided config file does not exist")

    config: Dict = json.load(open(args.base_config_file))
    override: Dict = json.load(open(args.config_file))

    config = deep_update(config, override)
    campaign = CampaignDate(config["campaign"])
    
    data_directory = Path(config["data_directory"])
    image_directory = Path(config["image_root"], f"{campaign.value}_images")
    
    footprint_strategy = YawAlignedEnclosingRectFootprint(
        should_enforce_aspect_ratio=False, enlargement=0.
    )
    
    reference_campaign_strategy = ClosestReferenceCampaign(
        ranking_order=RankingOrder.SEASON_YEAR_STATE
    )
    
    provider = OrthoProvider(
        campaign=campaign,
        data_directory=data_directory,
        footprint_strategy=footprint_strategy,
        reference_campaign_strategy=reference_campaign_strategy,
        prioritize_nrw_tiles=True,
        pixels_per_meter=config.get("pixels_per_meter", 5),
        provide_dsm=True
    )
    
    t_start_us = int(config["start_time"])
    t_end_us = int(config["end_time"])
    image_step = config.get("image_step", 10)
    
    camera_calib_dir = Path(config["path_camera_parameters_guess"]).parent
    camera_file_basename = Path(config["path_camera_parameters_guess"]).name
    
    extrinsics_calib_dir = Path(config["path_extrinsic_parameters_guess"]).parent
    extrinsics_file_basename = Path(config["path_extrinsic_parameters_guess"]).name
    
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
    
    print(f"\nCollecting observations for campaign {campaign.value}...")
    
    _PT = json.load(open(f"output/{campaign.value}/geofeatures/PT_{campaign.value}_{config['id_string']}.json"))
    _WP = json.load(open(f"output/{campaign.value}/geofeatures/WP_{campaign.value}_{config['id_string']}.json"))
    
    WP = defaultdict(list)
    for pid, coord in _WP.items():
        WP[int(pid)].append(coord)
    
    PT = defaultdict(list)
    num_observations = defaultdict(int)
    for pid, observations in _PT.items():
        for ob in observations:
            frame_id = ob[0]
            pixel_loc = ob[1]
            entry = [int(pid), pixel_loc, num_observations[int(pid)]]
            PT[int(frame_id)].append(entry)
            num_observations[int(pid)] += 1
    
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    
    for _, sample in tqdm(enumerate(it), total=len(it)):
        
        _pis = PT[sample["timestamp"]]
        ids, ploc = [], []
        
        for row in _pis:
            if row[2] > 0:
                ids.append(row[0])
                ploc.append(row[1])
        
        img_capture: np.ndarray = sample.get("image", None)
        if img_capture is None:
            raise ValueError("Failed to load image capture")
        img_capture = clahe.apply(img_capture)
        
        orthophotos = sample.get("orthophoto", [])
        orthophoto = None
        
        assert orthophotos is not None
        
        if len(orthophotos) > 1 or len(orthophotos) == 0:
            raise ValueError("Expected exactly only one orthophoto reference")
            
        orthophoto: np.ndarray = cv2.cvtColor(orthophotos[0], cv2.COLOR_RGB2GRAY)
        orthophoto = clahe.apply(orthophoto)
        
        dsm: np.ndarray = sample.get("dsm", None)
        assert dsm is not None
        
        calib_dir = Path(config["path_camera_parameters_guess"]).parent
        camera_file_basename = Path(config["path_camera_parameters_guess"]).name
        camera_model_guess = load_camera_model(calib_dir=calib_dir, camera_file_basename=camera_file_basename)
        
        with torch.inference_mode():
            
            if len(ploc) > 0:
                
                rmp = match_back(capture=img_capture, orthophoto=orthophoto, dsm=dsm,
                    tracked_features=ploc, tracked_ids=ids, pixel_to_utm=sample["pixel_to_utm"],
                    cam_model=camera_model_guess)
                
                print("Matched back: ", len(rmp))
                
                for pid, rmpoint in rmp.items():
                    
                    if np.abs(float(rmpoint[0]) - WP[int(pid)][0][0]) > config.get("outlier_threshold", 5) \
                        or np.abs(float(rmpoint[1]) - WP[int(pid)][0][1]) > config.get("outlier_threshold", 5):
                        continue
                    
                    WP[int(pid)].append(rmpoint.astype(float).tolist())

    output_folder = join("output", "spread")
    makedirs(output_folder, exist_ok=True)
    Path(join(output_folder, f"wp_spread_{campaign.value}.json")).write_text(json.dumps(WP, indent=4))

if __name__ == "__main__":
    main()
