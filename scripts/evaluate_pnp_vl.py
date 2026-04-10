import cv2
import json
import pycolmap
import argparse
import numpy as np
import matplotlib.pyplot as plt

from glob import glob
from os import makedirs
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
from os.path import join, isfile, isdir, basename

from scar import load_image_indices, \
    to_framewise_pixel_and_world_points, create_pose_matrix, inverse_transformation, \
    full_pose_error, error_statistics, extract_timestamp

from pygeon.datasets.vgl_iterator import VGLIterator
from pygeon.enums import FrameType, CampaignDate
from pygeon.types import CameraModel, Extrinsics

DATETIME_STRING = None

POSE_ACCURACY_THRESHOLDS = [ # provided as (m, deg)
    # typical thresholds for VL within community (but not a fix standard)
    # '-> see https://www.visuallocalization.net/benchmark/
    (0.25, 2), (0.5, 5), (5, 10),
    # proposed threshold for high-altitude UAV operations
    (2, 2), (5, 5), (10, 10)
]

def is_config_valid(config: Dict):
    """check fields of config dictionary"""
    return True

def read_opencv_yaml_matrix(path: str, key: str):
    
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    
    if not fs.isOpened():
        raise IOError(f"Cannot open {path}")
    
    node = fs.getNode(key)
    
    if node.empty():
        fs.release()
        raise KeyError(f"Key '{key}' not found in {path}")
    
    mat = node.mat()
    fs.release()
    
    return mat

def load_file_from_type(config: Dict | str):
    
    if isinstance(config, str):
        return config # filename
    
    input_file_type = config["type"]
    input_filename = None
    
    if input_file_type not in ["path", "load_latest_from_prefix"]:
        raise ValueError("Unknown file input type")
    
    if input_file_type == "load_latest_from_prefix":
    
        glob_directory = config["input_directory"]
        if not isdir(glob_directory):
            raise ValueError(f"'{glob_directory}' is not a directory")
        
        prefix = config["prefix"]
        possible_files = list(glob(join(glob_directory, f"{prefix}_*.json")))
        
        if len(possible_files) == 0:
            raise ValueError(f"There is no file with prefix '{prefix}' in '{glob_directory}'")
        
        input_filename = max(possible_files, key=extract_timestamp)
    
    elif input_file_type == "path":
        input_filename = config["path"]
    
    if not isfile(input_filename):
        raise ValueError(f"'{input_filename}' does not exist")
    
    return input_filename

def show_pnp_vl_summary(rotation_errors: List[float], translation_errors: List[float],
    number_inliers: List[float], image_indices: List[int], config: Dict):    

    global DATETIME_STRING

    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    fig.canvas.manager.set_window_title("Visual localization using PnP summary")
    
    ax1: plt.Axes = axes[0]
    ax2: plt.Axes = axes[1]
    ax3: plt.Axes = axes[2]

    ax1.plot(image_indices, rotation_errors, marker="x", c="teal")
    ax1.axhline(np.mean(rotation_errors), color="r", linestyle="--", label="Mean")
    ax1.set_title("Absolute rotation error per frame")
    ax1.set_ylabel("Error (deg)")
    ax1.grid(True, axis="y")
    ax1.legend()

    ax2.plot(image_indices, translation_errors, marker="x", c="coral")
    ax2.axhline(np.mean(translation_errors), color="r", linestyle="--", label="Mean")
    ax2.set_title("Absolute translation error per frame")
    ax2.set_ylabel("Error (m)")
    ax2.grid(True, axis="y")
    ax2.legend()

    ax3.plot(image_indices, number_inliers, marker="x", c="gold")
    ax3.axhline(np.mean(number_inliers), color="r", linestyle="--", label="Mean")
    ax3.set_xticks(image_indices[::len(image_indices)//10])
    ax3.set_title("Number of inliers per frame")
    ax3.set_xlabel("Frame index")
    ax3.set_ylabel("Count")
    ax3.grid(True, axis="y")
    ax3.legend()

    for ax in [ax1, ax2, ax3]:
        offset_text = ax.yaxis.get_offset_text()
        offset_text.set_fontweight("bold")
        offset_text.set_color("red")

    fig.suptitle("Visual localization error summary")
    fig.tight_layout()
    
    if config.get("save_plots", False):
        output_prefix = config.get("output_prefix", "")
        fig.savefig(join(config["output_directory"],
            f"{output_prefix}{'_' if len(output_prefix) > 0 else ''}pnp_vl_summary_{DATETIME_STRING}.png"), dpi=300)
    
    if config.get("show_plots", True):
        fig.show()

def load_pnp_data_realistic(config: Dict, campaign: CampaignDate):
    
    ### load sequence data and check sanity ###
    
    input_folder = join("output", campaign.value, "geofeatures")
    pixel_tracks: dict = json.load(open(join(input_folder, f"PT_{campaign.value}_{config['sequence']['sequence_id']}.json")))
    world_points: dict = json.load(open(join(input_folder, f"WP_{campaign.value}_{config['sequence']['sequence_id']}.json")))

    image_indices: List = load_image_indices(pixel_coords=pixel_tracks)

    pixel_tracks_FW, world_points_fw = to_framewise_pixel_and_world_points(
        pixel_tracks, world_points, image_indices)
            
    return world_points_fw, pixel_tracks_FW, image_indices

def deep_update(base, update):
    """overwrite base config entries"""
    
    for k, v in update.items():
        if isinstance(v, dict) and k in base:
            deep_update(base[k], v)
        else:
            base[k] = v

    return base

def main():

    parser = argparse.ArgumentParser(description="SCAR data evaluation using PnP for Visual Localization", add_help=True)
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
    dataset_config: Dict[str, Any] = config["dataset"]
    
    if not is_config_valid(config=config):
        raise ValueError("Provided config file is not valid")
    
    global DATETIME_STRING
    DATETIME_STRING = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    campaign = CampaignDate(dataset_config["campaign"])
    config["output_directory"] = join("output", campaign.value, "visloc", config["output_directory"])

    if config.get("export_results", False) or config.get("save_plots", False):
        makedirs(config["output_directory"], exist_ok=True)
    
    ### read data ###

    camera_calibration_file, extrinsics_calibration_file = None, None
    
    if "camera_calibration_file" not in dataset_config:
        raise ValueError("'camera_calibration_file' not in dataset config")
    
    if "extrinsics_calibration_file" not in dataset_config:
        raise ValueError("'read_extrinsics_from_file' not in in config")
    
    camera_calibration_file = load_file_from_type(config=dataset_config["camera_calibration_file"])
    print(f"\nReading camera calibration from {camera_calibration_file}...")

    extrinsics_calibration_file = load_file_from_type(config=dataset_config["extrinsics_calibration_file"])
    print(f"\nReading extrinsics from {extrinsics_calibration_file}...")
    
    if camera_calibration_file is None or extrinsics_calibration_file is None:
        raise ValueError("Calibrations could not be loaded successfully")
    
    camera_calibration_file: str
    extrinsics_calibration_file: str
    
    camera_calibration_directory = Path(camera_calibration_file).parent \
        if "colmap_sfm" not in camera_calibration_file and not basename(camera_calibration_file).startswith("ls_") else Path("config/")
    camera_file_basename = Path(camera_calibration_file).parts[-1] \
        if "colmap_sfm" not in camera_calibration_file and not basename(camera_calibration_file).startswith("ls_") else "camera_matrix_guess_kalibr.json"
    
    extrinsics_calibration_directory = Path(extrinsics_calibration_file).parent \
        if extrinsics_calibration_file.endswith(".json") and not basename(extrinsics_calibration_file).startswith("ls_") else Path("config/")
    extrinsics_file_basename = Path(extrinsics_calibration_file).parts[-1] \
        if extrinsics_calibration_file.endswith(".json") and not basename(extrinsics_calibration_file).startswith("ls_") else "extrinsics_guess_kalibr.json"
    
    image_directory = Path(dataset_config["image_directory"]) \
        if "image_directory" in dataset_config else None
    data_directory = Path(dataset_config["data_directory"]) \
        if "data_directory" in dataset_config else None
    
    it = VGLIterator(
        campaign=campaign,
        data_directory=data_directory,
        image_directory=image_directory,
        frame=FrameType.CAMERA,
        image_indexed_naming=True,
        with_landcover=False,
        with_image=False,
        camera_calibration_directory=camera_calibration_directory,
        camera_file_basename=camera_file_basename,
        extrinsics_calibration_directory=extrinsics_calibration_directory,
        extrinsics_file_basename=extrinsics_file_basename
    )
    
    if extrinsics_calibration_file.endswith(".yaml"): # VINS
        T_bc = read_opencv_yaml_matrix(extrinsics_calibration_file, key="body_T_cam0")
        T_cb = inverse_transformation(T_bc)
        it.ext = Extrinsics(T_bc=T_bc, T_cb=T_cb)
    elif basename(extrinsics_calibration_file).startswith("ls_"):
        T_bc = np.array(json.load(open(extrinsics_calibration_file))["T_ins_cam"])
        T_cb = inverse_transformation(T_bc)
        it.ext = Extrinsics(T_bc=T_bc, T_cb=T_cb)
    
    if "colmap_sfm" in camera_calibration_file or basename(camera_calibration_file).startswith("ls_"): # COLMAP or opt. result
        cal = json.load(open(camera_calibration_file))
        cal = cal["camera_calibration"] if basename(camera_calibration_file).startswith("ls_") else cal
        K = np.array([[cal["fx"], 0., cal["cx"]], [0., cal["fy"], cal["cy"]], [0., 0., 1.]], dtype=float)
        dist = np.array([cal["k1"], cal["k2"], cal["p1"], cal["p2"]], dtype=float)
        it.cam = CameraModel(K=K, dist=dist, size=(1600, 1100))
    
    if config["sequence"]["type"] not in ["realistic"]:
        raise ValueError(f"Unknown dataset type '{config['sequence']['type']}'")
    
    world_points_fw, pixel_coords_fw, image_indices = None, None, None
    
    if config["sequence"]["type"] == "realistic":
        world_points_fw, pixel_coords_fw, image_indices = load_pnp_data_realistic(
            config=config, campaign=campaign)
    
    if not(world_points_fw is not None and pixel_coords_fw is not None and image_indices is not None):
        raise ValueError("Data could not be loaded successfully")
    
    camera = pycolmap.Camera(
        model="OPENCV",
        width=it.cam.size[0],
        height=it.cam.size[1],
        params=[
            it.cam.K[0, 0],
            it.cam.K[1, 1],
            it.cam.K[0, 2],
            it.cam.K[1, 2],
            it.cam.dist[0],
            it.cam.dist[1],
            it.cam.dist[2],
            it.cam.dist[3]
        ]
    )
    
    ransac_options = pycolmap.RANSACOptions(
        dyn_num_trials_multiplier=config["perspective_n_point"]["dyn_num_trials_multiplier"],
        confidence=config["perspective_n_point"]["confidence"],
        max_error=config["perspective_n_point"]["maximum_error"]
    )

    options = pycolmap.AbsolutePoseEstimationOptions(
        estimate_focal_length=False,
        ransac=ransac_options
    )
    
    refinement_options = pycolmap.AbsolutePoseRefinementOptions()
    refinement_options.refine_extra_params = False
    refinement_options.refine_focal_length = False

    all_num_inliers = []
    all_image_indices = []
    all_rotation_errors = []
    all_translation_errors = []
    all_rotation_error_norms = []
    all_translation_error_norms = []
    
    # for hard-coded pose-acc@(m,deg) thresholds, count number of successes
    pose_errors_within_thresholds = {k: 0 for k in POSE_ACCURACY_THRESHOLDS}
    
    err_rot: np.ndarray
    err_trans: np.ndarray
    
    rotation_err_outlier_th = np.inf
    translation_err_outlier_th = np.inf
    
    if "visual_localization" in config:
        visual_localization_config: Dict = config["visual_localization"]
        rotation_err_outlier_th = visual_localization_config.get("rotation_error_outlier_threshold_degree", np.inf)
        translation_err_outlier_th = visual_localization_config.get("translation_error_outlier_threshold", np.inf)

    num_skipped_frames = {"require_minimum_points": 0, "require_minimum_inliers": 0, "error_threshold_outlier_rejection": 0}
    
    for i, image_ts in enumerate(image_indices):
        
        if "require_minimum_points" in config["perspective_n_point"]:
            if len(world_points_fw[i]) < config["perspective_n_point"]["require_minimum_points"]:
                print(f"\nOnly {len(world_points_fw[i])} points for image {image_ts}, continue with next image...")
                num_skipped_frames["require_minimum_points"] += 1
                continue

        objectPoints = np.array(world_points_fw[i], dtype=np.float32)
        imagePoints = np.array(pixel_coords_fw[i], dtype=np.float32)
        
        result = pycolmap.estimate_and_refine_absolute_pose(
            points2D=imagePoints,
            points3D=objectPoints,
            camera=camera,
            estimation_options=options,
            refinement_options=refinement_options
        )
        
        num_inliers = result["num_inliers"]
        
        if "require_minimum_inliers" in config["perspective_n_point"]:
            if num_inliers < config["perspective_n_point"]["require_minimum_inliers"]:
                print(f"\nOnly {num_inliers} inliers used in solution for image {image_ts}, continue with next image...")
                num_skipped_frames["require_minimum_inliers"] += 1
                continue
        
        pose = result["cam_from_world"]
        P_CAM_PNP = create_pose_matrix(pose.rotation.matrix(), pose.translation)
        P_CAM_PNP = inverse_transformation(P_CAM_PNP)
        
        ### build groundtruth pose for frame ###
        
        sample = it.get_by_timestamp(ts_us=image_ts, nearest_only=True, max_dt_us=0)
        p_cam_gt = sample["pose"]
        err_rot, err_trans = full_pose_error(P1=P_CAM_PNP, P2=p_cam_gt)
        
        err_rot_norm = np.linalg.norm(err_rot)
        err_trans_norm = np.linalg.norm(err_trans)
        
        print(f"\n################ 'image_index' = {image_ts} ################")
        print(f"#inliers = {num_inliers}, rot. error = {err_rot_norm:.2f}°, trans. error = {err_trans_norm:.2f}m")
        
        # VL result based outlier rejection (reject solution on hard errors)
        if err_rot_norm < rotation_err_outlier_th and err_trans_norm < translation_err_outlier_th:
        
            all_num_inliers.append(num_inliers)
            all_rotation_errors.append(err_rot.tolist())
            all_image_indices.append(image_ts)
            all_translation_errors.append(err_trans.tolist())
            
            all_rotation_error_norms.append(err_rot_norm.item())
            all_translation_error_norms.append(err_trans_norm.item())
            
            for k in POSE_ACCURACY_THRESHOLDS:
                if err_trans_norm < k[0] and err_rot_norm < k[1]:
                    pose_errors_within_thresholds[k] += 1

        else:
            num_skipped_frames["error_threshold_outlier_rejection"] += 1
            print(f"Rejecting VL solution based on error thresholds: th(rot) = {rotation_err_outlier_th}°, th(trans) = {translation_err_outlier_th}m...")
    
    num_processed_frames = len(image_indices) - num_skipped_frames['require_minimum_points']
    print(f"\nSkipped {num_skipped_frames['require_minimum_points']} of {len(image_indices)} total frames, b/c of too few points")
    print(f"Skipped {num_skipped_frames['require_minimum_inliers']} of {num_processed_frames} processed frames, b/c of too few inliers")
    
    num_solutions = num_processed_frames - num_skipped_frames['require_minimum_inliers']
    print(f"\nRejected {num_skipped_frames['error_threshold_outlier_rejection']} of {num_solutions} solutions, b/c of too high errors")

    print(f"\nMean rotation error norm: {np.mean(all_rotation_error_norms):.4f}°")
    print(f"Mean translation error norm: {np.mean(all_translation_error_norms):.4f}m")
    
    show_pnp_vl_summary(rotation_errors=all_rotation_error_norms,
        translation_errors=all_translation_error_norms, number_inliers=all_num_inliers,
        image_indices=all_image_indices, config=config)
    
    if config.get("show_plots", True):
        plt.show()
    
    if config.get("export_results", False):
        
        rotation_error_norm_statistics = error_statistics(
            errors=np.array(all_rotation_error_norms).flatten())
        
        translation_error_norm_statistics = error_statistics(
            errors=np.array(all_translation_error_norms).flatten())
        
        pose_accuracies = {
            f"{k[0]}/{k[1]}": float(pose_errors_within_thresholds[k]) / num_processed_frames \
            for k in POSE_ACCURACY_THRESHOLDS}
        
        results = {
            "rotation_error_norm_statistics_degree": rotation_error_norm_statistics,
            "translation_error_norm_statistics": translation_error_norm_statistics,
            "rotation_errors_degree": all_rotation_errors,
            "translation_errors": all_translation_errors,
            "pose_accuracies": pose_accuracies,
            "number_inliers": all_num_inliers,
            "image_indices": all_image_indices,
            "number_skipped_frames": num_skipped_frames
        }
        
        output_prefix = config.get("output_prefix", "")
        filename = f"{output_prefix}{'_' if len(output_prefix) > 0 else ''}pnp_vl_{DATETIME_STRING}.json"
        
        with open(join(config["output_directory"], filename), "w+") as fout:
            json.dump(results, fout, indent=4)

        print(f"\nWrote PnP VL results to {join(config['output_directory'], filename)}")
    
    print("\nDONE.\n")

if __name__ == "__main__":
    main()
